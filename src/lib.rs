use good_lp::Solution as LpSolution;
use good_lp::solvers::coin_cbc::coin_cbc;
use good_lp::{
    Expression, ProblemVariables, SolverModel, Variable, constraint, variable, variables,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct Problem {
    pub workloads: BTreeMap<String, WorkloadSpec>,
    pub clusters: BTreeMap<String, u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Solution {
    pub solution: BTreeMap<String, BTreeMap<String, u32>>,
}

#[derive(Debug, Deserialize)]
pub struct WorkloadSpec {
    pub replicas: u32,
    #[serde(rename = "groupSize")]
    pub group_size: Option<u32>,
    pub affinity: Option<Affinity>,
    #[serde(rename = "antiAffinity")]
    pub anti_affinity: Option<AntiAffinity>,
}

#[derive(Debug, Deserialize)]
pub struct Affinity {
    pub soft: Option<Vec<SoftRequirement>>,
    pub hard: Option<HardRequirement>,
}

#[derive(Debug, Deserialize)]
pub struct AntiAffinity {
    pub soft: Option<Vec<SoftRequirement>>,
    pub hard: Option<HardRequirement>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SoftRequirement {
    #[serde(default = "default_weight")]
    pub weight: f64,
    pub clusters: Vec<String>,
}
fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct HardRequirement {
    pub clusters: Vec<String>,
}

// TODO: Manually validate replica counts are multiples of group sizes

// In this MILP model, we handle divisibility constraints (e.g., groupSize) by using
// auxiliary variables. For example, to enforce that a replica count must be divisible by
// a group size, we create a "complete_groups" variable and constrain:
//   replica_count = group_size * complete_groups
// This forces replica_count to be a multiple of group_size, since complete_groups
// is an integer variable.
//
// Note: We can't directly express "replica_count % group_size == 0" in linear programming
// because modulo operations are not linear constraints. LP constraints must be expressible
// as linear equations or inequalities (ax + by + cz <= d). Modulo and integer division are
// non-linear operations, so we must reformulate them using auxiliary variables and linear
// relationships like we've done here.

impl Problem {
    pub fn solve(&self) -> Result<Solution, Box<dyn std::error::Error>> {
        let workloads = &self.workloads;
        let clusters = &self.clusters;

        // Create all variables, and LUTs of type (workload, cluster) → Variable
        let (variables, replicas_map, group_count_map) = init_variables(workloads, clusters);

        // Collect affinity and anti-affinity coefficients
        let mut soft_requirement_weights = BTreeMap::new();

        let affinity_coeff = 1.0;
        let anti_affinity_coeff = -1.0;
        // Process affinity (positive weights)
        process_soft_requirements(
            workloads,
            clusters,
            |spec| spec.affinity.as_ref().map(|aff| &aff.soft),
            affinity_coeff,
            &mut soft_requirement_weights,
        );

        // Process anti-affinity (negative weights)
        process_soft_requirements(
            workloads,
            clusters,
            |spec| spec.anti_affinity.as_ref().map(|anti| &anti.soft),
            anti_affinity_coeff,
            &mut soft_requirement_weights,
        );

        // Build objective function
        let objective = create_objective_function(&replicas_map, &soft_requirement_weights);

        // TODO: Try replacing coin_cbc with high (microlp couldn't solve some of our test cases)
        let model = variables.maximise(objective).using(coin_cbc);

        // Add constraints
        #[rustfmt::skip]
        let model =
            constrain_replica_counts_must_equal_desired_sizes(model, workloads, clusters, &replicas_map);
        let model = constrain_hard_placement_rules(model, workloads, clusters, &replicas_map);
        let model = constrain_cluster_capacities(model, workloads, clusters, &replicas_map);
        let model =
            constrain_group_sizes(model, workloads, clusters, &replicas_map, &group_count_map);

        // Solve
        let solution = model.solve()?;

        // Convert the solver's solution into our final workload assignment map
        let solution_map =
            create_workload_assignments(&solution, workloads, clusters, &replicas_map);

        Ok(Solution {
            solution: solution_map,
        })
    }
}

type ClusterWorkloadToVariableMap = BTreeMap<(String, String), Variable>;

fn init_variables(
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
) -> (
    ProblemVariables,
    ClusterWorkloadToVariableMap,
    ClusterWorkloadToVariableMap,
) {
    let mut problem_vars = variables!();
    let mut replicas_map = BTreeMap::new();
    let mut group_count_map = BTreeMap::new();

    // Create all variables upfront
    for (workload, spec) in workloads.iter() {
        for cluster in clusters.keys() {
            let key = (workload.clone(), cluster.clone());
            // Replica count variable - total replicas of this workload in this cluster
            let replica_count = problem_vars.add(variable().integer().min(0));
            replicas_map.insert(key.clone(), replica_count);

            // If this workload has a group size, also create a "complete groups" variable
            // This auxiliary variable represents how many complete groups are placed in this cluster
            if spec.group_size.is_some() && spec.group_size.unwrap() > 0 {
                let complete_groups = problem_vars.add(variable().integer().min(0));
                group_count_map.insert(key, complete_groups);
            }
        }
    }

    (problem_vars, replicas_map, group_count_map)
}

fn process_soft_requirements(
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
    get_reqs: fn(&WorkloadSpec) -> Option<&Option<Vec<SoftRequirement>>>,
    weight_factor: f64,
    obj_coeffs: &mut BTreeMap<(String, String), f64>,
) {
    workloads
        .iter()
        .filter_map(|(w, spec)| {
            get_reqs(spec)
                .and_then(|soft_opt| soft_opt.as_ref())
                .map(|soft_reqs| (w, soft_reqs))
        })
        .flat_map(|(w, soft_reqs)| {
            soft_reqs
                .iter()
                .flat_map(|soft| {
                    soft.clusters
                        .iter()
                        .filter(|c| clusters.contains_key(*c))
                        .map(move |c| (w.clone(), c.clone(), soft.weight * weight_factor))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .for_each(|(w, c, weight)| {
            *obj_coeffs.entry((w.clone(), c.clone())).or_insert(0.0) += weight;
        });
}

fn create_objective_function(
    replicas_map: &ClusterWorkloadToVariableMap,
    soft_requirement_weights: &BTreeMap<(String, String), f64>,
) -> Expression {
    soft_requirement_weights.iter().fold(
        Expression::from(0.0),
        |sum, ((workload, cluster), &soft_requirement_weight)| {
            let key = (workload.clone(), cluster.clone());
            let replica_count_var = replicas_map[&key];

            sum + replica_count_var * soft_requirement_weight
        },
    )
}

/// Add constraints that replica counts must equal desired sizes to the model
fn constrain_replica_counts_must_equal_desired_sizes<Model: SolverModel>(
    model: Model,
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
) -> Model {
    workloads.iter().fold(model, |m, (w, spec)| {
        let zero = Expression::from(0.0);
        let total_replicas_placed = clusters
            .keys()
            .map(|c| replicas_map[&(w.clone(), c.clone())])
            .fold(zero, |sum, replica_count| sum + replica_count);

        let required_replicas = spec.replicas as f64;
        let constraint = total_replicas_placed.eq(required_replicas);
        m.with(constraint)
    })
}

/// Add cluster capacity constraints to the model
fn constrain_cluster_capacities<Model: SolverModel>(
    model: Model,
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
) -> Model {
    clusters.iter().fold(model, |m, (c, &cap)| {
        let zero = Expression::from(0.0);
        let lhs = workloads
            .keys()
            .map(|w| replicas_map[&(w.clone(), c.clone())])
            .fold(zero, |sum, v| sum + v);
        m.with(lhs.leq(cap as f64))
    })
}

/// Add hard placement rules like affinity and anti-affinity to the model
fn constrain_hard_placement_rules<Model: SolverModel>(
    model: Model,
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
) -> Model {
    workloads.iter().fold(model, |m, (w, spec)| {
        let model =
            if let Some(valid_clusters) = get_valid_clusters_based_on_affinity(spec, clusters) {
                constrain_workload_to_clusters(m, w, &valid_clusters, clusters, replicas_map)
            } else {
                m
            };

        if let Some(forbidden_clusters) = get_valid_clusters_based_on_anti_affinity(spec, clusters)
        {
            constrain_workload_from_clusters(model, w, &forbidden_clusters, replicas_map)
        } else {
            model
        }
    })
}

fn get_valid_clusters_based_on_affinity(
    spec: &WorkloadSpec,
    clusters: &BTreeMap<String, u32>,
) -> Option<Vec<String>> {
    spec.affinity
        .as_ref()
        .and_then(|aff| aff.hard.as_ref())
        .map(|hard| {
            hard.clusters
                .iter()
                .filter(|c| clusters.contains_key(*c))
                .cloned()
                .collect()
        })
        .filter(|valid: &Vec<String>| !valid.is_empty())
}

fn get_valid_clusters_based_on_anti_affinity(
    spec: &WorkloadSpec,
    clusters: &BTreeMap<String, u32>,
) -> Option<Vec<String>> {
    spec.anti_affinity
        .as_ref()
        .and_then(|anti| anti.hard.as_ref())
        .map(|hard| {
            hard.clusters
                .iter()
                .filter(|c| clusters.contains_key(*c))
                .cloned()
                .collect()
        })
}

fn constrain_workload_to_clusters<Model: SolverModel>(
    model: Model,
    workload: &str,
    valid_clusters: &[String],
    all_clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
) -> Model {
    all_clusters.keys().fold(model, |m, c| {
        if !valid_clusters.contains(c) {
            let v = replicas_map[&(workload.to_owned(), c.to_owned())];
            m.with(constraint!(v == 0.0))
        } else {
            m
        }
    })
}

fn constrain_workload_from_clusters<Model: SolverModel>(
    model: Model,
    workload: &str,
    forbidden_clusters: &[String],
    replicas_map: &ClusterWorkloadToVariableMap,
) -> Model {
    forbidden_clusters.iter().fold(model, |m, c| {
        let v = replicas_map[&(workload.to_owned(), c.to_owned())];
        m.with(constraint!(v == 0.0))
    })
}

fn constrain_group_sizes<Model: SolverModel>(
    model: Model,
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
    group_count_map: &ClusterWorkloadToVariableMap,
) -> Model {
    workloads.iter().fold(model, |m, (w, spec)| {
        if let Some(group_size) = spec.group_size {
            if group_size > 0 {
                return clusters.keys().fold(m, |m2, c| {
                    let key = (w.clone(), c.clone());
                    let replica_var = replicas_map[&key];

                    // Get the corresponding group count variable (represents number of complete groups)
                    if let Some(&complete_groups_var) = group_count_map.get(&key) {
                        // Add divisibility constraint:
                        // replica_count = group_size * complete_groups
                        // This enforces that replica_count must be a multiple of group_size
                        //
                        // We can't use a constraint like "replica_var % group_size == 0" because
                        // modulo operations are not linear and can't be directly expressed in LP.
                        // Instead, we use this auxiliary variable approach which accomplishes the
                        // same mathematical requirement.
                        m2.with(constraint!(
                            replica_var == complete_groups_var * (group_size as f32)
                        ))
                    } else {
                        m2
                    }
                });
            }
        }
        m
    })
}

/// Create a map of workload assignments from the solver's solution
fn create_workload_assignments(
    solution: &impl LpSolution,
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
) -> BTreeMap<String, BTreeMap<String, u32>> {
    workloads
        .keys()
        .filter_map(|workload| {
            let cluster_assignments =
                get_cluster_assignments(solution, workload, clusters, replicas_map);

            // Only include workloads that have at least one assignment
            (!cluster_assignments.is_empty()).then_some((workload.clone(), cluster_assignments))
        })
        .collect()
}

/// Get the replica assignments for a workload across all clusters
fn get_cluster_assignments(
    solution: &impl LpSolution,
    workload: &str,
    clusters: &BTreeMap<String, u32>,
    replicas_map: &ClusterWorkloadToVariableMap,
) -> BTreeMap<String, u32> {
    clusters
        .keys()
        .filter_map(|cluster| {
            let key = (workload.to_string(), cluster.clone());
            let replica_count = solution.value(replicas_map[&key]).round() as u32;
            let cluster = key.1;

            // Only include clusters with at least one replica
            (replica_count > 0).then_some((cluster, replica_count))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{read_dir, read_to_string};
    use std::path::Path;

    // Helper function to run a test from a test file
    fn run_test_file(test_file: &Path) {
        println!("Running test for file: {:?}", test_file);

        let failure_message = format!("Failed to read test file: {}", test_file.display());
        let yaml_content = read_to_string(test_file).expect(&failure_message);

        // Split the file content at the "solution:" marker to separate input and expected output
        let parts: Vec<&str> = yaml_content.split("solution:").collect();

        // Parse the input part
        let failure_message = format!("Failed to parse input YAML: {}", test_file.display());
        let input_yaml = parts.first().expect("No input found in test file").trim();
        let input: Problem = serde_yaml::from_str(input_yaml).expect(&failure_message);

        let failure_message = format!("Failed to parse expected YAML: {}", test_file.display());
        let expected_yaml = format!("solution:{}", parts.get(1).expect(&failure_message));

        // Run the solver
        let failure_message = format!("Failed to solve test file: {}", test_file.display());
        let solution = input.solve().expect(&failure_message);
        let received_solution = serde_yaml::to_string(&solution).expect(&failure_message);

        // Compare the output (normalizing by parsing and re-serializing the expected)
        let failure_message = format!("Failed to parse expected YAML: {}", test_file.display());
        let expected_unnormalized: Solution =
            serde_yaml::from_str(&expected_yaml).expect(&failure_message);
        let failure_message = format!("Failed to normalize expected YAML: {}", test_file.display());
        let expected_solution =
            serde_yaml::to_string(&expected_unnormalized).expect(&failure_message);

        println!("expected: {}", expected_solution);
        println!("received: {}", received_solution);

        assert_eq!(
            expected_solution.trim(),
            received_solution.trim(),
            "{}",
            test_file.display()
        );
    }

    #[test]
    fn run_all_test_files() {
        // Read all files from the test_data directory
        let test_data_dir = Path::new("test_data");
        let mut entries: Vec<_> = read_dir(test_data_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.is_file() && path.extension().map(|ext| ext == "yaml").unwrap_or(false)
            })
            .collect();

        // Sort paths lexically by filename
        entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        // Process each file in sorted order
        for path in entries {
            run_test_file(&path);
        }
    }
}
