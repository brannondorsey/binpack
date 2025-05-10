use good_lp::Solution as LpSolution;
use good_lp::solvers::coin_cbc::coin_cbc;
use good_lp::{
    Expression, ProblemVariables, SolverModel, Variable, constraint, variable, variables,
};
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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

pub fn solve(input: Problem) -> Result<Solution, Box<dyn std::error::Error>> {
    let workloads = &input.workloads;
    let clusters = &input.clusters;
    let init_zero = Expression::from(0.0);

    // Store each (workload, cluster) → Variable
    let (replica_count_vars, workload_cluster_to_replica_count_var_map, group_count_vars) =
        init_replica_count_vars(workloads, clusters);

    // Collect affinity and anti-affinity coefficients
    let mut soft_requirement_weights = BTreeMap::new();

    // Process affinity (positive weights)
    process_soft_requirements(
        workloads,
        clusters,
        |spec| spec.affinity.as_ref().map(|aff| &aff.soft),
        1.0,
        &mut soft_requirement_weights,
    );

    // Process anti-affinity (negative weights)
    process_soft_requirements(
        workloads,
        clusters,
        |spec| spec.anti_affinity.as_ref().map(|anti| &anti.soft),
        -1.0,
        &mut soft_requirement_weights,
    );

    // Build objective function
    let objective = soft_requirement_weights.iter().fold(
        init_zero.clone(),
        |sum, ((workload, cluster), &weight)| {
            let replica_count_var =
                workload_cluster_to_replica_count_var_map[&(workload.clone(), cluster.clone())];

            sum + weight * replica_count_var
        },
    );

    // Build and solve the MILP with pure‑Rust microlp solver
    // TODO: Try high as well (microlp couldn't solve some of our test cases)
    let mut model = replica_count_vars.maximise(objective).using(coin_cbc);

    // Add replica‐count equality for each workload
    model = workloads.iter().fold(model, |m, (w, spec)| {
        let total_replicas_placed = clusters
            .keys()
            .map(|c| workload_cluster_to_replica_count_var_map[&(w.clone(), c.clone())])
            .fold(init_zero.clone(), |sum, replica_count| sum + replica_count);

        let required_replicas = spec.replicas as f64;
        let constraint = total_replicas_placed.eq(required_replicas);
        m.with(constraint)
    });

    // Enforce hard affinity constraints
    model = workloads.iter().fold(model, |m, (w, spec)| {
        if let Some(aff) = &spec.affinity {
            if let Some(hard) = &aff.hard {
                let valid_clusters: Vec<_> = hard
                    .clusters
                    .iter()
                    .filter(|c| clusters.contains_key(*c))
                    .cloned()
                    .collect();

                if !valid_clusters.is_empty() {
                    // If there are hard affinity constraints, workload must be placed only on those clusters
                    return clusters.keys().fold(m, |m2, c| {
                        if !valid_clusters.contains(c) {
                            // Force var == 0 for clusters not in the hard affinity list
                            let v =
                                workload_cluster_to_replica_count_var_map[&(w.clone(), c.clone())];
                            m2.with(constraint!(v == 0.0))
                        } else {
                            m2
                        }
                    });
                }
            }
        }
        m
    });

    // Enforce hard anti‑affinity (var == 0)
    model = workloads.iter().fold(model, |m, (w, spec)| {
        if let Some(anti) = &spec.anti_affinity {
            if let Some(hard) = &anti.hard {
                return hard.clusters.iter().fold(m, |m2, c| {
                    if clusters.contains_key(c) {
                        let v = workload_cluster_to_replica_count_var_map[&(w.clone(), c.clone())];
                        m2.with(constraint!(v == 0.0))
                    } else {
                        m2
                    }
                });
            }
        }
        m
    });

    // Enforce cluster capacities
    model = clusters.iter().fold(model, |m, (c, &cap)| {
        let lhs = workloads
            .keys()
            .map(|w| workload_cluster_to_replica_count_var_map[&(w.clone(), c.clone())])
            .fold(init_zero.clone(), |sum, v| sum + v);
        m.with(lhs.leq(cap as f64))
    });

    // Enforce group size constraints
    model = workloads.iter().fold(model, |m, (w, spec)| {
        if let Some(group_size) = spec.group_size {
            if group_size > 0 {
                return clusters.keys().fold(m, |m2, c| {
                    let replica_var =
                        workload_cluster_to_replica_count_var_map[&(w.clone(), c.clone())];

                    // Get the corresponding group count variable (represents number of complete groups)
                    if let Some(&complete_groups_var) =
                        group_count_vars.get(&(w.clone(), c.clone()))
                    {
                        // Add divisibility constraint:
                        // replica_count = group_size * complete_groups
                        // This enforces that replica_count must be a multiple of group_size
                        //
                        // We can't use a constraint like "replica_var % group_size == 0" because
                        // modulo operations are not linear and can't be directly expressed in LP.
                        // Instead, we use this auxiliary variable approach which accomplishes the
                        // same mathematical requirement.
                        let group_size_f = group_size as f64;

                        // Mathematical relationship: replicas = group_size * complete_groups
                        m2.with(constraint!(
                            replica_var == group_size_f * complete_groups_var
                        ))
                    } else {
                        m2
                    }
                });
            }
        }
        m
    });

    // Solve
    let solution = model.solve()?;

    // Collect and print solution as YAML
    let solution_map: BTreeMap<_, _> = workloads
        .keys()
        .filter_map(|w| {
            let assigns: BTreeMap<_, _> = clusters
                .keys()
                .filter_map(|c| {
                    let val = solution
                        .value(workload_cluster_to_replica_count_var_map[&(w.clone(), c.clone())])
                        .round() as u32;
                    (val > 0).then_some((c.clone(), val))
                })
                .collect();
            (!assigns.is_empty()).then_some((w.clone(), assigns))
        })
        .collect();

    Ok(Solution {
        solution: solution_map,
    })
}

fn init_replica_count_vars(
    workloads: &BTreeMap<String, WorkloadSpec>,
    clusters: &BTreeMap<String, u32>,
) -> (
    ProblemVariables,
    BTreeMap<(String, String), Variable>,
    BTreeMap<(String, String), Variable>, // Map for "complete groups" count variables
) {
    let mut problem_vars = variables!();
    let mut replica_count_var_map = BTreeMap::new();
    let mut complete_groups_var_map = BTreeMap::new(); // Holds variables for count of complete groups

    // Create all variables upfront
    for (workload, spec) in workloads.iter() {
        for cluster in clusters.keys() {
            // Replica count variable - total replicas of this workload in this cluster
            let replica_count = problem_vars.add(variable().integer().min(0));
            replica_count_var_map.insert((workload.clone(), cluster.clone()), replica_count);

            // If this workload has a group size, also create a "complete groups" variable
            // This auxiliary variable represents how many complete groups are placed in this cluster
            if spec.group_size.is_some() && spec.group_size.unwrap() > 0 {
                let complete_groups = problem_vars.add(variable().integer().min(0));
                complete_groups_var_map
                    .insert((workload.clone(), cluster.clone()), complete_groups);
            }
        }
    }

    (problem_vars, replica_count_var_map, complete_groups_var_map)
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
        let solution = solve(input).expect(&failure_message);
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
