use good_lp::Solution as LpSolution;
use good_lp::solvers::coin_cbc::coin_cbc;
use good_lp::{
    Expression, ProblemVariables, SolverModel, Variable, constraint, variable, variables,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Problem {
    pub items: BTreeMap<String, ItemSpec>,
    pub bins: BTreeMap<String, u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Solution {
    pub solution: BTreeMap<String, BTreeMap<String, u32>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ItemSpec {
    pub quantity: u32,
    #[serde(rename = "groupSize")]
    pub group_size: Option<u32>,
    pub affinity: Option<Affinity>,
    #[serde(rename = "antiAffinity")]
    pub anti_affinity: Option<AntiAffinity>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Affinity {
    pub soft: Option<Vec<SoftRequirement>>,
    pub hard: Option<HardRequirement>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AntiAffinity {
    pub soft: Option<Vec<SoftRequirement>>,
    pub hard: Option<HardRequirement>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SoftRequirement {
    #[serde(default = "default_weight")]
    pub weight: f64,
    pub bins: Vec<String>,
}
fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HardRequirement {
    pub bins: Vec<String>,
}

// TODO: Manually validate quantities are multiples of group sizes
//       We could have a series of simple_validations()
impl Problem {
    pub fn solve(&self) -> Result<Solution, Box<dyn std::error::Error>> {
        let items = &self.items;
        let bins = &self.bins;

        // Create all variables, and LUTs of type (item, bin) → Variable
        let (variables, quantity_map, group_count_map) = init_variables(items, bins);

        // Collect affinity and anti-affinity coefficients
        let mut soft_requirement_weights = BTreeMap::new();

        // Process affinity (positive weights)
        process_soft_requirements(
            items,
            bins,
            |spec| spec.affinity.as_ref().map(|aff| &aff.soft),
            1.0,
            &mut soft_requirement_weights,
        );

        // Process anti-affinity (negative weights)
        process_soft_requirements(
            items,
            bins,
            |spec| spec.anti_affinity.as_ref().map(|anti| &anti.soft),
            -1.0,
            &mut soft_requirement_weights,
        );

        // Build objective function
        let objective = create_objective_function(&quantity_map, &soft_requirement_weights);
        let model = create_model(variables, objective);

        // Add constraints
        #[rustfmt::skip]
        let model =
            constrain_quantities_must_equal_desired_sizes(model, items, bins, &quantity_map);
        let model = constrain_hard_placement_rules(model, items, bins, &quantity_map);
        let model = constrain_bin_capacities(model, items, bins, &quantity_map);
        let model = constrain_group_sizes(model, items, bins, &quantity_map, &group_count_map);

        // Solve
        let solution = model.solve()?;

        // Convert the solver's solution into our final item assignment map
        let solution_map = create_item_assignments(&solution, items, bins, &quantity_map);

        Ok(Solution {
            solution: solution_map,
        })
    }
}

type BinItemToVariableMap = BTreeMap<(String, String), Variable>;

fn init_variables(
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
) -> (ProblemVariables, BinItemToVariableMap, BinItemToVariableMap) {
    let mut problem_vars = variables!();
    let mut quantity_map = BTreeMap::new();
    let mut group_count_map = BTreeMap::new();

    // Create all variables upfront
    for (item, spec) in items.iter() {
        for bin in bins.keys() {
            let key = (item.clone(), bin.clone());
            // Quantity variable - total quantity of this item in this bin
            let quantity = problem_vars.add(variable().integer().min(0));
            quantity_map.insert(key.clone(), quantity);

            // If this item has a group size, also create a "complete groups" variable
            // This auxiliary variable represents how many complete groups are placed in this bin
            if spec.group_size.is_some() && spec.group_size.unwrap() > 0 {
                let complete_groups = problem_vars.add(variable().integer().min(0));
                group_count_map.insert(key, complete_groups);
            }
        }
    }

    (problem_vars, quantity_map, group_count_map)
}

fn process_soft_requirements(
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
    get_reqs: fn(&ItemSpec) -> Option<&Option<Vec<SoftRequirement>>>,
    weight_factor: f64,
    obj_coeffs: &mut BTreeMap<(String, String), f64>,
) {
    // For each item that has soft requirements
    for (item_name, item_spec) in items.iter() {
        // Get the soft requirements (e.g. affinity or anti-affinity)
        let soft_requirements = get_reqs(item_spec)
            .and_then(|maybe_reqs| maybe_reqs.as_ref())
            .into_iter()
            .flatten();

        // Process each soft requirement
        for preference in soft_requirements {
            // For each valid bin in the requirement
            for bin_name in &preference.bins {
                // Skip if bin doesn't exist
                if !bins.contains_key(bin_name) {
                    continue;
                }
                let key = (item_name.clone(), bin_name.clone());

                // Calculate the weighted score for this item-bin pair
                // and add it to the objective coefficients
                let weighted_score = preference.weight * weight_factor;
                *obj_coeffs.entry(key).or_insert(0.0) += weighted_score;
            }
        }
    }
}

fn create_objective_function(
    quantity_map: &BinItemToVariableMap,
    soft_requirement_weights: &BTreeMap<(String, String), f64>,
) -> Expression {
    soft_requirement_weights.iter().fold(
        Expression::from(0.0),
        |sum, ((item, bin), &soft_requirement_weight)| {
            let key = (item.clone(), bin.clone());
            let quantity_var = quantity_map[&key];

            sum + quantity_var * soft_requirement_weight
        },
    )
}

/// Create a model with the given objective function
fn create_model(variables: ProblemVariables, objective: Expression) -> impl SolverModel {
    #[allow(unused_mut)]
    let mut model = variables.maximise(objective).using(coin_cbc);
    #[cfg(not(debug_assertions))]
    model.set_parameter("loglevel", "0");
    model
}

/// Add constraints that quantities must equal desired sizes to the model
fn constrain_quantities_must_equal_desired_sizes<Model: SolverModel>(
    model: Model,
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
) -> Model {
    items.iter().fold(model, |m, (item, spec)| {
        let zero = Expression::from(0.0);
        let total_quantity_placed = bins
            .keys()
            .map(|bin| quantity_map[&(item.clone(), bin.clone())])
            .fold(zero, |sum, quantity| sum + quantity);

        let required_quantity = spec.quantity as f64;
        let constraint = total_quantity_placed.eq(required_quantity);
        m.with(constraint)
    })
}

/// Add bin capacity constraints to the model
fn constrain_bin_capacities<Model: SolverModel>(
    model: Model,
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
) -> Model {
    bins.iter().fold(model, |m, (bin, &cap)| {
        let zero = Expression::from(0.0);
        let lhs = items
            .keys()
            .map(|item| quantity_map[&(item.clone(), bin.clone())])
            .fold(zero, |sum, v| sum + v);
        m.with(lhs.leq(cap as f64))
    })
}

/// Add hard placement rules like affinity and anti-affinity to the model
fn constrain_hard_placement_rules<Model: SolverModel>(
    model: Model,
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
) -> Model {
    items.iter().fold(model, |m, (w, spec)| {
        let model = if let Some(valid_bins) = get_valid_bins_based_on_affinity(spec, bins) {
            constrain_item_to_bins(m, w, &valid_bins, bins, quantity_map)
        } else {
            m
        };

        if let Some(forbidden_bins) = get_valid_bins_based_on_anti_affinity(spec, bins) {
            constrain_item_from_bins(model, w, &forbidden_bins, quantity_map)
        } else {
            model
        }
    })
}

fn get_valid_bins_based_on_affinity(
    spec: &ItemSpec,
    bins: &BTreeMap<String, u32>,
) -> Option<Vec<String>> {
    spec.affinity
        .as_ref()
        .and_then(|aff| aff.hard.as_ref())
        .map(|hard| {
            hard.bins
                .iter()
                .filter(|c| bins.contains_key(*c))
                .cloned()
                .collect()
        })
        .filter(|valid: &Vec<String>| !valid.is_empty())
}

fn get_valid_bins_based_on_anti_affinity(
    spec: &ItemSpec,
    bins: &BTreeMap<String, u32>,
) -> Option<Vec<String>> {
    spec.anti_affinity
        .as_ref()
        .and_then(|anti| anti.hard.as_ref())
        .map(|hard| {
            hard.bins
                .iter()
                .filter(|c| bins.contains_key(*c))
                .cloned()
                .collect()
        })
}

fn constrain_item_to_bins<Model: SolverModel>(
    model: Model,
    item: &str,
    valid_bins: &[String],
    all_bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
) -> Model {
    all_bins.keys().fold(model, |m, c| {
        if !valid_bins.contains(c) {
            let v = quantity_map[&(item.to_owned(), c.to_owned())];
            m.with(constraint!(v == 0.0))
        } else {
            m
        }
    })
}

fn constrain_item_from_bins<Model: SolverModel>(
    model: Model,
    item: &str,
    forbidden_bins: &[String],
    quantity_map: &BinItemToVariableMap,
) -> Model {
    forbidden_bins.iter().fold(model, |m, c| {
        let v = quantity_map[&(item.to_owned(), c.to_owned())];
        m.with(constraint!(v == 0.0))
    })
}

fn constrain_group_sizes<Model: SolverModel>(
    model: Model,
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
    group_count_map: &BinItemToVariableMap,
) -> Model {
    items.iter().fold(model, |m, (item, spec)| {
        if let Some(group_size) = spec.group_size {
            if group_size > 0 {
                return bins.keys().fold(m, |m2, bin| {
                    let key = (item.clone(), bin.clone());
                    let quantity_var = quantity_map[&key];

                    // Get the corresponding group count variable (represents number of complete groups)
                    if let Some(&complete_groups_var) = group_count_map.get(&key) {
                        // Add divisibility constraint:
                        // quantity = group_size * complete_groups
                        // This enforces that quantity must be a multiple of group_size
                        //
                        // We can't use a constraint like "quantity_var % group_size == 0" because
                        // modulo operations are not linear and can't be directly expressed in LP.
                        // Instead, we use this auxiliary variable approach which accomplishes the
                        // same mathematical requirement.
                        m2.with(constraint!(
                            quantity_var == complete_groups_var * (group_size as f32)
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

/// Create a map of item assignments from the solver's solution
fn create_item_assignments(
    solution: &impl LpSolution,
    items: &BTreeMap<String, ItemSpec>,
    bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
) -> BTreeMap<String, BTreeMap<String, u32>> {
    items
        .keys()
        .filter_map(|item| {
            let bin_assignments = get_bin_assignments(solution, item, bins, quantity_map);

            // Only include items that have at least one assignment
            (!bin_assignments.is_empty()).then_some((item.clone(), bin_assignments))
        })
        .collect()
}

/// Get the quantity assignments for an item across all bins
fn get_bin_assignments(
    solution: &impl LpSolution,
    item: &str,
    bins: &BTreeMap<String, u32>,
    quantity_map: &BinItemToVariableMap,
) -> BTreeMap<String, u32> {
    bins.keys()
        .filter_map(|bin| {
            let key = (item.to_string(), bin.clone());
            let quantity = solution.value(quantity_map[&key]).round() as u32;
            let bin = key.1;

            // Only include bins with at least one item
            (quantity > 0).then_some((bin, quantity))
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
