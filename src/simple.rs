use good_lp::{Solution, SolverModel, constraint, default_solver, variables};

fn main() {
    // 1) Declare all assignment variables as non‑negative integers
    variables! {
        vars:
            // w1 → c1, c2, c3
            0 <= x_w1_c1 (integer);
            0 <= x_w1_c2 (integer);
            0 <= x_w1_c3 (integer);
            // w2 → c1, c2, c3
            0 <= x_w2_c1 (integer);
            0 <= x_w2_c2 (integer);
            0 <= x_w2_c3 (integer);
            // w3 → c1, c2, c3
            0 <= x_w3_c1 (integer);
            0 <= x_w3_c2 (integer);
            0 <= x_w3_c3 (integer);
    }

    // 2) Build and solve the MILP
    //    Objective: maximize 1*x_w1_c1 + 2*x_w2_c1 (soft affinities)
    let solution = vars
        .maximise(1 * x_w1_c1 + 2 * x_w2_c1)
        .using(default_solver)
        // replica counts
        .with(constraint!(x_w1_c1 + x_w1_c2 + x_w1_c3 == 30))
        .with(constraint!(x_w2_c1 + x_w2_c2 + x_w2_c3 == 110))
        .with(constraint!(x_w3_c1 + x_w3_c2 + x_w3_c3 == 10))
        // hard anti‑affinity: w1 must not go to c3
        .with(constraint!(x_w1_c3 == 0))
        // cluster capacities
        .with(constraint!(x_w1_c1 + x_w2_c1 + x_w3_c1 <= 100))
        .with(constraint!(x_w1_c2 + x_w2_c2 + x_w3_c2 <= 40))
        .with(constraint!(x_w1_c3 + x_w2_c3 + x_w3_c3 <= 10))
        .solve()
        .unwrap();

    // 3) Print the assignment
    println!("w1 → c1: {}", solution.value(x_w1_c1));
    println!("w1 → c2: {}", solution.value(x_w1_c2));
    println!("w1 → c3: {}", solution.value(x_w1_c3));

    println!("w2 → c1: {}", solution.value(x_w2_c1));
    println!("w2 → c2: {}", solution.value(x_w2_c2));
    println!("w2 → c3: {}", solution.value(x_w2_c3));

    println!("w3 → c1: {}", solution.value(x_w3_c1));
    println!("w3 → c2: {}", solution.value(x_w3_c2));
    println!("w3 → c3: {}", solution.value(x_w3_c3));
}
