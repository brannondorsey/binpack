use std::fs::read_to_string;

use binpack::Problem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: <program> <input_file.yaml>");

    let buf = read_to_string(path)?;
    let problem: Problem = serde_yaml::from_str(&buf)?;
    let solution = problem.solve()?;

    println!("{}", serde_yaml::to_string(&solution)?);
    Ok(())
}
