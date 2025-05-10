use std::fs::read_to_string;

use binpack::{Problem, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: <program> <input_file.yaml>");

    let buf = read_to_string(path)?;
    let input: Problem = serde_yaml::from_str(&buf)?;
    let solution = solve(input)?;

    println!("{}", serde_yaml::to_string(&solution)?);
    Ok(())
}
