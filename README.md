# Binpack

A rust crate for solving binpacking problems using Linear Programming.

## Usage

```rust
use binpack::Problem;

const PROBLEM: &str = r#"
bins:
  b1: 100
  b2: 40
  b3: 10

items:
  i1:
    quantity: 30
    affinity:
      soft:
        - weight: 1
          bins: [b1]
    antiAffinity:
      hard:
        bins: [b3]
  i2:
    quantity: 110
    affinity:
      soft:
        - weight: 2
          bins: [b1]
  i3:
    quantity: 10
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem: Problem = serde_yaml::from_str(PROBLEM)?;
    let solution = problem.solve()?;

    println!("{}", serde_yaml::to_string(&solution)?);
    assert_eq!(serde_yaml::to_string(&solution)?.trim(), SOLUTION.trim());
    Ok(())
}

const SOLUTION: &str = r#"
solution:
  i1:
    b2: 30
  i2:
    b1: 100
    b3: 10
  i3:
    b2: 10
"#;
```

## License

`binpack` is dual-licensed under either of:

* [MIT license](https://opensource.org/license/mit)
* [Apache License, Version 2.0](https://opensource.org/license/apache-2-0)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
