# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/brannondorsey/binpack/releases/tag/v0.1.0) - 2025-05-14

### Added

- *(ci)* Add release-plz action ([#1](https://github.com/brannondorsey/binpack/pull/1))

### Other

- Add inline docs ([#3](https://github.com/brannondorsey/binpack/pull/3))
- Update Cargo.toml
- Add license and README
- Re-export serde
- Add Serialize traits to all types
- Use quantity
- More renaming
- Refactor to only print coin_cbc output when debug_assertions are on
- More readability improvements
- Rename cluster/workload to bin/item
- Readability refactor
- Make solve a method
- Rename Input/Output Problem/Solution
- Split into main.rs and lib.rs
- Remove microlp feature because it is unused
- First commit after moving contents from rust-playground/cluster-binpacking
