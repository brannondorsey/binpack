use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct Input {
    pub workloads: BTreeMap<String, WorkloadSpec>,
    pub clusters: BTreeMap<String, u32>,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    pub solution: BTreeMap<String, BTreeMap<String, u32>>,
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
