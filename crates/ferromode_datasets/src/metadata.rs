//! Signal metadata types and structures

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Signal source category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSource {
    Synthetic,
    Geophysics,
    Biomedicine,
    Finance,
    Audio,
    Other,
}

/// Signal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
    pub sample_count: u32,
    pub sample_rate_hz: f64,
    pub source: SignalSource,
    pub license: String,
}

/// Metadata collection for a signal library
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataCollection {
    pub signals: Vec<SignalMetadata>,
}

impl MetadataCollection {
    pub fn new() -> Self {
        Self { signals: Vec::new() }
    }

    pub fn add_signal(&mut self, metadata: SignalMetadata) {
        self.signals.push(metadata);
    }

    pub fn get(&self, id: &str) -> Option<&SignalMetadata> {
        self.signals.iter().find(|s| s.id == id)
    }

    pub fn get_by_source(&self, source: SignalSource) -> Vec<&SignalMetadata> {
        self.signals.iter().filter(|s| s.source == source).collect()
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for MetadataCollection {
    fn default() -> Self {
        Self::new()
    }
}
