use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A standardized item ingested from any data source for Laya classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataItem {
    /// Unique identifier for the item.
    pub id: String,

    /// State payload presented to Laya (can be a string or a structured JSON object).
    pub state: serde_json::Value,

    /// Optional ground-truth targets (question_id -> expected label or score) for evaluation/testing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_truth: Option<BTreeMap<String, serde_json::Value>>,

    /// Origin identifier (e.g. file path, stream partition, URI).
    pub source_uri: String,
}

impl DataItem {
    pub fn new(id: impl Into<String>, state: serde_json::Value, source_uri: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            state,
            ground_truth: None,
            source_uri: source_uri.into(),
        }
    }

    pub fn with_ground_truth(mut self, gt: BTreeMap<String, serde_json::Value>) -> Self {
        self.ground_truth = Some(gt);
        self
    }

    /// Helper to convert state to a readable string for logging or preview.
    pub fn preview_text(&self, max_chars: usize) -> String {
        let s = match &self.state {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        let char_count = s.chars().count();
        if char_count > max_chars {
            let truncated: String = s.chars().take(max_chars).collect();
            format!("{}...", truncated)
        } else {
            s
        }
    }
}

/// Extensible trait for ingesting data into Lemma from any backend.
/// Custom sources (e.g. S3, Kafka, PostgreSQL, Snowflake) can implement this trait.
pub trait DataSource: Send + Sync {
    /// Human-readable identifier for the data source.
    fn name(&self) -> &str;

    /// Ingest and return all data items.
    fn read_items(&self) -> Result<Vec<DataItem>>;
}

/// In-memory data source for unit tests, synthetic benchmarks, or embedded calls.
pub struct MemorySource {
    name: String,
    items: Vec<DataItem>,
}

impl MemorySource {
    pub fn new(name: impl Into<String>, items: Vec<DataItem>) -> Self {
        Self {
            name: name.into(),
            items,
        }
    }
}

impl DataSource for MemorySource {
    fn name(&self) -> &str {
        &self.name
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        Ok(self.items.clone())
    }
}
