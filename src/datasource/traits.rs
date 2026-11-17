use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
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

    /// Optional non-model metadata (slice columns, leakage group keys, provenance tags).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,

    /// Origin identifier (e.g. file path, stream partition, URI).
    pub source_uri: String,
}

impl DataItem {
    pub fn new(
        id: impl Into<String>,
        state: serde_json::Value,
        source_uri: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            state,
            ground_truth: None,
            metadata: BTreeMap::new(),
            source_uri: source_uri.into(),
        }
    }

    pub fn with_ground_truth(mut self, gt: BTreeMap<String, serde_json::Value>) -> Self {
        self.ground_truth = Some(gt);
        self
    }

    pub fn with_metadata(mut self, metadata: BTreeMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Canonical model-visible text representation of a `state` JSON value.
    ///
    /// Unwraps single-key `{"text": "..."}` wrappers emitted by plain-text, CSV, and JSONL
    /// ingestion so encoders and routers see raw text instead of JSON punctuation, while
    /// serializing genuine multi-field structured objects as JSON.
    pub fn state_to_text(state: &serde_json::Value) -> Cow<'_, str> {
        match state {
            serde_json::Value::String(text) => Cow::Borrowed(text.as_str()),
            serde_json::Value::Object(map) if map.len() == 1 => {
                if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                    Cow::Borrowed(text)
                } else {
                    Cow::Owned(serde_json::to_string(state).unwrap_or_default())
                }
            }
            other => Cow::Owned(serde_json::to_string(other).unwrap_or_default()),
        }
    }

    /// Return the canonical model-visible text representation of `self.state`.
    pub fn state_text(&self) -> Cow<'_, str> {
        Self::state_to_text(&self.state)
    }

    /// Helper to convert state to a readable string for logging or preview.
    pub fn preview_text(&self, max_chars: usize) -> String {
        let s = self.state_text();
        let char_count = s.chars().count();
        if char_count > max_chars {
            let truncated: String = s.chars().take(max_chars).collect();
            format!("{}...", truncated)
        } else {
            s.into_owned()
        }
    }
}

/// Extensible trait for ingesting data into Lemmata from any backend.
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
