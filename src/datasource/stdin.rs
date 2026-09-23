use std::io::{self, BufRead};
use anyhow::Result;

use super::traits::{DataItem, DataSource};

/// Ingests data streamed from standard input.
pub struct StdinSource;

impl StdinSource {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StdinSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSource for StdinSource {
    fn name(&self) -> &str {
        "StdinSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        let stdin = io::stdin();
        let handle = stdin.lock();
        let mut items = Vec::new();

        for (idx, line) in handle.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let item = if let Ok(parsed_item) = serde_json::from_str::<DataItem>(trimmed) {
                parsed_item
            } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                super::file::FileSource::json_value_to_item(format!("stdin:{idx}"), val, "stdin")
            } else {
                DataItem::new(
                    format!("stdin:{idx}"),
                    serde_json::json!({ "text": trimmed }),
                    "stdin",
                )
            };
            items.push(item);
        }

        Ok(items)
    }
}
