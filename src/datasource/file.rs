use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::traits::{DataItem, DataSource};

/// Ingests data from a single local file.
pub struct FileSource {
    path: PathBuf,
}

impl FileSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn parse_file(path: &Path) -> Result<Vec<DataItem>> {
        if !path.exists() {
            bail!("Input file does not exist: {}", path.display());
        }

        if let Ok(meta) = std::fs::metadata(path)
            && meta.len() == 0
        {
            return Ok(vec![]);
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let path_str = path.display().to_string();

        match ext.as_str() {
            "json" => Self::parse_json(path),
            "jsonl" => Self::parse_jsonl(path),
            "csv" => Self::parse_csv(path),
            "txt" | "md" | "text" => Self::parse_plain_text(path),
            _ => {
                // Try JSON first, fallback to lines
                if let Ok(items) = Self::parse_json(path) {
                    Ok(items)
                } else {
                    Self::parse_plain_text(path)
                }
            }
        }
        .with_context(|| format!("Failed to parse data file: {path_str}"))
    }

    fn parse_json(path: &Path) -> Result<Vec<DataItem>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let val: serde_json::Value = serde_json::from_reader(reader)?;
        let path_str = path.display().to_string();

        match val {
            serde_json::Value::Array(arr) => {
                let mut items = Vec::with_capacity(arr.len());
                for (i, v) in arr.into_iter().enumerate() {
                    items.push(Self::json_value_to_item(
                        format!(
                            "{}:{}",
                            path.file_stem().and_then(|s| s.to_str()).unwrap_or("item"),
                            i
                        ),
                        v,
                        &path_str,
                    ));
                }
                Ok(items)
            }
            serde_json::Value::Object(_) => {
                let id = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("item")
                    .to_string();
                Ok(vec![Self::json_value_to_item(id, val, &path_str)])
            }
            _ => bail!("JSON must be an object or array of objects"),
        }
    }

    fn parse_jsonl(path: &Path) -> Result<Vec<DataItem>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let path_str = path.display().to_string();
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("item");
        let mut items = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(val) => {
                    items.push(Self::json_value_to_item(
                        format!("{stem}:{idx}"),
                        val,
                        &path_str,
                    ));
                }
                Err(err) => {
                    eprintln!(
                        "⚠️  [line {}] Skipping malformed JSONL in {}: {}",
                        idx + 1,
                        path_str,
                        err
                    );
                }
            }
        }
        Ok(items)
    }

    fn parse_csv(path: &Path) -> Result<Vec<DataItem>> {
        let file = File::open(path)?;
        let mut rdr = csv::Reader::from_reader(file);
        let headers = rdr.headers()?.clone();
        let path_str = path.display().to_string();
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("item");

        let mut items = Vec::new();
        for (idx, result) in rdr.records().enumerate() {
            let record = result?;
            let mut map = serde_json::Map::new();
            let mut gt = BTreeMap::new();
            let mut id = format!("{stem}:{idx}");

            for (header, val) in headers.iter().zip(record.iter()) {
                let header_lower = header.to_lowercase();
                if header_lower == "id" || header_lower == "item_id" {
                    id = val.to_string();
                } else if header_lower == "target"
                    || header_lower == "label"
                    || header_lower == "ground_truth"
                {
                    gt.insert(
                        "default".to_string(),
                        serde_json::Value::String(val.to_string()),
                    );
                } else if let Some(key) = header_lower
                    .strip_prefix("gt_")
                    .or_else(|| header_lower.strip_prefix("gt:"))
                    .or_else(|| header_lower.strip_prefix("label_"))
                    .or_else(|| header_lower.strip_prefix("label:"))
                    .or_else(|| header_lower.strip_prefix("target_"))
                    .or_else(|| header_lower.strip_prefix("target:"))
                {
                    gt.insert(key.to_string(), serde_json::Value::String(val.to_string()));
                } else {
                    map.insert(
                        header.to_string(),
                        serde_json::Value::String(val.to_string()),
                    );
                }
            }

            let mut item = DataItem::new(id, serde_json::Value::Object(map), &path_str);
            if !gt.is_empty() {
                item = item.with_ground_truth(gt);
            }
            items.push(item);
        }
        Ok(items)
    }

    fn parse_plain_text(path: &Path) -> Result<Vec<DataItem>> {
        let content = std::fs::read_to_string(path)?;
        let path_str = path.display().to_string();
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("doc")
            .to_string();

        let state = serde_json::json!({
            "text": content,
            "filename": path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown")
        });

        Ok(vec![DataItem::new(id, state, path_str)])
    }

    pub fn json_value_to_item(
        default_id: String,
        val: serde_json::Value,
        source_uri: &str,
    ) -> DataItem {
        match val {
            serde_json::Value::Object(mut map) => {
                let id = map
                    .remove("id")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or(default_id);

                let gt = map
                    .remove("ground_truth")
                    .or_else(|| map.remove("target"))
                    .or_else(|| map.remove("label"))
                    .and_then(|v| {
                        if let serde_json::Value::Object(obj) = v {
                            Some(obj.into_iter().collect::<BTreeMap<_, _>>())
                        } else if let serde_json::Value::String(s) = v {
                            let mut m = BTreeMap::new();
                            m.insert("default".to_string(), serde_json::Value::String(s));
                            Some(m)
                        } else if let serde_json::Value::Number(_) | serde_json::Value::Bool(_) = v
                        {
                            let mut m = BTreeMap::new();
                            m.insert("default".to_string(), v);
                            Some(m)
                        } else {
                            None
                        }
                    });

                let mut prefixed_gt = BTreeMap::new();
                for key in map.keys().cloned().collect::<Vec<_>>() {
                    let key_lower = key.to_ascii_lowercase();
                    if let Some(qid) = key_lower
                        .strip_prefix("gt_")
                        .or_else(|| key_lower.strip_prefix("gt:"))
                        .or_else(|| key_lower.strip_prefix("label_"))
                        .or_else(|| key_lower.strip_prefix("label:"))
                        .or_else(|| key_lower.strip_prefix("target_"))
                        .or_else(|| key_lower.strip_prefix("target:"))
                    {
                        if let Some(val) = map.remove(&key) {
                            prefixed_gt.insert(qid.to_string(), val);
                        }
                    }
                }

                let mut final_gt = gt.unwrap_or_default();
                final_gt.extend(prefixed_gt);

                // If "state" is explicitly provided as a field, use that; otherwise use the whole object
                let state = if let Some(st) = map.remove("state") {
                    st
                } else {
                    serde_json::Value::Object(map)
                };

                let mut item = DataItem::new(id, state, source_uri);
                if !final_gt.is_empty() {
                    item = item.with_ground_truth(final_gt);
                }
                item
            }
            serde_json::Value::String(text) => {
                DataItem::new(default_id, serde_json::json!({ "text": text }), source_uri)
            }
            other => DataItem::new(default_id, other, source_uri),
        }
    }
}

impl DataSource for FileSource {
    fn name(&self) -> &str {
        "FileSource"
    }

    fn read_items(&self) -> Result<Vec<DataItem>> {
        Self::parse_file(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_value_to_item_with_label_variants() {
        // String label
        let val_str = serde_json::json!({
            "id": "item1",
            "text": "sample query",
            "label": "billing"
        });
        let item_str = FileSource::json_value_to_item("default".into(), val_str, "test");
        assert_eq!(item_str.id, "item1");
        assert_eq!(
            item_str
                .ground_truth
                .unwrap()
                .get("default")
                .unwrap()
                .as_str()
                .unwrap(),
            "billing"
        );

        // Numeric label
        let val_num = serde_json::json!({
            "id": "item2",
            "text": "score query",
            "label": 3
        });
        let item_num = FileSource::json_value_to_item("default".into(), val_num, "test");
        assert_eq!(
            item_num
                .ground_truth
                .unwrap()
                .get("default")
                .unwrap()
                .as_u64()
                .unwrap(),
            3
        );

        // Boolean label
        let val_bool = serde_json::json!({
            "id": "item3",
            "text": "bool query",
            "label": true
        });
        let item_bool = FileSource::json_value_to_item("default".into(), val_bool, "test");
        assert!(
            item_bool
                .ground_truth
                .unwrap()
                .get("default")
                .unwrap()
                .as_bool()
                .unwrap()
        );

        // Multi-task prefixed labels
        let val_multi = serde_json::json!({
            "id": "item4",
            "text": "multi query",
            "gt_dept": "support",
            "label_urgency": 4
        });
        let item_multi = FileSource::json_value_to_item("default".into(), val_multi, "test");
        let gt = item_multi.ground_truth.unwrap();
        assert_eq!(gt.get("dept").unwrap().as_str().unwrap(), "support");
        assert_eq!(gt.get("urgency").unwrap().as_u64().unwrap(), 4);
    }
}
