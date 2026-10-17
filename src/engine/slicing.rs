use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::str::FromStr;

use super::classifier::LayaClassifier;
use super::evaluator::Evaluator;
use crate::datasource::DataItem;
use crate::model::{Checkpoint, Question, estimate_tokens};

/// Diagnostic slicing dimension for disaggregated evaluation in `lemmata test` and `lemmata compare`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceBy {
    /// Group items by the alphabetic prefix of `item.id` before `'_'` or `'-'` (e.g. `en`, `neg`, `multi`, `long`, `struct`).
    AutoPrefix,
    /// Group items by detected Unicode / lexical script route (`ascii_latin`, `latin_extended`, `cjk`, `cyrillic`, etc.).
    Script,
    /// Group items by subword token length bucket (`short (<=128t)`, `medium (129-512t)`, `long (513-1024t)`, `ultra (>1024t)`).
    LengthBucket,
    /// Group items by a structured payload or metadata field key.
    Field(String),
}

impl FromStr for SliceBy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err("Slice mode cannot be empty".to_string());
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "auto" | "prefix" | "auto_prefix" | "autoprefix" | "id" => Ok(Self::AutoPrefix),
            "script" | "lang" | "language" => Ok(Self::Script),
            "length" | "len" | "tokens" | "length_bucket" => Ok(Self::LengthBucket),
            _ => {
                let key = trimmed
                    .strip_prefix("field:")
                    .or_else(|| trimmed.strip_prefix("Field:"))
                    .unwrap_or(trimmed)
                    .trim();
                if key.is_empty() {
                    Err("Field slice key cannot be empty".to_string())
                } else {
                    Ok(Self::Field(key.to_string()))
                }
            }
        }
    }
}

/// Per-slice evaluation metrics for a single checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceMetricEntry {
    pub slice: String,
    pub sample_count: usize,
    pub accuracy: f64,
    pub ece: f64,
    pub avg_confidence: f64,
}

/// Per-checkpoint row in a multi-model slice accuracy matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaSliceRow {
    pub checkpoint: String,
    pub slice_accuracies: BTreeMap<String, f64>,
    pub overall_accuracy: f64,
}

fn item_text(item: &DataItem) -> String {
    match &item.state {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(map) if map.len() == 1 => map
            .get("text")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| serde_json::to_string(&item.state).unwrap_or_default()),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Extract the diagnostic slice label for a single `DataItem`.
pub fn extract_item_slice(item: &DataItem, mode: &SliceBy) -> String {
    match mode {
        SliceBy::AutoPrefix => {
            if let Some((prefix, _)) = item.id.split_once(['_', '-']) {
                let alpha: String = prefix.chars().filter(|c| c.is_ascii_alphabetic()).collect();
                if !alpha.is_empty() {
                    return alpha.to_ascii_lowercase();
                }
            }
            "all".to_string()
        }
        SliceBy::Script => Checkpoint::detect_script_route(&item_text(item))
            .1
            .to_string(),
        SliceBy::LengthBucket => {
            let tokens = estimate_tokens(&item_text(item));
            if tokens <= 128 {
                "short (<=128t)".to_string()
            } else if tokens <= 512 {
                "medium (129-512t)".to_string()
            } else if tokens <= 1024 {
                "long (513-1024t)".to_string()
            } else {
                "ultra (>1024t)".to_string()
            }
        }
        SliceBy::Field(key) => {
            let val = item
                .state
                .as_object()
                .and_then(|obj| {
                    obj.get(key)
                        .or_else(|| obj.get("metadata").and_then(|m| m.get(key)))
                })
                .or_else(|| item.ground_truth.as_ref().and_then(|gt| gt.get(key)));
            match val {
                Some(serde_json::Value::String(s)) if !s.trim().is_empty() => s.trim().to_string(),
                Some(serde_json::Value::Bool(b)) => b.to_string(),
                Some(serde_json::Value::Number(n)) => n.to_string(),
                _ => "unknown".to_string(),
            }
        }
    }
}

/// Evaluate a single classifier across all discovered data slices.
pub fn evaluate_slices(
    classifier: &LayaClassifier,
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    mode: &SliceBy,
) -> Result<Vec<SliceMetricEntry>> {
    if items.is_empty() {
        bail!("No items to evaluate for slice breakdown");
    }

    let mut grouped: BTreeMap<String, Vec<DataItem>> = BTreeMap::new();
    for item in items {
        grouped
            .entry(extract_item_slice(item, mode))
            .or_default()
            .push(item.clone());
    }

    let mut entries = Vec::with_capacity(grouped.len());
    for (slice, slice_items) in grouped {
        let eval = Evaluator::new(LayaClassifier::new(classifier.config().clone()));
        if let Ok(metrics) = eval.evaluate(&slice_items, questions) {
            entries.push(SliceMetricEntry {
                slice,
                sample_count: metrics.sample_count,
                accuracy: metrics.accuracy,
                ece: metrics.ece,
                avg_confidence: metrics.avg_confidence,
            });
        }
    }

    if entries.is_empty() {
        bail!("No ground-truth labels matched across any data slices");
    }

    Ok(entries)
}

/// Print a formatted diagnostic slice report table for a single checkpoint.
pub fn print_slice_report(entries: &[SliceMetricEntry]) {
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Slice").fg(Color::Cyan),
        Cell::new("Samples").fg(Color::White),
        Cell::new("Accuracy").fg(Color::Green),
        Cell::new("ECE").fg(Color::Yellow),
        Cell::new("Avg Confidence").fg(Color::Blue),
    ]);

    for e in entries {
        table.add_row(vec![
            Cell::new(&e.slice),
            Cell::new(e.sample_count.to_string()),
            Cell::new(format!("{:.2}%", e.accuracy * 100.0)),
            Cell::new(format!("{:.4}", e.ece)),
            Cell::new(format!("{:.4}", e.avg_confidence)),
        ]);
    }

    println!("🔎 Diagnostic Slice Breakdown:\n{table}\n");
}

/// Compute a Checkpoint × Slice accuracy matrix across multiple checkpoints.
pub fn compute_arena_slice_matrix(
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    checkpoints: &[Checkpoint],
    mode: &SliceBy,
) -> Result<(Vec<String>, Vec<ArenaSliceRow>)> {
    compute_arena_slice_matrix_with_options(
        items,
        questions,
        checkpoints,
        mode,
        crate::model::ChunkStrategy::default(),
        None,
        0,
    )
}

/// Compute a Checkpoint × Slice accuracy matrix with configurable chunking strategy, hypothesis template, and bi-encoder shortlisting.
pub fn compute_arena_slice_matrix_with_options(
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    checkpoints: &[Checkpoint],
    mode: &SliceBy,
    chunk_strategy: crate::model::ChunkStrategy,
    hypothesis_template: Option<&str>,
    shortlist_k: usize,
) -> Result<(Vec<String>, Vec<ArenaSliceRow>)> {
    if items.is_empty() {
        bail!("No items provided for arena slice matrix");
    }
    if checkpoints.is_empty() {
        bail!("At least one checkpoint is required for arena slice matrix");
    }

    let mut grouped: BTreeMap<String, Vec<DataItem>> = BTreeMap::new();
    for item in items {
        grouped
            .entry(extract_item_slice(item, mode))
            .or_default()
            .push(item.clone());
    }

    let slices: Vec<String> = grouped.keys().cloned().collect();
    let mut rows = Vec::with_capacity(checkpoints.len());

    for &ckpt in checkpoints {
        let make_cls = || {
            let mut c = LayaClassifier::with_checkpoint(ckpt)
                .with_chunk_strategy(chunk_strategy)
                .with_shortlist_k(shortlist_k);
            if let Some(tpl) = hypothesis_template {
                c = c.with_hypothesis_template(tpl);
            }
            c
        };
        let overall_accuracy = Evaluator::new(make_cls())
            .evaluate(items, questions)
            .map(|m| m.accuracy)
            .unwrap_or(0.0);

        let mut slice_accuracies = BTreeMap::new();
        for (slice, slice_items) in &grouped {
            if let Ok(m) = Evaluator::new(make_cls()).evaluate(slice_items, questions) {
                slice_accuracies.insert(slice.clone(), m.accuracy);
            }
        }

        rows.push(ArenaSliceRow {
            checkpoint: ckpt.repo_id().to_string(),
            slice_accuracies,
            overall_accuracy,
        });
    }

    Ok((slices, rows))
}

/// Print a compact Checkpoint × Slice accuracy matrix table.
pub fn print_arena_slice_matrix(slices: &[String], rows: &[ArenaSliceRow]) {
    let mut table = Table::new();
    let mut header = vec![Cell::new("Checkpoint").fg(Color::Cyan)];
    for s in slices {
        header.push(Cell::new(s).fg(Color::Yellow));
    }
    header.push(Cell::new("Overall").fg(Color::Green));
    table.set_header(header);

    for row in rows {
        let mut cells = vec![Cell::new(&row.checkpoint)];
        for s in slices {
            let cell_str = row
                .slice_accuracies
                .get(s)
                .map(|acc| format!("{:.1}%", acc * 100.0))
                .unwrap_or_else(|| "N/A".to_string());
            cells.push(Cell::new(cell_str));
        }
        cells.push(Cell::new(format!("{:.1}%", row.overall_accuracy * 100.0)));
        table.add_row(cells);
    }

    println!("🧩 Arena Per-Slice Accuracy Matrix:\n{table}\n");
}
