use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::datasource::DataItem;
use crate::model::{Checkpoint, Question, QuestionType, estimate_tokens};

/// Capability and cardinality-aware meta-routing decision across System 1 encoders.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetaRouteDecision {
    pub selected: Checkpoint,
    pub reason: &'static str,
    pub estimated_tokens: usize,
    pub max_label_cardinality: usize,
    pub script: &'static str,
}

/// Route an input state and question schema to the optimal System 1 encoder checkpoint
/// based on Unicode script, token length, label cardinality `K`, and typed decision heads.
pub fn route_auto_best(
    state_text: &str,
    questions: &BTreeMap<String, Question>,
) -> MetaRouteDecision {
    let (_script_ckpt, script) = Checkpoint::detect_script_route(state_text);
    let estimated_tokens = estimate_tokens(state_text);
    let max_label_cardinality = questions
        .values()
        .map(|q| q.render_options().len())
        .max()
        .unwrap_or(2);
    let has_typed_heads = questions
        .values()
        .any(|q| matches!(q.qtype, QuestionType::Score | QuestionType::Noul));

    let (selected, reason) = if script != "ascii_latin" {
        if matches!(script, "latin_extended" | "cyrillic")
            && estimated_tokens > 1024
            && !has_typed_heads
        {
            (Checkpoint::EuroBert, "euro_long_context_8k")
        } else {
            (Checkpoint::Multilingual, script)
        }
    } else if state_text.trim_start().starts_with('{') {
        (Checkpoint::TypedDecisions, "structured_json_payload")
    } else if estimated_tokens > 1024 {
        (Checkpoint::ModernBertZeroShot, "long_context_8k_nli")
    } else if estimated_tokens > 512 {
        if has_typed_heads {
            (Checkpoint::TypedDecisions, "medium_long_typed_1k")
        } else {
            (Checkpoint::ModernBertZeroShot, "medium_long_8k_nli")
        }
    } else if max_label_cardinality > 8 {
        (Checkpoint::English, "high_cardinality_single_pass")
    } else if !has_typed_heads && max_label_cardinality <= 5 {
        (Checkpoint::DebertaV3ZeroShot, "short_nli_disentangled_attn")
    } else {
        (Checkpoint::English, "multi_head_english_rlcd")
    };

    MetaRouteDecision {
        selected,
        reason,
        estimated_tokens,
        max_label_cardinality,
        script,
    }
}

/// Count how many items in a dataset route to each checkpoint under `route_auto_best`.
pub fn summarize_route_distribution(
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for item in items {
        let state_text = match &item.state {
            serde_json::Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
            other => std::borrow::Cow::Owned(serde_json::to_string(other).unwrap_or_default()),
        };
        let decision = route_auto_best(&state_text, questions);
        *counts
            .entry(decision.selected.repo_id().to_string())
            .or_insert(0) += 1;
    }
    counts
}
