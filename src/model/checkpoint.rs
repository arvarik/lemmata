use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use super::schema::{DecisionAnswer, DecisionResult, Question};

/// Available System 1 decision model checkpoints and comparison baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Checkpoint {
    /// ModernBERT-large backbone (421M params, 512 context). Root repo checkpoint.
    /// Best for English text classification, email triage, and guardrails.
    #[serde(rename = "convaiinnovations/laya")]
    English,

    /// mmBERT-base backbone (322M params, 1024 up to 8k context).
    /// Supports 100+ languages, ~2.2x faster inference.
    #[serde(rename = "convaiinnovations/laya-multilingual")]
    Multilingual,

    /// ModernBERT-large backbone (421M params, 1024 context).
    /// Specialized and fine-tuned for typed-decisions enterprise workflows.
    #[serde(rename = "convaiinnovations/laya-typed-decisions")]
    TypedDecisions,

    /// Laya `<0.5ms` script-aware router that inspects Unicode blocks to route
    /// non-Latin/multilingual inputs to `Multilingual` and ASCII/English to `English`.
    #[serde(rename = "convaiinnovations/laya-router")]
    AutoRoute,

    /// TypeSafe AI Jev 1.13.0 System 1 decision model profile for benchmarking/comparison.
    #[serde(rename = "typesafe/jev-1.13")]
    Jev,

    /// Zero-shot NLI encoder baseline for System 1 comparison.
    #[serde(rename = "tasksource/ModernBERT-large-nli")]
    ModernBertNli,
}

impl Checkpoint {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::English => "convaiinnovations/laya",
            Self::Multilingual => "convaiinnovations/laya-multilingual",
            Self::TypedDecisions => "convaiinnovations/laya-typed-decisions",
            Self::AutoRoute => "convaiinnovations/laya-router",
            Self::Jev => "typesafe/jev-1.13",
            Self::ModernBertNli => "tasksource/ModernBERT-large-nli",
        }
    }

    pub fn backbone(&self) -> &'static str {
        match self {
            Self::English | Self::TypedDecisions | Self::Jev => "ModernBERT-large",
            Self::Multilingual => "mmBERT-base",
            Self::AutoRoute => "ModernBERT-large + mmBERT-base",
            Self::ModernBertNli => "ModernBERT-large-NLI",
        }
    }

    pub fn param_count(&self) -> &'static str {
        match self {
            Self::English | Self::TypedDecisions | Self::Jev => "421M",
            Self::Multilingual => "322M",
            Self::AutoRoute => "421M/322M",
            Self::ModernBertNli => "395M",
        }
    }

    pub fn context_window(&self) -> usize {
        match self {
            Self::English | Self::Jev | Self::ModernBertNli => 512,
            Self::Multilingual | Self::TypedDecisions | Self::AutoRoute => 1024,
        }
    }

    pub fn head_budget(&self) -> usize {
        match self {
            Self::English | Self::Jev => 192,
            Self::Multilingual | Self::TypedDecisions | Self::AutoRoute => 256,
            Self::ModernBertNli => 128,
        }
    }

    pub fn local_subdir(&self) -> &'static str {
        match self {
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
            _ => "",
        }
    }

    /// Per-qtype temperatures `[choice, score, noul]` from `rl_agent_config.json`.
    pub fn default_temperatures(&self) -> [f64; 3] {
        match self {
            Self::English | Self::AutoRoute => {
                [1.6369030475616455, 1.2514300346374512, 1.983399510383606]
            }
            Self::Multilingual => [1.0, 1.0, 1.0],
            Self::TypedDecisions => [1.0148024559020996, 1.0374259948730469, 1.0575125217437744],
            Self::Jev => [1.45, 1.35, 1.65],
            Self::ModernBertNli => [1.85, 1.60, 2.10],
        }
    }

    /// Per-option-cardinality temperature buckets from `rl_agent_config.json`.
    pub fn default_temperature_by_options(&self) -> HashMap<String, f64> {
        match self {
            Self::English | Self::TypedDecisions | Self::AutoRoute => HashMap::from([
                ("choice:2".into(), 1.9063563346862793),
                ("choice:3-5".into(), 1.7601518630981445),
                ("choice:6-10".into(), 1.0000158548355103),
                ("choice:11+".into(), 0.10058280825614929),
                ("score:3-5".into(), 1.2514300346374512),
                ("noul:2".into(), 1.983399510383606),
            ]),
            Self::Multilingual | Self::ModernBertNli => HashMap::new(),
            Self::Jev => HashMap::from([
                ("choice:2".into(), 1.55),
                ("choice:3-5".into(), 1.45),
                ("choice:6-10".into(), 1.10),
                ("choice:11+".into(), 0.25),
                ("score:3-5".into(), 1.35),
                ("noul:2".into(), 1.65),
            ]),
        }
    }

    /// Sub-0.5ms Unicode block inspector routing non-Latin/multilingual text to `Multilingual`
    /// and English/ASCII text to `English`.
    pub fn detect_script_route(text: &str) -> (Checkpoint, &'static str) {
        let mut ext_latin = 0usize;
        let mut alpha = 0usize;
        for ch in text.chars() {
            let u = ch as u32;
            match u {
                0x4E00..=0x9FFF | 0x3400..=0x4DBF => return (Self::Multilingual, "cjk"),
                0x3040..=0x30FF => return (Self::Multilingual, "hiragana_katakana"),
                0xAC00..=0xD7AF | 0x1100..=0x11FF => return (Self::Multilingual, "hangul"),
                0x0900..=0x097F => return (Self::Multilingual, "devanagari"),
                0x0600..=0x06FF => return (Self::Multilingual, "arabic"),
                0x0400..=0x04FF => return (Self::Multilingual, "cyrillic"),
                0x1780..=0x17FF => return (Self::Multilingual, "khmer"),
                0x0E00..=0x0E7F => return (Self::Multilingual, "thai"),
                0x0590..=0x05FF => return (Self::Multilingual, "hebrew"),
                0x0370..=0x03FF => return (Self::Multilingual, "greek"),
                0x00C0..=0x024F | 0x1E00..=0x1EFF => {
                    ext_latin += 1;
                    alpha += 1;
                }
                _ if ch.is_ascii_alphabetic() => alpha += 1,
                _ => {}
            }
        }
        if ext_latin >= 2 || (alpha > 0 && (ext_latin as f64 / alpha as f64) > 0.03) {
            (Self::Multilingual, "latin_extended")
        } else {
            (Self::English, "ascii_latin")
        }
    }

    pub fn laya_checkpoints() -> &'static [Checkpoint] {
        &[Self::English, Self::Multilingual, Self::TypedDecisions]
    }

    pub fn all() -> &'static [Checkpoint] {
        &[
            Self::English,
            Self::Multilingual,
            Self::TypedDecisions,
            Self::AutoRoute,
            Self::Jev,
            Self::ModernBertNli,
        ]
    }
}

impl fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repo_id())
    }
}

impl std::str::FromStr for Checkpoint {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().trim() {
            "english" | "laya" | "convaiinnovations/laya" | "default" => Ok(Self::English),
            "multilingual"
            | "multi"
            | "laya-multilingual"
            | "convaiinnovations/laya-multilingual" => Ok(Self::Multilingual),
            "typed-decisions"
            | "typed"
            | "laya-typed-decisions"
            | "convaiinnovations/laya-typed-decisions" => Ok(Self::TypedDecisions),
            "route" | "auto" | "router" | "laya-router" | "convaiinnovations/laya-router" => {
                Ok(Self::AutoRoute)
            }
            "jev" | "jev-1.13" | "typesafe/jev" | "typesafe/jev-1.13" => Ok(Self::Jev),
            "modernbert-nli" | "nli" | "deberta-nli" | "tasksource/modernbert-large-nli" => {
                Ok(Self::ModernBertNli)
            }
            other => Err(format!(
                "Unknown checkpoint '{}'. Options: english, multilingual, typed-decisions, route, jev, modernbert-nli",
                other
            )),
        }
    }
}

/// Pluggable trait for System 1 non-autoregressive decision models.
pub trait DecisionModel: Send + Sync {
    fn model_id(&self) -> &str;
    fn checkpoint(&self) -> Checkpoint;
    fn classify(
        &self,
        state: &serde_json::Value,
        questions: &BTreeMap<String, Question>,
    ) -> anyhow::Result<DecisionResult>;

    fn classify_single(
        &self,
        state_text: &str,
        question: &Question,
    ) -> anyhow::Result<DecisionAnswer> {
        let mut map = BTreeMap::new();
        map.insert("single".to_string(), question.clone());
        let mut res = self.classify(&serde_json::Value::String(state_text.to_string()), &map)?;
        res.answers
            .remove("single")
            .ok_or_else(|| anyhow::anyhow!("Failed to extract decision answer"))
    }
}
