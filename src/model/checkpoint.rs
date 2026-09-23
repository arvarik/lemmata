use serde::{Deserialize, Serialize};
use std::fmt;

/// Available Laya System 1 decision model checkpoints hosted on Hugging Face Hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
}

impl Checkpoint {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::English => "convaiinnovations/laya",
            Self::Multilingual => "convaiinnovations/laya-multilingual",
            Self::TypedDecisions => "convaiinnovations/laya-typed-decisions",
        }
    }

    pub fn backbone(&self) -> &'static str {
        match self {
            Self::English | Self::TypedDecisions => "ModernBERT-large",
            Self::Multilingual => "mmBERT-base",
        }
    }

    pub fn param_count(&self) -> &'static str {
        match self {
            Self::English | Self::TypedDecisions => "421M",
            Self::Multilingual => "322M",
        }
    }

    pub fn context_window(&self) -> usize {
        match self {
            Self::English => 512,
            Self::Multilingual | Self::TypedDecisions => 1024,
        }
    }

    pub fn head_budget(&self) -> usize {
        match self {
            Self::English => 192,
            Self::Multilingual | Self::TypedDecisions => 256,
        }
    }

    pub fn all() -> &'static [Checkpoint] {
        &[Self::English, Self::Multilingual, Self::TypedDecisions]
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
        match s.to_lowercase().as_str() {
            "english" | "laya" | "convaiinnovations/laya" | "default" => Ok(Self::English),
            "multilingual" | "multi" | "laya-multilingual" | "convaiinnovations/laya-multilingual" => {
                Ok(Self::Multilingual)
            }
            "typed-decisions" | "typed" | "laya-typed-decisions"
            | "convaiinnovations/laya-typed-decisions" => Ok(Self::TypedDecisions),
            other => Err(format!(
                "Unknown checkpoint '{}'. Options: english, multilingual, typed-decisions",
                other
            )),
        }
    }
}
