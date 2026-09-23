use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Primitive question types supported natively by Laya System 1 models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    /// Pick one label from a set of discrete candidates.
    Choice,
    /// Return an ordinal rubric value (e.g. 0 to 4).
    Score,
    /// Return calibrated probability for a boolean assertion (True/False).
    Noul,
}

/// A typed question schema definition conforming to Laya / Jev request specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    pub instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<serde_json::Value>,
}

impl Question {
    pub fn new_choice(instructions: impl Into<String>, labels: Vec<String>) -> Self {
        let mut map = serde_json::Map::new();
        for label in labels {
            map.insert(label, serde_json::Value::Null);
        }
        Self {
            qtype: QuestionType::Choice,
            instructions: instructions.into(),
            criteria: Some(serde_json::Value::Object(map)),
        }
    }

    pub fn new_score(instructions: impl Into<String>, levels: Vec<String>) -> Self {
        let arr = levels.into_iter().map(serde_json::Value::String).collect();
        Self {
            qtype: QuestionType::Score,
            instructions: instructions.into(),
            criteria: Some(serde_json::Value::Array(arr)),
        }
    }

    pub fn new_noul(instructions: impl Into<String>) -> Self {
        Self {
            qtype: QuestionType::Noul,
            instructions: instructions.into(),
            criteria: None,
        }
    }

    /// Extract rendered option texts for this question.
    pub fn render_options(&self) -> Vec<String> {
        match self.qtype {
            QuestionType::Choice => {
                if let Some(serde_json::Value::Object(map)) = &self.criteria {
                    map.iter()
                        .map(|(k, v)| {
                            if let Some(desc) = v.as_str() {
                                format!("{k}: {desc}")
                            } else {
                                k.clone()
                            }
                        })
                        .collect()
                } else if let Some(serde_json::Value::Array(arr)) = &self.criteria {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                } else {
                    vec![]
                }
            }
            QuestionType::Score => {
                if let Some(serde_json::Value::Array(arr)) = &self.criteria {
                    arr.iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let text = v.as_str().unwrap_or("");
                            format!("level {i}: {text}")
                        })
                        .collect()
                } else {
                    vec!["level 0".into(), "level 1".into(), "level 2".into()]
                }
            }
            QuestionType::Noul => vec![
                "false: no, the statement does not hold".into(),
                "true: yes, the statement holds".into(),
            ],
        }
    }

    /// Validate that the question has sufficient options and fits within the Laya head token budget.
    pub fn validate(&self, qid: &str, head_budget: usize) -> Result<Vec<String>, String> {
        let mut warnings = Vec::new();
        let options = self.render_options();

        match self.qtype {
            QuestionType::Choice => {
                if options.len() < 2 {
                    return Err(format!("Choice question '{}' must specify at least 2 candidate options", qid));
                }
            }
            QuestionType::Score => {
                if options.len() < 2 {
                    return Err(format!("Score question '{}' requires at least 2 rubric levels", qid));
                }
            }
            QuestionType::Noul => {}
        }

        let est_tokens: usize = options
            .iter()
            .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
            .sum();

        if est_tokens > head_budget {
            warnings.push(format!(
                "Question '{}' options (~{} tokens) exceed head budget ({} tokens). Truncation will degrade accuracy.",
                qid, est_tokens, head_budget
            ));
        }

        Ok(warnings)
    }
}

/// The answer returned by Laya for a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAnswer {
    #[serde(rename = "type")]
    pub qtype: QuestionType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,

    pub confidence: f64,

    pub probabilities: BTreeMap<String, f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub act_probability: Option<f64>,
}

/// The overall response for a single state input across all questions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    pub model: String,
    pub answers: BTreeMap<String, DecisionAnswer>,
    pub input_tokens: usize,
    pub latency_ms: f64,
}
