use anyhow::{Context, Result};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::datasource::DataItem;
use crate::model::DecisionResult;

/// Deterministic guardrail enforcement action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardAction {
    Allow,
    Block,
    Escalate,
    Quarantine,
    Redact,
}

impl GuardAction {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::Allow => "🟢 ALLOW",
            Self::Block => "🔴 BLOCK",
            Self::Escalate => "🟡 ESCALATE",
            Self::Quarantine => "🟠 QUARANTINE",
            Self::Redact => "🟣 REDACT",
        }
    }
}

/// A single declarative policy rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    pub condition: String,
    pub action: GuardAction,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// A complete declarative policy specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailPolicy {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_action")]
    pub default_action: GuardAction,
    pub rules: Vec<PolicyRule>,
}

fn default_version() -> String {
    "1.0".to_string()
}

fn default_action() -> GuardAction {
    GuardAction::Allow
}

impl GuardrailPolicy {
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Cannot open guardrail policy file: {}", path.display()))?;
        let reader = BufReader::new(file);
        let policy: Self = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse guardrail policy in: {}", path.display()))?;
        Ok(policy)
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let policy: Self =
            serde_json::from_str(json).context("Failed to parse policy JSON string")?;
        Ok(policy)
    }
}

/// Immutable audit record emitted for each evaluated request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub audit_id: String,
    pub item_id: String,
    pub input_preview: String,
    pub input_hash: String,
    pub policy_name: String,
    pub matched_rule: Option<String>,
    pub final_action: GuardAction,
    pub target: Option<String>,
    pub reason: Option<String>,
    pub latency_ms: f64,
}

pub struct GuardrailEngine {
    policy: GuardrailPolicy,
}

impl GuardrailEngine {
    pub fn new(policy: GuardrailPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &GuardrailPolicy {
        &self.policy
    }

    /// Evaluate an input item and its Laya decision result against the guardrail policy.
    pub fn evaluate(&self, item: &DataItem, decision: &DecisionResult) -> AuditRecord {
        let mut matched_rule = None;
        let mut final_action = self.policy.default_action;
        let mut target = None;
        let mut reason = None;

        for rule in &self.policy.rules {
            if self.evaluate_condition(&rule.condition, decision) {
                matched_rule = Some(rule.id.clone());
                final_action = rule.action;
                target = rule.target.clone();
                reason = rule.reason.clone();
                break;
            }
        }

        // Generate cryptographic hash and audit ID
        let state_str = serde_json::to_string(&item.state).unwrap_or_default();
        let input_hash = format!(
            "{:016x}",
            crate::engine::metrics::fnv1a64(state_str.as_bytes())
        );
        let audit_id = format!("aud_{}", &input_hash[..12]);

        let input_preview = if final_action == GuardAction::Redact {
            "[REDACTED]".to_string()
        } else {
            item.preview_text(45)
        };

        AuditRecord {
            audit_id,
            item_id: item.id.clone(),
            input_preview,
            input_hash,
            policy_name: self.policy.name.clone(),
            matched_rule,
            final_action,
            target,
            reason,
            latency_ms: decision.latency_ms,
        }
    }

    /// Batch evaluation over multiple items and decisions.
    pub fn evaluate_batch(
        &self,
        items: &[DataItem],
        decisions: &[(String, DecisionResult)],
    ) -> Vec<AuditRecord> {
        let dec_map: BTreeMap<&str, &DecisionResult> = decisions
            .iter()
            .map(|(id, res)| (id.as_str(), res))
            .collect();

        items
            .iter()
            .filter_map(|item| {
                dec_map
                    .get(item.id.as_str())
                    .map(|dec| self.evaluate(item, dec))
            })
            .collect()
    }

    /// Internal condition evaluator supporting parenthesized clauses joined by OR / AND.
    fn evaluate_condition(&self, cond_str: &str, decision: &DecisionResult) -> bool {
        let trimmed = strip_outer_parens(cond_str);
        if trimmed.eq_ignore_ascii_case("true") || trimmed == "1" {
            return true;
        }
        if trimmed.eq_ignore_ascii_case("false") || trimmed == "0" {
            return false;
        }

        if let Some(or_clauses) = split_top_level(trimmed, " or ") {
            return or_clauses
                .into_iter()
                .any(|c| self.evaluate_condition(c, decision));
        }

        if let Some(and_clauses) = split_top_level(trimmed, " and ") {
            return and_clauses
                .into_iter()
                .all(|c| self.evaluate_condition(c, decision));
        }

        self.evaluate_single_predicate(trimmed, decision)
    }

    fn evaluate_single_predicate(&self, pred: &str, decision: &DecisionResult) -> bool {
        let pred = strip_outer_parens(pred);
        let operators = ["==", "!=", ">=", "<=", ">", "<"];
        for op in operators {
            if let Some(pos) = pred.find(op) {
                let left = pred[..pos].trim();
                let right = pred[pos + op.len()..]
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');

                let (actual_str, actual_num) = self.resolve_field(left, decision);

                return match op {
                    "==" => {
                        if let Some(s) = &actual_str {
                            s.eq_ignore_ascii_case(right)
                        } else if let (Some(n), Ok(r_num)) = (actual_num, right.parse::<f64>()) {
                            (n - r_num).abs() < 1e-4
                        } else {
                            false
                        }
                    }
                    "!=" => {
                        if let Some(s) = &actual_str {
                            !s.eq_ignore_ascii_case(right)
                        } else if let (Some(n), Ok(r_num)) = (actual_num, right.parse::<f64>()) {
                            (n - r_num).abs() >= 1e-4
                        } else {
                            true
                        }
                    }
                    ">=" => actual_num
                        .zip(right.parse::<f64>().ok())
                        .is_some_and(|(n, r)| n >= r - 1e-6),
                    "<=" => actual_num
                        .zip(right.parse::<f64>().ok())
                        .is_some_and(|(n, r)| n <= r + 1e-6),
                    ">" => actual_num
                        .zip(right.parse::<f64>().ok())
                        .is_some_and(|(n, r)| n > r),
                    "<" => actual_num
                        .zip(right.parse::<f64>().ok())
                        .is_some_and(|(n, r)| n < r),
                    _ => false,
                };
            }
        }

        false
    }

    /// Extract string or numeric value from decision result given a field selector.
    fn resolve_field(
        &self,
        field: &str,
        decision: &DecisionResult,
    ) -> (Option<String>, Option<f64>) {
        let f_lower = field.to_lowercase();

        // Global metrics
        if f_lower == "confidence" {
            let avg_conf: f64 = decision.answers.values().map(|a| a.confidence).sum::<f64>()
                / decision.answers.len().max(1) as f64;
            return (None, Some(avg_conf));
        }
        if f_lower == "act_probability" {
            let max_act: f64 = decision
                .answers
                .values()
                .filter_map(|a| a.act_probability)
                .fold(0.0, f64::max);
            return (None, Some(max_act));
        }

        // Subfield selectors: e.g. "department.choice", "urgency.score", "churn_risk.noul"
        let parts: Vec<&str> = field.split('.').collect();
        if parts.len() == 2 {
            let qid = parts[0];
            let sub = parts[1].to_lowercase();
            if let Some(ans) = decision.answers.get(qid) {
                match sub.as_str() {
                    "choice" => return (ans.choice.clone(), None),
                    "score" => return (None, ans.score),
                    "noul" => return (None, ans.noul),
                    "confidence" => return (None, Some(ans.confidence)),
                    "act_probability" => return (None, ans.act_probability),
                    _ => {}
                }
            }
        } else if let Some(ans) = decision.answers.get(field) {
            // Direct question selector
            if let Some(c) = &ans.choice {
                return (Some(c.clone()), None);
            }
            if let Some(s) = ans.score {
                return (None, Some(s));
            }
            if let Some(n) = ans.noul {
                return (None, Some(n));
            }
        }

        (None, None)
    }

    pub fn print_report(records: &[AuditRecord]) {
        println!("\n🛡️  Lemma Declarative Policy Guardrails Enforcement");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Item ID").fg(Color::Cyan),
            Cell::new("Input Snippet").fg(Color::White),
            Cell::new("Action").fg(Color::Yellow),
            Cell::new("Matched Rule").fg(Color::Blue),
            Cell::new("Routing Target").fg(Color::Magenta),
            Cell::new("Reason / Audit Details").fg(Color::DarkGrey),
        ]);

        let mut allow_count = 0;
        let mut block_count = 0;
        let mut escalate_count = 0;
        let mut quarantine_count = 0;

        for rec in records {
            let action_cell = match rec.final_action {
                GuardAction::Allow => {
                    allow_count += 1;
                    Cell::new(rec.final_action.badge()).fg(Color::Green)
                }
                GuardAction::Block => {
                    block_count += 1;
                    Cell::new(rec.final_action.badge()).fg(Color::Red)
                }
                GuardAction::Escalate => {
                    escalate_count += 1;
                    Cell::new(rec.final_action.badge()).fg(Color::Yellow)
                }
                GuardAction::Quarantine => {
                    quarantine_count += 1;
                    Cell::new(rec.final_action.badge()).fg(Color::DarkYellow)
                }
                GuardAction::Redact => Cell::new(rec.final_action.badge()).fg(Color::Magenta),
            };

            table.add_row(vec![
                Cell::new(&rec.item_id),
                Cell::new(&rec.input_preview),
                action_cell,
                Cell::new(rec.matched_rule.as_deref().unwrap_or("DEFAULT")),
                Cell::new(rec.target.as_deref().unwrap_or("-")),
                Cell::new(rec.reason.as_deref().unwrap_or("-")),
            ]);
        }

        println!("{table}\n");
        println!(
            "Summary: {} total | {} allowed | {} blocked | {} escalated | {} quarantined\n",
            records.len(),
            allow_count,
            block_count,
            escalate_count,
            quarantine_count
        );
    }
}

fn strip_outer_parens(mut s: &str) -> &str {
    loop {
        s = s.trim();
        if s.starts_with('(') && s.ends_with(')') {
            let inner = &s[1..s.len() - 1];
            let mut depth = 0i32;
            let mut in_quotes = false;
            let mut balanced = true;
            for c in inner.chars() {
                if c == '"' || c == '\'' {
                    in_quotes = !in_quotes;
                } else if !in_quotes {
                    if c == '(' {
                        depth += 1;
                    } else if c == ')' {
                        depth -= 1;
                        if depth < 0 {
                            balanced = false;
                            break;
                        }
                    }
                }
            }
            if balanced && depth == 0 {
                s = inner;
                continue;
            }
        }
        return s;
    }
}

fn split_top_level<'a>(s: &'a str, sep_lower: &str) -> Option<Vec<&'a str>> {
    let bytes = s.as_bytes();
    let sep_len = sep_lower.len();
    let mut depth = 0i32;
    let mut in_quotes = false;
    let mut parts = Vec::new();
    let mut last = 0;
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' || b == b'\'' {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if b == b'(' {
                depth += 1;
            } else if b == b')' {
                depth = (depth - 1).max(0);
            } else if depth == 0
                && i + sep_len <= bytes.len()
                && s.get(i..i + sep_len)
                    .is_some_and(|sub| sub.eq_ignore_ascii_case(sep_lower))
            {
                parts.push(s[last..i].trim());
                i += sep_len;
                last = i;
                continue;
            }
        }
        i += 1;
    }

    if parts.is_empty() {
        None
    } else {
        parts.push(s[last..].trim());
        Some(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DecisionAnswer, QuestionType};

    #[test]
    fn test_parenthesized_boolean_and_strict_inequalities() {
        let policy = GuardrailPolicy {
            name: "test".into(),
            version: "1.0".into(),
            default_action: GuardAction::Allow,
            rules: vec![PolicyRule {
                id: "R1".into(),
                description: None,
                condition: "(department.choice == \"billing\" OR churn_risk.noul > 0.70) AND urgency.score > 1.0".into(),
                action: GuardAction::Escalate,
                target: None,
                reason: None,
            }],
        };
        let engine = GuardrailEngine::new(policy);
        let item = DataItem::new("1", serde_json::json!({"text": "hi"}), "test");
        let make_ans =
            |choice: Option<&str>, score: Option<f64>, noul: Option<f64>| DecisionAnswer {
                qtype: QuestionType::Choice,
                choice: choice.map(str::to_string),
                selected_labels: None,
                score,
                noul,
                probabilities: BTreeMap::new(),
                confidence: 0.9,
                act_probability: Some(0.1),
            };
        let mut answers = BTreeMap::new();
        answers.insert("department".into(), make_ans(Some("sales"), None, None));
        answers.insert("churn_risk".into(), make_ans(None, None, Some(0.85)));
        answers.insert("urgency".into(), make_ans(None, Some(1.0), None));
        let mut dec = DecisionResult {
            model: "laya".into(),
            answers,
            input_tokens: 5,
            latency_ms: 1.0,
        };
        // urgency.score == 1.0 must NOT satisfy strict `> 1.0`
        assert_eq!(engine.evaluate(&item, &dec).matched_rule, None);
        // Raising urgency.score to 1.2 satisfies `(false OR true) AND true`
        dec.answers.get_mut("urgency").unwrap().score = Some(1.2);
        assert_eq!(
            engine.evaluate(&item, &dec).matched_rule.as_deref(),
            Some("R1")
        );
    }
}
