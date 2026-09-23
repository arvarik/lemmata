use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use anyhow::{Context, Result};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};

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
        let policy: Self = serde_json::from_str(json)
            .context("Failed to parse policy JSON string")?;
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

        // Generate synthetic cryptographic hash and audit ID
        let state_str = serde_json::to_string(&item.state).unwrap_or_default();
        let input_hash = simple_hash_hex(&state_str);
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

    /// Internal condition evaluator supporting clauses joined by OR / AND.
    fn evaluate_condition(&self, cond_str: &str, decision: &DecisionResult) -> bool {
        let trimmed = cond_str.trim();
        if trimmed.eq_ignore_ascii_case("true") || trimmed == "1" {
            return true;
        }
        if trimmed.eq_ignore_ascii_case("false") || trimmed == "0" {
            return false;
        }

        // Split top-level OR clauses
        let or_clauses: Vec<&str> = trimmed.split(" OR ").collect();
        for or_c in or_clauses {
            // Split AND clauses
            let and_clauses: Vec<&str> = or_c.split(" AND ").collect();
            let mut and_match = true;
            for clause in and_clauses {
                if !self.evaluate_single_predicate(clause.trim(), decision) {
                    and_match = false;
                    break;
                }
            }
            if and_match {
                return true;
            }
        }

        false
    }

    fn evaluate_single_predicate(&self, pred: &str, decision: &DecisionResult) -> bool {
        // Operators to match in precedence order: ==, !=, >=, <=, >, <
        let operators = ["==", "!=", ">=", "<=", ">", "<"];
        for op in operators {
            if let Some(pos) = pred.find(op) {
                let left = pred[..pos].trim();
                let right = pred[pos + op.len()..].trim().trim_matches('"').trim_matches('\'');

                let (actual_str, actual_num) = self.resolve_field(left, decision);

                return match op {
                    "==" => {
                        if let Some(s) = &actual_str {
                            s.eq_ignore_ascii_case(right)
                        } else if let Some(n) = actual_num {
                            if let Ok(r_num) = right.parse::<f64>() {
                                (n - r_num).abs() < 1e-4
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    }
                    "!=" => {
                        if let Some(s) = &actual_str {
                            !s.eq_ignore_ascii_case(right)
                        } else if let Some(n) = actual_num {
                            if let Ok(r_num) = right.parse::<f64>() {
                                (n - r_num).abs() >= 1e-4
                            } else {
                                true
                            }
                        } else {
                            true
                        }
                    }
                    ">=" => {
                        if let (Some(n), Ok(r_num)) = (actual_num, right.parse::<f64>()) {
                            n >= r_num - 1e-6
                        } else {
                            false
                        }
                    }
                    "<=" => {
                        if let (Some(n), Ok(r_num)) = (actual_num, right.parse::<f64>()) {
                            n <= r_num + 1e-6
                        } else {
                            false
                        }
                    }
                    ">" => {
                        if let (Some(n), Ok(r_num)) = (actual_num, right.parse::<f64>()) {
                            n > r_num + 1e-6
                        } else {
                            false
                        }
                    }
                    "<" => {
                        if let (Some(n), Ok(r_num)) = (actual_num, right.parse::<f64>()) {
                            n < r_num - 1e-6
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
            }
        }

        false
    }

    /// Extract string or numeric value from decision result given a field selector.
    fn resolve_field(&self, field: &str, decision: &DecisionResult) -> (Option<String>, Option<f64>) {
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
                GuardAction::Redact => {
                    Cell::new(rec.final_action.badge()).fg(Color::Magenta)
                }
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

/// Lightweight deterministic 64-bit FNV-1a hex digest for audit hashing.
fn simple_hash_hex(data: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}
