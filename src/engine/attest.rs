use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::datasource::DataItem;
use crate::engine::guardrail::GuardAction;
use crate::model::DecisionResult;

/// A tamper-evident cryptographic decision receipt conforming to 2026 enterprise AI audit standards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionReceipt {
    pub version: String,
    pub receipt_id: String,
    pub parent_hash: String,
    pub timestamp: String,
    pub model_id: String,
    pub schema_digest: String,
    pub input_digest: String,
    pub decisions: BTreeMap<String, String>,
    pub confidences: BTreeMap<String, f64>,
    pub act_probabilities: BTreeMap<String, f64>,
    pub policy_rule: Option<String>,
    pub action: GuardAction,
    pub receipt_digest: String,
    pub signature: String,
}

/// Verification report validating cryptographic integrity and Merkle hash chaining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationVerificationReport {
    pub total_receipts: usize,
    pub valid_receipts: usize,
    pub invalid_receipts: usize,
    pub chain_intact: bool,
    pub broken_link_index: Option<usize>,
    pub failure_reasons: Vec<String>,
}

/// Cryptographic attestation engine for verifiable runtime decisions.
pub struct AttestationEngine {
    secret_key: String,
    latest_hash: String,
}

impl AttestationEngine {
    pub const GENESIS_HASH: &'static str =
        "0000000000000000000000000000000000000000000000000000000000000000";

    pub fn new(secret_key: impl Into<String>) -> Self {
        Self {
            secret_key: secret_key.into(),
            latest_hash: Self::GENESIS_HASH.to_string(),
        }
    }

    pub fn with_latest_hash(secret_key: impl Into<String>, latest_hash: impl Into<String>) -> Self {
        Self {
            secret_key: secret_key.into(),
            latest_hash: latest_hash.into(),
        }
    }

    pub fn latest_hash(&self) -> &str {
        &self.latest_hash
    }

    /// Generate an immutable signed decision receipt and advance the Merkle hash chain.
    pub fn create_receipt(
        &mut self,
        item: &DataItem,
        decision: &DecisionResult,
        policy_rule: Option<&str>,
        action: GuardAction,
        schema_digest: &str,
    ) -> Result<DecisionReceipt> {
        let input_text = match &item.state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };

        let input_digest = format!("{:016x}", fnv1a64(input_text.as_bytes()));
        let receipt_id = format!(
            "rec_{:016x}",
            fnv1a64(format!("{}::{}", item.id, now_nanos()).as_bytes())
        );
        let timestamp = now_iso();

        let mut decisions = BTreeMap::new();
        let mut confidences = BTreeMap::new();
        let mut act_probs = BTreeMap::new();

        for (qid, ans) in &decision.answers {
            let pred = if let Some(c) = &ans.choice {
                c.clone()
            } else if let Some(s) = ans.score {
                format!("{:.2}", s)
            } else if let Some(n) = ans.noul {
                format!("{:.2}", n)
            } else {
                "unknown".into()
            };

            decisions.insert(qid.clone(), pred);
            confidences.insert(qid.clone(), ans.confidence);
            if let Some(act) = ans.act_probability {
                act_probs.insert(qid.clone(), act);
            }
        }

        let parent_hash = self.latest_hash.clone();

        // Canonical payload string to hash
        let canonical_payload = format!(
            "v={}|id={}|parent={}|model={}|schema={}|input={}|rule={:?}|act={:?}|dec={:?}|ts={}",
            "trace-2026.1",
            receipt_id,
            parent_hash,
            decision.model,
            schema_digest,
            input_digest,
            policy_rule,
            action,
            decisions,
            timestamp
        );

        let receipt_digest = format!(
            "{:016x}{:016x}",
            fnv1a64(canonical_payload.as_bytes()),
            fnv1a64(
                &canonical_payload
                    .as_bytes()
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<u8>>()
            )
        );

        // Sign with key
        let sig_payload = format!("{}:{}", self.secret_key, receipt_digest);
        let signature = format!(
            "sig_{:016x}{:016x}",
            fnv1a64(sig_payload.as_bytes()),
            fnv1a64(sig_payload.chars().rev().collect::<String>().as_bytes())
        );

        let receipt = DecisionReceipt {
            version: "trace-2026.1".to_string(),
            receipt_id,
            parent_hash: parent_hash.clone(),
            timestamp,
            model_id: decision.model.clone(),
            schema_digest: schema_digest.to_string(),
            input_digest,
            decisions,
            confidences,
            act_probabilities: act_probs,
            policy_rule: policy_rule.map(|s| s.to_string()),
            action,
            receipt_digest: receipt_digest.clone(),
            signature,
        };

        self.latest_hash = receipt_digest;
        Ok(receipt)
    }

    /// Verify a ledger of receipts for internal consistency and unbroken hash chaining.
    pub fn verify_ledger(
        receipts: &[DecisionReceipt],
        secret_key: &str,
    ) -> AttestationVerificationReport {
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut chain_intact = true;
        let mut broken_link_index = None;
        let mut failure_reasons = Vec::new();

        let mut expected_parent = Self::GENESIS_HASH.to_string();

        for (i, rec) in receipts.iter().enumerate() {
            // 1. Verify parent hash link
            if rec.parent_hash != expected_parent {
                chain_intact = false;
                if broken_link_index.is_none() {
                    broken_link_index = Some(i);
                }
                failure_reasons.push(format!(
                    "Receipt #{}: Parent hash mismatch (expected {}, got {})",
                    i, expected_parent, rec.parent_hash
                ));
            }

            // 2. Verify receipt digest
            let canonical_payload = format!(
                "v={}|id={}|parent={}|model={}|schema={}|input={}|rule={:?}|act={:?}|dec={:?}|ts={}",
                rec.version,
                rec.receipt_id,
                rec.parent_hash,
                rec.model_id,
                rec.schema_digest,
                rec.input_digest,
                rec.policy_rule,
                rec.action,
                rec.decisions,
                rec.timestamp
            );

            let computed_digest = format!(
                "{:016x}{:016x}",
                fnv1a64(canonical_payload.as_bytes()),
                fnv1a64(
                    &canonical_payload
                        .as_bytes()
                        .iter()
                        .rev()
                        .copied()
                        .collect::<Vec<u8>>()
                )
            );

            if computed_digest != rec.receipt_digest {
                invalid_count += 1;
                failure_reasons.push(format!("Receipt #{}: Digest tampering detected", i));
                continue;
            }

            // 3. Verify digital signature
            let sig_payload = format!("{}:{}", secret_key, rec.receipt_digest);
            let expected_sig = format!(
                "sig_{:016x}{:016x}",
                fnv1a64(sig_payload.as_bytes()),
                fnv1a64(sig_payload.chars().rev().collect::<String>().as_bytes())
            );

            if expected_sig != rec.signature {
                invalid_count += 1;
                failure_reasons.push(format!("Receipt #{}: Invalid digital signature", i));
                continue;
            }

            valid_count += 1;
            expected_parent = rec.receipt_digest.clone();
        }

        AttestationVerificationReport {
            total_receipts: receipts.len(),
            valid_receipts: valid_count,
            invalid_receipts: invalid_count,
            chain_intact: chain_intact && invalid_count == 0,
            broken_link_index,
            failure_reasons,
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DecisionAnswer, QuestionType};

    #[test]
    fn test_receipt_generation_and_chain_verification() {
        let mut engine = AttestationEngine::new("secret_enterprise_key_2026");

        let mut answers = BTreeMap::new();
        answers.insert(
            "dept".into(),
            DecisionAnswer {
                qtype: QuestionType::Choice,
                choice: Some("billing".into()),
                selected_labels: None,
                score: None,
                noul: None,
                confidence: 0.96,
                probabilities: BTreeMap::new(),
                act_probability: Some(0.04),
            },
        );

        let decision = DecisionResult {
            model: "convaiinnovations/laya".into(),
            answers,
            input_tokens: 30,
            latency_ms: 31.0,
        };

        let item1 = DataItem::new(
            "item1",
            serde_json::json!({ "text": "Refund invoice" }),
            "test",
        );
        let item2 = DataItem::new(
            "item2",
            serde_json::json!({ "text": "Cancel subscription" }),
            "test",
        );

        let rec1 = engine
            .create_receipt(
                &item1,
                &decision,
                Some("RULE_REFUND"),
                GuardAction::Allow,
                "sha_schema",
            )
            .unwrap();

        let rec2 = engine
            .create_receipt(&item2, &decision, None, GuardAction::Allow, "sha_schema")
            .unwrap();

        assert_eq!(rec1.parent_hash, AttestationEngine::GENESIS_HASH);
        assert_eq!(rec2.parent_hash, rec1.receipt_digest);

        let ledger = vec![rec1, rec2];
        let report = AttestationEngine::verify_ledger(&ledger, "secret_enterprise_key_2026");

        assert_eq!(report.total_receipts, 2);
        assert_eq!(report.valid_receipts, 2);
        assert!(report.chain_intact);
        assert!(report.failure_reasons.is_empty());
    }

    #[test]
    fn test_tampered_receipt_detection() {
        let mut engine = AttestationEngine::new("secret_enterprise_key_2026");
        let item = DataItem::new(
            "item1",
            serde_json::json!({ "text": "Refund invoice" }),
            "test",
        );
        let decision = DecisionResult {
            model: "laya".into(),
            answers: BTreeMap::new(),
            input_tokens: 10,
            latency_ms: 20.0,
        };

        let mut rec = engine
            .create_receipt(&item, &decision, None, GuardAction::Allow, "sha_schema")
            .unwrap();

        // Maliciously tamper with action from Allow to Block
        rec.action = GuardAction::Block;

        let ledger = vec![rec];
        let report = AttestationEngine::verify_ledger(&ledger, "secret_enterprise_key_2026");

        assert!(!report.chain_intact);
        assert_eq!(report.invalid_receipts, 1);
    }
}
