use anyhow::{Result, bail};
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::datasource::DataItem;
use crate::engine::guardrail::GuardAction;
use crate::engine::metrics::{fnv1a64, format_utc_iso8601};
use crate::model::{DecisionResult, Question};

/// Receipt format this build writes and verifies: SHA-256 digests and HMAC-SHA256 signatures.
pub const RECEIPT_VERSION: &str = "trace-2026.2";
const SIGNATURE_PREFIX: &str = "hmac-sha256:";

fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

fn hmac(secret_key: &str) -> Hmac<Sha256> {
    // HMAC accepts keys of any length, so this cannot fail.
    Hmac::<Sha256>::new_from_slice(secret_key.as_bytes()).expect("HMAC takes any key length")
}

/// SHA-256 of the serialized question schema, for receipts.
pub fn schema_digest(questions: &BTreeMap<String, Question>) -> Result<String> {
    Ok(sha256_hex(serde_json::to_string(questions)?.as_bytes()))
}

/// A tamper-evident decision receipt: a SHA-256 digest, an HMAC-SHA256 signature, and a hash
/// chain link to the previous receipt.
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

impl DecisionReceipt {
    pub fn canonical_payload(&self) -> String {
        let conf_bits: BTreeMap<&str, u64> = self
            .confidences
            .iter()
            .map(|(k, v)| (k.as_str(), ((v * 10000.0).round() as i64) as u64))
            .collect();
        let act_bits: BTreeMap<&str, u64> = self
            .act_probabilities
            .iter()
            .map(|(k, v)| (k.as_str(), ((v * 10000.0).round() as i64) as u64))
            .collect();
        format!(
            "v={}|id={}|parent={}|model={}|schema={}|input={}|rule={:?}|act={:?}|dec={:?}|conf={:?}|act_p={:?}|ts={}",
            self.version,
            self.receipt_id,
            self.parent_hash,
            self.model_id,
            self.schema_digest,
            self.input_digest,
            self.policy_rule,
            self.action,
            self.decisions,
            conf_bits,
            act_bits,
            self.timestamp
        )
    }

    fn compute_digest(&self) -> String {
        sha256_hex(self.canonical_payload().as_bytes())
    }

    fn compute_signature(&self, secret_key: &str) -> String {
        let mut mac = hmac(secret_key);
        mac.update(self.receipt_digest.as_bytes());
        format!("{SIGNATURE_PREFIX}{}", to_hex(&mac.finalize().into_bytes()))
    }

    /// Constant-time check of the HMAC-SHA256 signature.
    fn signature_is_valid(&self, secret_key: &str) -> bool {
        let Some(sig) = self
            .signature
            .strip_prefix(SIGNATURE_PREFIX)
            .and_then(from_hex)
        else {
            return false;
        };
        let mut mac = hmac(secret_key);
        mac.update(self.receipt_digest.as_bytes());
        mac.verify_slice(&sig).is_ok()
    }
}

/// Verification report for receipt digests, signatures, and hash chain links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationVerificationReport {
    pub total_receipts: usize,
    pub valid_receipts: usize,
    pub invalid_receipts: usize,
    pub chain_intact: bool,
    pub broken_link_index: Option<usize>,
    pub failure_reasons: Vec<String>,
}

/// Signs decision receipts and links each one to the previous receipt.
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

    /// Generate a signed decision receipt and advance the hash chain.
    pub fn create_receipt(
        &mut self,
        item: &DataItem,
        decision: &DecisionResult,
        policy_rule: Option<&str>,
        action: GuardAction,
        schema_digest: &str,
    ) -> Result<DecisionReceipt> {
        if self.secret_key.is_empty() {
            bail!("Attestation needs a signing key. Set LEMMATA_ATTEST_KEY or pass the key flag.");
        }
        let input_text = match &item.state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };

        let input_digest = sha256_hex(input_text.as_bytes());
        let receipt_id = format!(
            "rec_{:016x}",
            fnv1a64(format!("{}::{}", item.id, now_nanos()).as_bytes())
        );

        let mut decisions = BTreeMap::new();
        let mut confidences = BTreeMap::new();
        let mut act_probabilities = BTreeMap::new();

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
                act_probabilities.insert(qid.clone(), act);
            }
        }

        let mut receipt = DecisionReceipt {
            version: RECEIPT_VERSION.to_string(),
            receipt_id,
            parent_hash: self.latest_hash.clone(),
            timestamp: format_utc_iso8601(),
            model_id: decision.model.clone(),
            schema_digest: schema_digest.to_string(),
            input_digest,
            decisions,
            confidences,
            act_probabilities,
            policy_rule: policy_rule.map(|s| s.to_string()),
            action,
            receipt_digest: String::new(),
            signature: String::new(),
        };
        receipt.receipt_digest = receipt.compute_digest();
        receipt.signature = receipt.compute_signature(&self.secret_key);

        self.latest_hash = receipt.receipt_digest.clone();
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
            if rec.version != RECEIPT_VERSION {
                invalid_count += 1;
                failure_reasons.push(format!(
                    "Receipt #{}: Unsupported receipt version {} (this build verifies {})",
                    i, rec.version, RECEIPT_VERSION
                ));
                continue;
            }

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

            if rec.compute_digest() != rec.receipt_digest {
                invalid_count += 1;
                failure_reasons.push(format!("Receipt #{}: Digest tampering detected", i));
                continue;
            }

            if !rec.signature_is_valid(secret_key) {
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

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
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
                logits: None,
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

        // Without the key, nobody can produce a signature that verifies.
        let forged = AttestationEngine::verify_ledger(&ledger, "guessed_key");
        assert_eq!(forged.valid_receipts, 0);
        assert!(!forged.chain_intact);

        let mut keyless = AttestationEngine::new("");
        assert!(
            keyless
                .create_receipt(&item1, &decision, None, GuardAction::Allow, "sha_schema")
                .is_err()
        );
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
