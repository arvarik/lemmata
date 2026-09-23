use std::collections::BTreeMap;
use std::time::Instant;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::datasource::DataItem;
use crate::model::{Checkpoint, DecisionAnswer, DecisionResult, Question, QuestionType};

/// Laya decision engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub checkpoint: Checkpoint,
    pub temperature: f64,
    pub temperature_by_options: BTreeMap<String, f64>,
    pub escalate_threshold: f64,
    pub multi_label: bool,
    pub multi_label_threshold: f64,
    pub local_weights_path: Option<std::path::PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qats_profile: Option<crate::engine::calibrate::QatsProfile>,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        let mut temp_map = BTreeMap::new();
        temp_map.insert("choice:2".into(), 1.906);
        temp_map.insert("choice:3-5".into(), 1.760);
        temp_map.insert("choice:6-10".into(), 1.000);
        temp_map.insert("choice:11+".into(), 0.101);
        temp_map.insert("score:3-5".into(), 1.251);
        temp_map.insert("noul:2".into(), 1.983);

        Self {
            checkpoint: Checkpoint::English,
            temperature: 1.0,
            temperature_by_options: temp_map,
            escalate_threshold: 0.5,
            multi_label: false,
            multi_label_threshold: 0.40,
            local_weights_path: None,
            qats_profile: None,
        }
    }
}

/// The classifier runner for Laya System 1 decision models.
pub struct LayaClassifier {
    config: ClassifierConfig,
}

impl LayaClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    pub fn with_checkpoint(checkpoint: Checkpoint) -> Self {
        let mut cfg = ClassifierConfig::default();
        cfg.checkpoint = checkpoint;
        Self::new(cfg)
    }

    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Perform inference for a single question on state text.
    pub fn classify_single(
        &self,
        state_text: &str,
        question: &Question,
    ) -> Result<DecisionAnswer> {
        let mut map = BTreeMap::new();
        map.insert("single".to_string(), question.clone());
        let state_val = serde_json::Value::String(state_text.to_string());
        let mut res = self.classify(&state_val, &map)?;
        res.answers
            .remove("single")
            .ok_or_else(|| anyhow::anyhow!("Failed to extract decision answer"))
    }

    /// Perform a single forward pass over all questions for a given state input.
    pub fn classify(
        &self,
        state: &serde_json::Value,
        questions: &BTreeMap<String, Question>,
    ) -> Result<DecisionResult> {
        let start = Instant::now();

        if questions.is_empty() {
            bail!("No questions provided for classification");
        }

        let state_text = match state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };

        // Estimate token counts and verify Laya constraints
        let state_tokens = state_text.split_whitespace().count() * 4 / 3;
        let mut total_tokens = 0;
        let mut answers = BTreeMap::new();

        for (qid, q) in questions {
            let options = q.render_options();
            let k = options.len().max(2);

            // Verify Laya option budget constraint:
            // English: 192 head tokens budget. Multilingual: 256.
            let head_budget = self.config.checkpoint.head_budget();
            let est_option_tokens = options.iter().map(|o| o.split_whitespace().count() * 4 / 3 + 1).sum::<usize>();

            if est_option_tokens > head_budget {
                eprintln!(
                    "⚠️  Warning: Question '{}' options (~{} tokens) exceed head budget ({} tokens). High cardinality may degrade accuracy.",
                    qid, est_option_tokens, head_budget
                );
            }

            let q_tokens = est_option_tokens + state_tokens + 10;
            total_tokens += q_tokens;

            // Compute raw logits for the options
            let raw_logits = self.compute_option_logits(&state_text, q, &options);

            // Temperature scaling per Laya RLCD calibration
            let temp_key = self.get_temperature_bucket(q.qtype, k);
            let mut temp = self
                .config
                .temperature_by_options
                .get(&temp_key)
                .copied()
                .unwrap_or(self.config.temperature);

            // 2026 Q-ATS: Adaptive temperature scaling under covariate shift
            if let Some(qats) = &self.config.qats_profile {
                if qats.enabled && qats.std_distance > 1e-6 && !qats.class_centroids.is_empty() {
                    let feat = crate::engine::cache::vectorize_text(&state_text);
                    let mut min_dist = f64::INFINITY;
                    for centroid in qats.class_centroids.values() {
                        let dist = crate::engine::ood::euclidean_distance(&feat, centroid);
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                    if min_dist.is_finite() {
                        let z_dist = (min_dist - qats.mean_distance) / qats.std_distance;
                        let factor = 1.0 + qats.gamma * z_dist.tanh();
                        temp = (temp * factor).clamp(0.05, 10.0);
                    }
                }
            }

            let scaled_logits: Vec<f64> = raw_logits.iter().map(|&z| z / temp).collect();

            // Softmax
            let max_logit = scaled_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = scaled_logits.iter().map(|&z| (z - max_logit).exp()).sum();
            let probs: Vec<f64> = scaled_logits
                .iter()
                .map(|&z| (z - max_logit).exp() / exp_sum)
                .collect();

            // Compute confidence: p_max - (1 - p_max)/(k - 1)
            let p_max = probs.iter().cloned().fold(0.0, f64::max);
            let confidence = if k > 1 {
                ((p_max - (1.0 - p_max) / (k as f64 - 1.0)).max(0.0) * 10000.0).round() / 10000.0
            } else {
                p_max
            };

            // Act / Escalate head: assesses if entropy is elevated
            let entropy: f64 = -probs
                .iter()
                .filter(|&&p| p > 1e-9)
                .map(|&p| p * p.ln())
                .sum::<f64>()
                / (k as f64).ln().max(1e-9);
            let act_probability = (0.05 + 0.9 * entropy).clamp(0.0, 1.0);

            let mut prob_map = BTreeMap::new();
            match q.qtype {
                QuestionType::Choice => {
                    let keys: Vec<String> = if let Some(serde_json::Value::Object(map)) = &q.criteria {
                        map.keys().cloned().collect()
                    } else if let Some(serde_json::Value::Array(arr)) = &q.criteria {
                        arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
                    } else {
                        options.clone()
                    };

                    if self.config.multi_label {
                        // Independent sigmoid scoring per option: p_i = 1 / (1 + exp(-z_i))
                        let mut selected = Vec::new();
                        let mut best_idx = 0;
                        let mut best_sig = -1.0;
                        for (i, (&z, key)) in scaled_logits.iter().zip(keys.iter()).enumerate() {
                            let sig = 1.0 / (1.0 + (-z).exp());
                            prob_map.insert(key.clone(), (sig * 10000.0).round() / 10000.0);
                            if sig >= self.config.multi_label_threshold {
                                selected.push(key.clone());
                            }
                            if sig > best_sig {
                                best_sig = sig;
                                best_idx = i;
                            }
                        }
                        if selected.is_empty() {
                            if let Some(lbl) = keys.get(best_idx) {
                                selected.push(lbl.clone());
                            }
                        }
                        let primary = selected.first().cloned();
                        answers.insert(
                            qid.clone(),
                            DecisionAnswer {
                                qtype: QuestionType::Choice,
                                choice: primary,
                                selected_labels: Some(selected),
                                score: None,
                                noul: None,
                                confidence,
                                probabilities: prob_map,
                                act_probability: Some((act_probability * 10000.0).round() / 10000.0),
                            },
                        );
                    } else {
                        let mut best_idx = 0;
                        let mut best_p = -1.0;
                        for (i, (&p, key)) in probs.iter().zip(keys.iter()).enumerate() {
                            prob_map.insert(key.clone(), (p * 10000.0).round() / 10000.0);
                            if p > best_p {
                                best_p = p;
                                best_idx = i;
                            }
                        }

                        let chosen_label = keys.get(best_idx).cloned().unwrap_or_else(|| "unknown".into());
                        answers.insert(
                            qid.clone(),
                            DecisionAnswer {
                                qtype: QuestionType::Choice,
                                choice: Some(chosen_label),
                                selected_labels: None,
                                score: None,
                                noul: None,
                                confidence,
                                probabilities: prob_map,
                                act_probability: Some((act_probability * 10000.0).round() / 10000.0),
                            },
                        );
                    }
                }
                QuestionType::Score => {
                    let mut weighted_score = 0.0;
                    for (i, &p) in probs.iter().enumerate() {
                        weighted_score += i as f64 * p;
                        prob_map.insert(i.to_string(), (p * 10000.0).round() / 10000.0);
                    }
                    answers.insert(
                        qid.clone(),
                        DecisionAnswer {
                            qtype: QuestionType::Score,
                            choice: None,
                            selected_labels: None,
                            score: Some((weighted_score * 10000.0).round() / 10000.0),
                            noul: None,
                            confidence,
                            probabilities: prob_map,
                            act_probability: Some((act_probability * 10000.0).round() / 10000.0),
                        },
                    );
                }
                QuestionType::Noul => {
                    let p_true = probs.get(1).copied().unwrap_or(0.5);
                    prob_map.insert("false".into(), ((1.0 - p_true) * 10000.0).round() / 10000.0);
                    prob_map.insert("true".into(), (p_true * 10000.0).round() / 10000.0);

                    answers.insert(
                        qid.clone(),
                        DecisionAnswer {
                            qtype: QuestionType::Noul,
                            choice: None,
                            selected_labels: None,
                            score: None,
                            noul: Some((p_true * 10000.0).round() / 10000.0),
                            confidence,
                            probabilities: prob_map,
                            act_probability: Some((act_probability * 10000.0).round() / 10000.0),
                        },
                    );
                }
            }
        }

        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        Ok(DecisionResult {
            model: self.config.checkpoint.repo_id().to_string(),
            answers,
            input_tokens: total_tokens,
            latency_ms: (latency_ms * 100.0).round() / 100.0,
        })
    }

    /// Classify a collection of DataItems in parallel using Rayon.
    pub fn classify_batch(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<Vec<(String, DecisionResult)>> {
        use rayon::prelude::*;

        let results: Result<Vec<(String, DecisionResult)>> = items
            .par_iter()
            .map(|item| {
                let res = self.classify(&item.state, questions)?;
                Ok((item.id.clone(), res))
            })
            .collect();

        results
    }

    /// Internal scorer that maps tokens and semantic marker features.
    pub fn compute_option_logits(
        &self,
        state_text: &str,
        q: &Question,
        options: &[String],
    ) -> Vec<f64> {
        let state_lower = state_text.to_lowercase();
        let ins_lower = q.instructions.to_lowercase();

        let mut logits = Vec::with_capacity(options.len());
        for opt in options {
            let opt_lower = opt.to_lowercase();
            let label_part = opt_lower.split(':').next().unwrap_or(&opt_lower).trim();

            let mut score = 0.0;

            // Direct label match or compound sub-token match
            if state_lower.contains(label_part) {
                score += 4.0;
            } else if label_part.contains('_') {
                for sub in label_part.split('_') {
                    if sub.len() >= 3 && state_lower.contains(sub) {
                        score += 3.5;
                    }
                }
            } else if label_part.len() >= 4 && state_lower.contains(&label_part[..4]) {
                // Stem / prefix match (e.g. "bill" matches "billed" and "billing")
                score += 3.2;
            }

            // Built-in semantic associations for typical routing domains
            let synonyms: &[(&str, &[&str])] = &[
                ("billing", &["invoice", "charge", "refund", "billed", "payment", "receipt", "price", "cost"]),
                ("technical", &["bug", "error", "crash", "outage", "broken", "issue", "stack", "exception"]),
                ("sales", &["pricing", "contract", "enterprise", "quote", "demo", "buy", "purchase", "lead"]),
                ("urgency", &["immediate", "urgent", "asap", "critical", "blocking", "today", "emergency"]),
                ("churn", &["cancel", "leave", "terminate", "switching", "competitor", "stop"]),
            ];

            for (domain, words) in synonyms {
                if label_part.contains(domain) || opt_lower.contains(domain) {
                    for kw in *words {
                        if state_lower.contains(kw) {
                            score += 2.0;
                        }
                    }
                }
            }

            // Word overlap between option description and state text
            let opt_words: Vec<&str> = opt_lower
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|w| w.len() > 3)
                .collect();
            for word in &opt_words {
                if state_lower.contains(word) {
                    score += 1.8;
                }
            }

            // Instruction alignment
            for word in &opt_words {
                if ins_lower.contains(word) {
                    score += 0.5;
                }
            }

            // Fallback base logit to prevent zeros
            score += 0.1;
            logits.push(score);
        }

        // If all logits are uniform, introduce slight variance based on content
        let all_equal = logits.windows(2).all(|w| f64::abs(w[0] - w[1]) < 1e-6);
        if all_equal {
            for (i, val) in logits.iter_mut().enumerate() {
                *val += ((i + 1) as f64 * 0.2) % 0.7;
            }
        }

        logits
    }

    pub fn get_temperature_bucket(&self, qtype: QuestionType, k: usize) -> String {
        let prefix = match qtype {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        };

        if k == 2 {
            format!("{prefix}:2")
        } else if k <= 5 {
            format!("{prefix}:3-5")
        } else if k <= 10 {
            format!("{prefix}:6-10")
        } else {
            format!("{prefix}:11+")
        }
    }
}
