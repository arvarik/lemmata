use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::classifier::LayaClassifier;
use crate::datasource::DataItem;
use crate::model::{Checkpoint, Question, QuestionType};

/// Status of multi-checkpoint agreement for an item or question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusStatus {
    Agreed,
    Disputed,
}

impl ConsensusStatus {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::Agreed => "✅ AUTO-EXECUTE (AGREED)",
            Self::Disputed => "🚨 ESCALATE (DISPUTE)",
        }
    }
}

/// Consensus outcome for an individual question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionConsensus {
    pub qid: String,
    pub qtype: QuestionType,
    pub consensus_prediction: String,
    pub consensus_confidence: f64,
    pub epistemic_uncertainty: f64,
    pub aleatoric_uncertainty: f64,
    pub status: ConsensusStatus,
    pub checkpoint_votes: BTreeMap<String, String>,
}

/// Consolidated consensus result for an entire item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub item_id: String,
    pub questions: BTreeMap<String, QuestionConsensus>,
    pub overall_status: ConsensusStatus,
    pub total_latency_ms: f64,
}

pub struct ConsensusEngine {
    checkpoints: Vec<Checkpoint>,
    weights: Vec<f64>,
    disagreement_threshold: f64,
}

impl ConsensusEngine {
    pub fn new(checkpoints: Vec<Checkpoint>) -> Self {
        let n = checkpoints.len().max(1);
        let equal_weight = 1.0 / n as f64;
        let weights = vec![equal_weight; n];
        Self {
            checkpoints,
            weights,
            disagreement_threshold: 0.15,
        }
    }

    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        if weights.len() != self.checkpoints.len() {
            bail!(
                "Weights count ({}) does not match checkpoints count ({})",
                weights.len(),
                self.checkpoints.len()
            );
        }
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            bail!("Weights sum must be strictly positive");
        }
        self.weights = weights.iter().map(|w| w / sum).collect();
        Ok(self)
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.disagreement_threshold = threshold;
        self
    }

    /// Run multi-checkpoint consensus evaluation across a batch of data items.
    pub fn evaluate_batch(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<Vec<ConsensusResult>> {
        if self.checkpoints.is_empty() {
            bail!("No checkpoints configured for consensus");
        }

        // Run inference in parallel across each checkpoint
        let per_ckpt_classifiers: Vec<LayaClassifier> = self
            .checkpoints
            .iter()
            .map(|&ckpt| LayaClassifier::with_checkpoint(ckpt))
            .collect();

        // ckpt_results[c] = Vec<(id, DecisionResult)>
        let ckpt_results: Vec<Vec<(String, crate::model::DecisionResult)>> = per_ckpt_classifiers
            .par_iter()
            .map(|clf| clf.classify_batch(items, questions))
            .collect::<Result<Vec<_>>>()?;

        let consensus_results: Vec<ConsensusResult> = (0..items.len())
            .into_par_iter()
            .map(|item_idx| {
                let item_id = &items[item_idx].id;
                let mut q_consensus_map = BTreeMap::new();
                let mut overall_disputed = false;
                let mut max_latency = 0.0f64;

                for (qid, q) in questions {
                    let mut votes: BTreeMap<String, String> = BTreeMap::new();
                    let mut all_prob_maps: Vec<BTreeMap<String, f64>> = Vec::new();

                    for (c_idx, ckpt) in self.checkpoints.iter().enumerate() {
                        let (_, dec) = &ckpt_results[c_idx][item_idx];
                        max_latency = max_latency.max(dec.latency_ms);
                        if let Some(ans) = dec.answers.get(qid) {
                            let pred_str = if let Some(c) = &ans.choice {
                                c.clone()
                            } else if let Some(s) = ans.score {
                                format!("{:.0}", s.round())
                            } else if let Some(n) = ans.noul {
                                if n > 0.5 {
                                    "true".into()
                                } else {
                                    "false".into()
                                }
                            } else {
                                "N/A".into()
                            };
                            votes.insert(ckpt.repo_id().to_string(), pred_str);
                            all_prob_maps.push(ans.probabilities.clone());
                        }
                    }

                    // Compute pooled ensemble probabilities via Bayesian Model Averaging
                    let mut pooled_probs: BTreeMap<String, f64> = BTreeMap::new();
                    for (c_idx, prob_map) in all_prob_maps.iter().enumerate() {
                        let w = self.weights[c_idx];
                        let sum_p: f64 = prob_map.values().sum::<f64>().max(1e-12);
                        for (k, &p) in prob_map {
                            *pooled_probs.entry(k.clone()).or_default() += w * (p / sum_p);
                        }
                    }

                    // Epistemic uncertainty = H(ensemble) - sum_m(w_m * H(p_m))
                    let ens_entropy = shannon_entropy(pooled_probs.values().copied());
                    let member_entropies: f64 = all_prob_maps
                        .iter()
                        .enumerate()
                        .map(|(i, pm)| {
                            let s = pm.values().sum::<f64>().max(1e-12);
                            self.weights[i] * shannon_entropy(pm.values().map(|&p| p / s))
                        })
                        .sum();
                    let epistemic = (ens_entropy - member_entropies).max(0.0);
                    let aleatoric = member_entropies;

                    // Determine consensus winner
                    let mut best_label = "unknown".to_string();
                    let mut best_p = -1.0;
                    for (k, &p) in &pooled_probs {
                        if p > best_p {
                            best_p = p;
                            best_label = k.clone();
                        }
                    }

                    let k = pooled_probs.len().max(2);
                    let consensus_confidence = if k > 1 {
                        ((best_p - (1.0 - best_p) / (k as f64 - 1.0)).max(0.0) * 10000.0).round()
                            / 10000.0
                    } else {
                        best_p
                    };

                    // Check for vote discordance: do distinct checkpoints choose different top labels?
                    let distinct_votes: std::collections::BTreeSet<_> = votes.values().collect();
                    let has_dispute = epistemic > self.disagreement_threshold
                        || (distinct_votes.len() > 1 && consensus_confidence < 0.85);

                    let status = if has_dispute {
                        overall_disputed = true;
                        ConsensusStatus::Disputed
                    } else {
                        ConsensusStatus::Agreed
                    };

                    let pred_display = match q.qtype {
                        QuestionType::Choice => best_label,
                        QuestionType::Score => {
                            let weighted_score: f64 = pooled_probs
                                .iter()
                                .filter_map(|(k, &p)| k.parse::<f64>().ok().map(|idx| idx * p))
                                .sum();
                            format!("{:.2}", weighted_score)
                        }
                        QuestionType::Noul => {
                            let p_true = pooled_probs.get("true").copied().unwrap_or(best_p);
                            format!("{:.4}", p_true)
                        }
                    };

                    q_consensus_map.insert(
                        qid.clone(),
                        QuestionConsensus {
                            qid: qid.clone(),
                            qtype: q.qtype,
                            consensus_prediction: pred_display,
                            consensus_confidence,
                            epistemic_uncertainty: (epistemic * 10000.0).round() / 10000.0,
                            aleatoric_uncertainty: (aleatoric * 10000.0).round() / 10000.0,
                            status,
                            checkpoint_votes: votes,
                        },
                    );
                }

                ConsensusResult {
                    item_id: item_id.clone(),
                    questions: q_consensus_map,
                    overall_status: if overall_disputed {
                        ConsensusStatus::Disputed
                    } else {
                        ConsensusStatus::Agreed
                    },
                    total_latency_ms: (max_latency * 100.0).round() / 100.0,
                }
            })
            .collect();

        Ok(consensus_results)
    }

    pub fn print_report(results: &[ConsensusResult]) {
        println!("\n🏛️  Lemma Multi-Checkpoint Consensus & Epistemic Uncertainty");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Item ID").fg(Color::Cyan),
            Cell::new("Question").fg(Color::Blue),
            Cell::new("Consensus Winner").fg(Color::Green),
            Cell::new("Confidence").fg(Color::Yellow),
            Cell::new("Epistemic Uncertainty").fg(Color::Magenta),
            Cell::new("Consensus Status").fg(Color::White),
            Cell::new("Checkpoint Breakdown").fg(Color::DarkGrey),
        ]);

        let mut agreed_count = 0;
        let mut disputed_count = 0;

        for res in results {
            for (qid, qc) in &res.questions {
                let status_cell = match qc.status {
                    ConsensusStatus::Agreed => {
                        agreed_count += 1;
                        Cell::new(qc.status.badge()).fg(Color::Green)
                    }
                    ConsensusStatus::Disputed => {
                        disputed_count += 1;
                        Cell::new(qc.status.badge()).fg(Color::Red)
                    }
                };

                let votes_str: Vec<String> = qc
                    .checkpoint_votes
                    .iter()
                    .map(|(ckpt, vote)| {
                        let short_name = ckpt.split('/').next_back().unwrap_or(ckpt);
                        format!("{short_name}: {vote}")
                    })
                    .collect();

                table.add_row(vec![
                    Cell::new(&res.item_id),
                    Cell::new(qid),
                    Cell::new(&qc.consensus_prediction),
                    Cell::new(format!("{:.1}%", qc.consensus_confidence * 100.0)),
                    Cell::new(format!("{:.4}", qc.epistemic_uncertainty)),
                    status_cell,
                    Cell::new(votes_str.join(" | ")),
                ]);
            }
        }

        println!("{table}\n");
        println!(
            "Summary: {} decisions | {} agreed (auto-executable) | {} disputed (flagged for review)\n",
            agreed_count + disputed_count,
            agreed_count,
            disputed_count
        );
    }
}

fn shannon_entropy(probs: impl IntoIterator<Item = f64>) -> f64 {
    let mut h = 0.0;
    for p in probs {
        if p > 1e-9 {
            h -= p * p.ln();
        }
    }
    h.max(0.0)
}
