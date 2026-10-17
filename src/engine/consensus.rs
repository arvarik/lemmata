use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::classifier::LayaClassifier;
use crate::datasource::DataItem;
use crate::model::{Checkpoint, DecisionResult, Question, QuestionType};

/// Whether the members agree on an item or question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusStatus {
    Agreed,
    Disputed,
}

impl ConsensusStatus {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::Agreed => "✅ AGREED",
            Self::Disputed => "🚨 DISPUTED (review)",
        }
    }
}

/// The pooled decision for one question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionConsensus {
    pub qid: String,
    pub qtype: QuestionType,
    pub consensus_prediction: String,
    pub consensus_confidence: f64,
    /// Mutual information between the members (nats): the pooled entropy minus the weighted mean
    /// of the member entropies. It is 0 when every member gives the same distribution.
    pub disagreement: f64,
    /// Weighted mean of the member entropies (nats).
    pub member_entropy: f64,
    pub status: ConsensusStatus,
    pub checkpoint_votes: BTreeMap<String, String>,
}

/// The pooled decisions for one item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub item_id: String,
    pub questions: BTreeMap<String, QuestionConsensus>,
    pub overall_status: ConsensusStatus,
    pub total_latency_ms: f64,
}

/// Member weights fitted on labelled data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightFit {
    pub checkpoints: Vec<String>,
    pub weights: Vec<f64>,
    /// Mean negative log probability of the true label under the fitted weights.
    pub log_loss: f64,
    /// The same with equal weights, for comparison.
    pub equal_weights_log_loss: f64,
    pub labelled_decisions: usize,
}

/// Each member's results, in item order: `member_results[c][i]` is checkpoint `c` on item `i`.
pub type MemberResults = Vec<Vec<(String, DecisionResult)>>;

/// Pools several checkpoints into one ensemble with a linear pool: the weighted average of the
/// members' probabilities.
pub struct ConsensusEngine {
    checkpoints: Vec<Checkpoint>,
    weights: Vec<f64>,
    disagreement_threshold: f64,
}

impl ConsensusEngine {
    pub fn new(checkpoints: Vec<Checkpoint>) -> Self {
        let n = checkpoints.len().max(1);
        Self {
            checkpoints,
            weights: vec![1.0 / n as f64; n],
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
        if sum <= 0.0 || weights.iter().any(|w| *w < 0.0) {
            bail!("Weights must be non-negative, with a positive sum");
        }
        self.weights = weights.iter().map(|w| w / sum).collect();
        Ok(self)
    }

    /// Flag a question as disputed when the members' disagreement exceeds `threshold` nats.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.disagreement_threshold = threshold;
        self
    }

    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Run every member on every item.
    pub fn member_results(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<MemberResults> {
        if self.checkpoints.is_empty() {
            bail!("No checkpoints configured for the ensemble");
        }
        let classifiers: Vec<LayaClassifier> = self
            .checkpoints
            .iter()
            .map(|&ckpt| LayaClassifier::with_checkpoint(ckpt))
            .collect();
        crate::model::native::map_items(&classifiers, |clf| clf.classify_batch(items, questions))
            .into_iter()
            .collect()
    }

    /// Pool the members' decisions for every item.
    pub fn evaluate_batch(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<Vec<ConsensusResult>> {
        let members = self.member_results(items, questions)?;
        Ok(self.pool(items, questions, &members))
    }

    /// Pool already computed member results.
    pub fn pool(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        members: &MemberResults,
    ) -> Vec<ConsensusResult> {
        (0..items.len())
            .into_par_iter()
            .map(|item_idx| self.pool_item(&items[item_idx].id, item_idx, questions, members))
            .collect()
    }

    fn pool_item(
        &self,
        item_id: &str,
        item_idx: usize,
        questions: &BTreeMap<String, Question>,
        members: &MemberResults,
    ) -> ConsensusResult {
        let mut q_consensus_map = BTreeMap::new();
        let mut overall_disputed = false;
        let mut max_latency = 0.0f64;

        for (qid, q) in questions {
            let mut votes: BTreeMap<String, String> = BTreeMap::new();
            let mut member_probs: Vec<(f64, BTreeMap<String, f64>)> = Vec::new();
            for (c_idx, ckpt) in self.checkpoints.iter().enumerate() {
                let (_, dec) = &members[c_idx][item_idx];
                max_latency = max_latency.max(dec.latency_ms);
                if let Some(ans) = dec.answers.get(qid) {
                    let pred = if let Some(c) = &ans.choice {
                        c.clone()
                    } else if let Some(s) = ans.score {
                        format!("{:.0}", s.round())
                    } else if let Some(n) = ans.noul {
                        (n > 0.5).to_string()
                    } else {
                        "N/A".into()
                    };
                    votes.insert(ckpt.repo_id().to_string(), pred);
                    member_probs.push((self.weights[c_idx], normalized(&ans.probabilities)));
                }
            }

            let pooled = linear_pool(&member_probs);
            let pooled_entropy = shannon_entropy(pooled.values().copied());
            let member_entropy: f64 = member_probs
                .iter()
                .map(|(w, p)| w * shannon_entropy(p.values().copied()))
                .sum();
            let disagreement = (pooled_entropy - member_entropy).max(0.0);

            let (best_label, best_p) = pooled
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, &p)| (k.clone(), p))
                .unwrap_or_else(|| ("unknown".to_string(), 0.0));
            let k = pooled.len().max(2) as f64;
            // Chance-corrected confidence: 0 at a uniform pool, 1 at certainty.
            let consensus_confidence =
                ((best_p - (1.0 - best_p) / (k - 1.0)).max(0.0) * 10000.0).round() / 10000.0;

            let status = if disagreement > self.disagreement_threshold {
                overall_disputed = true;
                ConsensusStatus::Disputed
            } else {
                ConsensusStatus::Agreed
            };

            let prediction = match q.qtype {
                QuestionType::Choice => best_label,
                QuestionType::Score => {
                    let expected: f64 = pooled
                        .iter()
                        .filter_map(|(k, &p)| k.parse::<f64>().ok().map(|v| v * p))
                        .sum();
                    format!("{expected:.2}")
                }
                QuestionType::Noul => {
                    format!("{:.4}", pooled.get("true").copied().unwrap_or(best_p))
                }
            };

            q_consensus_map.insert(
                qid.clone(),
                QuestionConsensus {
                    qid: qid.clone(),
                    qtype: q.qtype,
                    consensus_prediction: prediction,
                    consensus_confidence,
                    disagreement: (disagreement * 10000.0).round() / 10000.0,
                    member_entropy: (member_entropy * 10000.0).round() / 10000.0,
                    status,
                    checkpoint_votes: votes,
                },
            );
        }

        ConsensusResult {
            item_id: item_id.to_string(),
            questions: q_consensus_map,
            overall_status: if overall_disputed {
                ConsensusStatus::Disputed
            } else {
                ConsensusStatus::Agreed
            },
            total_latency_ms: (max_latency * 100.0).round() / 100.0,
        }
    }

    /// Search the weight simplex in steps of 0.05 for the weights with the lowest log loss of the
    /// true label on the labelled choice questions.
    pub fn fit_weights(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        members: &MemberResults,
    ) -> Result<WeightFit> {
        let m = self.checkpoints.len();
        if !(2..=5).contains(&m) {
            bail!("--fit-weights needs 2 to 5 checkpoints, not {m}");
        }
        // Per labelled decision: each member's probability for the true label.
        let mut true_label_probs: Vec<Vec<f64>> = Vec::new();
        for (i, item) in items.iter().enumerate() {
            for (qid, q) in questions {
                if q.qtype != QuestionType::Choice {
                    continue;
                }
                let Some(truth) = crate::engine::metrics::resolve_ground_truth_for_question(
                    item.ground_truth.as_ref(),
                    qid,
                    q,
                    questions.len(),
                )
                .and_then(|v| v.as_str().map(str::to_lowercase)) else {
                    continue;
                };
                let row: Vec<f64> = (0..m)
                    .map(|c| {
                        members[c][i].1.answers.get(qid).map_or(0.0, |a| {
                            normalized(&a.probabilities)
                                .iter()
                                .find(|(k, _)| k.to_lowercase() == truth)
                                .map_or(0.0, |(_, &p)| p)
                        })
                    })
                    .collect();
                true_label_probs.push(row);
            }
        }
        if true_label_probs.is_empty() {
            bail!("--fit-weights found no labelled choice questions in the data");
        }

        let log_loss = |w: &[f64]| {
            true_label_probs
                .iter()
                .map(|row| {
                    let p: f64 = row.iter().zip(w).map(|(p, w)| p * w).sum();
                    -p.max(1e-12).ln()
                })
                .sum::<f64>()
                / true_label_probs.len() as f64
        };
        const STEPS: usize = 20;
        let mut best = (vec![1.0 / m as f64; m], f64::INFINITY);
        for counts in simplex_points(m, STEPS) {
            let w: Vec<f64> = counts.iter().map(|&c| c as f64 / STEPS as f64).collect();
            let loss = log_loss(&w);
            if loss < best.1 {
                best = (w, loss);
            }
        }
        Ok(WeightFit {
            checkpoints: self
                .checkpoints
                .iter()
                .map(|c| c.repo_id().to_string())
                .collect(),
            equal_weights_log_loss: log_loss(&vec![1.0 / m as f64; m]),
            weights: best.0,
            log_loss: best.1,
            labelled_decisions: true_label_probs.len(),
        })
    }

    pub fn print_report(results: &[ConsensusResult]) {
        println!("\n🏛️  lemmata ensemble: pooled decisions and member disagreement");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Item ID").fg(Color::Cyan),
            Cell::new("Question").fg(Color::Blue),
            Cell::new("Pooled Decision").fg(Color::Green),
            Cell::new("Confidence").fg(Color::Yellow),
            Cell::new("Disagreement (MI)").fg(Color::Magenta),
            Cell::new("Status").fg(Color::White),
            Cell::new("Member Votes").fg(Color::DarkGrey),
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
                let votes: Vec<String> = qc
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
                    Cell::new(format!("{:.4}", qc.disagreement)),
                    status_cell,
                    Cell::new(votes.join(" | ")),
                ]);
            }
        }

        println!("{table}\n");
        println!(
            "Summary: {} decisions | {} agreed | {} disputed (flagged for review)\n",
            agreed_count + disputed_count,
            agreed_count,
            disputed_count
        );
    }
}

/// A probability map scaled to sum to 1.
fn normalized(probs: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let sum = probs.values().sum::<f64>().max(1e-12);
    probs.iter().map(|(k, &p)| (k.clone(), p / sum)).collect()
}

/// The weighted average of the member distributions.
fn linear_pool(members: &[(f64, BTreeMap<String, f64>)]) -> BTreeMap<String, f64> {
    let mut pooled = BTreeMap::new();
    for (w, probs) in members {
        for (k, &p) in probs {
            *pooled.entry(k.clone()).or_insert(0.0) += w * p;
        }
    }
    pooled
}

/// Every way to split `steps` units over `m` members: the grid points of the weight simplex.
fn simplex_points(m: usize, steps: usize) -> Vec<Vec<usize>> {
    if m == 1 {
        return vec![vec![steps]];
    }
    (0..=steps)
        .flat_map(|first| {
            simplex_points(m - 1, steps - first)
                .into_iter()
                .map(move |mut rest| {
                    rest.insert(0, first);
                    rest
                })
        })
        .collect()
}

fn shannon_entropy(probs: impl IntoIterator<Item = f64>) -> f64 {
    probs
        .into_iter()
        .filter(|&p| p > 1e-9)
        .map(|p| -p * p.ln())
        .sum::<f64>()
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_grid_covers_every_split() {
        let points = simplex_points(3, 20);
        assert_eq!(
            points.len(),
            231,
            "C(22, 2) splits of 20 units over 3 members"
        );
        assert!(points.iter().all(|p| p.iter().sum::<usize>() == 20));
    }

    #[test]
    fn disagreement_is_mutual_information() {
        let a = BTreeMap::from([("x".to_string(), 1.0), ("y".to_string(), 0.0)]);
        let b = BTreeMap::from([("x".to_string(), 0.0), ("y".to_string(), 1.0)]);
        let pooled = linear_pool(&[(0.5, a.clone()), (0.5, b)]);
        // Two certain members that disagree: the pool is uniform, so MI = ln 2.
        let mi = shannon_entropy(pooled.values().copied());
        assert!((mi - 2f64.ln()).abs() < 1e-9);
        let same = linear_pool(&[(0.5, a.clone()), (0.5, a)]);
        assert!(shannon_entropy(same.values().copied()).abs() < 1e-9);
    }
}
