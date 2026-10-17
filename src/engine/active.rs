use anyhow::{Context, Result, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

use crate::datasource::DataItem;
use crate::engine::classifier::LayaClassifier;
use crate::engine::ood::{OodDetector, compute_free_energy};
use crate::model::Question;

/// Uncertainty acquisition strategy for active learning sample prioritization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyStrategy {
    /// Weighted composite combining entropy, margin gap, conformal set size, and free energy
    Composite,
    /// Normalized Shannon Entropy across candidate options (highest uncertainty first)
    Entropy,
    /// Top-2 Candidate Margin Gap (lowest margin first)
    Margin,
    /// Conformal Prediction Set Cardinality (|C_{1-alpha}(x)| > 1 or empty)
    Conformal,
    /// Free-Energy Out-Of-Distribution Score (highest free energy first)
    Energy,
}

impl FromStr for UncertaintyStrategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "composite" | "hybrid" => Ok(UncertaintyStrategy::Composite),
            "entropy" | "shannon" => Ok(UncertaintyStrategy::Entropy),
            "margin" | "top2" => Ok(UncertaintyStrategy::Margin),
            "conformal" | "set" => Ok(UncertaintyStrategy::Conformal),
            "energy" | "ood" => Ok(UncertaintyStrategy::Energy),
            other => Err(format!(
                "Unknown uncertainty strategy '{}'. Options: composite, entropy, margin, conformal, energy",
                other
            )),
        }
    }
}

/// Detailed prediction diagnostics for items in the supervisor review queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewQueuePrediction {
    pub chosen_option: String,
    pub confidence: f64,
    pub act_probability: f64,
    pub top2_margin: f64,
    pub normalized_entropy: f64,
    pub conformal_set: Vec<String>,
}

/// Out-of-distribution diagnostics for review queue items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodDiagnostics {
    pub free_energy: f64,
    pub is_ood: bool,
}

/// A prioritized review queue item matching Section 5.1 of the RLCD specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewQueueItem {
    pub id: String,
    pub text: String,
    pub question_id: String,
    pub prediction: ReviewQueuePrediction,
    pub ood_diagnostics: OodDiagnostics,
    pub acquisition_score: f64,
    pub escalated: bool,
    pub escalation_reason: String,
    pub timestamp: String,
}

/// An RLCD fine-tuning contrastive training pair matching Section 5.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlcdSample {
    pub input_text: String,
    pub question_id: String,
    pub candidate_options: Vec<String>,
    pub target_label: String,
    pub negative_label: String,
    pub margin_weight: f64,
    pub source: String,
    pub split: String,
}

/// Report summarizing an active learning acquisition run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningReport {
    pub strategy: UncertaintyStrategy,
    pub total_evaluated: usize,
    pub acquired_count: usize,
    pub mean_entropy: f64,
    pub mean_margin: f64,
    pub ood_sample_count: usize,
    pub multi_label_conformal_count: usize,
    pub mean_acquisition_score: f64,
}

/// Report summarizing an RLCD distillation dataset generation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveDistillReport {
    pub total_samples: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub average_margin_weight: f64,
    pub max_margin_weight: f64,
    pub questions_covered: Vec<String>,
    pub export_path: String,
}

/// Engine managing uncertainty sampling, supervisor queue generation, and RLCD distillation.
pub struct ActiveLearningEngine {
    escalate_threshold: f64,
    conformal_alpha: f64,
    weight_entropy: f64,
    weight_margin: f64,
    weight_conformal: f64,
    weight_energy: f64,
}

impl Default for ActiveLearningEngine {
    fn default() -> Self {
        Self {
            escalate_threshold: 0.50,
            conformal_alpha: 0.10,
            weight_entropy: 0.35,
            weight_margin: 0.30,
            weight_conformal: 0.20,
            weight_energy: 0.15,
        }
    }
}

impl ActiveLearningEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_escalate_threshold(mut self, threshold: f64) -> Self {
        self.escalate_threshold = threshold;
        self
    }

    pub fn with_conformal_alpha(mut self, alpha: f64) -> Self {
        self.conformal_alpha = alpha;
        self
    }

    pub fn with_weights(mut self, entropy: f64, margin: f64, conformal: f64, energy: f64) -> Self {
        self.weight_entropy = entropy;
        self.weight_margin = margin;
        self.weight_conformal = conformal;
        self.weight_energy = energy;
        self
    }

    /// Evaluates items across question schemas, computes uncertainty metrics,
    /// ranks candidate requests, and extracts the top `budget` items for review.
    pub fn sample(
        &self,
        classifier: &LayaClassifier,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        strategy: UncertaintyStrategy,
        budget: usize,
        ood_detector: Option<&OodDetector>,
    ) -> Result<(Vec<ReviewQueueItem>, ActiveLearningReport)> {
        if items.is_empty() {
            bail!("No items provided for active learning sampling");
        }

        let timestamp = crate::engine::metrics::format_utc_iso8601();

        let mut scored_items = Vec::new();
        let mut total_entropy_sum = 0.0;
        let mut total_margin_sum = 0.0;
        let mut ood_count = 0;
        let mut multi_conformal_count = 0;
        let mut total_evaluations = 0;

        for item in items {
            let item_text = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_else(|_| item.preview_text(250)),
            };
            let decision = classifier.classify(&item.state, questions)?;

            for (qid, question) in questions {
                if let Some(ans) = decision.answers.get(qid) {
                    total_evaluations += 1;

                    let candidate_options = question.render_options();
                    let k = candidate_options.len().max(2);

                    let mut sorted_probs: Vec<f64> = ans.probabilities.values().copied().collect();
                    sorted_probs
                        .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                    let p_top1 = sorted_probs.first().copied().unwrap_or(0.5);
                    let p_top2 = sorted_probs.get(1).copied().unwrap_or(0.0);

                    let margin = (p_top1 - p_top2).clamp(0.0, 1.0);
                    total_margin_sum += margin;

                    let entropy: f64 = -ans
                        .probabilities
                        .values()
                        .filter(|&&p| p > 1e-9)
                        .map(|&p| p * p.ln())
                        .sum::<f64>()
                        / (k as f64).ln().max(1e-9);
                    let normalized_entropy = entropy.clamp(0.0, 1.0);
                    total_entropy_sum += normalized_entropy;

                    let act_prob = ans
                        .act_probability
                        .unwrap_or((0.05 + 0.9 * normalized_entropy).clamp(0.0, 1.0));

                    let threshold = 1.0 - self.conformal_alpha;
                    let mut cum_prob = 0.0;
                    let mut conformal_set = Vec::new();

                    let mut label_probs: Vec<(&String, &f64)> = ans.probabilities.iter().collect();
                    label_probs
                        .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                    for (label, &prob) in label_probs {
                        conformal_set.push(label.clone());
                        cum_prob += prob;
                        if cum_prob >= threshold {
                            break;
                        }
                    }

                    if conformal_set.len() > 1 {
                        multi_conformal_count += 1;
                    }

                    let free_energy = ans.logits.as_deref().map_or(0.0, compute_free_energy);
                    // Energy has no fixed OOD threshold: only a fitted profile can flag OOD items.
                    let is_ood = ood_detector.is_some_and(|ood| {
                        ood.evaluate_with_decision(&item.id, &item_text, &decision)
                            .is_ok_and(|res| res.is_ood)
                    });

                    if is_ood {
                        ood_count += 1;
                    }

                    let acquisition_score = match strategy {
                        UncertaintyStrategy::Entropy => normalized_entropy,
                        UncertaintyStrategy::Margin => 1.0 - margin,
                        UncertaintyStrategy::Conformal => {
                            if conformal_set.is_empty() {
                                1.0
                            } else {
                                conformal_set.len() as f64 / k as f64
                            }
                        }
                        UncertaintyStrategy::Energy => 1.0 / (1.0 + (-free_energy).exp()),
                        UncertaintyStrategy::Composite => {
                            let norm_energy = 1.0 / (1.0 + (-free_energy).exp());
                            let norm_conformal = conformal_set.len() as f64 / k as f64;
                            self.weight_entropy * normalized_entropy
                                + self.weight_margin * (1.0 - margin)
                                + self.weight_conformal * norm_conformal
                                + self.weight_energy * norm_energy
                        }
                    };

                    let escalated = act_prob > self.escalate_threshold
                        || ans.confidence < self.escalate_threshold
                        || is_ood
                        || conformal_set.len() > 1;

                    let mut reason_parts = Vec::new();
                    if act_prob > self.escalate_threshold {
                        reason_parts.push(format!("act_prob={:.1}%", act_prob * 100.0));
                    }
                    if ans.confidence < self.escalate_threshold {
                        reason_parts.push(format!("low_conf={:.1}%", ans.confidence * 100.0));
                    }
                    if conformal_set.len() > 1 {
                        reason_parts.push(format!(
                            "ambiguous_conformal_cardinality={}",
                            conformal_set.len()
                        ));
                    }
                    if is_ood {
                        reason_parts.push("ood_distribution_shift".into());
                    }
                    let escalation_reason = if reason_parts.is_empty() {
                        "uncertainty_sampling".into()
                    } else {
                        reason_parts.join("; ")
                    };

                    let chosen_opt = if let Some(labels) = &ans.selected_labels {
                        labels.join(", ")
                    } else if let Some(c) = &ans.choice {
                        c.clone()
                    } else if let Some(s) = ans.score {
                        format!("{:.2}", s)
                    } else if let Some(n) = ans.noul {
                        format!("true: {:.2}%", n * 100.0)
                    } else {
                        "unknown".into()
                    };

                    scored_items.push(ReviewQueueItem {
                        id: item.id.clone(),
                        text: item_text.clone(),
                        question_id: qid.clone(),
                        prediction: ReviewQueuePrediction {
                            chosen_option: chosen_opt,
                            confidence: ans.confidence,
                            act_probability: act_prob,
                            top2_margin: margin,
                            normalized_entropy,
                            conformal_set,
                        },
                        ood_diagnostics: OodDiagnostics {
                            free_energy,
                            is_ood,
                        },
                        acquisition_score,
                        escalated,
                        escalation_reason,
                        timestamp: timestamp.clone(),
                    });
                }
            }
        }

        scored_items.sort_by(|a, b| {
            b.acquisition_score
                .partial_cmp(&a.acquisition_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_items.truncate(budget);

        let acquired_count = scored_items.len();
        let mean_acq = if acquired_count > 0 {
            scored_items
                .iter()
                .map(|it| it.acquisition_score)
                .sum::<f64>()
                / acquired_count as f64
        } else {
            0.0
        };
        let mean_ent = if total_evaluations > 0 {
            total_entropy_sum / total_evaluations as f64
        } else {
            0.0
        };
        let mean_mar = if total_evaluations > 0 {
            total_margin_sum / total_evaluations as f64
        } else {
            0.0
        };

        let report = ActiveLearningReport {
            strategy,
            total_evaluated: total_evaluations,
            acquired_count,
            mean_entropy: mean_ent,
            mean_margin: mean_mar,
            ood_sample_count: ood_count,
            multi_label_conformal_count: multi_conformal_count,
            mean_acquisition_score: mean_acq,
        };

        Ok((scored_items, report))
    }

    /// Synthesizes an RLCD contrastive fine-tuning corpus from labeled or supervisor-reviewed items.
    /// Pairs positive ground-truth targets with negative model predictions and calculates contrastive margin weights.
    pub fn distill(
        &self,
        classifier: &LayaClassifier,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        val_split: f64,
        source_label: &str,
    ) -> Result<(Vec<RlcdSample>, ActiveDistillReport)> {
        if items.is_empty() {
            bail!("No items provided for RLCD distillation");
        }

        let mut samples = Vec::new();
        let mut questions_seen = std::collections::BTreeSet::new();
        let mut total_weight = 0.0;
        let mut max_weight = 0.0;

        for (idx, item) in items.iter().enumerate() {
            let item_text = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_else(|_| item.preview_text(300)),
            };
            let decision = classifier.classify(&item.state, questions)?;

            for (qid, question) in questions {
                let candidate_options = question.render_options();

                let opt_gt = if let Some(gt_map) = &item.ground_truth {
                    gt_map
                        .get(qid)
                        .cloned()
                        .or_else(|| gt_map.get("default").cloned())
                } else {
                    None
                };

                let target_label = match opt_gt {
                    Some(serde_json::Value::String(s)) => s,
                    Some(serde_json::Value::Number(n)) => n.to_string(),
                    Some(serde_json::Value::Bool(b)) => b.to_string(),
                    _ => continue,
                };

                if let Some(ans) = decision.answers.get(qid) {
                    let chosen_opt = if let Some(labels) = &ans.selected_labels {
                        labels.first().cloned().unwrap_or_default()
                    } else if let Some(c) = &ans.choice {
                        c.clone()
                    } else if let Some(s) = ans.score {
                        format!("{:.2}", s)
                    } else if let Some(n) = ans.noul {
                        format!("{}", n > 0.5)
                    } else {
                        "unknown".into()
                    };

                    let clean_target = target_label
                        .split(':')
                        .next()
                        .unwrap_or(&target_label)
                        .trim()
                        .to_lowercase();
                    let clean_chosen = chosen_opt
                        .split(':')
                        .next()
                        .unwrap_or(&chosen_opt)
                        .trim()
                        .to_lowercase();

                    let negative_label = if clean_chosen != clean_target && !chosen_opt.is_empty() {
                        chosen_opt
                    } else {
                        let mut alt_candidates: Vec<(&String, &f64)> = ans
                            .probabilities
                            .iter()
                            .filter(|(label, _)| {
                                label
                                    .split(':')
                                    .next()
                                    .unwrap_or(label)
                                    .trim()
                                    .to_lowercase()
                                    != clean_target
                            })
                            .collect();
                        alt_candidates.sort_by(|a, b| {
                            b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        alt_candidates
                            .first()
                            .map(|(label, _)| (*label).clone())
                            .unwrap_or_else(|| {
                                candidate_options
                                    .iter()
                                    .find(|&c| {
                                        c.split(':').next().unwrap_or(c).trim().to_lowercase()
                                            != clean_target
                                    })
                                    .cloned()
                                    .unwrap_or_else(|| "other".into())
                            })
                    };

                    let entropy: f64 = -ans
                        .probabilities
                        .values()
                        .filter(|&&p| p > 1e-9)
                        .map(|&p| p * p.ln())
                        .sum::<f64>()
                        / (candidate_options.len().max(2) as f64).ln().max(1e-9);

                    let lookup_prob = |lbl: &str| -> f64 {
                        let clean_lbl = lbl.split(':').next().unwrap_or(lbl).trim();
                        ans.probabilities
                            .iter()
                            .find(|(k, _)| {
                                k.eq_ignore_ascii_case(lbl)
                                    || k.split(':')
                                        .next()
                                        .unwrap_or(k)
                                        .trim()
                                        .eq_ignore_ascii_case(clean_lbl)
                            })
                            .map(|(_, &v)| v)
                            .unwrap_or(0.0)
                    };
                    let p_target = lookup_prob(&target_label);
                    let p_neg = lookup_prob(&negative_label);
                    let margin_gap = p_target - p_neg;

                    let margin_weight =
                        ((1.0 + entropy + (1.0 - margin_gap).max(0.0)) * 100.0).round() / 100.0;
                    total_weight += margin_weight;
                    if margin_weight > max_weight {
                        max_weight = margin_weight;
                    }

                    let split = if ((idx * 17) % 100) as f64 / 100.0 < val_split {
                        "val".to_string()
                    } else {
                        "train".to_string()
                    };

                    questions_seen.insert(qid.clone());

                    samples.push(RlcdSample {
                        input_text: item_text.clone(),
                        question_id: qid.clone(),
                        candidate_options,
                        target_label,
                        negative_label,
                        margin_weight,
                        source: source_label.to_string(),
                        split,
                    });
                }
            }
        }

        if samples.is_empty() {
            bail!(
                "No labeled ground-truth targets found in dataset. Ensure items have ground truth."
            );
        }

        let total_samples = samples.len();
        let val_samples = samples.iter().filter(|s| s.split == "val").count();
        let train_samples = total_samples - val_samples;
        let avg_weight = total_weight / total_samples as f64;

        let report = ActiveDistillReport {
            total_samples,
            train_samples,
            val_samples,
            average_margin_weight: avg_weight,
            max_margin_weight: max_weight,
            questions_covered: questions_seen.into_iter().collect(),
            export_path: String::new(),
        };

        Ok((samples, report))
    }

    /// Serializes review queue items to a JSONL file.
    pub fn export_queue(items: &[ReviewQueueItem], path: &Path) -> Result<()> {
        write_jsonl_slice(items, path)
    }

    /// Serializes RLCD distillation samples to a JSONL file.
    pub fn export_rlcd(samples: &[RlcdSample], path: &Path) -> Result<()> {
        write_jsonl_slice(samples, path)
    }

    /// Formats active learning sample report as a terminal table.
    pub fn format_sample_table(
        report: &ActiveLearningReport,
        items: &[ReviewQueueItem],
        export_path: Option<&Path>,
    ) -> String {
        let mut table = Table::new();
        table.load_style(comfy_table::presets::UTF8_FULL);
        table.set_header(vec![
            "#",
            "Item ID",
            "Question",
            "Model Prediction",
            "Confidence",
            "Act Prob",
            "Top2 Margin",
            "Entropy",
            "Acq Score",
            "Reason",
        ]);

        for (idx, it) in items.iter().take(15).enumerate() {
            table.add_row(vec![
                (idx + 1).to_string(),
                it.id.clone(),
                it.question_id.clone(),
                it.prediction.chosen_option.clone(),
                format!("{:.1}%", it.prediction.confidence * 100.0),
                format!("{:.1}%", it.prediction.act_probability * 100.0),
                format!("{:.3}", it.prediction.top2_margin),
                format!("{:.3}", it.prediction.normalized_entropy),
                format!("{:.4}", it.acquisition_score),
                it.escalation_reason.clone(),
            ]);
        }

        let mut output = format!(
            "\n🎯 Active Learning Uncertainty Prioritization\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nStrategy:       {:?} (Acquired {} of {} candidate decisions)\nDiagnostics:    Mean Entropy={:.3}, Mean Margin={:.3}, OOD Items={}, Multi-Conformal={}\n\n{}",
            report.strategy,
            report.acquired_count,
            report.total_evaluated,
            report.mean_entropy,
            report.mean_margin,
            report.ood_sample_count,
            report.multi_label_conformal_count,
            table
        );

        if items.len() > 15 {
            output.push_str(&format!(
                "\n... and {} more items in review queue\n",
                items.len() - 15
            ));
        }
        if let Some(p) = export_path {
            output.push_str(&format!("\n💾 Saved review queue to {}\n", p.display()));
        }
        output
    }

    /// Formats RLCD distillation report as a terminal table.
    pub fn format_distill_table(report: &ActiveDistillReport) -> String {
        let mut table = Table::new();
        table.load_style(comfy_table::presets::UTF8_FULL);
        table.set_header(vec![
            Cell::new("RLCD Distillation Metric").fg(Color::Cyan),
            Cell::new("Value").fg(Color::Yellow),
        ]);
        let total = report.total_samples.max(1) as f64;
        for (k, v) in [
            ("Total Contrastive Pairs", report.total_samples.to_string()),
            (
                "Train Split (RLCD Training)",
                format!(
                    "{} ({:.1}%)",
                    report.train_samples,
                    (report.train_samples as f64 / total) * 100.0
                ),
            ),
            (
                "Validation Split (Evaluation)",
                format!(
                    "{} ({:.1}%)",
                    report.val_samples,
                    (report.val_samples as f64 / total) * 100.0
                ),
            ),
            (
                "Average Margin Weight",
                format!("{:.3}", report.average_margin_weight),
            ),
            (
                "Max Margin Weight",
                format!("{:.3}", report.max_margin_weight),
            ),
            ("Questions Covered", report.questions_covered.join(", ")),
            ("Export Destination", report.export_path.clone()),
        ] {
            table.add_row(vec![Cell::new(k), Cell::new(v)]);
        }
        format!(
            "\n🧪 RLCD Contrastive Distillation Corpus Ready:\n{}",
            table
        )
    }

    /// Formats sampling report as JSON.
    pub fn format_sample_json(
        report: &ActiveLearningReport,
        items: &[ReviewQueueItem],
    ) -> Result<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "report": report,
            "review_queue": items,
        }))?)
    }

    /// Formats distillation report as JSON.
    pub fn format_distill_json(
        report: &ActiveDistillReport,
        samples: &[RlcdSample],
    ) -> Result<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "report": report,
            "distillation_samples": samples,
        }))?)
    }
}

fn write_jsonl_slice<T: Serialize>(items: &[T], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)
        .with_context(|| format!("Failed to create JSONL file: {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for item in items {
        writeln!(writer, "{}", serde_json::to_string(item)?)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Checkpoint;

    #[test]
    fn test_uncertainty_strategies_parsing() {
        assert_eq!(
            UncertaintyStrategy::from_str("composite").unwrap(),
            UncertaintyStrategy::Composite
        );
        assert_eq!(
            UncertaintyStrategy::from_str("entropy").unwrap(),
            UncertaintyStrategy::Entropy
        );
        assert_eq!(
            UncertaintyStrategy::from_str("margin").unwrap(),
            UncertaintyStrategy::Margin
        );
        assert_eq!(
            UncertaintyStrategy::from_str("conformal").unwrap(),
            UncertaintyStrategy::Conformal
        );
        assert_eq!(
            UncertaintyStrategy::from_str("energy").unwrap(),
            UncertaintyStrategy::Energy
        );
    }

    #[test]
    fn test_active_learning_sample_and_review_queue() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        );

        let items = vec![
            DataItem::new(
                "t1",
                serde_json::json!({"text": "urgent invoice refund"}),
                "test",
            ),
            DataItem::new(
                "t2",
                serde_json::json!({"text": "server crash stack trace"}),
                "test",
            ),
            DataItem::new(
                "t3",
                serde_json::json!({"text": "unclear request maybe billing maybe tech"}),
                "test",
            ),
        ];

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let engine = ActiveLearningEngine::new().with_escalate_threshold(0.70);

        let (queue, report) = engine
            .sample(
                &classifier,
                &items,
                &questions,
                UncertaintyStrategy::Composite,
                2,
                None,
            )
            .unwrap();

        assert_eq!(queue.len(), 2);
        assert_eq!(report.acquired_count, 2);
        assert_eq!(report.total_evaluated, 3);
        assert!(queue[0].acquisition_score >= queue[1].acquisition_score);
    }

    #[test]
    fn test_rlcd_distillation_corpus_generation() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice(
                "Department",
                vec!["billing".into(), "technical".into(), "sales".into()],
            ),
        );

        let mut gt = BTreeMap::new();
        gt.insert("dept".into(), serde_json::Value::String("technical".into()));

        let items = vec![
            DataItem::new(
                "t1",
                serde_json::json!({"text": "server crash error"}),
                "test",
            )
            .with_ground_truth(gt),
        ];

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let engine = ActiveLearningEngine::new();

        let (samples, report) = engine
            .distill(&classifier, &items, &questions, 0.20, "human_supervisor")
            .unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].target_label, "technical");
        assert_eq!(samples[0].source, "human_supervisor");
        assert!(samples[0].margin_weight >= 1.0);
        assert_eq!(report.total_samples, 1);
    }
}
