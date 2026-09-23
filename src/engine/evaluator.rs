use std::collections::BTreeMap;
use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};

use super::classifier::LayaClassifier;
use super::metrics::{EvaluationMetrics, calculate_brier_score, calculate_ece};
use crate::datasource::DataItem;
use crate::model::{Question, QuestionType};

/// Evaluator that computes Laya metrics against labeled ground truth data.
pub struct Evaluator {
    classifier: LayaClassifier,
}

impl Evaluator {
    pub fn new(classifier: LayaClassifier) -> Self {
        Self { classifier }
    }

    pub fn evaluate(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<EvaluationMetrics> {
        if items.is_empty() {
            bail!("No items to evaluate");
        }

        let mut total_samples = 0;
        let mut correct_count = 0;
        let mut soft_acc_sum = 0.0;
        let mut escalate_count = 0;

        let mut confidences = Vec::new();
        let mut correctness = Vec::new();

        let mut all_probs = Vec::new();
        let mut all_targets = Vec::new();

        let mut score_err_sum = 0.0;
        let mut score_count = 0;

        for item in items {
            let gt_map = match &item.ground_truth {
                Some(map) => map,
                None => continue, // Skip items without ground-truth labels
            };

            let result = self.classifier.classify(&item.state, questions)?;

            for (qid, answer) in &result.answers {
                let target_val = match gt_map.get(qid).or_else(|| gt_map.get("default")) {
                    Some(v) => v,
                    None => continue,
                };

                total_samples += 1;
                let conf = answer.confidence;
                confidences.push(conf);

                if let Some(act) = answer.act_probability {
                    if act > self.classifier.config().escalate_threshold {
                        escalate_count += 1;
                    }
                }

                match answer.qtype {
                    QuestionType::Choice => {
                        let pred_choice = answer.choice.as_deref().unwrap_or("");
                        let target_str = target_val.as_str().unwrap_or("");
                        let is_correct = pred_choice.eq_ignore_ascii_case(target_str);

                        if is_correct {
                            correct_count += 1;
                        }
                        correctness.push(is_correct);

                        // Soft accuracy: probability assigned to the target class
                        let p_target = answer.probabilities.get(target_str).copied().unwrap_or(0.0);
                        soft_acc_sum += p_target;

                        // Brier score prep
                        let keys: Vec<&String> = answer.probabilities.keys().collect();
                        let target_idx = keys.iter().position(|k| k.eq_ignore_ascii_case(target_str));
                        if let Some(t_idx) = target_idx {
                            let p_vec: Vec<f64> = keys.iter().map(|k| answer.probabilities.get(*k).copied().unwrap_or(0.0)).collect();
                            all_probs.push(p_vec);
                            all_targets.push(t_idx);
                        }
                    }
                    QuestionType::Score => {
                        let pred_score = answer.score.unwrap_or(0.0);
                        let target_num = target_val.as_f64().unwrap_or_else(|| {
                            target_val.as_str().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
                        });

                        let err = (pred_score - target_num).abs();
                        score_err_sum += err;
                        score_count += 1;

                        let is_close = err < 0.5;
                        if is_close {
                            correct_count += 1;
                        }
                        correctness.push(is_close);
                        soft_acc_sum += 1.0 - (err / 5.0).min(1.0);
                    }
                    QuestionType::Noul => {
                        let pred_noul = answer.noul.unwrap_or(0.5);
                        let target_bool = target_val.as_bool().unwrap_or_else(|| {
                            target_val.as_str().map(|s| s.eq_ignore_ascii_case("true") || s == "1").unwrap_or(false)
                        });

                        let is_correct = (pred_noul >= 0.5) == target_bool;
                        if is_correct {
                            correct_count += 1;
                        }
                        correctness.push(is_correct);

                        let p_target = if target_bool { pred_noul } else { 1.0 - pred_noul };
                        soft_acc_sum += p_target;

                        all_probs.push(vec![1.0 - pred_noul, pred_noul]);
                        all_targets.push(if target_bool { 1 } else { 0 });
                    }
                }
            }
        }

        if total_samples == 0 {
            bail!("No ground-truth labels matched between data items and questions");
        }

        let accuracy = correct_count as f64 / total_samples as f64;
        let soft_accuracy = soft_acc_sum / total_samples as f64;
        let act_escalate_rate = escalate_count as f64 / total_samples as f64;
        let ece = calculate_ece(&confidences, &correctness, 10);
        let brier_score = calculate_brier_score(&all_probs, &all_targets);
        let score_mae = if score_count > 0 {
            Some(score_err_sum / score_count as f64)
        } else {
            None
        };

        Ok(EvaluationMetrics {
            accuracy: (accuracy * 10000.0).round() / 10000.0,
            soft_accuracy: (soft_accuracy * 10000.0).round() / 10000.0,
            brier_score: (brier_score * 10000.0).round() / 10000.0,
            ece: (ece * 10000.0).round() / 10000.0,
            score_mae: score_mae.map(|v| (v * 10000.0).round() / 10000.0),
            act_escalate_rate: (act_escalate_rate * 10000.0).round() / 10000.0,
            sample_count: total_samples,
        })
    }

    pub fn print_report(metrics: &EvaluationMetrics) {
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Metric").fg(Color::Cyan),
            Cell::new("Value").fg(Color::Green),
            Cell::new("Description / Interpretation").fg(Color::DarkGrey),
        ]);

        table.add_row(vec![
            Cell::new("Total Samples"),
            Cell::new(format!("{}", metrics.sample_count)),
            Cell::new("Evaluated test instances"),
        ]);
        table.add_row(vec![
            Cell::new("Accuracy"),
            Cell::new(format!("{:.2}%", metrics.accuracy * 100.0)),
            Cell::new("Argmax class agreement with ground truth"),
        ]);
        table.add_row(vec![
            Cell::new("Soft Accuracy"),
            Cell::new(format!("{:.4}", metrics.soft_accuracy)),
            Cell::new("Probability mass on true classes (distribution agreement)"),
        ]);
        table.add_row(vec![
            Cell::new("ECE (Expected Calibration Error)"),
            Cell::new(format!("{:.4}", metrics.ece)),
            Cell::new("Lower is better; measures over/under-confidence"),
        ]);
        table.add_row(vec![
            Cell::new("Brier Score"),
            Cell::new(format!("{:.4}", metrics.brier_score)),
            Cell::new("Strictly proper quadratic scoring rule (lower is better)"),
        ]);

        if let Some(mae) = metrics.score_mae {
            table.add_row(vec![
                Cell::new("Score MAE"),
                Cell::new(format!("{:.4}", mae)),
                Cell::new("Mean Absolute Error on ordinal rubrics"),
            ]);
        }

        table.add_row(vec![
            Cell::new("Escalate / Act Rate"),
            Cell::new(format!("{:.2}%", metrics.act_escalate_rate * 100.0)),
            Cell::new("Decisions flagged for human/supervisor escalation"),
        ]);

        println!("\n{table}\n");
    }
}
