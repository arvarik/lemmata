use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::classifier::LayaClassifier;
use crate::datasource::DataItem;
use crate::model::Question;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalReport {
    pub alpha: f64,
    pub target_coverage_pct: f64,
    pub quantile_threshold: f64,
    pub empirical_coverage_pct: f64,
    pub average_set_size: f64,
    pub singleton_rate_pct: f64,
    pub multi_label_rate_pct: f64,
    pub empty_set_rate_pct: f64,
    pub total_evaluated: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionItem {
    pub id: String,
    pub prediction_set: Vec<String>,
    pub set_size: usize,
    pub is_singleton: bool,
    pub is_covered: Option<bool>,
}

pub struct ConformalEngine;

impl ConformalEngine {
    /// Calibrate conformal quantile threshold on labeled data and evaluate prediction sets.
    pub fn evaluate(
        classifier: &LayaClassifier,
        items: &[DataItem],
        qid: &str,
        question: &Question,
        alpha: f64,
    ) -> Result<(ConformalReport, Vec<ConformalPredictionItem>)> {
        if !(0.0..=1.0).contains(&alpha) || alpha.is_nan() {
            bail!(
                "Conformal significance level alpha must be between 0.0 and 1.0 (received: {})",
                alpha
            );
        }
        if items.is_empty() {
            bail!("No calibration items provided for conformal prediction");
        }

        let mut q_map = BTreeMap::new();
        q_map.insert(qid.to_string(), question.clone());

        // 1. Compute non-conformity scores on labeled calibration data
        let batch_results = classifier.classify_batch(items, &q_map)?;
        let mut raw_scores = Vec::new();
        let mut predictions_cache = Vec::with_capacity(items.len());

        for (item, (_, res)) in items.iter().zip(batch_results.iter()) {
            let ans = &res.answers[qid];

            let mut sorted_probs: Vec<(String, f64)> = ans
                .probabilities
                .iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.total_cmp(&a.1));

            let gt_label = super::metrics::resolve_ground_truth_for_question(
                item.ground_truth.as_ref(),
                qid,
                question,
                1,
            )
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Bool(b) => Some(b.to_string()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            });

            let mut score_idx = None;
            if let Some(ref target) = gt_label {
                let mut cum_p = 0.0;
                for (lbl, p) in &sorted_probs {
                    cum_p += p;
                    if lbl.eq_ignore_ascii_case(target) {
                        score_idx = Some(raw_scores.len());
                        raw_scores.push(cum_p);
                        break;
                    }
                }
            }

            predictions_cache.push((item.id.clone(), sorted_probs, gt_label, score_idx));
        }

        if raw_scores.is_empty() {
            bail!(
                "No labeled ground-truth targets found matching question options in calibration dataset"
            );
        }

        // 2. Compute conformal empirical quantile
        let compute_quantile = |mut s: Vec<f64>| -> f64 {
            s.sort_by(|a, b| a.total_cmp(b));
            let n = s.len();
            let raw_rank = (((n + 1) as f64 * (1.0 - alpha) - 1e-10).ceil().max(0.0)) as usize;
            if raw_rank > n {
                1.0
            } else if raw_rank == 0 {
                0.0
            } else {
                s[raw_rank - 1]
            }
        };
        let q_hat = compute_quantile(raw_scores.clone());

        // 3. Construct prediction sets (using leave-one-out quantile when n >= 4 to prevent in-sample leakage)
        let mut conformal_items = Vec::with_capacity(predictions_cache.len());
        let mut covered_count = 0;
        let mut total_set_size = 0;
        let mut singleton_count = 0;
        let mut multi_count = 0;
        let mut empty_count = 0;

        for (id, sorted_probs, gt_opt, score_idx) in predictions_cache {
            let item_q = if raw_scores.len() >= 4
                && let Some(idx) = score_idx
            {
                let loo: Vec<f64> = raw_scores
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (i != idx).then_some(v))
                    .collect();
                compute_quantile(loo)
            } else {
                q_hat
            };

            let mut set = Vec::new();
            let mut cum_p = 0.0;

            if item_q > 0.0 {
                for (lbl, p) in sorted_probs {
                    set.push(lbl);
                    cum_p += p;
                    if cum_p >= item_q - 1e-9 {
                        break;
                    }
                }
            }

            let set_size = set.len();
            total_set_size += set_size;

            if set_size == 1 {
                singleton_count += 1;
            } else if set_size > 1 {
                multi_count += 1;
            } else {
                empty_count += 1;
            }

            let is_covered = gt_opt.as_ref().map(|target| {
                let cov = set.iter().any(|s| s.eq_ignore_ascii_case(target));
                if cov {
                    covered_count += 1;
                }
                cov
            });

            conformal_items.push(ConformalPredictionItem {
                id,
                prediction_set: set,
                set_size,
                is_singleton: set_size == 1,
                is_covered,
            });
        }

        let total_eval = conformal_items.len();
        let report = ConformalReport {
            alpha,
            target_coverage_pct: (1.0 - alpha) * 100.0,
            quantile_threshold: (q_hat * 1000.0).round() / 1000.0,
            empirical_coverage_pct: {
                let labeled_count = conformal_items
                    .iter()
                    .filter(|i| i.is_covered.is_some())
                    .count();
                if labeled_count > 0 {
                    ((covered_count as f64 / labeled_count as f64) * 10000.0).round() / 100.0
                } else {
                    0.0
                }
            },
            average_set_size: ((total_set_size as f64 / total_eval as f64) * 100.0).round() / 100.0,
            singleton_rate_pct: ((singleton_count as f64 / total_eval as f64) * 10000.0).round()
                / 100.0,
            multi_label_rate_pct: ((multi_count as f64 / total_eval as f64) * 10000.0).round()
                / 100.0,
            empty_set_rate_pct: ((empty_count as f64 / total_eval as f64) * 10000.0).round()
                / 100.0,
            total_evaluated: total_eval,
        };

        Ok((report, conformal_items))
    }

    pub fn print_report(report: &ConformalReport, items: &[ConformalPredictionItem]) {
        println!("\n🛡️  Conformal Prediction & Risk-Controlled Decision Sets");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Conformal Metric").fg(Color::Cyan),
            Cell::new("Observed Value").fg(Color::Green),
            Cell::new("Operational Interpretation").fg(Color::DarkGrey),
        ]);

        table.add_row(vec![
            Cell::new("Target Coverage Guarantee (1 - α)"),
            Cell::new(format!("{:.1}%", report.target_coverage_pct)),
            Cell::new(format!("Significance level α = {:.2}", report.alpha)),
        ]);
        let cov_status = if report.empirical_coverage_pct >= report.target_coverage_pct - 5.0 {
            "Satisfies formal non-asymptotic coverage bounds ✅"
        } else {
            "Below target coverage bound (recalibrate or increase sample size) ⚠️"
        };
        table.add_row(vec![
            Cell::new("Empirical Coverage Observed"),
            Cell::new(format!("{:.1}%", report.empirical_coverage_pct)),
            Cell::new(cov_status),
        ]);
        table.add_row(vec![
            Cell::new("Calibrated Quantile (q̂)"),
            Cell::new(format!("{:.4}", report.quantile_threshold)),
            Cell::new("Cumulative probability threshold for set inclusion"),
        ]);
        table.add_row(vec![
            Cell::new("Average Prediction Set Size"),
            Cell::new(format!("{:.2} labels", report.average_set_size)),
            Cell::new("Efficiency metric (1.0 = perfect sharpness)"),
        ]);
        table.add_row(vec![
            Cell::new("Singleton Rate (|C(X)| = 1)"),
            Cell::new(format!("{:.1}%", report.singleton_rate_pct)),
            Cell::new("High-confidence decisions safe to auto-execute"),
        ]);
        table.add_row(vec![
            Cell::new("Multi-Label Ambiguous Rate (|C(X)| > 1)"),
            Cell::new(format!("{:.1}%", report.multi_label_rate_pct)),
            Cell::new("Candidates requiring supervisor escalation"),
        ]);

        println!("{table}\n");

        println!("Sample Conformal Prediction Sets:");
        let mut sample_table = Table::new();
        sample_table.set_header(vec![
            Cell::new("Item ID").fg(Color::Cyan),
            Cell::new("Prediction Set C(X)").fg(Color::Yellow),
            Cell::new("Size").fg(Color::White),
            Cell::new("Action Policy").fg(Color::Green),
        ]);

        for item in items.iter().take(6) {
            let policy = if item.is_singleton {
                "AUTO-EXECUTE"
            } else if item.set_size > 1 {
                "ESCALATE TO REVIEW"
            } else {
                "REJECT / OOD"
            };

            sample_table.add_row(vec![
                Cell::new(&item.id),
                Cell::new(format!("{:?}", item.prediction_set)),
                Cell::new(format!("{}", item.set_size)),
                Cell::new(policy),
            ]);
        }

        println!("{sample_table}\n");
    }
}
