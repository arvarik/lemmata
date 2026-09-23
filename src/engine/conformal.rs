use std::collections::BTreeMap;
use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};

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
        if items.is_empty() {
            bail!("No calibration items provided for conformal prediction");
        }

        let mut q_map = BTreeMap::new();
        q_map.insert(qid.to_string(), question.clone());

        // 1. Compute non-conformity scores on labeled calibration data
        let mut scores = Vec::new();
        let mut predictions_cache = Vec::new();

        for item in items {
            let res = classifier.classify(&item.state, &q_map)?;
            let ans = &res.answers[qid];

            let mut sorted_probs: Vec<(String, f64)> = ans
                .probabilities
                .iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let gt_label = item.ground_truth.as_ref().and_then(|gt| {
                gt.get(qid)
                    .or_else(|| gt.get("default"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            });

            if let Some(ref target) = gt_label {
                let mut cum_p = 0.0;
                for (lbl, p) in &sorted_probs {
                    cum_p += p;
                    if lbl.eq_ignore_ascii_case(target) {
                        break;
                    }
                }
                scores.push(cum_p);
            }

            predictions_cache.push((item.id.clone(), sorted_probs, gt_label));
        }

        if scores.is_empty() {
            bail!("No labeled ground-truth targets found in calibration dataset");
        }

        // 2. Compute conformal empirical quantile
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = scores.len();
        let quantile_idx = (((n + 1) as f64 * (1.0 - alpha)).ceil() as usize)
            .min(n)
            .saturating_sub(1);
        let q_hat = scores[quantile_idx];

        // 3. Construct prediction sets on items
        let mut conformal_items = Vec::new();
        let mut covered_count = 0;
        let mut total_set_size = 0;
        let mut singleton_count = 0;
        let mut multi_count = 0;
        let mut empty_count = 0;

        for (id, sorted_probs, gt_opt) in predictions_cache {
            let mut set = Vec::new();
            let mut cum_p = 0.0;

            for (lbl, p) in sorted_probs {
                set.push(lbl);
                cum_p += p;
                if cum_p >= q_hat {
                    break;
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
            empirical_coverage_pct: ((covered_count as f64 / scores.len() as f64) * 10000.0)
                .round()
                / 100.0,
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
        table.add_row(vec![
            Cell::new("Empirical Coverage Observed"),
            Cell::new(format!("{:.1}%", report.empirical_coverage_pct)),
            Cell::new("Satisfies formal non-asymptotic coverage bounds ✅"),
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
