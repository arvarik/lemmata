use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::classifier::LayaClassifier;
use crate::datasource::DataItem;
use crate::model::{Question, QuestionType};

/// Drift severity status for production observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftSeverity {
    Stable,
    Moderate,
    Severe,
}

impl DriftSeverity {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::Stable => "🟢 STABLE",
            Self::Moderate => "🟡 MODERATE DRIFT",
            Self::Severe => "🔴 SEVERE DRIFT",
        }
    }
}

/// Detailed drift metrics for an individual question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionDriftMetrics {
    pub qid: String,
    pub qtype: QuestionType,
    pub confidence_psi: f64,
    pub confidence_wasserstein: f64,
    pub act_head_wasserstein: Option<f64>,
    pub class_prior_psi: Option<f64>,
    pub class_prior_jsd: Option<f64>,
    pub baseline_distribution: BTreeMap<String, f64>,
    pub current_distribution: BTreeMap<String, f64>,
    pub overall_status: DriftSeverity,
}

/// Comprehensive report comparing baseline vs live production input distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub baseline_count: usize,
    pub current_count: usize,
    pub psi_threshold: f64,
    pub wasserstein_threshold: f64,
    pub question_metrics: BTreeMap<String, QuestionDriftMetrics>,
    pub has_critical_drift: bool,
    pub has_moderate_drift: bool,
}

pub struct DriftEngine<'a> {
    classifier: &'a LayaClassifier,
    psi_threshold: f64,
    wasserstein_threshold: f64,
}

impl<'a> DriftEngine<'a> {
    pub fn new(classifier: &'a LayaClassifier) -> Self {
        Self {
            classifier,
            psi_threshold: 0.20,
            wasserstein_threshold: 0.15,
        }
    }

    pub fn with_thresholds(mut self, psi_threshold: f64, wasserstein_threshold: f64) -> Self {
        self.psi_threshold = psi_threshold;
        self.wasserstein_threshold = wasserstein_threshold;
        self
    }

    /// Compare reference baseline dataset with current production dataset.
    pub fn detect_drift(
        &self,
        baseline_items: &[DataItem],
        current_items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<DriftReport> {
        if baseline_items.is_empty() {
            bail!("Baseline items dataset is empty");
        }
        if current_items.is_empty() {
            bail!("Current items dataset is empty");
        }

        let base_results = self.classifier.classify_batch(baseline_items, questions)?;
        let curr_results = self.classifier.classify_batch(current_items, questions)?;

        let mut question_metrics = BTreeMap::new();
        let mut has_critical_drift = false;
        let mut has_moderate_drift = false;

        for (qid, q) in questions {
            // 1. Gather baseline and current confidences
            let base_confs: Vec<f64> = base_results
                .iter()
                .filter_map(|(_, r)| r.answers.get(qid).map(|a| a.confidence))
                .collect();
            let curr_confs: Vec<f64> = curr_results
                .iter()
                .filter_map(|(_, r)| r.answers.get(qid).map(|a| a.confidence))
                .collect();

            // 2. Gather act probabilities
            let base_acts: Vec<f64> = base_results
                .iter()
                .filter_map(|(_, r)| r.answers.get(qid).and_then(|a| a.act_probability))
                .collect();
            let curr_acts: Vec<f64> = curr_results
                .iter()
                .filter_map(|(_, r)| r.answers.get(qid).and_then(|a| a.act_probability))
                .collect();

            // Continuous Wasserstein-1 Distance on confidence distributions
            let conf_wasserstein = calculate_wasserstein_1(&base_confs, &curr_confs, 100);

            // Binned PSI on confidence (10 uniform bins: [0.0..0.1), [0.1..0.2), ...)
            let base_binned_conf = bin_continuous_values(&base_confs, 10);
            let curr_binned_conf = bin_continuous_values(&curr_confs, 10);
            let conf_psi = calculate_psi(&base_binned_conf, &curr_binned_conf);

            // Wasserstein on Act/Escalate head
            let act_wasserstein = if !base_acts.is_empty() && !curr_acts.is_empty() {
                Some(calculate_wasserstein_1(&base_acts, &curr_acts, 100))
            } else {
                None
            };

            // Categorical distributions for Choice / Score / Noul
            let mut base_dist = BTreeMap::new();
            let mut curr_dist = BTreeMap::new();
            let mut class_prior_psi = None;
            let mut class_prior_jsd = None;

            if q.qtype == QuestionType::Choice || q.qtype == QuestionType::Noul {
                let options = q.render_options();
                let keys: Vec<String> = if q.qtype == QuestionType::Noul {
                    vec!["false".into(), "true".into()]
                } else if let Some(serde_json::Value::Object(map)) = &q.criteria {
                    map.keys().cloned().collect()
                } else {
                    options.clone()
                };

                let count_labels = |res: &[(String, crate::model::DecisionResult)]| {
                    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
                    for (_, r) in res {
                        if let Some(ans) = r.answers.get(qid) {
                            if let Some(c) = &ans.choice {
                                *counts.entry(c.clone()).or_default() += 1;
                            } else if let Some(n) = ans.noul {
                                *counts
                                    .entry(if n > 0.5 { "true" } else { "false" }.to_string())
                                    .or_default() += 1;
                            }
                        }
                    }
                    counts
                };
                let base_counts = count_labels(&base_results);
                let curr_counts = count_labels(&curr_results);

                let base_total = base_results.len().max(1) as f64;
                let curr_total = curr_results.len().max(1) as f64;

                let mut base_probs = Vec::new();
                let mut curr_probs = Vec::new();

                for k in &keys {
                    let bp = *base_counts.get(k).unwrap_or(&0) as f64 / base_total;
                    let cp = *curr_counts.get(k).unwrap_or(&0) as f64 / curr_total;
                    base_dist.insert(k.clone(), (bp * 10000.0).round() / 10000.0);
                    curr_dist.insert(k.clone(), (cp * 10000.0).round() / 10000.0);
                    base_probs.push(bp);
                    curr_probs.push(cp);
                }

                class_prior_psi = Some(calculate_psi(&base_probs, &curr_probs));
                class_prior_jsd = Some(calculate_jsd(&base_probs, &curr_probs));
            }

            // Determine status
            let is_severe = conf_psi >= self.psi_threshold
                || conf_wasserstein >= self.wasserstein_threshold
                || class_prior_psi
                    .map(|p| p >= self.psi_threshold)
                    .unwrap_or(false);

            let is_moderate = conf_psi >= self.psi_threshold * 0.5
                || conf_wasserstein >= self.wasserstein_threshold * 0.6
                || class_prior_psi
                    .map(|p| p >= self.psi_threshold * 0.5)
                    .unwrap_or(false);

            let status = if is_severe {
                DriftSeverity::Severe
            } else if is_moderate {
                DriftSeverity::Moderate
            } else {
                DriftSeverity::Stable
            };

            if status == DriftSeverity::Severe {
                has_critical_drift = true;
            } else if status == DriftSeverity::Moderate {
                has_moderate_drift = true;
            }

            question_metrics.insert(
                qid.clone(),
                QuestionDriftMetrics {
                    qid: qid.clone(),
                    qtype: q.qtype,
                    confidence_psi: (conf_psi * 10000.0).round() / 10000.0,
                    confidence_wasserstein: (conf_wasserstein * 10000.0).round() / 10000.0,
                    act_head_wasserstein: act_wasserstein.map(|w| (w * 10000.0).round() / 10000.0),
                    class_prior_psi: class_prior_psi.map(|p| (p * 10000.0).round() / 10000.0),
                    class_prior_jsd: class_prior_jsd.map(|j| (j * 10000.0).round() / 10000.0),
                    baseline_distribution: base_dist,
                    current_distribution: curr_dist,
                    overall_status: status,
                },
            );
        }

        Ok(DriftReport {
            baseline_count: baseline_items.len(),
            current_count: current_items.len(),
            psi_threshold: self.psi_threshold,
            wasserstein_threshold: self.wasserstein_threshold,
            question_metrics,
            has_critical_drift,
            has_moderate_drift,
        })
    }

    pub fn print_report(report: &DriftReport) {
        println!(
            "\n📊 Lemma Distribution Drift & Concept Shift Report\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nReference Baseline: {} samples | Live Production: {} samples\nConfigured Thresholds: PSI >= {:.2} (Critical), W1 >= {:.2} (Critical)\n",
            report.baseline_count,
            report.current_count,
            report.psi_threshold,
            report.wasserstein_threshold
        );

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Question ID").fg(Color::Cyan),
            Cell::new("Type").fg(Color::DarkGrey),
            Cell::new("Conf PSI").fg(Color::Yellow),
            Cell::new("Conf W1 (EMD)").fg(Color::Yellow),
            Cell::new("Prior PSI").fg(Color::Blue),
            Cell::new("Prior JSD").fg(Color::Blue),
            Cell::new("Act Head W1").fg(Color::Magenta),
            Cell::new("Drift Status").fg(Color::White),
        ]);

        let fmt_opt = |v: Option<f64>| v.map_or_else(|| "N/A".into(), |x| format!("{x:.4}"));
        for (qid, m) in &report.question_metrics {
            let color = match m.overall_status {
                DriftSeverity::Stable => Color::Green,
                DriftSeverity::Moderate => Color::Yellow,
                DriftSeverity::Severe => Color::Red,
            };
            table.add_row(vec![
                Cell::new(qid),
                Cell::new(format!("{:?}", m.qtype)),
                Cell::new(format!("{:.4}", m.confidence_psi)),
                Cell::new(format!("{:.4}", m.confidence_wasserstein)),
                Cell::new(fmt_opt(m.class_prior_psi)),
                Cell::new(fmt_opt(m.class_prior_jsd)),
                Cell::new(fmt_opt(m.act_head_wasserstein)),
                Cell::new(m.overall_status.badge()).fg(color),
            ]);
        }

        println!("{table}\n");
        if report.has_critical_drift {
            println!(
                "🚨 CRITICAL DRIFT DETECTED!\n   Production distributions have diverged significantly from baseline.\n   Calibrated probabilities may fail silently without recalibration.\n"
            );
        } else if report.has_moderate_drift {
            println!(
                "⚠️  Moderate drift detected in one or more questions.\n   Monitor calibration ECE and consider scheduling temperature refitting.\n"
            );
        } else {
            println!("✅ Distribution is STABLE: no significant concept drift detected.\n");
        }
    }
}

/// Calculate Population Stability Index (PSI) between two discrete distributions.
/// PSI = sum( (P_curr - P_base) * ln( (P_curr + eps) / (P_base + eps) ) )
pub fn calculate_psi(baseline_probs: &[f64], current_probs: &[f64]) -> f64 {
    let eps = 1e-6;
    (0..baseline_probs.len().max(current_probs.len()))
        .map(|i| {
            let p_base = baseline_probs.get(i).copied().unwrap_or(0.0);
            let p_curr = current_probs.get(i).copied().unwrap_or(0.0);
            (p_curr - p_base) * ((p_curr + eps) / (p_base + eps)).ln()
        })
        .sum::<f64>()
        .max(0.0)
}

/// Two-sample Kolmogorov-Smirnov (KS) test statistic D = sup_x |F_1(x) - F_2(x)|
pub fn calculate_ks_statistic(sample_a: &[f64], sample_b: &[f64]) -> f64 {
    if sample_a.is_empty() || sample_b.is_empty() {
        return 0.0;
    }
    let mut a = sample_a.to_vec();
    let mut b = sample_b.to_vec();
    a.sort_by(|x, y| x.total_cmp(y));
    b.sort_by(|x, y| x.total_cmp(y));

    let (n_a, n_b) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut max_diff) = (0, 0, 0.0f64);
    while i < a.len() || j < b.len() {
        let val = match (a.get(i), b.get(j)) {
            (Some(&va), Some(&vb)) => va.min(vb),
            (Some(&va), None) => va,
            (None, Some(&vb)) => vb,
            (None, None) => break,
        };
        while i < a.len() && a[i] <= val {
            i += 1;
        }
        while j < b.len() && b[j] <= val {
            j += 1;
        }
        max_diff = max_diff.max((i as f64 / n_a - j as f64 / n_b).abs());
    }
    max_diff.clamp(0.0, 1.0)
}

/// Two-sample Kolmogorov-Smirnov asymptotic p-value using the Kolmogorov limiting distribution.
pub fn calculate_ks_p_value(d_stat: f64, n_a: usize, n_b: usize) -> f64 {
    if n_a == 0 || n_b == 0 || d_stat <= 0.0 {
        return 1.0;
    }
    let n_eff = (n_a * n_b) as f64 / (n_a + n_b) as f64;
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * d_stat;
    if lambda < 0.20 {
        return 1.0;
    }
    let mut p = 0.0;
    for k in 1..=100 {
        let term = 2.0 * (-1.0f64).powi(k - 1) * (-2.0 * (k as f64).powi(2) * lambda.powi(2)).exp();
        p += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    p.clamp(0.0, 1.0)
}

/// Calculate Wasserstein-1 Distance (Earth Mover's Distance) on 1D empirical distributions
/// using quantile sampling.
pub fn calculate_wasserstein_1(sample_a: &[f64], sample_b: &[f64], num_quantiles: usize) -> f64 {
    if sample_a.is_empty() || sample_b.is_empty() || num_quantiles == 0 {
        return 0.0;
    }
    let mut a_sorted = sample_a.to_vec();
    let mut b_sorted = sample_b.to_vec();
    a_sorted.sort_by(|x, y| x.total_cmp(y));
    b_sorted.sort_by(|x, y| x.total_cmp(y));

    let sum_diff: f64 = (0..num_quantiles)
        .map(|i| {
            let q = (i as f64 + 0.5) / num_quantiles as f64;
            let idx_a = ((a_sorted.len() - 1) as f64 * q).round() as usize;
            let idx_b = ((b_sorted.len() - 1) as f64 * q).round() as usize;
            (a_sorted[idx_a] - b_sorted[idx_b]).abs()
        })
        .sum();
    sum_diff / num_quantiles as f64
}

/// Calculate Jensen-Shannon Divergence between two probability distributions.
pub fn calculate_jsd(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len().max(q.len());
    if n == 0 {
        return 0.0;
    }
    let eps = 1e-9;
    let mut jsd = 0.0;
    for i in 0..n {
        let pi = p.get(i).copied().unwrap_or(0.0);
        let qi = q.get(i).copied().unwrap_or(0.0);
        let mi = 0.5 * (pi + qi);
        if pi > eps {
            jsd += 0.5 * pi * (pi / (mi + eps)).ln();
        }
        if qi > eps {
            jsd += 0.5 * qi * (qi / (mi + eps)).ln();
        }
    }
    if jsd.is_nan() || jsd < 0.0 { 0.0 } else { jsd }
}

/// Bin continuous values into `num_bins` uniform intervals over [0.0, 1.0].
fn bin_continuous_values(values: &[f64], num_bins: usize) -> Vec<f64> {
    let mut counts = vec![0usize; num_bins];
    if values.is_empty() {
        return vec![1.0 / num_bins as f64; num_bins];
    }

    for &v in values {
        let clamped = v.clamp(0.0, 0.9999);
        let bin_idx = (clamped * num_bins as f64).floor() as usize;
        if bin_idx < num_bins {
            counts[bin_idx] += 1;
        }
    }

    let total = values.len() as f64;
    counts.into_iter().map(|c| c as f64 / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_psi_unequal_lengths() {
        let base = vec![0.5, 0.5];
        let curr = vec![0.4, 0.4, 0.2];
        let psi = calculate_psi(&base, &curr);
        assert!(psi >= 0.0);
    }

    #[test]
    fn test_calculate_ks_statistic() {
        let s1 = vec![0.1, 0.2, 0.3, 0.4];
        let s2 = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(calculate_ks_statistic(&s1, &s2), 0.0);

        // Identical distributions with tied values and different lengths must have 0.0 distance
        let tied_a = vec![1.0, 1.0, 1.0];
        let tied_b = vec![1.0, 1.0];
        assert_eq!(calculate_ks_statistic(&tied_a, &tied_b), 0.0);

        let left = vec![0.1, 0.2, 0.3];
        let right = vec![0.8, 0.9, 1.0];
        assert_eq!(calculate_ks_statistic(&left, &right), 1.0);

        let p_val_ident = calculate_ks_p_value(0.0, 100, 100);
        assert_eq!(p_val_ident, 1.0);

        let p_val_drift = calculate_ks_p_value(1.0, 50, 50);
        assert!(p_val_drift < 0.001);
    }

    #[test]
    fn test_bin_continuous_values() {
        let vals = vec![0.05, 0.15, 0.85];
        let bins = bin_continuous_values(&vals, 10);
        assert_eq!(bins.len(), 10);
        let sum: f64 = bins.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
