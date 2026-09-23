use serde::{Deserialize, Serialize};

/// Comprehensive statistical metrics for evaluating Laya System 1 decision models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Percentage of top predictions matching the ground-truth target.
    pub accuracy: f64,

    /// Soft accuracy: probability mass assigned to the correct class.
    pub soft_accuracy: f64,

    /// Brier score: strictly proper mean squared error of predicted probabilities vs one-hot targets.
    /// Lower is better.
    pub brier_score: f64,

    /// Expected Calibration Error (ECE) across confidence bins.
    /// Measures the gap between predicted confidence and observed empirical accuracy.
    /// Lower is better.
    pub ece: f64,

    /// Mean Absolute Error for ordinal score questions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_mae: Option<f64>,

    /// Percentage of requests where the model recommended human escalation / supervisor review.
    pub act_escalate_rate: f64,

    /// Total number of evaluated decisions.
    pub sample_count: usize,
}

/// Calculate Expected Calibration Error (ECE) across confidence bins.
/// Formula: ECE = \sum_{b=1}^B \frac{|B_b|}{N} | acc(B_b) - conf(B_b) |
pub fn calculate_ece(confidences: &[f64], correctness: &[bool], bins: usize) -> f64 {
    if confidences.is_empty() || confidences.len() != correctness.len() {
        return 0.0;
    }

    let n = confidences.len() as f64;
    let mut ece = 0.0;

    for b in 0..bins {
        let lo = b as f64 / bins as f64;
        let hi = (b + 1) as f64 / bins as f64;

        let mut bin_count = 0;
        let mut bin_correct = 0;
        let mut bin_conf_sum = 0.0;

        for (&conf, &corr) in confidences.iter().zip(correctness.iter()) {
            let in_bin = if b == bins - 1 {
                conf >= lo && conf <= hi
            } else {
                conf >= lo && conf < hi
            };

            if in_bin {
                bin_count += 1;
                bin_conf_sum += conf;
                if corr {
                    bin_correct += 1;
                }
            }
        }

        if bin_count > 0 {
            let acc = bin_correct as f64 / bin_count as f64;
            let avg_conf = bin_conf_sum / bin_count as f64;
            let weight = bin_count as f64 / n;
            ece += weight * (acc - avg_conf).abs();
        }
    }

    ece
}

/// Calculate Brier Score: \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2
pub fn calculate_brier_score(probabilities: &[Vec<f64>], targets: &[usize]) -> f64 {
    if probabilities.is_empty() || probabilities.len() != targets.len() {
        return 0.0;
    }

    let mut sum_squared_errors = 0.0;
    for (probs, &target) in probabilities.iter().zip(targets.iter()) {
        for (k, &p) in probs.iter().enumerate() {
            let y = if k == target { 1.0 } else { 0.0 };
            sum_squared_errors += (p - y).powi(2);
        }
    }

    sum_squared_errors / probabilities.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_ece() {
        let confs = vec![1.0, 1.0, 0.5, 0.5];
        let correct = vec![true, true, true, false];
        let ece = calculate_ece(&confs, &correct, 10);
        assert!(ece < 0.01);
    }

    #[test]
    fn test_brier_score() {
        let probs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![0, 1];
        let brier = calculate_brier_score(&probs, &targets);
        assert_eq!(brier, 0.0);
    }
}
