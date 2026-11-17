use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::model::{DecisionAnswer, Question, QuestionType};

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

    /// Mean predicted confidence across evaluated decisions.
    #[serde(default)]
    pub avg_confidence: f64,

    /// Mean per-item inference latency in milliseconds.
    #[serde(default)]
    pub avg_latency_ms: f64,
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

/// Winning predicted probability for top-label calibration (ECE).
///
/// Uses the predicted option probability (`p_win` for `Choice`, accounting for `cost_weights`;
/// `max(p, 1 - p)` for `Noul`; and the probability of the rounded ordinal level for `Score`)
/// rather than heuristic negentropy/chance-adjusted confidence.
pub fn winning_probability(answer: &DecisionAnswer) -> f64 {
    match answer.qtype {
        QuestionType::Choice => {
            if let Some(choice) = answer.choice.as_deref()
                && let Some(p) = answer.probabilities.get(choice).copied().or_else(|| {
                    answer
                        .probabilities
                        .iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(choice))
                        .map(|(_, &v)| v)
                })
            {
                return p.clamp(0.0, 1.0);
            }
            answer
                .probabilities
                .values()
                .copied()
                .fold(0.0, f64::max)
                .clamp(0.0, 1.0)
        }
        QuestionType::Noul => {
            let p = answer.noul.unwrap_or(0.5);
            p.max(1.0 - p).clamp(0.0, 1.0)
        }
        QuestionType::Score => {
            if let Some(s) = answer.score {
                let rounded_key = format!("{}", s.round() as i64);
                if let Some(&p) = answer.probabilities.get(&rounded_key) {
                    return p.clamp(0.0, 1.0);
                }
            }
            answer
                .probabilities
                .values()
                .copied()
                .fold(0.0, f64::max)
                .clamp(0.0, 1.0)
        }
    }
}

/// Parse boolean / numeric / string ground-truth values for `Noul` questions consistently.
pub fn parse_noul_ground_truth(val: &serde_json::Value) -> Option<bool> {
    if let Some(b) = val.as_bool() {
        return Some(b);
    }
    if let Some(s) = val.as_str() {
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("true")
            || trimmed.eq_ignore_ascii_case("yes")
            || trimmed.eq_ignore_ascii_case("t")
            || trimmed == "1"
        {
            return Some(true);
        }
        if trimmed.eq_ignore_ascii_case("false")
            || trimmed.eq_ignore_ascii_case("no")
            || trimmed.eq_ignore_ascii_case("f")
            || trimmed == "0"
        {
            return Some(false);
        }
    }
    if let Some(n) = val.as_i64() {
        return Some(n != 0);
    }
    if let Some(n) = val.as_u64() {
        return Some(n != 0);
    }
    if let Some(n) = val.as_f64() {
        return Some(n != 0.0);
    }
    None
}

/// Extract normalized, sorted, deduplicated lowercase label strings when `val` is a JSON array.
pub fn extract_multi_label_targets(val: &serde_json::Value) -> Option<Vec<String>> {
    let arr = val.as_array()?;
    let mut out: Vec<String> = arr
        .iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_lowercase()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    Some(out)
}

/// Resolves the ground-truth value for a specific question from the item's ground-truth map.
/// If an exact key exists (e.g. "department"), it is returned.
/// If only a "default" key exists (e.g. from a single-column CSV "target"):
/// - In a 1-question schema, it is used directly.
/// - In a multi-question schema, it is only matched to a question whose schema type and options
///   are compatible with the value, avoiding false zero-accuracy or spurious evaluations on unrelated questions.
pub fn resolve_ground_truth_for_question<'a>(
    gt_map: Option<&'a BTreeMap<String, serde_json::Value>>,
    qid: &str,
    question: &Question,
    total_questions: usize,
) -> Option<&'a serde_json::Value> {
    let map = gt_map?;
    if let Some(val) = map.get(qid).or_else(|| map.get(&qid.to_ascii_lowercase())) {
        return Some(val);
    }
    let default_val = map.get("default")?;
    if total_questions == 1 {
        return Some(default_val);
    }
    match question.qtype {
        QuestionType::Choice => {
            let options = question.render_options();
            let option_matches = |target_str: &str| {
                options.iter().any(|opt| {
                    let opt_key = opt.split(':').next().unwrap_or(opt).trim();
                    opt_key.eq_ignore_ascii_case(target_str) || opt.eq_ignore_ascii_case(target_str)
                })
            };
            match default_val {
                serde_json::Value::String(s) if option_matches(s) => Some(default_val),
                serde_json::Value::Array(arr)
                    if !arr.is_empty()
                        && arr.iter().all(|v| v.as_str().is_some_and(option_matches)) =>
                {
                    Some(default_val)
                }
                _ => None,
            }
        }
        QuestionType::Score => {
            if default_val.is_number() {
                Some(default_val)
            } else if let Some(s) = default_val.as_str() {
                if s.parse::<f64>().is_ok() {
                    Some(default_val)
                } else {
                    None
                }
            } else {
                None
            }
        }
        QuestionType::Noul => parse_noul_ground_truth(default_val).map(|_| default_val),
    }
}

/// Deterministic 64-bit FNV-1a hash function.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// Format current system time as an ISO-8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
pub fn format_utc_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let z = (secs / 86400) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if mo <= 2 { y + 1 } else { y };
    format!("{year:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
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

    #[test]
    fn test_resolve_ground_truth_for_question() {
        let q_choice = Question::new_choice(
            "Department",
            vec!["billing".to_string(), "tech".to_string()],
        );
        let q_score = Question::new_score("Urgency", vec!["low".to_string(), "high".to_string()]);

        let mut gt = BTreeMap::new();
        gt.insert("default".to_string(), serde_json::json!("billing"));

        // Multi-question schema: "billing" matches dept, but NOT urgency
        assert_eq!(
            resolve_ground_truth_for_question(Some(&gt), "dept", &q_choice, 2),
            Some(&serde_json::json!("billing"))
        );
        assert_eq!(
            resolve_ground_truth_for_question(Some(&gt), "urgency", &q_score, 2),
            None
        );

        // Single-question schema: fallback matches directly
        assert_eq!(
            resolve_ground_truth_for_question(Some(&gt), "dept", &q_choice, 1),
            Some(&serde_json::json!("billing"))
        );
    }
}
