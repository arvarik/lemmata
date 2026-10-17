//! Deferral to System 2: decide which items System 1 keeps and which it hands to a larger model or
//! a person.
//!
//! `fit` picks the loosest risk threshold whose kept items reach a target accuracy with a stated
//! confidence. It tests a fixed grid of thresholds with an exact binomial test and a Bonferroni
//! correction, as in Learn then Test (Angelopoulos et al., arXiv:2110.01052). lemmata never
//! calls System 2 itself.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::datasource::DataItem;
use crate::engine::classifier::LayaClassifier;
use crate::model::{DecisionAnswer, DecisionResult, Question};

/// Where an item goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemTier {
    System1Fast,
    System2Deliberative,
}

impl SystemTier {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::System1Fast => "⚡ KEEP (System 1)",
            Self::System2Deliberative => "🧠 DEFER (System 2)",
        }
    }
}

/// Settings for `defer run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeConfig {
    /// Keep an item when its risk is at or below this value.
    pub risk_threshold: f64,
    pub system2_cost_per_call: f64,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            risk_threshold: 0.45,
            system2_cost_per_call: 0.005,
        }
    }
}

/// The routing decision for one item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeDecision {
    pub item_id: String,
    pub tier: SystemTier,
    /// 1 minus the smallest top-option probability across the item's questions.
    pub risk: f64,
    pub system1_result: DecisionResult,
    pub cost_usd: f64,
    /// A prompt for System 2 with the System 1 probabilities, for deferred items only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_seed: Option<String>,
}

/// The fixed risk thresholds that `fit` tests. Fixing them before the data arrives keeps the
/// Bonferroni guarantee valid.
pub const RISK_GRID: [f64; 20] = [
    0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00,
];

/// One tested threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferPoint {
    pub threshold: f64,
    pub kept: usize,
    pub kept_correct: usize,
    /// Share of the labelled items that System 1 keeps.
    pub coverage: f64,
    pub kept_accuracy: f64,
    /// One-sided Clopper-Pearson lower bound on the accuracy of kept items.
    pub accuracy_lower_bound: f64,
    pub deferral_cost_per_1000: f64,
    /// The lower bound reaches the target accuracy.
    pub certified: bool,
}

/// The threshold that `fit` chose. `defer run --policy` reads this file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferPolicy {
    pub checkpoint: String,
    pub threshold: f64,
    pub target_accuracy: f64,
    /// With this probability, the kept items reach the target accuracy on new data from the same
    /// distribution.
    pub confidence: f64,
    pub labelled_items: usize,
    pub coverage: f64,
    pub kept_accuracy: f64,
    pub accuracy_lower_bound: f64,
    pub deferral_rate: f64,
    pub deferral_cost_per_1000: f64,
}

/// The result of `fit`: every tested threshold, and the chosen one when one is certified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferFit {
    pub labelled_items: usize,
    /// Per-threshold error level: (1 - confidence) / number of thresholds.
    pub alpha: f64,
    pub points: Vec<DeferPoint>,
    pub policy: Option<DeferPolicy>,
    /// Kept items, all correct, that the smallest certifiable set needs.
    pub min_items_needed: usize,
}

pub struct CascadeRouter {
    config: CascadeConfig,
}

impl CascadeRouter {
    pub fn new(config: CascadeConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CascadeConfig {
        &self.config
    }

    /// 1 minus the smallest top-option probability across the questions: an item is only as safe
    /// as its least certain answer. An answer without option probabilities uses its confidence.
    pub fn risk(result: &DecisionResult) -> f64 {
        if result.answers.is_empty() {
            return 1.0;
        }
        let least_certain = result
            .answers
            .values()
            .map(top_probability)
            .fold(1.0, f64::min);
        (1.0 - least_certain).clamp(0.0, 1.0)
    }

    /// Classify one item and keep it or defer it.
    pub fn route(
        &self,
        item: &DataItem,
        questions: &BTreeMap<String, Question>,
        classifier: &LayaClassifier,
    ) -> Result<CascadeDecision> {
        let res = classifier.classify(&item.state, questions)?;
        let risk = Self::risk(&res);
        let tier = if risk <= self.config.risk_threshold {
            SystemTier::System1Fast
        } else {
            SystemTier::System2Deliberative
        };
        let (cost_usd, prompt_seed) = match tier {
            SystemTier::System1Fast => (0.0, None),
            SystemTier::System2Deliberative => {
                let state = match &item.state {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other)?,
                };
                let prompt = format!(
                    "Input context:\n{state}\n\nSystem 1 probabilities:\n{}\nSystem 1 risk: {:.1}%\n\nPlease deliberate and provide the final structured classification.",
                    format_priors(&res),
                    risk * 100.0
                );
                (self.config.system2_cost_per_call, Some(prompt))
            }
        };
        Ok(CascadeDecision {
            item_id: item.id.clone(),
            tier,
            risk: (risk * 10000.0).round() / 10000.0,
            system1_result: res,
            cost_usd,
            prompt_seed,
        })
    }

    /// Test every threshold in `RISK_GRID` on labelled items, and choose the loosest one whose
    /// kept items reach `target_accuracy` with probability `confidence`.
    pub fn fit(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        classifier: &LayaClassifier,
        target_accuracy: f64,
        confidence: f64,
        system2_cost: f64,
    ) -> Result<DeferFit> {
        if !(0.0..1.0).contains(&target_accuracy) || !(0.0..1.0).contains(&confidence) {
            bail!("--target-accuracy and --confidence must be between 0 and 1");
        }
        let samples = collect_samples(items, questions, classifier)?;
        Ok(fit_samples(
            &samples,
            target_accuracy,
            confidence,
            system2_cost,
            classifier.config().checkpoint.repo_id(),
        ))
    }
}

/// `fit` on (risk, correct) pairs that are already scored.
pub fn fit_samples(
    samples: &[(f64, bool)],
    target_accuracy: f64,
    confidence: f64,
    system2_cost: f64,
    checkpoint: &str,
) -> DeferFit {
    {
        let n = samples.len();
        let alpha = (1.0 - confidence) / RISK_GRID.len() as f64;

        let points: Vec<DeferPoint> = RISK_GRID
            .iter()
            .map(|&threshold| {
                let kept: Vec<bool> = samples
                    .iter()
                    .filter(|(risk, _)| *risk <= threshold)
                    .map(|&(_, correct)| correct)
                    .collect();
                let kept_correct = kept.iter().filter(|&&c| c).count();
                let lower_bound = binomial_lower_bound(kept_correct, kept.len(), alpha);
                let coverage = kept.len() as f64 / n as f64;
                DeferPoint {
                    threshold,
                    kept: kept.len(),
                    kept_correct,
                    coverage,
                    kept_accuracy: if kept.is_empty() {
                        0.0
                    } else {
                        kept_correct as f64 / kept.len() as f64
                    },
                    accuracy_lower_bound: lower_bound,
                    deferral_cost_per_1000: (1.0 - coverage) * 1000.0 * system2_cost,
                    certified: !kept.is_empty() && lower_bound >= target_accuracy,
                }
            })
            .collect();

        let policy = points
            .iter()
            .filter(|p| p.certified)
            .max_by(|a, b| a.coverage.total_cmp(&b.coverage))
            .map(|p| DeferPolicy {
                checkpoint: checkpoint.to_string(),
                threshold: p.threshold,
                target_accuracy,
                confidence,
                labelled_items: n,
                coverage: p.coverage,
                kept_accuracy: p.kept_accuracy,
                accuracy_lower_bound: p.accuracy_lower_bound,
                deferral_rate: 1.0 - p.coverage,
                deferral_cost_per_1000: p.deferral_cost_per_1000,
            });

        DeferFit {
            labelled_items: n,
            alpha,
            points,
            policy,
            min_items_needed: (alpha.ln() / target_accuracy.ln()).ceil() as usize,
        }
    }
}

/// The highest option probability of an answer.
fn top_probability(answer: &DecisionAnswer) -> f64 {
    if let Some(p) = answer.probabilities.values().copied().reduce(f64::max) {
        return p;
    }
    if let Some(p) = answer.noul {
        return p.max(1.0 - p);
    }
    answer.confidence
}

/// (risk, every question correct) for each item with ground truth.
fn collect_samples(
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    classifier: &LayaClassifier,
) -> Result<Vec<(f64, bool)>> {
    if items.is_empty() {
        bail!("Cannot fit a deferral threshold on an empty dataset");
    }
    let mut samples = Vec::new();
    for (item, (_, res)) in items
        .iter()
        .zip(classifier.classify_batch(items, questions)?.iter())
    {
        let Some(gt) = item.ground_truth.as_ref().filter(|gt| !gt.is_empty()) else {
            continue;
        };
        let mut correct = true;
        let mut matched_any = false;
        for (qid, question) in questions {
            let Some(val) = crate::engine::metrics::resolve_ground_truth_for_question(
                Some(gt),
                qid,
                question,
                questions.len(),
            ) else {
                continue;
            };
            matched_any = true;
            let matches = res.answers.get(qid).is_some_and(|ans| match &ans.qtype {
                crate::model::QuestionType::Choice => {
                    let expected = val.as_str().unwrap_or("");
                    ans.choice
                        .as_deref()
                        .is_some_and(|c| c.eq_ignore_ascii_case(expected))
                        || ans
                            .selected_labels
                            .as_ref()
                            .is_some_and(|v| v.iter().any(|s| s.eq_ignore_ascii_case(expected)))
                }
                crate::model::QuestionType::Score => val
                    .as_f64()
                    .or_else(|| val.as_str().and_then(|s| s.parse::<f64>().ok()))
                    .zip(ans.score)
                    .is_some_and(|(e, s)| (e - s).abs() < 0.5),
                crate::model::QuestionType::Noul => {
                    let expected = val
                        .as_bool()
                        .or_else(|| {
                            val.as_str()
                                .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
                        })
                        .or_else(|| val.as_i64().map(|n| n != 0));
                    expected.is_some() && expected == ans.noul.map(|n| n >= 0.5)
                }
            });
            if !matches {
                correct = false;
                break;
            }
        }
        if matched_any {
            samples.push((CascadeRouter::risk(res), correct));
        }
    }
    if samples.is_empty() {
        bail!("No labelled items with ground truth found. `defer fit` needs labelled data");
    }
    Ok(samples)
}

fn ln_factorials(n: usize) -> Vec<f64> {
    let mut table = vec![0.0; n + 1];
    for i in 1..=n {
        table[i] = table[i - 1] + (i as f64).ln();
    }
    table
}

/// P(X >= k) for X ~ Binomial(n, p).
fn binomial_upper_tail(k: usize, n: usize, p: f64, ln_fact: &[f64]) -> f64 {
    if k == 0 || p >= 1.0 {
        return 1.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    let (ln_p, ln_q) = (p.ln(), (1.0 - p).ln());
    (k..=n)
        .map(|i| {
            (ln_fact[n] - ln_fact[i] - ln_fact[n - i] + i as f64 * ln_p + (n - i) as f64 * ln_q)
                .exp()
        })
        .sum::<f64>()
        .min(1.0)
}

/// One-sided Clopper-Pearson lower bound on a success rate: with probability 1 - `alpha`, the
/// true rate is at least this value. `correct` of `n` trials succeeded.
pub fn binomial_lower_bound(correct: usize, n: usize, alpha: f64) -> f64 {
    if n == 0 || correct == 0 {
        return 0.0;
    }
    let ln_fact = ln_factorials(n);
    // The tail grows with p. Bisect for the p where it reaches alpha, and keep the lower end so
    // the bound never overstates accuracy.
    let (mut lo, mut hi) = (0.0, correct as f64 / n as f64);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if binomial_upper_tail(correct, n, mid, &ln_fact) < alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn format_priors(res: &DecisionResult) -> String {
    let mut out = Vec::new();
    for (qid, ans) in &res.answers {
        let mut prob_list: Vec<String> = ans
            .probabilities
            .iter()
            .map(|(k, v)| format!("{}: {:.1}%", k, v * 100.0))
            .collect();
        prob_list.sort();
        out.push(format!("- {}: [{}]", qid, prob_list.join(", ")));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Checkpoint;

    fn dept_questions() -> BTreeMap<String, Question> {
        BTreeMap::from([(
            "dept".to_string(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        )])
    }

    #[test]
    fn lower_bound_matches_exact_clopper_pearson() {
        // Reference values from exact integer binomial sums in Python.
        assert!((binomial_lower_bound(95, 100, 0.05) - 0.897747).abs() < 1e-5);
        assert!((binomial_lower_bound(95, 100, 0.025) - 0.887165).abs() < 1e-5);
        // All correct: the bound is alpha^(1/n).
        assert!((binomial_lower_bound(20, 20, 0.05) - 0.05f64.powf(1.0 / 20.0)).abs() < 1e-6);
        assert_eq!(binomial_lower_bound(0, 10, 0.05), 0.0);
        assert_eq!(binomial_lower_bound(0, 0, 0.05), 0.0);
    }

    #[test]
    fn route_keeps_low_risk_and_defers_high_risk() {
        let item = DataItem::new(
            "c1",
            serde_json::json!({ "text": "Invoice payment billing refund charge immediate" }),
            "test",
        );
        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let keep_all = CascadeRouter::new(CascadeConfig {
            risk_threshold: 1.0,
            ..Default::default()
        });
        let decision = keep_all
            .route(&item, &dept_questions(), &classifier)
            .unwrap();
        assert_eq!(decision.tier, SystemTier::System1Fast);
        assert_eq!(decision.cost_usd, 0.0);

        let defer_all = CascadeRouter::new(CascadeConfig {
            risk_threshold: -1.0,
            ..Default::default()
        });
        let decision = defer_all
            .route(&item, &dept_questions(), &classifier)
            .unwrap();
        assert_eq!(decision.tier, SystemTier::System2Deliberative);
        assert!(decision.cost_usd > 0.0);
        assert!(decision.prompt_seed.is_some());
    }

    #[test]
    fn fit_chooses_the_loosest_certified_threshold() {
        // 200 confident items, all correct, and 100 risky items, half correct.
        let mut samples = vec![(0.01, true); 200];
        samples.extend((0..100).map(|i| (0.5, i % 2 == 0)));
        let fit = fit_samples(&samples, 0.95, 0.95, 0.01, "test");
        let policy = fit.policy.expect("the confident items alone reach 95%");
        // Thresholds from 0.01 to 0.45 keep the same 200 items, and the loosest of them wins.
        assert!((policy.threshold - 0.45).abs() < 1e-9);
        assert!((policy.coverage - 200.0 / 300.0).abs() < 1e-9);
        assert!(policy.accuracy_lower_bound >= 0.95);
        let at = |t: f64| {
            fit.points
                .iter()
                .find(|p| (p.threshold - t).abs() < 1e-9)
                .unwrap()
        };
        assert!(!at(0.005).certified, "keeps nothing");
        assert!(at(0.45).certified);
        assert!(
            !at(0.5).certified,
            "the risky items drop accuracy to 250 / 300"
        );
        assert!(!at(1.0).certified);
    }

    #[test]
    fn fit_needs_labels_and_enough_items() {
        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let unlabelled = DataItem::new("c3", serde_json::json!({ "text": "Invoice" }), "test");
        assert!(
            CascadeRouter::fit(
                &[unlabelled],
                &dept_questions(),
                &classifier,
                0.95,
                0.95,
                0.005
            )
            .is_err()
        );

        let gt = BTreeMap::from([("dept".to_string(), serde_json::json!("billing"))]);
        let item = DataItem::new(
            "c4",
            serde_json::json!({ "text": "Invoice payment billing refund immediate" }),
            "test",
        )
        .with_ground_truth(gt);
        let fit =
            CascadeRouter::fit(&[item], &dept_questions(), &classifier, 0.95, 0.95, 0.005).unwrap();
        // One item cannot show 95% accuracy at 95% confidence.
        assert!(fit.policy.is_none());
        assert_eq!(fit.points.len(), RISK_GRID.len());
        assert_eq!(fit.min_items_needed, 117);
    }
}
