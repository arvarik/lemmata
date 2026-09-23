use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::datasource::DataItem;
use crate::engine::classifier::LayaClassifier;
use crate::model::{DecisionResult, Question};

/// Target execution tier in a hybrid System 1 / System 2 cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemTier {
    System1Fast,
    System2Deliberative,
}

impl SystemTier {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::System1Fast => "⚡ SYSTEM 1 (Laya 33ms)",
            Self::System2Deliberative => "🧠 SYSTEM 2 (Frontier LLM)",
        }
    }
}

/// Configuration for cost-aware cascading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeConfig {
    pub entropy_weight: f64,
    pub margin_weight: f64,
    pub act_weight: f64,
    pub risk_threshold: f64,
    pub system2_cost_per_call: f64,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            entropy_weight: 0.40,
            margin_weight: 0.35,
            act_weight: 0.25,
            risk_threshold: 0.45,
            system2_cost_per_call: 0.005, // $0.005 per System 2 call
        }
    }
}

/// Cascaded routing decision for a single input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeDecision {
    pub item_id: String,
    pub tier: SystemTier,
    pub composite_risk: f64,
    pub normalized_entropy: f64,
    pub top2_margin: f64,
    pub act_probability: f64,
    pub system1_result: DecisionResult,
    pub cost_usd: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_seed: Option<String>,
}

/// Summary report of a Pareto frontier optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadePolicy {
    pub optimal_threshold: f64,
    pub target_accuracy: f64,
    pub expected_accuracy: f64,
    pub system1_coverage_rate: f64,
    pub system2_escalation_rate: f64,
    pub cost_savings_percent: f64,
    pub estimated_cost_per_1000: f64,
}

/// Cost-aware hybrid System 1 / System 2 router.
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

    /// Calculate composite risk score R(x) for a decision result.
    pub fn compute_risk(&self, result: &DecisionResult) -> (f64, f64, f64, f64) {
        if result.answers.is_empty() {
            return (1.0, 1.0, 0.0, 1.0);
        }

        let mut entropies = Vec::new();
        let mut margins = Vec::new();
        let mut act_probs = Vec::new();

        for ans in result.answers.values() {
            let k = ans.probabilities.len().max(2);
            let mut probs: Vec<f64> = ans.probabilities.values().copied().collect();
            probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Normalized Shannon Entropy: H(P) / ln(K)
            let h: f64 = -probs
                .iter()
                .filter(|&&p| p > 1e-9)
                .map(|&p| p * p.ln())
                .sum::<f64>()
                / (k as f64).ln().max(1e-9);
            entropies.push(h.clamp(0.0, 1.0));

            // Top-2 Probability Margin: p_1 - p_2
            let p1 = probs.first().copied().unwrap_or(0.0);
            let p2 = probs.get(1).copied().unwrap_or(0.0);
            let margin = (p1 - p2).clamp(0.0, 1.0);
            margins.push(margin);

            let act = ans.act_probability.unwrap_or(0.1);
            act_probs.push(act.clamp(0.0, 1.0));
        }

        let avg_h = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let avg_margin = margins.iter().sum::<f64>() / margins.len() as f64;
        let avg_act = act_probs.iter().sum::<f64>() / act_probs.len() as f64;

        // Composite risk R(x) in [0, 1]
        let risk = self.config.entropy_weight * avg_h
            + self.config.margin_weight * (1.0 - avg_margin)
            + self.config.act_weight * avg_act;

        (
            (risk * 10000.0).round() / 10000.0,
            (avg_h * 10000.0).round() / 10000.0,
            (avg_margin * 10000.0).round() / 10000.0,
            (avg_act * 10000.0).round() / 10000.0,
        )
    }

    /// Route a single item through the cascade.
    pub fn route(
        &self,
        item: &DataItem,
        questions: &BTreeMap<String, Question>,
        classifier: &LayaClassifier,
    ) -> Result<CascadeDecision> {
        let res = classifier.classify(&item.state, questions)?;
        let (risk, norm_entropy, margin, act_prob) = self.compute_risk(&res);

        let tier = if risk <= self.config.risk_threshold {
            SystemTier::System1Fast
        } else {
            SystemTier::System2Deliberative
        };

        let cost_usd = match tier {
            SystemTier::System1Fast => 0.0,
            SystemTier::System2Deliberative => self.config.system2_cost_per_call,
        };

        let prompt_seed = if tier == SystemTier::System2Deliberative {
            let state_str = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other)?,
            };
            let avg_conf: f64 = if !res.answers.is_empty() {
                res.answers.values().map(|a| a.confidence).sum::<f64>() / res.answers.len() as f64
            } else {
                1.0 - risk
            };
            Some(format!(
                "Input context:\n{}\n\nSystem 1 Priors:\n{}\nSystem 1 Confidence: {:.1}%\nAct/Escalate Signal: {:.1}%\n\nPlease deliberate and provide final structured classification.",
                state_str,
                format_priors(&res),
                avg_conf * 100.0,
                act_prob * 100.0
            ))
        } else {
            None
        };

        Ok(CascadeDecision {
            item_id: item.id.clone(),
            tier,
            composite_risk: risk,
            normalized_entropy: norm_entropy,
            top2_margin: margin,
            act_probability: act_prob,
            system1_result: res,
            cost_usd,
            prompt_seed,
        })
    }

    /// Tune the Pareto-optimal threshold tau* to satisfy target accuracy with minimal cost using default config.
    pub fn tune_pareto(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        classifier: &LayaClassifier,
        target_accuracy: f64,
        system2_cost: f64,
    ) -> Result<CascadePolicy> {
        Self::tune_pareto_with_config(
            items,
            questions,
            classifier,
            target_accuracy,
            system2_cost,
            &CascadeConfig::default(),
        )
    }

    /// Tune the Pareto-optimal threshold tau* with custom router config weights.
    pub fn tune_pareto_with_config(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        classifier: &LayaClassifier,
        target_accuracy: f64,
        system2_cost: f64,
        config: &CascadeConfig,
    ) -> Result<CascadePolicy> {
        if items.is_empty() {
            bail!("Cannot tune cascade on empty dataset");
        }

        let router = Self::new(config.clone());
        let mut sample_evals: Vec<(f64, bool)> = Vec::new(); // (risk, is_s1_correct)

        for item in items {
            let gt = match &item.ground_truth {
                Some(g) if !g.is_empty() => g,
                _ => continue, // Skip unlabeled items for accuracy tuning
            };

            let res = classifier.classify(&item.state, questions)?;
            let (risk, _, _, _) = router.compute_risk(&res);

            // Determine if System 1 prediction matches ground truth across all questions
            let mut correct = true;
            for (qid, val) in gt {
                if let Some(ans) = res.answers.get(qid) {
                    let matches = match &ans.qtype {
                        crate::model::QuestionType::Choice => {
                            let expected = match val {
                                serde_json::Value::String(s) => s.as_str(),
                                _ => "",
                            };
                            ans.choice.as_deref() == Some(expected)
                                || ans
                                    .selected_labels
                                    .as_ref()
                                    .map(|v| v.iter().any(|s| s == expected))
                                    .unwrap_or(false)
                        }
                        crate::model::QuestionType::Score => {
                            let expected = match val {
                                serde_json::Value::Number(n) => n.as_f64(),
                                serde_json::Value::String(s) => s.parse::<f64>().ok(),
                                _ => None,
                            };
                            match (expected, ans.score) {
                                (Some(e), Some(s)) => (e - s).abs() < 0.5,
                                _ => false,
                            }
                        }
                        crate::model::QuestionType::Noul => {
                            let expected = match val {
                                serde_json::Value::Bool(b) => Some(*b),
                                serde_json::Value::String(s) => s.parse::<bool>().ok(),
                                serde_json::Value::Number(n) => Some(n.as_i64().unwrap_or(0) != 0),
                                _ => None,
                            };
                            let pred_bool = ans.noul.map(|n| n > 0.5);
                            expected.is_some() && expected == pred_bool
                        }
                    };
                    if !matches {
                        correct = false;
                        break;
                    }
                } else {
                    correct = false;
                    break;
                }
            }
            sample_evals.push((risk, correct));
        }

        if sample_evals.is_empty() {
            bail!("No labeled items with ground truth found for Pareto tuning");
        }

        // Sweep candidate thresholds tau in [0.10, 0.90]
        let candidate_taus: Vec<f64> = (10..=90).map(|i| i as f64 / 100.0).collect();
        let mut best_tau = 0.45;
        let mut best_cost = f64::INFINITY;
        let mut best_acc = 0.0;
        let mut best_s1_cov = 0.0;

        let n = sample_evals.len() as f64;
        let assumed_s2_acc = 0.98; // Frontier LLM average accuracy

        for &tau in &candidate_taus {
            let mut s1_count = 0.0;
            let mut s1_correct = 0.0;
            let mut s2_count = 0.0;

            for &(risk, is_correct) in &sample_evals {
                if risk <= tau {
                    s1_count += 1.0;
                    if is_correct {
                        s1_correct += 1.0;
                    }
                } else {
                    s2_count += 1.0;
                }
            }

            let s1_acc = if s1_count > 0.0 {
                s1_correct / s1_count
            } else {
                1.0
            };

            let s1_rate = s1_count / n;
            let s2_rate = s2_count / n;
            let hybrid_acc = s1_rate * s1_acc + s2_rate * assumed_s2_acc;
            let total_cost = s2_count * system2_cost;

            if hybrid_acc >= target_accuracy && total_cost < best_cost {
                best_cost = total_cost;
                best_tau = tau;
                best_acc = hybrid_acc;
                best_s1_cov = s1_rate;
            }
        }

        if best_cost.is_infinite() {
            bail!(
                "Target accuracy {:.2}% cannot be achieved by cascade (maximum achievable: {:.2}%). Consider lowering target accuracy.",
                target_accuracy * 100.0,
                assumed_s2_acc * 100.0
            );
        }

        let s2_esc_rate = 1.0 - best_s1_cov;
        let cost_per_1000 = s2_esc_rate * 1000.0 * system2_cost;
        let savings_percent = if system2_cost > 0.0 {
            best_s1_cov * 100.0
        } else {
            0.0
        };

        Ok(CascadePolicy {
            optimal_threshold: best_tau,
            target_accuracy,
            expected_accuracy: (best_acc * 10000.0).round() / 10000.0,
            system1_coverage_rate: (best_s1_cov * 10000.0).round() / 10000.0,
            system2_escalation_rate: (s2_esc_rate * 10000.0).round() / 10000.0,
            cost_savings_percent: (savings_percent * 100.0).round() / 100.0,
            estimated_cost_per_1000: (cost_per_1000 * 100.0).round() / 100.0,
        })
    }
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

    #[test]
    fn test_cascade_routing_high_confidence() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        );

        let item = DataItem::new(
            "c1",
            serde_json::json!({ "text": "Invoice payment billing refund charge immediate" }),
            "test",
        );

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let router = CascadeRouter::new(CascadeConfig::default());
        let decision = router.route(&item, &questions, &classifier).unwrap();

        // Clear billing intent should execute autonomously on System 1
        assert_eq!(decision.tier, SystemTier::System1Fast);
        assert_eq!(decision.cost_usd, 0.0);
        assert!(decision.composite_risk < 0.45);
    }

    #[test]
    fn test_cascade_routing_ambiguous_escalation() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        );

        // Ambiguous mix of billing and technical words
        let item = DataItem::new(
            "c2",
            serde_json::json!({ "text": "maybe something or other random words with no clear intent" }),
            "test",
        );

        let cfg = CascadeConfig {
            risk_threshold: 0.20, // Very strict threshold forces System 2
            ..Default::default()
        };
        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let router = CascadeRouter::new(cfg);
        let decision = router.route(&item, &questions, &classifier).unwrap();

        assert_eq!(decision.tier, SystemTier::System2Deliberative);
        assert!(decision.cost_usd > 0.0);
        assert!(decision.prompt_seed.is_some());
    }

    #[test]
    fn test_tune_pareto_unlabeled_items_rejected() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        );
        let item = DataItem::new(
            "c3",
            serde_json::json!({ "text": "Invoice payment billing refund" }),
            "test",
        ); // Ground truth is None
        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        assert!(CascadeRouter::tune_pareto(&[item], &questions, &classifier, 0.95, 0.005).is_err());
    }

    #[test]
    fn test_tune_pareto_with_custom_config() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        );
        let mut gt = BTreeMap::new();
        gt.insert("dept".into(), serde_json::Value::String("billing".into()));

        let item = DataItem::new(
            "c4",
            serde_json::json!({ "text": "Invoice payment billing refund immediate" }),
            "test",
        )
        .with_ground_truth(gt);

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let custom_cfg = CascadeConfig {
            entropy_weight: 0.50,
            margin_weight: 0.30,
            act_weight: 0.20,
            risk_threshold: 0.50,
            system2_cost_per_call: 0.01,
        };
        let policy = CascadeRouter::tune_pareto_with_config(
            &[item],
            &questions,
            &classifier,
            0.90,
            0.01,
            &custom_cfg,
        )
        .unwrap();

        assert!(policy.optimal_threshold > 0.0);
        assert!(policy.expected_accuracy >= 0.90);
    }
}
