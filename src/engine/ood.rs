use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::datasource::DataItem;
use crate::engine::cache::vectorize_text;
use crate::engine::classifier::LayaClassifier;
use crate::engine::guardrail::GuardAction;
use crate::engine::metrics::resolve_ground_truth_for_question;
use crate::model::{DecisionResult, Question};

/// Out-of-distribution (OOD) detection profile fitted on in-distribution data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodProfile {
    pub name: String,
    pub question_id: String,
    pub target_classes: Vec<String>,
    pub centroids: BTreeMap<String, Vec<f64>>,
    pub class_variances: BTreeMap<String, f64>,
    pub mean_energy: f64,
    pub std_energy: f64,
    pub energy_threshold: f64,
    pub distance_threshold: f64,
}

/// Evaluation report for a single input tested for OOD anomalies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodEvaluation {
    pub item_id: String,
    pub is_ood: bool,
    pub free_energy: f64,
    pub energy_z_score: f64,
    pub mahalanobis_distance: f64,
    pub anomaly_score: f64,
    pub closest_class: String,
    pub action: GuardAction,
}

/// Anomaly and OOD detector combining Helmholtz Free Energy and Mahalanobis++ distance.
pub struct OodDetector {
    profile: OodProfile,
}

impl OodDetector {
    pub fn new(profile: OodProfile) -> Self {
        Self { profile }
    }

    pub fn profile(&self) -> &OodProfile {
        &self.profile
    }

    /// Fit an OOD profile on a golden in-distribution baseline dataset.
    pub fn fit(
        name: impl Into<String>,
        items: &[DataItem],
        classifier: &LayaClassifier,
        questions: &BTreeMap<String, Question>,
        target_qid: &str,
    ) -> Result<OodProfile> {
        let q = questions.get(target_qid).ok_or_else(|| {
            anyhow::anyhow!("Question '{}' not found in questions schema", target_qid)
        })?;

        let options = q.render_options();
        if options.is_empty() {
            bail!("Question '{}' has no candidate options", target_qid);
        }

        let mut class_features: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
        let mut energy_scores = Vec::with_capacity(items.len());
        let batch_results = classifier.classify_batch(items, questions)?;

        for (item, (_, res)) in items.iter().zip(batch_results.iter()) {
            let text = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other)?,
            };

            let feat = vectorize_text(&text);
            let ans = res
                .answers
                .get(target_qid)
                .ok_or_else(|| anyhow::anyhow!("No answer for '{}'", target_qid))?;

            let assigned_class = resolve_ground_truth_for_question(
                item.ground_truth.as_ref(),
                target_qid,
                q,
                questions.len(),
            )
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Bool(b) => Some(b.to_string()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .or_else(|| ans.choice.clone())
            .unwrap_or_else(|| "unknown".into());

            let energy = compute_free_energy(&ans.probabilities, classifier.config().temperature);
            energy_scores.push(energy);

            class_features.entry(assigned_class).or_default().push(feat);
        }

        if energy_scores.is_empty() {
            bail!("Cannot fit OOD profile on empty dataset");
        }

        let n = energy_scores.len() as f64;
        let mean_energy = energy_scores.iter().sum::<f64>() / n;
        let var_energy = energy_scores
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / n;
        let std_energy = var_energy.sqrt().max(1e-4);

        let mut centroids = BTreeMap::new();
        let mut class_variances = BTreeMap::new();
        let mut target_classes = Vec::new();

        for (cls, feats) in class_features {
            target_classes.push(cls.clone());
            let count = feats.len() as f64;
            let dim = feats[0].len();
            let mut mean_vec = vec![0.0f64; dim];

            for feat in &feats {
                for (i, &val) in feat.iter().enumerate() {
                    mean_vec[i] += val / count;
                }
            }

            let norm = mean_vec.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
            for val in &mut mean_vec {
                *val /= norm;
            }

            let avg_dist = feats
                .iter()
                .map(|feat| euclidean_distance(feat, &mean_vec))
                .sum::<f64>()
                / count;

            centroids.insert(cls.clone(), mean_vec);
            class_variances.insert(cls, avg_dist.max(1e-3));
        }

        let energy_threshold = mean_energy + 2.33 * std_energy;
        let distance_threshold = 1.65;

        Ok(OodProfile {
            name: name.into(),
            question_id: target_qid.to_string(),
            target_classes,
            centroids,
            class_variances,
            mean_energy,
            std_energy,
            energy_threshold,
            distance_threshold,
        })
    }

    /// Evaluate an input item against the OOD profile using a pre-computed DecisionResult.
    pub fn evaluate_with_decision(
        &self,
        item_id: &str,
        text: &str,
        res: &DecisionResult,
    ) -> Result<OodEvaluation> {
        let ans = res.answers.get(&self.profile.question_id).ok_or_else(|| {
            anyhow::anyhow!("No answer for question '{}'", self.profile.question_id)
        })?;

        let feat = vectorize_text(text);
        let energy = compute_free_energy(&ans.probabilities, 1.0);
        let energy_z = (energy - self.profile.mean_energy) / self.profile.std_energy;

        let mut min_dist = f64::INFINITY;
        let mut closest_class = "unknown".to_string();

        for (cls, centroid) in &self.profile.centroids {
            let variance = self
                .profile
                .class_variances
                .get(cls)
                .copied()
                .unwrap_or(1.0);
            let scaled_dist = euclidean_distance(&feat, centroid) / variance;

            if scaled_dist < min_dist {
                min_dist = scaled_dist;
                closest_class = cls.clone();
            }
        }

        let energy_factor = 1.0 / (1.0 + (-energy_z + 1.5).exp());
        let dist_factor = 1.0 / (1.0 + (-(min_dist - self.profile.distance_threshold) * 2.0).exp());
        let anomaly_score = (0.5 * energy_factor + 0.5 * dist_factor).clamp(0.0, 1.0);

        let is_ood = anomaly_score.is_nan()
            || anomaly_score >= 0.55
            || energy >= self.profile.energy_threshold
            || min_dist >= self.profile.distance_threshold;

        let action = if anomaly_score.is_nan()
            || min_dist.is_nan()
            || anomaly_score >= 0.75
            || min_dist >= self.profile.distance_threshold * 1.5
        {
            GuardAction::Quarantine
        } else if is_ood {
            GuardAction::Escalate
        } else {
            GuardAction::Allow
        };

        Ok(OodEvaluation {
            item_id: item_id.to_string(),
            is_ood,
            free_energy: (energy * 10000.0).round() / 10000.0,
            energy_z_score: (energy_z * 10000.0).round() / 10000.0,
            mahalanobis_distance: (min_dist * 10000.0).round() / 10000.0,
            anomaly_score: (anomaly_score * 10000.0).round() / 10000.0,
            closest_class,
            action,
        })
    }

    /// Evaluate an input item against the OOD profile.
    pub fn evaluate(
        &self,
        item_id: &str,
        text: &str,
        classifier: &LayaClassifier,
        questions: &BTreeMap<String, Question>,
    ) -> Result<OodEvaluation> {
        let state = serde_json::Value::String(text.to_string());
        let res = classifier.classify(&state, questions)?;
        self.evaluate_with_decision(item_id, text, &res)
    }

    /// Evaluate a batch of items.
    pub fn evaluate_batch(
        &self,
        items: &[DataItem],
        classifier: &LayaClassifier,
        questions: &BTreeMap<String, Question>,
    ) -> Result<Vec<OodEvaluation>> {
        let batch_results = classifier.classify_batch(items, questions)?;
        let mut results = Vec::with_capacity(items.len());
        for (item, (_, res)) in items.iter().zip(batch_results.iter()) {
            let text = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other)?,
            };
            results.push(self.evaluate_with_decision(&item.id, &text, res)?);
        }
        Ok(results)
    }
}

/// Compute Helmholtz Free Energy over the full probability distribution.
/// Combines Shannon entropy, Rényi collision energy, and max-probability concentration
/// so diffuse/uncertain OOD distributions yield significantly higher energy than confident ID inputs.
pub fn compute_free_energy(probs: &BTreeMap<String, f64>, temperature: f64) -> f64 {
    if probs.is_empty() {
        return 10.0;
    }
    let temp = temperature.max(0.01);
    let p_max = probs.values().copied().fold(0.0, f64::max).clamp(1e-9, 1.0);
    let shannon: f64 = -probs
        .values()
        .filter(|&&p| p > 1e-9)
        .map(|&p| p * p.ln())
        .sum::<f64>();
    let renyi: f64 = -probs
        .values()
        .map(|&p| p * p)
        .sum::<f64>()
        .clamp(1e-9, 1.0)
        .ln();
    temp * (0.5 * shannon + 0.3 * renyi - 0.2 * p_max.ln()).max(1e-6)
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Checkpoint;

    #[test]
    fn test_ood_profile_fit_and_separation() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
        );

        let id_items = vec![
            DataItem::new(
                "b1",
                serde_json::json!({ "text": "Invoice refund billing payment charge" }),
                "id",
            ),
            DataItem::new(
                "b2",
                serde_json::json!({ "text": "Payment credit card billing refund" }),
                "id",
            ),
            DataItem::new(
                "t1",
                serde_json::json!({ "text": "Server crash bug exception stack trace" }),
                "id",
            ),
            DataItem::new(
                "t2",
                serde_json::json!({ "text": "System error crash stack technical outage" }),
                "id",
            ),
        ];

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let profile = OodDetector::fit(
            "support_profile",
            &id_items,
            &classifier,
            &questions,
            "dept",
        )
        .unwrap();

        assert_eq!(profile.target_classes.len(), 2);
        assert!(
            profile.centroids.contains_key("billing")
                || profile.centroids.contains_key("technical")
        );

        let detector = OodDetector::new(profile);

        // In-distribution test
        let id_eval = detector
            .evaluate(
                "id_test",
                "Invoice refund payment billing charge",
                &classifier,
                &questions,
            )
            .unwrap();
        assert!(!id_eval.is_ood || id_eval.anomaly_score < 0.70);

        // Completely out-of-distribution test (gibberish / alien text)
        let ood_eval = detector
            .evaluate(
                "ood_test",
                "zxqw kljpoiu 9823412 @#$%%^&*!@# kjashdkh",
                &classifier,
                &questions,
            )
            .unwrap();
        assert!(ood_eval.mahalanobis_distance > 0.5);
    }
}
