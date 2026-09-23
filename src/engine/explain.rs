use std::collections::BTreeMap;
use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use rayon::prelude::*;

use super::classifier::LayaClassifier;
use crate::model::Question;

#[derive(Debug, Clone)]
pub struct TokenAttribution {
    pub token: String,
    pub delta_margin: f64,
    pub relative_influence_pct: f64,
}

#[derive(Debug, Clone)]
pub struct ExplanationReport {
    pub input_text: String,
    pub question_id: String,
    pub target_choice: String,
    pub contrast_choice: String,
    pub baseline_margin: f64,
    pub attributions: Vec<TokenAttribution>,
}

pub struct AttributionEngine<'a> {
    classifier: &'a LayaClassifier,
}

impl<'a> AttributionEngine<'a> {
    pub fn new(classifier: &'a LayaClassifier) -> Self {
        Self { classifier }
    }

    /// Explain the decision margin between `target_choice` and `contrast_choice`
    /// via parallel leave-one-out contrastive token ablation.
    pub fn explain(
        &self,
        text: &str,
        qid: &str,
        question: &Question,
        target_choice: &str,
        contrast_choice: &str,
    ) -> Result<ExplanationReport> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            bail!("Input text is empty");
        }

        let mut q_map = BTreeMap::new();
        q_map.insert(qid.to_string(), question.clone());

        // Baseline classification
        let state_val = serde_json::Value::String(text.to_string());
        let baseline = self.classifier.classify(&state_val, &q_map)?;
        let base_ans = baseline
            .answers
            .get(qid)
            .ok_or_else(|| anyhow::anyhow!("Question '{}' missing in results", qid))?;

        let p_target = get_prob_ci(&base_ans.probabilities, target_choice);
        let p_contrast = get_prob_ci(&base_ans.probabilities, contrast_choice);

        let baseline_margin = (p_target + 1e-9).ln() - (p_contrast + 1e-9).ln();

        // Parallel leave-one-out token ablation using Rayon
        let mut attributions: Vec<TokenAttribution> = words
            .par_iter()
            .enumerate()
            .map(|(idx, &word)| {
                let mut perturbed_words = words.clone();
                perturbed_words.remove(idx);
                let perturbed_text = perturbed_words.join(" ");

                let pert_state = serde_json::Value::String(perturbed_text);
                let pert_res = self.classifier.classify(&pert_state, &q_map).unwrap();
                let pert_ans = &pert_res.answers[qid];

                let pert_p_target = get_prob_ci(&pert_ans.probabilities, target_choice);
                let pert_p_contrast = get_prob_ci(&pert_ans.probabilities, contrast_choice);

                let pert_margin = (pert_p_target + 1e-9).ln() - (pert_p_contrast + 1e-9).ln();
                let delta = baseline_margin - pert_margin;

                TokenAttribution {
                    token: word.to_string(),
                    delta_margin: (delta * 1000.0).round() / 1000.0,
                    relative_influence_pct: 0.0,
                }
            })
            .collect();

        // Calculate relative contribution percentage
        let positive_sum: f64 = attributions
            .iter()
            .filter(|a| a.delta_margin > 0.0)
            .map(|a| a.delta_margin)
            .sum();

        for attr in &mut attributions {
            if positive_sum > 0.0 && attr.delta_margin > 0.0 {
                attr.relative_influence_pct =
                    ((attr.delta_margin / positive_sum) * 1000.0).round() / 10.0;
            }
        }

        Ok(ExplanationReport {
            input_text: text.to_string(),
            question_id: qid.to_string(),
            target_choice: target_choice.to_string(),
            contrast_choice: contrast_choice.to_string(),
            baseline_margin: (baseline_margin * 1000.0).round() / 1000.0,
            attributions,
        })
    }

    pub fn print_report(report: &ExplanationReport) {
        println!("\n🔍 Laya Contrastive Saliency Attribution");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "Target: '{}' vs Contrast: '{}' (Baseline Margin: {:+.3})",
            report.target_choice, report.contrast_choice, report.baseline_margin
        );
        println!("Question: '{}'", report.question_id);

        let mut sorted = report.attributions.clone();
        sorted.sort_by(|a, b| b.delta_margin.total_cmp(&a.delta_margin));

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Rank").fg(Color::Cyan),
            Cell::new("Token / Word").fg(Color::White),
            Cell::new("Margin Delta").fg(Color::Green),
            Cell::new("Relative Influence").fg(Color::Yellow),
            Cell::new("Significance Bar").fg(Color::Magenta),
        ]);

        for (i, attr) in sorted.iter().take(10).enumerate() {
            let bar_len = ((attr.relative_influence_pct / 10.0).round() as usize).min(10);
            let bar = "█".repeat(bar_len);

            table.add_row(vec![
                Cell::new(format!("{}", i + 1)),
                Cell::new(&attr.token),
                Cell::new(format!("{:+.3}", attr.delta_margin)),
                Cell::new(format!("{:.1}%", attr.relative_influence_pct)),
                Cell::new(bar),
            ]);
        }

        println!("\n{table}\n");
    }
}

fn get_prob_ci(map: &BTreeMap<String, f64>, target: &str) -> f64 {
    if let Some(&p) = map.get(target) {
        return p;
    }
    for (k, &p) in map {
        if k.eq_ignore_ascii_case(target) {
            return p;
        }
    }
    1e-6
}
