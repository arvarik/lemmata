use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use std::collections::BTreeMap;

use super::classifier::LayaClassifier;
use super::evaluator::Evaluator;
use crate::datasource::DataItem;
use crate::model::{Checkpoint, Question};

pub struct ExperimentRunner;

impl ExperimentRunner {
    /// Compare performance across System 1 model checkpoints.
    pub fn compare_checkpoints(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<()> {
        let report = Self::run_arena(
            items,
            questions,
            &crate::model::native::runnable_checkpoints(),
            5,
        )?;
        Self::print_arena_report(&report);
        Ok(())
    }

    /// Head-to-Head Model Arena comparing 2+ System 1 checkpoints (agreement, Cohen's kappa, McNemar's chi2, ECE, speedup).
    pub fn run_arena(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        checkpoints: &[Checkpoint],
        top_disagreements: usize,
    ) -> Result<ModelArenaReport> {
        Self::run_arena_with_options(
            items,
            questions,
            checkpoints,
            top_disagreements,
            crate::model::ChunkStrategy::default(),
            None,
            0,
        )
    }

    /// Head-to-Head Model Arena with configurable chunking strategy, hypothesis template, and bi-encoder shortlisting.
    pub fn run_arena_with_options(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        checkpoints: &[Checkpoint],
        top_disagreements: usize,
        chunk_strategy: crate::model::ChunkStrategy,
        hypothesis_template: Option<&str>,
        shortlist_k: usize,
    ) -> Result<ModelArenaReport> {
        if items.is_empty() {
            bail!("No items provided for Model Arena comparison");
        }
        if checkpoints.is_empty() {
            bail!("At least one checkpoint is required for Model Arena");
        }

        let mut model_preds: Vec<Vec<(String, String, bool, f64)>> = Vec::new(); // per model: (key, pred, is_correct, conf)
        let mut raw_entries = Vec::new();

        for &ckpt in checkpoints {
            let mut cls = LayaClassifier::with_checkpoint(ckpt)
                .with_chunk_strategy(chunk_strategy)
                .with_shortlist_k(shortlist_k);
            if let Some(tpl) = hypothesis_template {
                cls = cls.with_hypothesis_template(tpl);
            }
            let t0 = std::time::Instant::now();
            let batch = cls.classify_batch(items, questions)?;
            let avg_lat = (t0.elapsed().as_secs_f64() * 1000.0) / items.len().max(1) as f64;
            let eval_metrics = Evaluator::new(cls).evaluate(items, questions).ok();
            // The arena runs one checkpoint at a time. Keeping all nine models loaded took 14.6 GB.
            crate::model::native::release_models();

            let mut preds = Vec::new();
            for (item, (_, res)) in items.iter().zip(batch.iter()) {
                for (qid, q) in questions {
                    let Some(ans) = res.answers.get(qid) else {
                        continue;
                    };
                    let pred_str = if let Some(c) = &ans.choice {
                        c.to_lowercase()
                    } else if let Some(s) = ans.score {
                        format!("{:.0}", s.round())
                    } else if let Some(n) = ans.noul {
                        if n >= 0.5 { "true" } else { "false" }.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    let is_correct = item
                        .ground_truth
                        .as_ref()
                        .and_then(|gt| {
                            super::metrics::resolve_ground_truth_for_question(
                                Some(gt),
                                qid,
                                q,
                                questions.len(),
                            )
                        })
                        .is_some_and(|tv| {
                            let target_s = tv
                                .as_str()
                                .map(str::to_lowercase)
                                .or_else(|| tv.as_bool().map(|b| b.to_string()))
                                .or_else(|| tv.as_f64().map(|n| format!("{:.0}", n.round())))
                                .unwrap_or_default();
                            pred_str == target_s
                        });
                    preds.push((
                        format!("{}:{qid}", item.id),
                        pred_str,
                        is_correct,
                        ans.confidence,
                    ));
                }
            }
            model_preds.push(preds);
            raw_entries.push((ckpt, eval_metrics, avg_lat));
        }

        let ref_preds = &model_preds[0];
        let ref_lat = raw_entries[0].2.max(1e-6);
        let mut entries = Vec::with_capacity(checkpoints.len());

        for (idx, (ckpt, metrics, avg_lat)) in raw_entries.into_iter().enumerate() {
            let cur_preds = &model_preds[idx];
            let n = ref_preds.len().min(cur_preds.len()).max(1);
            let mut agree = 0usize;
            let mut ref_counts: BTreeMap<&str, usize> = BTreeMap::new();
            let mut cur_counts: BTreeMap<&str, usize> = BTreeMap::new();
            let mut b_ref_only = 0usize;
            let mut c_cur_only = 0usize;

            for ((_, p0, c0, _), (_, p1, c1, _)) in ref_preds.iter().zip(cur_preds.iter()) {
                if p0 == p1 {
                    agree += 1;
                }
                *ref_counts.entry(p0.as_str()).or_insert(0) += 1;
                *cur_counts.entry(p1.as_str()).or_insert(0) += 1;
                if *c0 && !*c1 {
                    b_ref_only += 1;
                } else if !*c0 && *c1 {
                    c_cur_only += 1;
                }
            }

            let p_o = agree as f64 / n as f64;
            let p_e: f64 = ref_counts
                .iter()
                .map(|(k, &c_ref)| {
                    let c_cur = cur_counts.get(k).copied().unwrap_or(0);
                    (c_ref as f64 / n as f64) * (c_cur as f64 / n as f64)
                })
                .sum();
            let kappa = if (1.0 - p_e).abs() < 1e-9 {
                1.0
            } else {
                ((p_o - p_e) / (1.0 - p_e)).clamp(-1.0, 1.0)
            };

            let bc = b_ref_only + c_cur_only;
            let mcnemar_chi2 = if bc > 0 {
                let diff = ((b_ref_only as f64 - c_cur_only as f64).abs() - 1.0).max(0.0);
                (diff * diff) / bc as f64
            } else {
                0.0
            };

            entries.push(ModelArenaEntry {
                checkpoint: ckpt.repo_id().to_string(),
                backbone: ckpt.backbone().to_string(),
                params: ckpt.param_count().to_string(),
                accuracy: metrics.as_ref().map(|m| m.accuracy),
                ece: metrics.as_ref().map(|m| m.ece),
                brier_score: metrics.as_ref().map(|m| m.brier_score),
                avg_latency_ms: (avg_lat * 1000.0).round() / 1000.0,
                speedup_vs_ref: ((ref_lat / avg_lat.max(1e-6)) * 100.0).round() / 100.0,
                agreement_with_ref: (p_o * 10000.0).round() / 10000.0,
                cohens_kappa: (kappa * 10000.0).round() / 10000.0,
                mcnemar_chi2: (mcnemar_chi2 * 1000.0).round() / 1000.0,
                significant_diff: mcnemar_chi2 >= 3.841,
            });
        }

        let mut disagreements = Vec::new();
        if model_preds.len() >= 2 && top_disagreements > 0 {
            for (i, ref_p) in ref_preds.iter().enumerate() {
                let first_pred = &ref_p.1;
                let mut distinct = false;
                let mut min_conf = ref_p.3;
                let mut max_conf = ref_p.3;
                let mut by_model = BTreeMap::new();
                for (m_idx, &ckpt) in checkpoints.iter().enumerate() {
                    if let Some((_, p, _, conf)) = model_preds[m_idx].get(i) {
                        if p != first_pred {
                            distinct = true;
                        }
                        min_conf = min_conf.min(*conf);
                        max_conf = max_conf.max(*conf);
                        by_model.insert(
                            ckpt.repo_id().to_string(),
                            format!("{p} ({:.0}%)", conf * 100.0),
                        );
                    }
                }
                if distinct {
                    disagreements.push(ArenaDisagreement {
                        item_question: ref_p.0.clone(),
                        confidence_spread: ((max_conf - min_conf) * 10000.0).round() / 10000.0,
                        predictions: by_model,
                    });
                }
            }
            disagreements.sort_by(|a, b| {
                b.confidence_spread
                    .partial_cmp(&a.confidence_spread)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            disagreements.truncate(top_disagreements);
        }

        Ok(ModelArenaReport {
            reference_model: checkpoints[0].repo_id().to_string(),
            sample_count: items.len(),
            models: entries,
            top_disagreements: disagreements,
        })
    }

    pub fn print_arena_report(report: &ModelArenaReport) {
        println!(
            "\n🏛️  System 1 Model Arena (Reference: {}, Samples: {})",
            report.reference_model, report.sample_count
        );
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Checkpoint").fg(Color::Cyan),
            Cell::new("Backbone").fg(Color::Blue),
            Cell::new("Accuracy").fg(Color::Green),
            Cell::new("ECE").fg(Color::Magenta),
            Cell::new("Agree %").fg(Color::White),
            Cell::new("Cohen's κ").fg(Color::White),
            Cell::new("McNemar χ²").fg(Color::Yellow),
            Cell::new("Latency (Speedup)").fg(Color::Yellow),
        ]);

        for m in &report.models {
            let acc_s = m
                .accuracy
                .map(|a| format!("{:.2}%", a * 100.0))
                .unwrap_or_else(|| "N/A".into());
            let ece_s = m
                .ece
                .map(|e| format!("{:.4}", e))
                .unwrap_or_else(|| "N/A".into());
            let sig = if m.significant_diff { "*" } else { "" };
            table.add_row(vec![
                Cell::new(&m.checkpoint),
                Cell::new(&m.backbone),
                Cell::new(acc_s),
                Cell::new(ece_s),
                Cell::new(format!("{:.1}%", m.agreement_with_ref * 100.0)),
                Cell::new(format!("{:.3}", m.cohens_kappa)),
                Cell::new(format!("{:.2}{}", m.mcnemar_chi2, sig)),
                Cell::new(format!(
                    "{:.2} ms ({:.2}x)",
                    m.avg_latency_ms, m.speedup_vs_ref
                )),
            ]);
        }
        println!("{table}");
        if !report.top_disagreements.is_empty() {
            println!("\n⚡ Top Model Disagreements (by confidence spread):");
            for d in &report.top_disagreements {
                let preds: Vec<String> = d
                    .predictions
                    .iter()
                    .map(|(k, v)| {
                        let short = k.rsplit('/').next().unwrap_or(k);
                        format!("{short}={v}")
                    })
                    .collect();
                println!(
                    "   • [{}] (Δconf={:.2}): {}",
                    d.item_question,
                    d.confidence_spread,
                    preds.join(" | ")
                );
            }
        }
        println!();
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelArenaEntry {
    pub checkpoint: String,
    pub backbone: String,
    pub params: String,
    pub accuracy: Option<f64>,
    pub ece: Option<f64>,
    pub brier_score: Option<f64>,
    pub avg_latency_ms: f64,
    pub speedup_vs_ref: f64,
    pub agreement_with_ref: f64,
    pub cohens_kappa: f64,
    pub mcnemar_chi2: f64,
    pub significant_diff: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArenaDisagreement {
    pub item_question: String,
    pub confidence_spread: f64,
    pub predictions: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelArenaReport {
    pub reference_model: String,
    pub sample_count: usize,
    pub models: Vec<ModelArenaEntry>,
    pub top_disagreements: Vec<ArenaDisagreement>,
}
