use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use rand::seq::SliceRandom;
use std::collections::BTreeMap;

use super::classifier::LayaClassifier;
use super::evaluator::Evaluator;
use crate::datasource::DataItem;
use crate::model::{Checkpoint, Question, QuestionType};

pub struct ExperimentRunner;

impl ExperimentRunner {
    /// Test option order permutation invariance (testing for positional bias in [MASK] markers).
    pub fn test_option_order_invariance(
        classifier: &LayaClassifier,
        items: &[DataItem],
        qid: &str,
        question: &Question,
        permutations: usize,
    ) -> Result<()> {
        println!(
            "\n🧪 Running Option Order Invariance Experiment (Permutations: {})",
            permutations
        );
        println!("   Question ID: '{}'", qid);
        println!("   Instructions: \"{}\"", question.instructions);

        let original_options = question.render_options();
        if original_options.len() < 2 {
            bail!("Option order invariance requires at least 2 options");
        }

        let mut rng = rand::rng();
        let mut flip_count = 0;
        let mut total_checks = 0;
        let mut conf_deltas = Vec::new();

        for item in items {
            let mut q_map_orig = BTreeMap::new();
            q_map_orig.insert(qid.to_string(), question.clone());
            let baseline = classifier.classify(&item.state, &q_map_orig)?;
            let base_ans = &baseline.answers[qid];
            let base_choice = base_ans.choice.clone();
            let base_conf = base_ans.confidence;

            for _ in 0..permutations {
                let mut shuffled_options = original_options.clone();
                shuffled_options.shuffle(&mut rng);

                let mut shuffled_q = question.clone();
                match shuffled_q.qtype {
                    QuestionType::Choice => {
                        let arr = shuffled_options
                            .iter()
                            .cloned()
                            .map(serde_json::Value::String)
                            .collect();
                        shuffled_q.criteria = Some(serde_json::Value::Array(arr));
                    }
                    QuestionType::Score => {
                        let arr = shuffled_options
                            .iter()
                            .map(|opt| {
                                let text =
                                    opt.split_once(':').map(|(_, r)| r.trim()).unwrap_or(opt);
                                serde_json::Value::String(text.to_string())
                            })
                            .collect();
                        shuffled_q.criteria = Some(serde_json::Value::Array(arr));
                    }
                    _ => continue,
                }

                let mut q_map_shuffled = BTreeMap::new();
                q_map_shuffled.insert(qid.to_string(), shuffled_q);
                let perm_result = classifier.classify(&item.state, &q_map_shuffled)?;
                let perm_ans = &perm_result.answers[qid];

                total_checks += 1;
                if perm_ans.choice != base_choice {
                    flip_count += 1;
                }
                conf_deltas.push((perm_ans.confidence - base_conf).abs());
            }
        }

        let flip_rate = if total_checks > 0 {
            flip_count as f64 / total_checks as f64
        } else {
            0.0
        };

        let avg_conf_delta = if !conf_deltas.is_empty() {
            conf_deltas.iter().sum::<f64>() / conf_deltas.len() as f64
        } else {
            0.0
        };

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Experiment Metric").fg(Color::Cyan),
            Cell::new("Observed Value").fg(Color::Green),
            Cell::new("Implication").fg(Color::DarkGrey),
        ]);

        table.add_row(vec![
            Cell::new("Total Permutation Tests"),
            Cell::new(format!("{}", total_checks)),
            Cell::new("Items × Permutations"),
        ]);
        table.add_row(vec![
            Cell::new("Label Flip Rate"),
            Cell::new(format!("{:.2}%", flip_rate * 100.0)),
            Cell::new(if flip_rate < 0.05 {
                "High Stability (positional bias is negligible)"
            } else {
                "Sensitivity detected (option ordering shifts decisions)"
            }),
        ]);
        table.add_row(vec![
            Cell::new("Mean Confidence Delta"),
            Cell::new(format!("{:.4}", avg_conf_delta)),
            Cell::new("Average absolute deviation in confidence"),
        ]);

        println!("{table}\n");
        Ok(())
    }

    /// Sweep over a grid of temperature values to find optimal calibration (lowest ECE).
    pub fn sweep_temperature_calibration(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        temperatures: &[f64],
    ) -> Result<()> {
        println!("\n🌡️  Running Temperature Calibration Sweep");
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Temperature (T)").fg(Color::Cyan),
            Cell::new("ECE").fg(Color::Yellow),
            Cell::new("Accuracy").fg(Color::Green),
            Cell::new("Avg Confidence").fg(Color::Blue),
            Cell::new("Status / Recommendation").fg(Color::DarkGrey),
        ]);

        let mut best_t = 1.0;
        let mut min_ece = f64::MAX;
        let mut rows = Vec::with_capacity(temperatures.len());

        for &t in temperatures {
            let cls = LayaClassifier::new(classifier_config_with_t(t));
            let batch_res = cls.classify_batch(items, questions)?;
            let mut conf_sum = 0.0;
            let mut conf_cnt = 0usize;
            for (_, res) in &batch_res {
                for ans in res.answers.values() {
                    conf_sum += ans.confidence;
                    conf_cnt += 1;
                }
            }
            let avg_conf = if conf_cnt > 0 {
                conf_sum / conf_cnt as f64
            } else {
                0.0
            };

            let evaluator = Evaluator::new(cls);
            match evaluator.evaluate(items, questions) {
                Ok(metrics) => {
                    if metrics.ece < min_ece {
                        min_ece = metrics.ece;
                        best_t = t;
                    }
                    rows.push((t, metrics.ece, metrics.accuracy, avg_conf));
                }
                Err(err) => {
                    eprintln!("Skipping T={}: {}", t, err);
                }
            }
        }

        for (t, ece, acc, avg_conf) in rows {
            let status = if (t - best_t).abs() < 1e-6 {
                "⭐ Optimal (Lowest ECE)"
            } else if t < 1.0 {
                "Sharpened (High Confidence)"
            } else {
                "Smoothed (Calibrated)"
            };

            table.add_row(vec![
                Cell::new(format!("{:.2}", t)),
                Cell::new(format!("{:.4}", ece)),
                Cell::new(format!("{:.2}%", acc * 100.0)),
                Cell::new(format!("{:.4}", avg_conf)),
                Cell::new(status),
            ]);
        }

        println!("{table}");
        println!(
            "✨ Recommended Calibration Temperature: T = {:.2} (ECE: {:.4})\n",
            best_t, min_ece
        );
        Ok(())
    }

    /// Compare performance across System 1 model checkpoints.
    pub fn compare_checkpoints(
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<()> {
        println!("\n🏛️  Comparing System 1 Model Checkpoints");
        let checkpoints = Checkpoint::all();

        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Checkpoint").fg(Color::Cyan),
            Cell::new("Backbone").fg(Color::Blue),
            Cell::new("Params").fg(Color::DarkGrey),
            Cell::new("Context").fg(Color::DarkGrey),
            Cell::new("Accuracy").fg(Color::Green),
            Cell::new("Avg Latency (ms)").fg(Color::Yellow),
            Cell::new("ECE").fg(Color::Magenta),
        ]);

        for &ckpt in checkpoints {
            let cls = LayaClassifier::with_checkpoint(ckpt);
            let evaluator = Evaluator::new(cls);
            let t0 = std::time::Instant::now();

            let (acc_str, ece_str, avg_lat) = match evaluator.evaluate(items, questions) {
                Ok(m) => {
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    (
                        format!("{:.2}%", m.accuracy * 100.0),
                        format!("{:.4}", m.ece),
                        elapsed_ms / items.len().max(1) as f64,
                    )
                }
                Err(_) => {
                    let cls_lat = LayaClassifier::with_checkpoint(ckpt);
                    let sample = &items[..items.len().min(20)];
                    let t_sub = std::time::Instant::now();
                    let _ = cls_lat.classify_batch(sample, questions);
                    let elapsed_ms = t_sub.elapsed().as_secs_f64() * 1000.0;
                    (
                        "N/A".into(),
                        "N/A".into(),
                        elapsed_ms / sample.len().max(1) as f64,
                    )
                }
            };

            table.add_row(vec![
                Cell::new(ckpt.repo_id()),
                Cell::new(ckpt.backbone()),
                Cell::new(ckpt.param_count()),
                Cell::new(format!("{}", ckpt.context_window())),
                Cell::new(acc_str),
                Cell::new(format!("{:.2} ms", avg_lat)),
                Cell::new(ece_str),
            ]);
        }

        println!("{table}\n");
        Ok(())
    }
}

fn classifier_config_with_t(t: f64) -> crate::engine::classifier::ClassifierConfig {
    let mut cfg = crate::engine::classifier::ClassifierConfig {
        temperature: t,
        ..Default::default()
    };
    for val in cfg.temperature_by_options.values_mut() {
        *val = t;
    }
    cfg
}
