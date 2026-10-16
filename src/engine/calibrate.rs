use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use crate::datasource::DataItem;
use crate::engine::cache::vectorize_text;
use crate::engine::classifier::LayaClassifier;
use crate::engine::metrics::calculate_ece;
use crate::engine::ood::euclidean_distance;
use crate::model::{Checkpoint, Question, QuestionType};

/// Loss function minimized during temperature calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CalibrationLoss {
    /// Negative Log-Likelihood (Cross-Entropy Loss)
    Nll,
    /// Strictly Proper Mean Squared Error of Probabilities
    Brier,
    /// Expected Calibration Error across confidence bins
    Ece,
}

impl std::str::FromStr for CalibrationLoss {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().trim() {
            "nll" | "cross_entropy" | "log_loss" => Ok(Self::Nll),
            "brier" | "mse" => Ok(Self::Brier),
            "ece" => Ok(Self::Ece),
            other => bail!(
                "Unknown loss function '{}'. Supported: nll, brier, ece",
                other
            ),
        }
    }
}

use comfy_table::Table;

/// A single validation instance used for temperature optimization.
#[derive(Debug, Clone)]
pub struct ValidationSample {
    pub raw_logits: Vec<f64>,
    pub target_index: usize,
    pub target_label: String,
    pub option_weights: Vec<f64>,
}

/// Parse cost weights from "label=weight,label2=weight2" or a JSON file path.
pub fn parse_cost_weights(spec: &str) -> Result<BTreeMap<String, f64>> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Ok(BTreeMap::new());
    }
    let path = Path::new(trimmed);
    if path.exists() {
        let content = std::fs::read_to_string(path)?;
        let raw: BTreeMap<String, f64> = serde_json::from_str(&content)?;
        return Ok(raw
            .into_iter()
            .map(|(k, v)| (k.trim().to_lowercase(), v.max(0.01)))
            .collect());
    }
    let mut map = BTreeMap::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some((k, v_str)) = part.split_once('=').or_else(|| part.split_once(':')) else {
            bail!(
                "Invalid cost weight entry '{}'. Expected format: label=weight",
                part
            );
        };
        let w: f64 = v_str
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid numeric cost weight in '{}'", part))?;
        if w <= 0.0 {
            bail!(
                "Cost weight for '{}' must be positive (got {})",
                k.trim(),
                w
            );
        }
        map.insert(k.trim().to_lowercase(), w);
    }
    Ok(map)
}

/// Detailed calibration outcome for a single question bucket (e.g. choice:3-5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketCalibrationReport {
    pub bucket: String,
    pub sample_count: usize,
    pub raw_temperature: f64,
    pub raw_ece: f64,
    pub raw_nll: f64,
    pub raw_brier: f64,
    pub optimal_temperature: f64,
    pub calibrated_ece: f64,
    pub calibrated_nll: f64,
    pub calibrated_brier: f64,
    pub ece_improvement_pct: f64,
    pub iterations: usize,
}

/// 2026 Quantile-Adaptive Temperature Scaling (Q-ATS) profile for modulation under covariate shift.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatsProfile {
    pub enabled: bool,
    pub gamma: f64,
    pub mean_distance: f64,
    pub std_distance: f64,
    pub class_centroids: BTreeMap<String, Vec<f64>>,
}

/// Per-question calibration summary row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuestionCalibrationSummary {
    pub question_id: String,
    pub bucket_key: String,
    pub optimal_temperature: f64,
    pub ece_before: f64,
    pub ece_after: f64,
}

/// Overall calibration run report across all buckets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub dataset_name: String,
    pub total_samples: usize,
    pub loss_function: CalibrationLoss,
    pub buckets: Vec<BucketCalibrationReport>,
    pub average_raw_ece: f64,
    pub average_calibrated_ece: f64,
    pub overall_improvement_pct: f64,
    pub tuning_duration_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qats_profile: Option<QatsProfile>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cost_weights: BTreeMap<String, f64>,
    #[serde(default)]
    pub raw_expected_cost: f64,
    #[serde(default)]
    pub calibrated_expected_cost: f64,
    #[serde(default)]
    pub cost_reduction_pct: f64,
    #[serde(default)]
    pub uncalibrated_ece: f64,
    #[serde(default)]
    pub calibrated_ece: f64,
    #[serde(default)]
    pub ece_reduction_pct: f64,
    #[serde(default)]
    pub uncalibrated_brier: f64,
    #[serde(default)]
    pub calibrated_brier: f64,
    #[serde(default)]
    pub global_temperature: f64,
    #[serde(default)]
    pub per_question: Vec<QuestionCalibrationSummary>,
}

/// Calibration metadata serialized into rl_agent_config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    pub dataset: String,
    pub samples_evaluated: usize,
    pub raw_mean_ece: f64,
    pub calibrated_mean_ece: f64,
    pub improvement_pct: f64,
    pub calibrated_at: String,
}

/// Calibrated configuration structure matching rl_agent_config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedAgentConfig {
    pub checkpoint: String,
    pub temperature: f64,
    pub temperature_by_options: BTreeMap<String, f64>,
    pub loss_function: String,
    pub metadata: CalibrationMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qats: Option<QatsProfile>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cost_weights: BTreeMap<String, f64>,
}

/// Automated temperature tuner optimizing per-bucket calibration parameters.
pub struct CalibrationTuner {
    checkpoint: Checkpoint,
    loss: CalibrationLoss,
    bins: usize,
    enable_qats: bool,
    qats_gamma: f64,
    cost_weights: BTreeMap<String, f64>,
}

impl CalibrationTuner {
    pub fn new(checkpoint: Checkpoint) -> Self {
        Self {
            checkpoint,
            loss: CalibrationLoss::Nll,
            bins: 10,
            enable_qats: false,
            qats_gamma: 0.30,
            cost_weights: BTreeMap::new(),
        }
    }

    pub fn with_loss(mut self, loss: CalibrationLoss) -> Self {
        self.loss = loss;
        self
    }

    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins.max(2);
        self
    }

    pub fn with_qats(mut self, enable: bool, gamma: f64) -> Self {
        self.enable_qats = enable;
        self.qats_gamma = gamma.clamp(0.01, 1.0);
        self
    }

    pub fn with_cost_weights(mut self, weights: BTreeMap<String, f64>) -> Self {
        self.cost_weights = weights
            .into_iter()
            .map(|(k, v)| (k.to_lowercase(), v.max(0.01)))
            .collect();
        self
    }

    /// Fit domain-specific calibration temperatures from labeled validation data.
    pub fn fit(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        dataset_name: &str,
    ) -> Result<(CalibrationReport, CalibratedAgentConfig)> {
        if items.is_empty() {
            bail!("Validation dataset cannot be empty for calibration");
        }
        if questions.is_empty() {
            bail!("Questions schema cannot be empty for calibration");
        }

        let start_time = Instant::now();
        let classifier = LayaClassifier::with_checkpoint(self.checkpoint);

        let mut bucket_samples: BTreeMap<String, Vec<ValidationSample>> = BTreeMap::new();
        let mut class_features: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
        let mut item_features: Vec<Vec<f64>> = Vec::new();
        let mut total_samples_collected = 0;

        for item in items {
            let Some(gt_map) = &item.ground_truth else {
                continue;
            };
            let state_text = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other)?,
            };
            let feat_vec = self.enable_qats.then(|| vectorize_text(&state_text));
            if let Some(ref fv) = feat_vec {
                item_features.push(fv.clone());
            }

            for (qid, q) in questions {
                let Some(target_val) = gt_map.get(qid).or_else(|| gt_map.get("default")) else {
                    continue;
                };
                let options = q.render_options();
                let k = options.len().max(2);
                let bucket_key = classifier.get_temperature_bucket(q.qtype, k);
                let raw_logits = classifier.compute_option_logits(&state_text, q, &options);

                let (target_idx, target_label, option_weights) = match q.qtype {
                    QuestionType::Choice => {
                        let keys: Vec<String> =
                            if let Some(serde_json::Value::Object(map)) = &q.criteria {
                                map.keys().cloned().collect()
                            } else if let Some(serde_json::Value::Array(arr)) = &q.criteria {
                                arr.iter()
                                    .filter_map(|v| {
                                        v.as_str().map(|s| {
                                            s.split(':').next().unwrap_or(s).trim().to_string()
                                        })
                                    })
                                    .collect()
                            } else {
                                options
                                    .iter()
                                    .map(|s| s.split(':').next().unwrap_or(s).trim().to_string())
                                    .collect()
                            };
                        let target_str = target_val.as_str().unwrap_or("");
                        let opt_idx = keys
                            .iter()
                            .position(|k| k.eq_ignore_ascii_case(target_str))
                            .or_else(|| {
                                options
                                    .iter()
                                    .position(|opt| opt.eq_ignore_ascii_case(target_str))
                            });
                        let weights: Vec<f64> = keys
                            .iter()
                            .map(|k| {
                                self.cost_weights
                                    .get(&k.to_lowercase())
                                    .copied()
                                    .unwrap_or(1.0)
                            })
                            .collect();
                        match opt_idx {
                            Some(idx) => (idx, target_str.to_string(), weights),
                            None => continue,
                        }
                    }
                    QuestionType::Score => {
                        let score_val = target_val
                            .as_u64()
                            .or_else(|| target_val.as_str().and_then(|s| s.parse::<u64>().ok()))
                            .unwrap_or(0) as usize;
                        let clamped = score_val.min(raw_logits.len().saturating_sub(1));
                        (clamped, clamped.to_string(), vec![1.0; raw_logits.len()])
                    }
                    QuestionType::Noul => {
                        let b = target_val
                            .as_bool()
                            .or_else(|| {
                                target_val
                                    .as_str()
                                    .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
                            })
                            .or_else(|| target_val.as_u64().map(|n| n != 0))
                            .unwrap_or(false);
                        let w_false = self.cost_weights.get("false").copied().unwrap_or(1.0);
                        let w_true = self
                            .cost_weights
                            .get("true")
                            .or_else(|| self.cost_weights.get(&qid.to_lowercase()))
                            .copied()
                            .unwrap_or(1.0);
                        (
                            usize::from(b),
                            if b { "true" } else { "false" }.into(),
                            vec![w_false, w_true],
                        )
                    }
                };

                if let Some(ref fv) = feat_vec
                    && !target_label.is_empty()
                {
                    class_features
                        .entry(format!("{qid}:{target_label}"))
                        .or_default()
                        .push(fv.clone());
                }

                bucket_samples
                    .entry(bucket_key)
                    .or_default()
                    .push(ValidationSample {
                        raw_logits,
                        target_index: target_idx,
                        target_label,
                        option_weights,
                    });
                total_samples_collected += 1;
            }
        }

        if total_samples_collected == 0 {
            bail!(
                "No labeled ground-truth targets found in dataset. Ensure items specify 'ground_truth' fields."
            );
        }

        let baseline_temps: BTreeMap<String, f64> = [
            ("choice:2", 1.906),
            ("choice:3-5", 1.760),
            ("choice:6-10", 1.000),
            ("choice:11+", 0.101),
            ("score:2", 1.500),
            ("score:3-5", 1.251),
            ("score:6-10", 1.000),
            ("score:11+", 0.800),
            ("noul:2", 1.983),
            ("noul:3-5", 1.500),
            ("noul:6-10", 1.000),
            ("noul:11+", 0.800),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let mut bucket_reports = Vec::new();
        let mut fitted_temperatures = BTreeMap::new();
        let mut raw_cost_sum = 0.0;
        let mut cal_cost_sum = 0.0;

        for (bucket_name, samples) in &bucket_samples {
            let (raw_ece, raw_nll, raw_brier) = compute_bucket_metrics(samples, 1.0, self.bins);

            let (optimal_t, iterations) = if samples.len() >= 2 {
                // Evaluate log-spaced grid in [0.05, 5.0] to handle non-smooth ECE step plateaus
                let grid_n = 40usize;
                let ln_min = 0.05f64.ln();
                let ln_max = 5.0f64.ln();
                let mut grid: Vec<f64> = (0..grid_n)
                    .map(|i| (ln_min + (ln_max - ln_min) * (i as f64 / (grid_n - 1) as f64)).exp())
                    .collect();
                grid.push(1.0);
                grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let mut best_grid_idx = 0;
                let mut best_grid_loss = f64::INFINITY;
                for (i, &t) in grid.iter().enumerate() {
                    let l = compute_bucket_loss(samples, t, self.loss, self.bins);
                    if l < best_grid_loss {
                        best_grid_loss = l;
                        best_grid_idx = i;
                    }
                }
                let lo = grid[best_grid_idx.saturating_sub(1)];
                let hi = grid[(best_grid_idx + 1).min(grid.len() - 1)];
                let (gss_t, _, iters) = golden_section_search(
                    |t| compute_bucket_loss(samples, t, self.loss, self.bins),
                    lo,
                    hi,
                    1e-4,
                    30,
                );
                let (full_gss_t, _, _) = golden_section_search(
                    |t| compute_bucket_loss(samples, t, self.loss, self.bins),
                    0.05,
                    5.0,
                    1e-4,
                    30,
                );
                grid.push(gss_t);
                grid.push(full_gss_t);

                // Select candidate minimizing target loss without degrading raw ECE
                let mut best_t = 1.0;
                let mut best_tuple = (
                    compute_bucket_loss(samples, 1.0, self.loss, self.bins),
                    raw_ece,
                );
                for cand in grid {
                    let t_round = (cand * 1000.0).round() / 1000.0;
                    let (c_ece, _, _) = compute_bucket_metrics(samples, t_round, self.bins);
                    if c_ece <= raw_ece + 1e-9 {
                        let c_loss = compute_bucket_loss(samples, t_round, self.loss, self.bins);
                        if c_loss < best_tuple.0 - 1e-9
                            || ((c_loss - best_tuple.0).abs() <= 1e-9 && c_ece < best_tuple.1)
                        {
                            best_tuple = (c_loss, c_ece);
                            best_t = t_round;
                        }
                    }
                }
                (best_t, iters)
            } else {
                let fallback = baseline_temps.get(bucket_name).copied().unwrap_or(1.0);
                ((fallback * 1000.0).round() / 1000.0, 0)
            };

            let (cal_ece, cal_nll, cal_brier) =
                compute_bucket_metrics(samples, optimal_t, self.bins);
            raw_cost_sum += compute_expected_asymmetric_cost(samples, 1.0, false);
            cal_cost_sum += compute_expected_asymmetric_cost(samples, optimal_t, true);
            let improvement_pct = if raw_ece > 1e-6 {
                ((raw_ece - cal_ece) / raw_ece) * 100.0
            } else {
                0.0
            };

            fitted_temperatures.insert(bucket_name.clone(), optimal_t);
            bucket_reports.push(BucketCalibrationReport {
                bucket: bucket_name.clone(),
                sample_count: samples.len(),
                raw_temperature: 1.0,
                raw_ece,
                raw_nll,
                raw_brier,
                optimal_temperature: optimal_t,
                calibrated_ece: cal_ece,
                calibrated_nll: cal_nll,
                calibrated_brier: cal_brier,
                ece_improvement_pct: improvement_pct,
                iterations,
            });
        }

        for (std_b, &def_t) in &baseline_temps {
            fitted_temperatures.entry(std_b.clone()).or_insert(def_t);
        }

        let total_b = bucket_reports.len() as f64;
        let avg_raw_ece = if total_b > 0.0 {
            bucket_reports.iter().map(|b| b.raw_ece).sum::<f64>() / total_b
        } else {
            0.0
        };
        let avg_cal_ece = if total_b > 0.0 {
            bucket_reports.iter().map(|b| b.calibrated_ece).sum::<f64>() / total_b
        } else {
            0.0
        };
        let overall_improvement = if avg_raw_ece > 1e-6 {
            ((avg_raw_ece - avg_cal_ece) / avg_raw_ece) * 100.0
        } else {
            0.0
        };

        let qats_profile = if self.enable_qats && !class_features.is_empty() {
            let mut centroids = BTreeMap::new();
            for (cls, feats) in &class_features {
                if feats.is_empty() {
                    continue;
                }
                let dim = feats[0].len();
                let mut mean = vec![0.0f64; dim];
                for f in feats {
                    for (i, &v) in f.iter().enumerate() {
                        mean[i] += v / feats.len() as f64;
                    }
                }
                let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-6);
                for v in &mut mean {
                    *v /= norm;
                }
                centroids.insert(cls.clone(), mean);
            }

            let distances: Vec<f64> = item_features
                .iter()
                .filter_map(|feat| {
                    centroids
                        .values()
                        .map(|c| euclidean_distance(feat, c))
                        .fold(f64::INFINITY, f64::min)
                        .into()
                })
                .filter(|d: &f64| d.is_finite())
                .collect();

            let n_dist = distances.len() as f64;
            let mean_d = if n_dist > 0.0 {
                distances.iter().sum::<f64>() / n_dist
            } else {
                1.0
            };
            let var_d = if n_dist > 1.0 {
                distances.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / (n_dist - 1.0)
            } else {
                0.04
            };

            Some(QatsProfile {
                enabled: true,
                gamma: self.qats_gamma,
                mean_distance: (mean_d * 1000.0).round() / 1000.0,
                std_distance: (var_d.sqrt().max(0.01) * 1000.0).round() / 1000.0,
                class_centroids: centroids,
            })
        } else {
            None
        };

        let tuning_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let avg_raw_brier = if total_b > 0.0 {
            bucket_reports.iter().map(|b| b.raw_brier).sum::<f64>() / total_b
        } else {
            0.0
        };
        let avg_cal_brier = if total_b > 0.0 {
            bucket_reports
                .iter()
                .map(|b| b.calibrated_brier)
                .sum::<f64>()
                / total_b
        } else {
            0.0
        };
        let per_question: Vec<QuestionCalibrationSummary> = questions
            .iter()
            .map(|(qid, q)| {
                let bkey =
                    classifier.get_temperature_bucket(q.qtype, q.render_options().len().max(2));
                let brep = bucket_reports.iter().find(|b| b.bucket == bkey);
                QuestionCalibrationSummary {
                    question_id: qid.clone(),
                    bucket_key: bkey.clone(),
                    optimal_temperature: fitted_temperatures.get(&bkey).copied().unwrap_or(1.0),
                    ece_before: brep.map_or(avg_raw_ece, |b| b.raw_ece),
                    ece_after: brep.map_or(avg_cal_ece, |b| b.calibrated_ece),
                }
            })
            .collect();

        let n_samp = total_samples_collected.max(1) as f64;
        let raw_expected_cost = ((raw_cost_sum / n_samp) * 10000.0).round() / 10000.0;
        let calibrated_expected_cost = ((cal_cost_sum / n_samp) * 10000.0).round() / 10000.0;
        let cost_reduction_pct = if raw_expected_cost > 1e-6 {
            (((raw_expected_cost - calibrated_expected_cost) / raw_expected_cost) * 10000.0).round()
                / 100.0
        } else {
            0.0
        };

        let report = CalibrationReport {
            dataset_name: dataset_name.to_string(),
            total_samples: total_samples_collected,
            loss_function: self.loss,
            buckets: bucket_reports,
            average_raw_ece: avg_raw_ece,
            average_calibrated_ece: avg_cal_ece,
            overall_improvement_pct: overall_improvement,
            tuning_duration_ms,
            qats_profile: qats_profile.clone(),
            cost_weights: self.cost_weights.clone(),
            raw_expected_cost,
            calibrated_expected_cost,
            cost_reduction_pct,
            uncalibrated_ece: avg_raw_ece,
            calibrated_ece: avg_cal_ece,
            ece_reduction_pct: overall_improvement,
            uncalibrated_brier: avg_raw_brier,
            calibrated_brier: avg_cal_brier,
            global_temperature: 1.0,
            per_question,
        };

        let config = CalibratedAgentConfig {
            checkpoint: self.checkpoint.repo_id().to_string(),
            temperature: 1.0,
            temperature_by_options: fitted_temperatures,
            loss_function: format!("{:?}", self.loss).to_lowercase(),
            metadata: CalibrationMetadata {
                dataset: dataset_name.to_string(),
                samples_evaluated: total_samples_collected,
                raw_mean_ece: avg_raw_ece,
                calibrated_mean_ece: avg_cal_ece,
                improvement_pct: overall_improvement,
                calibrated_at: crate::engine::metrics::format_utc_iso8601(),
            },
            qats: qats_profile,
            cost_weights: self.cost_weights.clone(),
        };

        Ok((report, config))
    }

    /// Format calibration report as a readable ASCII table.
    pub fn format_table(report: &CalibrationReport, export_path: Option<&Path>) -> String {
        let mut out = format!(
            "\n🌡️  Lemma Temperature Calibration Tuner\nDataset:       {} ({} samples)\nLoss Function: {:?} Loss ({:.1}ms)\n\n",
            report.dataset_name,
            report.total_samples,
            report.loss_function,
            report.tuning_duration_ms
        );

        let mut table = Table::new();
        table.set_header(vec![
            "Bucket",
            "Samples",
            "Raw ECE",
            "Fitted Temp (T*)",
            "Calibrated ECE",
            "Improvement",
        ]);
        for b in &report.buckets {
            table.add_row(vec![
                b.bucket.clone(),
                b.sample_count.to_string(),
                format!("{:.4}", b.raw_ece),
                format!("{:.3}", b.optimal_temperature),
                format!("{:.4}", b.calibrated_ece),
                format!("{:+.1}%", b.ece_improvement_pct),
            ]);
        }
        table.add_row(vec![
            "Average".to_string(),
            report.total_samples.to_string(),
            format!("{:.4}", report.average_raw_ece),
            "-".to_string(),
            format!("{:.4}", report.average_calibrated_ece),
            format!("{:+.1}%", report.overall_improvement_pct),
        ]);
        out.push_str(&table.to_string());
        out.push('\n');

        if let Some(qats) = &report.qats_profile {
            out.push_str(&format!(
                "\n📡 2026 Q-ATS Profile:\n   • Adaptive scaling factor gamma: {:.2}\n   • In-distribution mean distance:  {:.3}\n   • Distance standard deviation:    {:.3}\n   • Tracked class centroids:        {}\n",
                qats.gamma, qats.mean_distance, qats.std_distance, qats.class_centroids.len()
            ));
        }

        if !report.cost_weights.is_empty() {
            let weights_str: Vec<String> = report
                .cost_weights
                .iter()
                .map(|(k, v)| format!("{k}={v:.2}x"))
                .collect();
            out.push_str(&format!(
                "\n⚖️  Asymmetric Cost-Sensitive Threshold Optimization:\n   • Class Cost Weights:      {}\n   • Raw Expected Risk:       {:.4}\n   • Calibrated Bayes Risk:   {:.4} ({:+.1}% cost reduction)\n",
                weights_str.join(", "),
                report.raw_expected_cost,
                report.calibrated_expected_cost,
                report.cost_reduction_pct
            ));
        }

        if let Some(path) = export_path {
            out.push_str(&format!(
                "\n✅ Saved calibrated configuration to {}\nUse with: lemma classify --config {}\n",
                path.display(),
                path.display()
            ));
        }

        out
    }

    /// Format calibration report as JSON string.
    pub fn format_json(report: &CalibrationReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }
}

/// Compute scalar loss for a bucket at temperature T.
fn compute_bucket_loss(
    samples: &[ValidationSample],
    t: f64,
    loss: CalibrationLoss,
    bins: usize,
) -> f64 {
    let t_safe = t.max(0.01);
    match loss {
        CalibrationLoss::Nll => compute_bucket_metrics(samples, t_safe, bins).1,
        CalibrationLoss::Brier => compute_bucket_metrics(samples, t_safe, bins).2,
        CalibrationLoss::Ece => compute_bucket_metrics(samples, t_safe, bins).0,
    }
}

/// Compute expected asymmetric Bayes cost across samples at temperature T.
fn compute_expected_asymmetric_cost(
    samples: &[ValidationSample],
    t: f64,
    use_cost_rule: bool,
) -> f64 {
    let t_safe = t.max(0.01);
    let mut total_cost = 0.0;
    for s in samples {
        let max_z = s
            .raw_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = s
            .raw_logits
            .iter()
            .map(|&z| ((z - max_z) / t_safe).exp())
            .sum();
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &z) in s.raw_logits.iter().enumerate() {
            let p = ((z - max_z) / t_safe).exp() / exp_sum.max(1e-12);
            let w = if use_cost_rule {
                s.option_weights.get(i).copied().unwrap_or(1.0)
            } else {
                1.0
            };
            if p * w > best_score {
                best_score = p * w;
                best_idx = i;
            }
        }
        let target_w = s.option_weights.get(s.target_index).copied().unwrap_or(1.0);
        // Expected conditional risk + realized asymmetric penalty
        let p_target = s
            .raw_logits
            .get(s.target_index)
            .map(|&z| ((z - max_z) / t_safe).exp() / exp_sum.max(1e-12))
            .unwrap_or(0.0);
        let realized = if best_idx != s.target_index {
            target_w
        } else {
            0.0
        };
        total_cost += 0.5 * realized + 0.5 * target_w * (1.0 - p_target);
    }
    total_cost
}

/// Compute (ECE, NLL, Brier) metrics for a bucket at temperature T.
fn compute_bucket_metrics(samples: &[ValidationSample], t: f64, bins: usize) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let t_safe = t.max(0.01);
    let mut confs = Vec::with_capacity(samples.len());
    let mut corr = Vec::with_capacity(samples.len());
    let mut total_nll = 0.0;
    let mut total_brier = 0.0;
    let mut weight_sum = 0.0;

    for s in samples {
        let max_z = s
            .raw_logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = s
            .raw_logits
            .iter()
            .map(|&z| ((z - max_z) / t_safe).exp())
            .sum();
        let probs: Vec<f64> = s
            .raw_logits
            .iter()
            .map(|&z| ((z - max_z) / t_safe).exp() / exp_sum.max(1e-12))
            .collect();

        let mut best_idx = 0;
        let mut max_p = -1.0;
        let mut best_weighted = -1.0;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_p {
                max_p = p;
            }
            let w = s.option_weights.get(i).copied().unwrap_or(1.0);
            if p * w > best_weighted {
                best_weighted = p * w;
                best_idx = i;
            }
        }

        let k = probs.len().max(2);
        let conf = if k > 1 {
            ((max_p - (1.0 - max_p) / (k as f64 - 1.0)).max(0.0) * 10000.0).round() / 10000.0
        } else {
            max_p
        };
        confs.push(conf);
        corr.push(best_idx == s.target_index);

        let sample_w = s.option_weights.get(s.target_index).copied().unwrap_or(1.0);
        weight_sum += sample_w;
        let target_p = probs.get(s.target_index).copied().unwrap_or(1e-12);
        total_nll -= sample_w * target_p.max(1e-12).ln();

        for (i, &p) in probs.iter().enumerate() {
            let y = if i == s.target_index { 1.0 } else { 0.0 };
            total_brier += sample_w * (p - y).powi(2);
        }
    }

    let ece = calculate_ece(&confs, &corr, bins);
    let denom = weight_sum.max(1e-9);
    let nll = total_nll / denom;
    let brier = total_brier / denom;
    (ece, nll, brier)
}

/// 1D Golden-Section Search minimization algorithm.
pub fn golden_section_search<F>(
    mut f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64, usize)
where
    F: FnMut(f64) -> f64,
{
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    let mut iter = 0;

    while (b - a).abs() > tol && iter < max_iter {
        iter += 1;
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }

    let best_x = (a + b) / 2.0;
    let best_f = f(best_x);
    (best_x, best_f, iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_section_parabola_minimum() {
        let (min_x, min_y, iters) =
            golden_section_search(|x| (x - 2.5).powi(2) + 1.0, 0.0, 5.0, 1e-5, 30);
        assert!((min_x - 2.5).abs() < 1e-4);
        assert!((min_y - 1.0).abs() < 1e-4);
        assert!(iters > 5);
    }

    #[test]
    fn test_bucket_metrics_computation() {
        let sample = ValidationSample {
            raw_logits: vec![2.0, 0.5, -0.5],
            target_index: 0,
            target_label: "class_a".into(),
            option_weights: vec![1.0, 1.0, 1.0],
        };
        let (ece, nll, brier) = compute_bucket_metrics(&[sample], 1.0, 5);
        assert!(ece >= 0.0);
        assert!(nll > 0.0);
        assert!(brier >= 0.0);
    }

    #[test]
    fn test_parse_cost_weights_and_asymmetric_reduction() {
        let cw = parse_cost_weights("billing=5.0, security:10.0").unwrap();
        assert_eq!(cw.get("billing"), Some(&5.0));
        assert_eq!(cw.get("security"), Some(&10.0));

        // Sample where target is index 1 (raw logit slightly lower than index 0, but cost weight is 5x)
        let sample = ValidationSample {
            raw_logits: vec![1.0, 0.7],
            target_index: 1,
            target_label: "billing".into(),
            option_weights: vec![1.0, 5.0],
        };
        let raw_cost = compute_expected_asymmetric_cost(std::slice::from_ref(&sample), 1.0, false);
        let cal_cost = compute_expected_asymmetric_cost(&[sample], 1.0, true);
        assert!(cal_cost < raw_cost);
    }
}
