use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use crate::datasource::DataItem;
use crate::engine::classifier::LayaClassifier;
use crate::engine::features::vectorize_text;
use crate::engine::metrics::calculate_ece;
use crate::engine::ood::euclidean_distance;
use crate::model::{Checkpoint, Question, QuestionType};

/// Lowest temperature lemmata fits or applies to a bucket.
pub const MIN_TEMPERATURE: f64 = 0.5;
/// Highest temperature lemmata fits or applies to a bucket.
pub const MAX_TEMPERATURE: f64 = 5.0;
/// Fewer labeled samples than this cannot support a fitted bucket temperature.
pub const DEFAULT_MIN_BUCKET_SAMPLES: usize = 30;

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
    min_bucket_samples: usize,
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
            min_bucket_samples: DEFAULT_MIN_BUCKET_SAMPLES,
        }
    }

    /// Buckets with fewer labeled samples keep their prior temperature.
    pub fn with_min_bucket_samples(mut self, n: usize) -> Self {
        self.min_bucket_samples = n.max(2);
        self
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
        let (report, config, _) = self.fit_full(items, questions, dataset_name, false)?;
        Ok((report, config))
    }

    /// Single-pass temperature calibration that optionally also fits per-label log-biases
    /// from the exact same collected option logits without a second forward pass.
    pub fn fit_full(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        dataset_name: &str,
        include_label_bias: bool,
    ) -> Result<(
        CalibrationReport,
        CalibratedAgentConfig,
        Option<LabelBiasFit>,
    )> {
        if items.is_empty() {
            bail!("Validation dataset cannot be empty for calibration");
        }
        if questions.is_empty() {
            bail!("Questions schema cannot be empty for calibration");
        }

        let start_time = Instant::now();
        let classifier = LayaClassifier::with_checkpoint(self.checkpoint);

        type ChoiceBiasEntry = (Vec<String>, String, Vec<(Vec<f64>, usize)>);
        let mut bucket_samples: BTreeMap<String, Vec<ValidationSample>> = BTreeMap::new();
        let mut choice_bias_inputs: BTreeMap<String, ChoiceBiasEntry> = BTreeMap::new();
        let mut class_features: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
        let mut item_features: Vec<Vec<f64>> = Vec::new();
        let mut total_samples_collected = 0;

        let labeled_items: Vec<&DataItem> = items
            .iter()
            .filter(|it| it.ground_truth.is_some())
            .collect();
        let per_item_scored =
            crate::model::native::map_items(&labeled_items, |&item| -> Result<_> {
                let gt_map = item.ground_truth.as_ref().unwrap();
                let state_text = item.state_text();
                let feat_vec = self.enable_qats.then(|| vectorize_text(&state_text));
                let mut item_qs = Vec::new();

                for (qid, q) in questions {
                    let Some(target_val) = super::metrics::resolve_ground_truth_for_question(
                        Some(gt_map),
                        qid,
                        q,
                        questions.len(),
                    ) else {
                        continue;
                    };
                    let options = q.render_options();
                    let k = options.len().max(2);
                    let bucket_key = classifier.get_temperature_bucket(q.qtype, k);
                    let raw_logits = classifier.compute_option_logits(&state_text, q, &options)?;

                    let (target_idx, target_label, option_weights, choice_keys) = match q.qtype {
                        QuestionType::Choice => {
                            let keys: Vec<String> = if let Some(serde_json::Value::Object(map)) =
                                &q.criteria
                            {
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
                                Some(idx) => (idx, target_str.to_string(), weights, Some(keys)),
                                None => continue,
                            }
                        }
                        QuestionType::Score => {
                            let score_val = target_val
                                .as_u64()
                                .or_else(|| target_val.as_str().and_then(|s| s.parse::<u64>().ok()))
                                .unwrap_or(0) as usize;
                            let clamped = score_val.min(raw_logits.len().saturating_sub(1));
                            (
                                clamped,
                                clamped.to_string(),
                                vec![1.0; raw_logits.len()],
                                None,
                            )
                        }
                        QuestionType::Noul => {
                            let Some(b) = super::metrics::parse_noul_ground_truth(target_val)
                            else {
                                continue;
                            };
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
                                None,
                            )
                        }
                    };

                    item_qs.push((
                        qid.clone(),
                        bucket_key,
                        choice_keys,
                        ValidationSample {
                            raw_logits,
                            target_index: target_idx,
                            target_label,
                            option_weights,
                        },
                    ));
                }
                Ok((feat_vec, item_qs))
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        for (feat_vec, item_qs) in per_item_scored {
            if let Some(ref fv) = feat_vec {
                item_features.push(fv.clone());
            }
            for (qid, bucket_key, choice_keys, sample) in item_qs {
                if let Some(ref fv) = feat_vec
                    && !sample.target_label.is_empty()
                {
                    class_features
                        .entry(format!("{qid}:{}", sample.target_label))
                        .or_default()
                        .push(fv.clone());
                }
                if include_label_bias && let Some(keys) = choice_keys {
                    let entry = choice_bias_inputs
                        .entry(qid.clone())
                        .or_insert_with(|| (keys, bucket_key.clone(), Vec::new()));
                    entry
                        .2
                        .push((sample.raw_logits.clone(), sample.target_index));
                }
                bucket_samples.entry(bucket_key).or_default().push(sample);
                total_samples_collected += 1;
            }
        }

        if total_samples_collected == 0 {
            bail!(
                "No labeled ground-truth targets found in dataset. Ensure items specify 'ground_truth' fields."
            );
        }

        let default_qtype_temps = self.checkpoint.default_temperatures();
        let default_bucket_temps = self.checkpoint.default_temperature_by_options();
        let std_buckets = [
            ("choice:2", 0usize),
            ("choice:3-5", 0),
            ("choice:6-10", 0),
            ("choice:11+", 0),
            ("score:2", 1),
            ("score:3-5", 1),
            ("score:6-10", 1),
            ("score:11+", 1),
            ("noul:2", 2),
            ("noul:3-5", 2),
            ("noul:6-10", 2),
            ("noul:11+", 2),
        ];
        let baseline_temps: BTreeMap<String, f64> = std_buckets
            .into_iter()
            .map(|(k, q_idx)| {
                let raw = default_bucket_temps
                    .get(k)
                    .copied()
                    .unwrap_or(default_qtype_temps[q_idx])
                    .clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
                (k.to_string(), (raw * 1000.0).round() / 1000.0)
            })
            .collect();

        let mut bucket_reports = Vec::new();
        let mut fitted_temperatures = BTreeMap::new();
        let mut raw_cost_sum = 0.0;
        let mut cal_cost_sum = 0.0;

        for (bucket_name, samples) in &bucket_samples {
            let prior_t = baseline_temps.get(bucket_name).copied().unwrap_or(1.0);
            let (raw_ece, raw_nll, raw_brier) = compute_bucket_metrics(samples, prior_t, self.bins);

            let (optimal_t, iterations) = if samples.len() >= self.min_bucket_samples {
                // Evaluate a log-spaced grid over the safe range to handle non-smooth ECE step plateaus
                let grid_n = 40usize;
                let ln_min = MIN_TEMPERATURE.ln();
                let ln_max = MAX_TEMPERATURE.ln();
                let mut grid: Vec<f64> = (0..grid_n)
                    .map(|i| (ln_min + (ln_max - ln_min) * (i as f64 / (grid_n - 1) as f64)).exp())
                    .collect();
                grid.push(1.0);
                grid.push(prior_t);
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
                    MIN_TEMPERATURE,
                    MAX_TEMPERATURE,
                    1e-4,
                    30,
                );
                grid.push(gss_t);
                grid.push(full_gss_t);

                // Select candidate minimizing target loss without degrading raw ECE
                let mut best_t = prior_t;
                let mut best_tuple = (
                    compute_bucket_loss(samples, prior_t, self.loss, self.bins),
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
                eprintln!(
                    "⚠️  Bucket {bucket_name} has {} labeled samples (minimum {}). It keeps its prior temperature.",
                    samples.len(),
                    self.min_bucket_samples
                );
                (prior_t, 0)
            };

            let (cal_ece, cal_nll, cal_brier) =
                compute_bucket_metrics(samples, optimal_t, self.bins);
            raw_cost_sum += compute_expected_asymmetric_cost(samples, prior_t, false);
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
                raw_temperature: prior_t,
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

        let total_sample_weight: usize = bucket_reports.iter().map(|b| b.sample_count).sum();
        let avg_raw_ece = if total_sample_weight > 0 {
            bucket_reports
                .iter()
                .map(|b| b.raw_ece * b.sample_count as f64)
                .sum::<f64>()
                / total_sample_weight as f64
        } else {
            0.0
        };
        let avg_cal_ece = if total_sample_weight > 0 {
            bucket_reports
                .iter()
                .map(|b| b.calibrated_ece * b.sample_count as f64)
                .sum::<f64>()
                / total_sample_weight as f64
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
        let avg_raw_brier = if total_sample_weight > 0 {
            bucket_reports
                .iter()
                .map(|b| b.raw_brier * b.sample_count as f64)
                .sum::<f64>()
                / total_sample_weight as f64
        } else {
            0.0
        };
        let avg_cal_brier = if total_sample_weight > 0 {
            bucket_reports
                .iter()
                .map(|b| b.calibrated_brier * b.sample_count as f64)
                .sum::<f64>()
                / total_sample_weight as f64
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

        let label_bias_fit = include_label_bias.then(|| {
            let mut fit = LabelBiasFit {
                weights: BTreeMap::new(),
                questions: BTreeMap::new(),
            };
            for (qid, (labels, bkey, raw_vec)) in choice_bias_inputs {
                if raw_vec.is_empty() || labels.is_empty() {
                    continue;
                }
                let t = fitted_temperatures
                    .get(&bkey)
                    .copied()
                    .unwrap_or(1.0)
                    .clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
                let samples: Vec<(Vec<f64>, usize)> = raw_vec
                    .into_iter()
                    .map(|(raw_logits, target_idx)| {
                        let scaled: Vec<f64> = raw_logits
                            .iter()
                            .enumerate()
                            .map(|(i, &z)| {
                                let w = labels
                                    .get(i)
                                    .and_then(|l| self.cost_weights.get(&l.to_lowercase()))
                                    .copied()
                                    .unwrap_or(1.0)
                                    .max(1e-9);
                                z / t + w.ln()
                            })
                            .collect();
                        let max_s = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let lse = max_s
                            + scaled
                                .iter()
                                .map(|&v| (v - max_s).exp())
                                .sum::<f64>()
                                .max(1e-12)
                                .ln();
                        let log_p = scaled.into_iter().map(|v| v - lse).collect();
                        (log_p, target_idx)
                    })
                    .collect();
                record_label_bias_question(&mut fit, &qid, &labels, &samples);
            }
            fit
        });

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

        Ok((report, config, label_bias_fit))
    }

    /// Format calibration report as a readable ASCII table.
    pub fn format_table(report: &CalibrationReport, export_path: Option<&Path>) -> String {
        let mut out = format!(
            "\n🌡️  Lemmata Temperature Calibration Tuner\nDataset:       {} ({} samples)\nLoss Function: {:?} Loss ({:.1}ms)\n\n",
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
                "\n✅ Saved calibrated configuration to {}\nUse with: lemmata classify --config {}\n",
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
        let mut best_weighted = -1.0;
        for (i, &p) in probs.iter().enumerate() {
            let w = s.option_weights.get(i).copied().unwrap_or(1.0);
            if p * w > best_weighted {
                best_weighted = p * w;
                best_idx = i;
            }
        }

        // Top-label ECE requires the predicted probability of the selected class `best_idx`.
        let p_win = probs.get(best_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        confs.push(p_win);
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

/// Result of [`fit_label_bias`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelBiasFit {
    /// `exp(bias)` per lowercase choice label, for `cost_weights`.
    pub weights: BTreeMap<String, f64>,
    /// Per choice question: (labelled samples, accuracy before, accuracy after) on the fit data.
    pub questions: BTreeMap<String, (usize, f64, f64)>,
}

/// L2 penalty on the label biases. It keeps labels with few or no examples near zero.
const LABEL_BIAS_L2: f64 = 0.01;

fn softmax_with_bias(log_p: &[f64], bias: &[f64]) -> Vec<f64> {
    let z: Vec<f64> = log_p.iter().zip(bias).map(|(a, b)| a + b).collect();
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

/// Gradient descent on the mean negative log-likelihood of `softmax(log_p + bias)`, plus L2.
/// The objective is convex in the bias. The result is centered, which softmax ignores.
fn fit_bias(samples: &[(Vec<f64>, usize)], k: usize) -> Vec<f64> {
    let n = samples.len().max(1) as f64;
    let mut bias = vec![0.0; k];
    for _ in 0..500 {
        let mut grad: Vec<f64> = bias.iter().map(|b| 2.0 * LABEL_BIAS_L2 * b).collect();
        for (log_p, target) in samples {
            for (j, q_j) in softmax_with_bias(log_p, &bias).into_iter().enumerate() {
                grad[j] += (q_j - f64::from(j == *target)) / n;
            }
        }
        for (b, g) in bias.iter_mut().zip(grad) {
            *b -= g;
        }
    }
    let mean = bias.iter().sum::<f64>() / k as f64;
    bias.iter().map(|b| b - mean).collect()
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn record_label_bias_question(
    fit: &mut LabelBiasFit,
    qid: &str,
    labels: &[String],
    samples: &[(Vec<f64>, usize)],
) {
    if samples.is_empty() || labels.is_empty() {
        return;
    }
    let k = labels.len();
    let n = samples.len() as f64;
    let bias = fit_bias(samples, k);
    let zero = vec![0.0; k];
    let accuracy = |b: &[f64]| {
        samples
            .iter()
            .filter(|(log_p, t)| argmax(&softmax_with_bias(log_p, b)) == *t)
            .count() as f64
            / n
    };
    fit.questions.insert(
        qid.to_string(),
        (samples.len(), accuracy(&zero), accuracy(&bias)),
    );
    for (label, b) in labels.iter().zip(&bias) {
        let key = label.to_lowercase();
        if fit.weights.contains_key(&key) {
            eprintln!(
                "⚠️  Label '{label}' appears in more than one choice question. Its bias comes from the first one."
            );
            continue;
        }
        fit.weights.insert(key, b.exp());
    }
}

/// Fit one log-bias per choice label that maximizes the likelihood of the labelled answers
/// (vector scaling with a bias term, Guo et al. 2017), with L2 regularization. Zero-shot models
/// favor some label words regardless of the text, and the bias removes that prior. The result
/// goes into `cost_weights`, because `classify` picks `argmax p × w`, which is `p` shifted by
/// `ln w`. Fit on held-out labelled data, not on the data you will test on.
pub fn fit_label_bias(
    classifier: &LayaClassifier,
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
) -> Result<LabelBiasFit> {
    let results = classifier.classify_batch(items, questions)?;
    let mut fit = LabelBiasFit {
        weights: BTreeMap::new(),
        questions: BTreeMap::new(),
    };
    for (qid, q) in questions {
        if q.qtype != QuestionType::Choice {
            continue;
        }
        let option_keys: Vec<String> = q
            .render_options()
            .into_iter()
            .map(|o| o.split(':').next().unwrap_or(&o).trim().to_string())
            .collect();
        let bkey = classifier.get_temperature_bucket(q.qtype, option_keys.len().max(2));
        let temp = classifier
            .config()
            .temperature_by_options
            .get(&bkey)
            .copied()
            .unwrap_or(classifier.config().temperature)
            .clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);

        let mut labels: Vec<String> = Vec::new();
        let mut samples: Vec<(Vec<f64>, usize)> = Vec::new();
        for (item, (_, res)) in items.iter().zip(&results) {
            let Some(ans) = res.answers.get(qid) else {
                continue;
            };
            if labels.is_empty() {
                labels = if ans
                    .logits
                    .as_ref()
                    .is_some_and(|l| l.len() == option_keys.len())
                {
                    option_keys.clone()
                } else {
                    ans.probabilities.keys().cloned().collect()
                };
            }
            let truth = crate::engine::metrics::resolve_ground_truth_for_question(
                item.ground_truth.as_ref(),
                qid,
                q,
                questions.len(),
            )
            .and_then(|v| v.as_str().map(str::to_lowercase));
            let Some(target) =
                truth.and_then(|t| labels.iter().position(|l| l.to_lowercase() == t))
            else {
                continue;
            };
            let log_p: Vec<f64> =
                if let Some(logits) = ans.logits.as_ref().filter(|l| l.len() == labels.len()) {
                    let scaled: Vec<f64> = logits
                        .iter()
                        .enumerate()
                        .map(|(i, &z)| {
                            let w = classifier
                                .config()
                                .cost_weights
                                .get(&labels[i].to_lowercase())
                                .copied()
                                .unwrap_or(1.0)
                                .max(1e-9);
                            z / temp + w.ln()
                        })
                        .collect();
                    let max_s = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let lse = max_s
                        + scaled
                            .iter()
                            .map(|&v| (v - max_s).exp())
                            .sum::<f64>()
                            .max(1e-12)
                            .ln();
                    scaled.into_iter().map(|v| v - lse).collect()
                } else {
                    labels
                        .iter()
                        .map(|l| {
                            ans.probabilities
                                .get(l)
                                .copied()
                                .unwrap_or(0.0)
                                .max(1e-6)
                                .ln()
                        })
                        .collect()
                };
            samples.push((log_p, target));
        }
        record_label_bias_question(&mut fit, qid, &labels, &samples);
    }
    Ok(fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_bias_removes_a_constant_label_preference() {
        // The model always gives label 0 a +1.0 head start, so it wins even when label 1 is true.
        let samples: Vec<(Vec<f64>, usize)> = (0..40)
            .map(|i| {
                let target = i % 2;
                let mut log_p = vec![-2.0, -2.0];
                log_p[target] += 0.6;
                log_p[0] += 1.0;
                (log_p, target)
            })
            .collect();
        let before = samples.iter().filter(|(lp, t)| argmax(lp) == *t).count();
        let bias = fit_bias(&samples, 2);
        let after = samples
            .iter()
            .filter(|(lp, t)| argmax(&softmax_with_bias(lp, &bias)) == *t)
            .count();
        assert_eq!(before, 20);
        assert_eq!(after, 40);
        assert!(bias[1] > bias[0]);
    }

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

        // Perfectly calibrated binary predictions ([0.60, 0.40] with 60% empirical accuracy)
        // must yield top-label ECE == 0.0 (not 0.40!).
        let logit_60 = (0.6_f64 / 0.4_f64).ln();
        let cal_samples: Vec<ValidationSample> = (0..10)
            .map(|i| ValidationSample {
                raw_logits: vec![logit_60, 0.0],
                target_index: if i < 6 { 0 } else { 1 },
                target_label: if i < 6 { "a".into() } else { "b".into() },
                option_weights: vec![1.0, 1.0],
            })
            .collect();
        let (cal_ece, _, _) = compute_bucket_metrics(&cal_samples, 1.0, 10);
        assert!(
            cal_ece < 1e-9,
            "Perfectly calibrated 60% binary predictions should have ECE ~ 0.0, got {cal_ece}"
        );
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
