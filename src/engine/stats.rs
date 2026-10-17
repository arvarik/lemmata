use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, Color, ContentArrangement, Table};
use serde::{Deserialize, Serialize};

use crate::datasource::DataItem;
use crate::engine::classifier::LayaClassifier;
use crate::engine::metrics::{
    calculate_brier_score, calculate_ece, resolve_ground_truth_for_question,
};
use crate::engine::{CacheLookupResult, DecisionCache, SemanticCacheConfig, compute_schema_hash};
use crate::model::{Question, QuestionType};

/// Speed and latency statistics for single-item decisions and batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedStats {
    pub total_elapsed_secs: f64,
    pub latency_min_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p90_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_max_ms: f64,
    pub latency_avg_ms: f64,
    pub latency_std_dev_ms: f64,
}

/// System throughput and hardware efficiency statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub throughput_items_per_sec: f64,
    pub throughput_decisions_per_sec: f64,
    pub total_items_processed: usize,
    pub total_decisions_evaluated: usize,
    pub avg_head_token_utilization_pct: f64,
    pub cache_summary: Option<CacheStatsSummary>,
}

/// Sub-millisecond decision cache efficiency and latency savings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatsSummary {
    pub enabled: bool,
    pub total_lookups: u64,
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
    pub hit_rate_pct: f64,
    pub cached_avg_latency_ms: f64,
    pub uncached_avg_latency_ms: f64,
    pub speedup_ratio: f64,
}

/// Accuracy, calibration quality, and strictly proper scoring rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyStats {
    pub labeled_sample_count: usize,
    pub overall_accuracy: f64,
    pub soft_accuracy: f64,
    pub expected_calibration_error: f64,
    pub brier_score: f64,
}

/// Epistemic and aleatoric uncertainty distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyStats {
    pub mean_confidence: f64,
    pub mean_top2_margin: f64,
    pub mean_entropy: f64,
    pub act_escalate_rate: f64,
}

/// Granular evaluation breakdown for an individual question schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionStats {
    pub question_id: String,
    pub question_type: String,
    pub options_count: usize,
    pub est_head_tokens: usize,
    pub head_budget: usize,
    pub head_utilization_pct: f64,
    pub evaluated_samples: usize,
    pub accuracy: Option<f64>,
    pub soft_accuracy: Option<f64>,
    pub mean_confidence: f64,
    pub mean_top2_margin: f64,
    pub mean_entropy: f64,
    pub act_escalate_rate: f64,
}

/// Comprehensive, unified system statistics report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatsReport {
    pub checkpoint: String,
    pub head_budget: usize,
    pub dataset_items: usize,
    pub benchmark_iterations: usize,
    pub warmup_runs: usize,
    pub speed: SpeedStats,
    pub performance: PerformanceStats,
    pub accuracy: Option<AccuracyStats>,
    pub uncertainty: UncertaintyStats,
    pub questions: BTreeMap<String, QuestionStats>,
}

/// Configuration options for the stats evaluation engine.
#[derive(Debug, Clone)]
pub struct StatsEngineConfig {
    pub iterations: usize,
    pub warmup_runs: usize,
    pub enable_cache: bool,
    pub similarity_threshold: f32,
    pub escalate_threshold: f64,
    pub ece_bins: usize,
}

impl Default for StatsEngineConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            warmup_runs: 2,
            enable_cache: false,
            similarity_threshold: 0.92,
            escalate_threshold: 0.50,
            ece_bins: 10,
        }
    }
}

/// Unified system evaluation and benchmarking engine for Laya models.
pub struct StatsEngine {
    classifier: LayaClassifier,
    config: StatsEngineConfig,
}

impl StatsEngine {
    pub fn new(classifier: LayaClassifier) -> Self {
        Self {
            classifier,
            config: StatsEngineConfig::default(),
        }
    }

    pub fn with_config(mut self, config: StatsEngineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.config.iterations = iterations.max(1);
        self
    }

    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.config.warmup_runs = warmup;
        self
    }

    pub fn with_cache(mut self, enable_cache: bool, similarity_threshold: f32) -> Self {
        self.config.enable_cache = enable_cache;
        self.config.similarity_threshold = similarity_threshold;
        self
    }

    pub fn with_escalate_threshold(mut self, threshold: f64) -> Self {
        self.config.escalate_threshold = threshold;
        self
    }

    /// Evaluates speed, system performance, calibration, and classification accuracy.
    pub fn evaluate(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<SystemStatsReport> {
        if items.is_empty() {
            bail!("No items provided for system statistics evaluation");
        }
        if questions.is_empty() {
            bail!("No question schema provided for system statistics evaluation");
        }

        let checkpoint = self.classifier.config().checkpoint;
        let head_budget = checkpoint.head_budget();
        let total_items = items.len();

        // 1. Calculate question head token utilization
        let mut question_token_utils = BTreeMap::new();
        let mut total_head_util_sum = 0.0;
        for (qid, q) in questions {
            let options = q.render_options();
            let est_tokens: usize = options
                .iter()
                .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
                .sum();
            let util_pct = (est_tokens as f64 / head_budget as f64 * 100.0).clamp(0.0, 200.0);
            total_head_util_sum += util_pct;
            question_token_utils.insert(qid.clone(), (options.len(), est_tokens, util_pct));
        }
        let avg_head_token_utilization_pct = total_head_util_sum / questions.len() as f64;

        // 2. Warmup phase: run warmup iterations to hot-cache memory and CPU pipelines
        for _ in 0..self.config.warmup_runs {
            for item in items.iter().take(5) {
                let _ = self.classifier.classify(&item.state, questions);
            }
        }

        // 3. Speed & Performance benchmarking phase
        let mut latencies_ms = Vec::with_capacity(total_items * self.config.iterations);
        let mut cached_latencies_ms = Vec::new();
        let mut uncached_latencies_ms = Vec::new();

        let mut opt_cache = if self.config.enable_cache {
            let cache_cfg = SemanticCacheConfig {
                semantic: true,
                max_entries: 10_000,
                similarity_threshold: self.config.similarity_threshold as f64,
                min_confidence: 0.0,
                ttl_seconds: 86_400,
                max_escalate_probability: 1.0,
            };
            Some(DecisionCache::new(cache_cfg))
        } else {
            None
        };

        let schema_hash = compute_schema_hash(questions);
        let start_total = Instant::now();
        let mut last_decisions = Vec::with_capacity(total_items);

        for iter_idx in 0..self.config.iterations {
            let is_last = iter_idx + 1 == self.config.iterations;
            for item in items {
                let t0 = Instant::now();

                let (decision, was_cached, pure_inf_ms) = if let Some(cache) = &mut opt_cache {
                    let item_text = match &item.state {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    match cache.lookup(&item_text, &schema_hash) {
                        CacheLookupResult::ExactHit(cached_res) => (cached_res, true, None),
                        CacheLookupResult::SemanticHit { result, .. } => (result, true, None),
                        CacheLookupResult::Miss => {
                            let t_inf = Instant::now();
                            let res = self.classifier.classify(&item.state, questions)?;
                            let inf_ms = t_inf.elapsed().as_secs_f64() * 1000.0;
                            cache.insert(&item_text, &schema_hash, &res);
                            (res, false, Some(inf_ms))
                        }
                    }
                } else {
                    let t_inf = Instant::now();
                    let res = self.classifier.classify(&item.state, questions)?;
                    let inf_ms = t_inf.elapsed().as_secs_f64() * 1000.0;
                    (res, false, Some(inf_ms))
                };

                let elapsed_item_ms = t0.elapsed().as_secs_f64() * 1000.0;
                latencies_ms.push(elapsed_item_ms);

                if was_cached {
                    cached_latencies_ms.push(elapsed_item_ms);
                } else {
                    uncached_latencies_ms.push(pure_inf_ms.unwrap_or(elapsed_item_ms));
                }

                if is_last {
                    last_decisions.push(decision);
                }
            }
        }

        let total_measured_elapsed_secs = start_total.elapsed().as_secs_f64();
        let total_items_processed = total_items * self.config.iterations;
        let total_decisions_evaluated = total_items_processed * questions.len();

        latencies_ms.retain(|v| v.is_finite());
        latencies_ms.sort_by(|a, b| a.total_cmp(b));

        let latency_min_ms = *latencies_ms.first().unwrap_or(&0.0);
        let latency_max_ms = *latencies_ms.last().unwrap_or(&0.0);
        let latency_avg_ms = if !latencies_ms.is_empty() {
            latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64
        } else {
            0.0
        };

        let latency_p50_ms = interpolated_percentile(&latencies_ms, 0.50);
        let latency_p90_ms = interpolated_percentile(&latencies_ms, 0.90);
        let latency_p95_ms = interpolated_percentile(&latencies_ms, 0.95);
        let latency_p99_ms = interpolated_percentile(&latencies_ms, 0.99);
        let latency_std_dev_ms = compute_std_dev(&latencies_ms, latency_avg_ms);

        let throughput_items_per_sec =
            total_items_processed as f64 / total_measured_elapsed_secs.max(1e-9);
        let throughput_decisions_per_sec =
            total_decisions_evaluated as f64 / total_measured_elapsed_secs.max(1e-9);

        let cache_summary = opt_cache.as_ref().map(|c| {
            let stats = c.stats();
            let cached_avg = if !cached_latencies_ms.is_empty() {
                cached_latencies_ms.iter().sum::<f64>() / cached_latencies_ms.len() as f64
            } else {
                0.0
            };
            let uncached_avg = if !uncached_latencies_ms.is_empty() {
                uncached_latencies_ms.iter().sum::<f64>() / uncached_latencies_ms.len() as f64
            } else {
                latency_avg_ms
            };
            CacheStatsSummary {
                enabled: true,
                total_lookups: stats.total_lookups(),
                exact_hits: stats.exact_hits,
                semantic_hits: stats.semantic_hits,
                misses: stats.misses,
                hit_rate_pct: stats.hit_rate() * 100.0,
                cached_avg_latency_ms: cached_avg,
                uncached_avg_latency_ms: uncached_avg,
                speedup_ratio: if cached_avg > 1e-6 {
                    uncached_avg / cached_avg
                } else {
                    1.0
                },
            }
        });

        // 4. Accuracy, Calibration, and Per-Question Quality Evaluation (Reusing last iteration decisions)
        let mut overall_correct = 0;
        let mut overall_evaluated = 0;
        let mut overall_soft_acc_sum = 0.0;

        let mut confidences_all = Vec::new();
        let mut correctness_all = Vec::new();
        let mut probs_all: Vec<Vec<f64>> = Vec::new();
        let mut targets_all: Vec<usize> = Vec::new();

        let mut confidence_sum = 0.0;
        let mut entropy_sum = 0.0;
        let mut margin_sum = 0.0;
        let mut escalate_sum = 0;
        let mut total_question_answers = 0;

        #[derive(Default)]
        struct QuestionAccumulator {
            sample_count: usize,
            gt_count: usize,
            correct_count: usize,
            soft_acc_sum: f64,
            conf_sum: f64,
            margin_sum: f64,
            entropy_sum: f64,
            escalate_count: usize,
        }

        let mut q_accumulators: BTreeMap<String, QuestionAccumulator> = questions
            .keys()
            .map(|qid| (qid.clone(), QuestionAccumulator::default()))
            .collect();

        for (item, decision) in items.iter().zip(last_decisions.iter()) {
            let gt_map = item.ground_truth.as_ref();

            for (qid, question) in questions {
                if let Some(ans) = decision.answers.get(qid) {
                    total_question_answers += 1;
                    confidence_sum += ans.confidence;

                    let k = ans.probabilities.len().max(2);
                    let entropy: f64 = -ans
                        .probabilities
                        .values()
                        .filter(|&&p| p > 1e-9)
                        .map(|&p| p * p.ln())
                        .sum::<f64>()
                        / (k as f64).ln().max(1e-9);
                    let norm_entropy = entropy.clamp(0.0, 1.0);
                    entropy_sum += norm_entropy;

                    let mut sorted_probs: Vec<f64> = ans.probabilities.values().copied().collect();
                    sorted_probs
                        .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let p_top1 = sorted_probs.first().copied().unwrap_or(1.0);
                    let p_top2 = sorted_probs.get(1).copied().unwrap_or(0.0);
                    let top2_margin = (p_top1 - p_top2).clamp(0.0, 1.0);
                    margin_sum += top2_margin;

                    let act_prob = ans.act_probability.unwrap_or(0.05 + 0.9 * norm_entropy);
                    let is_escalated = act_prob >= self.config.escalate_threshold;
                    if is_escalated {
                        escalate_sum += 1;
                    }

                    let opt_gt =
                        resolve_ground_truth_for_question(gt_map, qid, question, questions.len());

                    if let Some(acc) = q_accumulators.get_mut(qid) {
                        acc.sample_count += 1;
                        acc.conf_sum += ans.confidence;
                        acc.margin_sum += top2_margin;
                        acc.entropy_sum += norm_entropy;
                        if is_escalated {
                            acc.escalate_count += 1;
                        }

                        if let Some(target_val) = opt_gt {
                            acc.gt_count += 1;
                            overall_evaluated += 1;
                            confidences_all.push(ans.confidence);

                            let (is_correct, soft_acc) = match question.qtype {
                                QuestionType::Choice => {
                                    let pred_str = ans.choice.as_deref().unwrap_or("");
                                    let target_str = target_val.as_str().unwrap_or("");
                                    let correct = !target_str.is_empty()
                                        && (pred_str.eq_ignore_ascii_case(target_str)
                                            || ans.selected_labels.as_ref().is_some_and(|v| {
                                                v.iter().any(|s| s.eq_ignore_ascii_case(target_str))
                                            }));
                                    let soft = ans
                                        .probabilities
                                        .iter()
                                        .find(|(k, _)| k.eq_ignore_ascii_case(target_str))
                                        .map(|(_, &v)| v)
                                        .unwrap_or(0.0);

                                    let keys: Vec<&String> = ans.probabilities.keys().collect();
                                    if let Some(t_idx) =
                                        keys.iter().position(|k| k.eq_ignore_ascii_case(target_str))
                                    {
                                        probs_all.push(
                                            keys.iter()
                                                .map(|k| {
                                                    ans.probabilities
                                                        .get(*k)
                                                        .copied()
                                                        .unwrap_or(0.0)
                                                })
                                                .collect(),
                                        );
                                        targets_all.push(t_idx);
                                    }

                                    (correct, soft)
                                }
                                QuestionType::Score => {
                                    let pred_num = ans.score.unwrap_or(0.0);
                                    let target_num = target_val.as_f64().unwrap_or_else(|| {
                                        target_val
                                            .as_str()
                                            .and_then(|s| s.parse::<f64>().ok())
                                            .unwrap_or(0.0)
                                    });
                                    let err = (pred_num - target_num).abs();
                                    (err < 0.5, 1.0 - (err / 5.0).min(1.0))
                                }
                                QuestionType::Noul => {
                                    let target_bool = target_val
                                        .as_bool()
                                        .or_else(|| {
                                            target_val
                                                .as_str()
                                                .map(|s| s.eq_ignore_ascii_case("true") || s == "1")
                                        })
                                        .or_else(|| target_val.as_u64().map(|n| n != 0))
                                        .unwrap_or(false);
                                    let pred_prob = ans.noul.unwrap_or(0.0);
                                    let pred_bool = pred_prob >= 0.5;
                                    let correct = pred_bool == target_bool;
                                    let soft = if target_bool {
                                        pred_prob
                                    } else {
                                        1.0 - pred_prob
                                    };

                                    probs_all.push(vec![1.0 - pred_prob, pred_prob]);
                                    targets_all.push(usize::from(target_bool));

                                    (correct, soft)
                                }
                            };

                            if is_correct {
                                overall_correct += 1;
                                acc.correct_count += 1;
                            }
                            correctness_all.push(is_correct);
                            overall_soft_acc_sum += soft_acc;
                            acc.soft_acc_sum += soft_acc;
                        }
                    }
                }
            }
        }

        let mut questions_stats = BTreeMap::new();
        for (qid, q) in questions {
            let (options_count, est_head_tokens, head_util_pct) = question_token_utils
                .get(qid)
                .cloned()
                .unwrap_or((2, 10, 5.0));

            let acc = q_accumulators.get(qid).unwrap();
            let count = acc.sample_count.max(1);
            let gt_count = acc.gt_count;

            questions_stats.insert(
                qid.clone(),
                QuestionStats {
                    question_id: qid.clone(),
                    question_type: match q.qtype {
                        QuestionType::Choice => "choice",
                        QuestionType::Score => "score",
                        QuestionType::Noul => "noul",
                    }
                    .to_string(),
                    options_count,
                    est_head_tokens,
                    head_budget,
                    head_utilization_pct: head_util_pct,
                    evaluated_samples: acc.sample_count,
                    accuracy: (gt_count > 0).then(|| acc.correct_count as f64 / gt_count as f64),
                    soft_accuracy: (gt_count > 0).then(|| acc.soft_acc_sum / gt_count as f64),
                    mean_confidence: acc.conf_sum / count as f64,
                    mean_top2_margin: acc.margin_sum / count as f64,
                    mean_entropy: acc.entropy_sum / count as f64,
                    act_escalate_rate: acc.escalate_count as f64 / count as f64,
                },
            );
        }

        let accuracy_stats = (overall_evaluated > 0).then(|| AccuracyStats {
            labeled_sample_count: overall_evaluated,
            overall_accuracy: overall_correct as f64 / overall_evaluated as f64,
            soft_accuracy: overall_soft_acc_sum / overall_evaluated as f64,
            expected_calibration_error: calculate_ece(
                &confidences_all,
                &correctness_all,
                self.config.ece_bins,
            ),
            brier_score: calculate_brier_score(&probs_all, &targets_all),
        });

        let denom = total_question_answers.max(1) as f64;
        let report = SystemStatsReport {
            checkpoint: checkpoint.repo_id().to_string(),
            head_budget,
            dataset_items: total_items,
            benchmark_iterations: self.config.iterations,
            warmup_runs: self.config.warmup_runs,
            speed: SpeedStats {
                total_elapsed_secs: total_measured_elapsed_secs,
                latency_min_ms,
                latency_p50_ms,
                latency_p90_ms,
                latency_p95_ms,
                latency_p99_ms,
                latency_max_ms,
                latency_avg_ms,
                latency_std_dev_ms,
            },
            performance: PerformanceStats {
                throughput_items_per_sec,
                throughput_decisions_per_sec,
                total_items_processed,
                total_decisions_evaluated,
                avg_head_token_utilization_pct,
                cache_summary,
            },
            accuracy: accuracy_stats,
            uncertainty: UncertaintyStats {
                mean_confidence: confidence_sum / denom,
                mean_top2_margin: margin_sum / denom,
                mean_entropy: entropy_sum / denom,
                act_escalate_rate: escalate_sum as f64 / denom,
            },
            questions: questions_stats,
        };

        Ok(report)
    }

    /// Renders the statistics report as a formatted ANSI table.
    pub fn format_table(report: &SystemStatsReport) -> String {
        let mut out = format!(
            "\n📊 Lemmata System 1 Diagnostics & Performance Report\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nModel Checkpoint : {} (Head Budget: {} tokens)\nDataset Workload : {} items, {} questions (Iterations: {}, Warmup: {})\n\n⚡ Speed & Latency Distribution\n",
            report.checkpoint,
            report.head_budget,
            report.dataset_items,
            report.questions.len(),
            report.benchmark_iterations,
            report.warmup_runs
        );

        let new_utf8_table = |headers: &[(&str, Color)]| {
            let mut t = Table::new();
            t.load_style(comfy_table::presets::UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic);
            t.set_header(
                headers
                    .iter()
                    .map(|(h, c)| Cell::new(*h).fg(*c))
                    .collect::<Vec<_>>(),
            );
            t
        };

        let mut speed_table = new_utf8_table(&[
            ("Latency Metric", Color::Cyan),
            ("Duration (ms)", Color::Green),
            ("Operational Reference / SLA", Color::DarkGrey),
        ]);
        for (label, val, note) in [
            (
                "P50 (Median Response)",
                format!("{:.3} ms", report.speed.latency_p50_ms),
                "Sub-35ms non-autoregressive target".to_string(),
            ),
            (
                "P90 Latency",
                format!("{:.3} ms", report.speed.latency_p90_ms),
                "High-load tail latency boundary".to_string(),
            ),
            (
                "P95 Latency",
                format!("{:.3} ms", report.speed.latency_p95_ms),
                "95th percentile SLA threshold".to_string(),
            ),
            (
                "P99 Latency",
                format!("{:.3} ms", report.speed.latency_p99_ms),
                "Extreme tail latency".to_string(),
            ),
            (
                "Average (Mean)",
                format!("{:.3} ms", report.speed.latency_avg_ms),
                format!("Std Dev (σ): ±{:.3} ms", report.speed.latency_std_dev_ms),
            ),
            (
                "Min / Max Bounds",
                format!(
                    "{:.3} / {:.3} ms",
                    report.speed.latency_min_ms, report.speed.latency_max_ms
                ),
                "Observed execution envelope".to_string(),
            ),
        ] {
            speed_table.add_row(vec![Cell::new(label), Cell::new(val), Cell::new(note)]);
        }
        out.push_str(&speed_table.to_string());
        out.push_str("\n\n🚀 System Throughput & Efficiency\n");

        let mut perf_table = new_utf8_table(&[
            ("Performance Metric", Color::Cyan),
            ("Observed Value", Color::Green),
            ("Efficiency Assessment", Color::DarkGrey),
        ]);
        perf_table.add_row(vec![
            Cell::new("Ingestion Throughput"),
            Cell::new(format!(
                "{:.1} items/sec",
                report.performance.throughput_items_per_sec
            ))
            .fg(Color::Green),
            Cell::new(format!(
                "{} items processed in {:.3}s",
                report.performance.total_items_processed, report.speed.total_elapsed_secs
            )),
        ]);
        perf_table.add_row(vec![
            Cell::new("Decision Throughput"),
            Cell::new(format!(
                "{:.1} decisions/sec",
                report.performance.throughput_decisions_per_sec
            ))
            .fg(Color::Green),
            Cell::new(format!(
                "{} total question decisions evaluated",
                report.performance.total_decisions_evaluated
            )),
        ]);
        perf_table.add_row(vec![
            Cell::new("Head Token Utilization"),
            Cell::new(format!(
                "{:.1}%",
                report.performance.avg_head_token_utilization_pct
            )),
            Cell::new(format!(
                "Average across questions against {} token budget",
                report.head_budget
            )),
        ]);
        if let Some(c) = &report.performance.cache_summary {
            perf_table.add_row(vec![
                Cell::new("Semantic Cache Hit Rate"),
                Cell::new(format!("{:.1}%", c.hit_rate_pct)),
                Cell::new(format!(
                    "Exact: {}, Semantic: {}, Misses: {}",
                    c.exact_hits, c.semantic_hits, c.misses
                )),
            ]);
            perf_table.add_row(vec![
                Cell::new("Cache Latency Speedup"),
                Cell::new(format!(
                    "{:.2}x ({:.3} ms vs {:.3} ms)",
                    c.speedup_ratio, c.cached_avg_latency_ms, c.uncached_avg_latency_ms
                )),
                Cell::new("Cached sub-millisecond retrieval boost"),
            ]);
        }
        out.push_str(&perf_table.to_string());
        out.push_str("\n\n🎯 Decision Accuracy, Calibration & Safety\n");

        let mut acc_table = new_utf8_table(&[
            ("Quality & Safety Metric", Color::Cyan),
            ("Value", Color::Green),
            ("Target Standard / Interpretation", Color::DarkGrey),
        ]);
        if let Some(acc) = &report.accuracy {
            for (m, v, note) in [
                (
                    "Top-1 Exact Accuracy",
                    format!("{:.2}%", acc.overall_accuracy * 100.0),
                    format!(
                        "Evaluated on {} ground-truth decisions",
                        acc.labeled_sample_count
                    ),
                ),
                (
                    "Soft Accuracy (Target Prob)",
                    format!("{:.2}%", acc.soft_accuracy * 100.0),
                    "Average probability mass assigned to true target".to_string(),
                ),
                (
                    "Expected Calibration Error (ECE)",
                    format!("{:.4}", acc.expected_calibration_error),
                    "Lower is better (ECE < 0.10 indicates strong calibration)".to_string(),
                ),
                (
                    "Brier Score",
                    format!("{:.4}", acc.brier_score),
                    "Strictly proper mean squared error (lower is better)".to_string(),
                ),
            ] {
                acc_table.add_row(vec![Cell::new(m), Cell::new(v), Cell::new(note)]);
            }
        } else {
            acc_table.add_row(vec![
                Cell::new("Classification Accuracy"),
                Cell::new("Ground Truth Not Provided").fg(Color::Yellow),
                Cell::new("Provide dataset with ground_truth to evaluate accuracy & ECE"),
            ]);
        }
        for (m, v, note) in [
            (
                "Mean Predicted Confidence",
                format!("{:.2}%", report.uncertainty.mean_confidence * 100.0),
                "Average winning option probability",
            ),
            (
                "Mean Top-2 Margin Gap (Δ)",
                format!("{:.3}", report.uncertainty.mean_top2_margin),
                "Separation between top-1 and top-2 candidate choices",
            ),
            (
                "Mean Normalized Entropy (H̃)",
                format!("{:.3}", report.uncertainty.mean_entropy),
                "Option distribution spread (0 = decisive, 1 = uniform)",
            ),
            (
                "Supervisor Escalation Rate",
                format!("{:.1}%", report.uncertainty.act_escalate_rate * 100.0),
                "Percentage routed to human/System 2 by Act head",
            ),
        ] {
            acc_table.add_row(vec![Cell::new(m), Cell::new(v), Cell::new(note)]);
        }
        out.push_str(&acc_table.to_string());
        out.push_str("\n\n📋 Per-Question Schema Breakdown\n");

        let mut q_table = Table::new();
        q_table
            .load_style(comfy_table::presets::UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                "Question ID",
                "Type",
                "Options",
                "Head Util",
                "Accuracy",
                "Confidence",
                "Margin (Δ)",
                "Entropy (H̃)",
                "Escalate %",
            ]);
        for (qid, q_stat) in &report.questions {
            q_table.add_row(vec![
                qid.clone(),
                q_stat.question_type.clone(),
                q_stat.options_count.to_string(),
                format!("{:.1}%", q_stat.head_utilization_pct),
                q_stat
                    .accuracy
                    .map_or_else(|| "N/A".into(), |a| format!("{:.1}%", a * 100.0)),
                format!("{:.1}%", q_stat.mean_confidence * 100.0),
                format!("{:.3}", q_stat.mean_top2_margin),
                format!("{:.3}", q_stat.mean_entropy),
                format!("{:.1}%", q_stat.act_escalate_rate * 100.0),
            ]);
        }
        out.push_str(&q_table.to_string());
        out.push('\n');
        out
    }

    /// Renders the statistics report as formatted JSON string.
    pub fn format_json(report: &SystemStatsReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    /// Renders a single-line or compact summary string suitable for logs or CI/CD pipelines.
    pub fn format_compact(report: &SystemStatsReport) -> String {
        let acc_str = if let Some(acc) = &report.accuracy {
            format!(
                "ACC: {:.1}% (Soft: {:.1}%, ECE: {:.3}, Brier: {:.3})",
                acc.overall_accuracy * 100.0,
                acc.soft_accuracy * 100.0,
                acc.expected_calibration_error,
                acc.brier_score
            )
        } else {
            "ACC: N/A (unlabeled)".into()
        };

        format!(
            "⚡ SPEED: {:.2}ms p50 (p95: {:.2}ms, avg: {:.2}ms) | 🚀 PERF: {:.1} items/s ({:.1} dec/s, {:.1}% head) | 🎯 {} | 🛡️ RISK: {:.1}% escalate (conf: {:.1}%)",
            report.speed.latency_p50_ms,
            report.speed.latency_p95_ms,
            report.speed.latency_avg_ms,
            report.performance.throughput_items_per_sec,
            report.performance.throughput_decisions_per_sec,
            report.performance.avg_head_token_utilization_pct,
            acc_str,
            report.uncertainty.act_escalate_rate * 100.0,
            report.uncertainty.mean_confidence * 100.0
        )
    }

    /// Renders key metrics as CSV rows.
    pub fn format_csv(report: &SystemStatsReport) -> String {
        let mut wtr = csv::Writer::from_writer(vec![]);
        let _ = wtr.write_record(["category", "metric", "value", "unit"]);

        let mut rows: Vec<(&str, &str, String, &str)> = vec![
            (
                "speed",
                "latency_p50",
                format!("{:.4}", report.speed.latency_p50_ms),
                "ms",
            ),
            (
                "speed",
                "latency_p90",
                format!("{:.4}", report.speed.latency_p90_ms),
                "ms",
            ),
            (
                "speed",
                "latency_p95",
                format!("{:.4}", report.speed.latency_p95_ms),
                "ms",
            ),
            (
                "speed",
                "latency_p99",
                format!("{:.4}", report.speed.latency_p99_ms),
                "ms",
            ),
            (
                "speed",
                "latency_avg",
                format!("{:.4}", report.speed.latency_avg_ms),
                "ms",
            ),
            (
                "speed",
                "latency_min",
                format!("{:.4}", report.speed.latency_min_ms),
                "ms",
            ),
            (
                "speed",
                "latency_max",
                format!("{:.4}", report.speed.latency_max_ms),
                "ms",
            ),
            (
                "speed",
                "latency_std_dev",
                format!("{:.4}", report.speed.latency_std_dev_ms),
                "ms",
            ),
            (
                "performance",
                "throughput_items",
                format!("{:.2}", report.performance.throughput_items_per_sec),
                "items/sec",
            ),
            (
                "performance",
                "throughput_decisions",
                format!("{:.2}", report.performance.throughput_decisions_per_sec),
                "decisions/sec",
            ),
            (
                "performance",
                "head_token_utilization",
                format!("{:.2}", report.performance.avg_head_token_utilization_pct),
                "%",
            ),
        ];

        if let Some(c) = &report.performance.cache_summary {
            rows.push((
                "performance",
                "cache_hit_rate",
                format!("{:.2}", c.hit_rate_pct),
                "%",
            ));
            rows.push((
                "performance",
                "cache_speedup",
                format!("{:.2}", c.speedup_ratio),
                "x",
            ));
        }
        if let Some(acc) = &report.accuracy {
            rows.extend([
                (
                    "accuracy",
                    "overall_accuracy",
                    format!("{:.4}", acc.overall_accuracy),
                    "ratio",
                ),
                (
                    "accuracy",
                    "soft_accuracy",
                    format!("{:.4}", acc.soft_accuracy),
                    "ratio",
                ),
                (
                    "accuracy",
                    "ece",
                    format!("{:.4}", acc.expected_calibration_error),
                    "error",
                ),
                (
                    "accuracy",
                    "brier_score",
                    format!("{:.4}", acc.brier_score),
                    "score",
                ),
            ]);
        }
        rows.extend([
            (
                "uncertainty",
                "mean_confidence",
                format!("{:.4}", report.uncertainty.mean_confidence),
                "ratio",
            ),
            (
                "uncertainty",
                "mean_top2_margin",
                format!("{:.4}", report.uncertainty.mean_top2_margin),
                "margin",
            ),
            (
                "uncertainty",
                "mean_entropy",
                format!("{:.4}", report.uncertainty.mean_entropy),
                "entropy",
            ),
            (
                "uncertainty",
                "act_escalate_rate",
                format!("{:.4}", report.uncertainty.act_escalate_rate),
                "ratio",
            ),
        ]);

        for (cat, met, val, unit) in rows {
            let _ = wtr.write_record([cat, met, &val, unit]);
        }
        String::from_utf8(wtr.into_inner().unwrap_or_default()).unwrap_or_default()
    }

    /// Exports the stats report to a destination file based on extension.
    pub fn export_report(report: &SystemStatsReport, destination: &Path) -> Result<()> {
        if let Some(parent) = destination.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let is_csv = destination
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.eq_ignore_ascii_case("csv"))
            .unwrap_or(false);

        let content = if is_csv {
            Self::format_csv(report)
        } else {
            Self::format_json(report)?
        };

        std::fs::write(destination, content).with_context(|| {
            format!("Failed to export stats report to {}", destination.display())
        })?;

        Ok(())
    }
}

/// Compute interpolated percentile between ranks.
fn interpolated_percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = pct.clamp(0.0, 1.0);
    let rank = p * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let weight = rank - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Compute sample standard deviation.
fn compute_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let sum_sq_diff: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();
    (sum_sq_diff / (values.len() - 1) as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Checkpoint;

    #[test]
    fn test_interpolated_percentile() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = interpolated_percentile(&vals, 0.50);
        assert!((p50 - 3.0).abs() < 1e-6);

        let p0 = interpolated_percentile(&vals, 0.0);
        assert!((p0 - 1.0).abs() < 1e-6);

        let p100 = interpolated_percentile(&vals, 1.0);
        assert!((p100 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_std_dev() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let std_dev = compute_std_dev(&vals, mean);
        assert!((std_dev - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_stats_engine_with_mock_data() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice(
                "Pick department",
                vec!["billing".into(), "technical".into()],
            ),
        );
        questions.insert("urgent".into(), Question::new_noul("Is urgent?"));

        let mut gt = BTreeMap::new();
        gt.insert("dept".into(), serde_json::json!("billing"));
        gt.insert("urgent".into(), serde_json::json!(true));

        let items = vec![
            DataItem::new(
                "item_1",
                serde_json::json!({ "text": "Invoice payment billing issue refund" }),
                "mock",
            )
            .with_ground_truth(gt.clone()),
            DataItem::new(
                "item_2",
                serde_json::json!({ "text": "Server connection failed technical crash" }),
                "mock",
            )
            .with_ground_truth(gt),
        ];

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let engine = StatsEngine::new(classifier)
            .with_iterations(2)
            .with_warmup(1)
            .with_cache(true, 0.92);

        let report = engine.evaluate(&items, &questions).unwrap();

        assert_eq!(report.dataset_items, 2);
        assert_eq!(report.benchmark_iterations, 2);
        assert_eq!(report.performance.total_items_processed, 4);
        assert!(report.speed.latency_avg_ms > 0.0);
        assert!(report.speed.latency_p50_ms > 0.0);
        assert!(report.performance.throughput_items_per_sec > 0.0);
        assert!(report.performance.throughput_decisions_per_sec > 0.0);

        assert!(report.accuracy.is_some());
        let acc = report.accuracy.as_ref().unwrap();
        assert!(acc.overall_accuracy >= 0.0 && acc.overall_accuracy <= 1.0);
        assert!(acc.soft_accuracy >= 0.0 && acc.soft_accuracy <= 1.0);

        // Check formatters
        let table = StatsEngine::format_table(&report);
        assert!(table.contains("Speed & Latency Distribution"));
        assert!(table.contains("System Throughput & Efficiency"));
        assert!(table.contains("Decision Accuracy, Calibration & Safety"));

        let compact = StatsEngine::format_compact(&report);
        assert!(compact.contains("SPEED:"));
        assert!(compact.contains("PERF:"));
        assert!(compact.contains("ACC:"));

        let json = StatsEngine::format_json(&report).unwrap();
        assert!(json.contains("throughput_items_per_sec"));

        let csv = StatsEngine::format_csv(&report);
        assert!(csv.contains("latency_p50"));
    }
}
