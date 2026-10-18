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
}

/// Accuracy, calibration quality, and strictly proper scoring rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyStats {
    pub labeled_sample_count: usize,
    pub overall_accuracy: f64,
    pub soft_accuracy: f64,
    pub expected_calibration_error: f64,
    pub brier_score: f64,
    /// Area under the risk-coverage curve over all labelled decisions (lower is better).
    pub aurc: f64,
}

/// Precision, recall, and F1 for one label of a choice question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelStats {
    pub label: String,
    /// Items whose true label is this label.
    pub support: usize,
    /// Items that the model gave this label.
    pub predicted: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// A true label that the model gave another label, and how often.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusedPair {
    pub truth: String,
    pub predicted: String,
    pub count: usize,
}

/// Accuracy over the most confident `coverage` share of the labelled items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveragePoint {
    pub coverage: f64,
    pub accuracy: f64,
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
    /// Per-label precision, recall, and F1 (choice questions with labels only).
    #[serde(default)]
    pub labels: Vec<LabelStats>,
    /// F1 averaged over the labels that occur in the data.
    pub macro_f1: Option<f64>,
    /// F1 averaged over the labels, weighted by support.
    pub weighted_f1: Option<f64>,
    /// The most frequent mistakes, most frequent first.
    #[serde(default)]
    pub confused_pairs: Vec<ConfusedPair>,
    /// Area under the risk-coverage curve (lower is better).
    pub aurc: Option<f64>,
    /// Accuracy at 10%, 20%, ..., 100% coverage, most confident items first.
    #[serde(default)]
    pub risk_coverage: Vec<CoveragePoint>,
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
    pub escalate_threshold: f64,
    pub ece_bins: usize,
}

impl Default for StatsEngineConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            warmup_runs: 2,
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
        let start_total = Instant::now();
        let mut last_decisions = Vec::with_capacity(total_items);

        for iter_idx in 0..self.config.iterations {
            let is_last = iter_idx + 1 == self.config.iterations;
            for item in items {
                let t0 = Instant::now();
                let decision = self.classifier.classify(&item.state, questions)?;
                latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
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
            confidence_correct: Vec<(f64, bool)>,
            truth_predicted: Vec<(String, String)>,
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
                                    if !target_str.is_empty() {
                                        acc.truth_predicted.push((
                                            target_str.to_lowercase(),
                                            pred_str.to_lowercase(),
                                        ));
                                    }
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
                            acc.confidence_correct.push((ans.confidence, is_correct));
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
            let labels = label_stats(&acc.truth_predicted);
            let (macro_f1, weighted_f1) = f1_averages(&labels);
            let (aurc, risk_coverage) = risk_coverage(&acc.confidence_correct);

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
                    labels,
                    macro_f1,
                    weighted_f1,
                    confused_pairs: confused_pairs(&acc.truth_predicted, 5),
                    aurc,
                    risk_coverage,
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
            aurc: risk_coverage(
                &confidences_all
                    .iter()
                    .copied()
                    .zip(correctness_all.iter().copied())
                    .collect::<Vec<_>>(),
            )
            .0
            .unwrap_or(0.0),
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
            "\n📊 lemmata Evaluation Report\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nModel Checkpoint : {} (Head Budget: {} tokens)\nDataset Workload : {} items, {} questions (Iterations: {}, Warmup: {})\n\n⚡ Speed & Latency Distribution\n",
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
                "Median latency per item".to_string(),
            ),
            (
                "P90 Latency",
                format!("{:.3} ms", report.speed.latency_p90_ms),
                "90th percentile".to_string(),
            ),
            (
                "P95 Latency",
                format!("{:.3} ms", report.speed.latency_p95_ms),
                "95th percentile".to_string(),
            ),
            (
                "P99 Latency",
                format!("{:.3} ms", report.speed.latency_p99_ms),
                "99th percentile".to_string(),
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
                "Fastest and slowest item".to_string(),
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
                    "Lower is better. 0 means confidence matches accuracy".to_string(),
                ),
                (
                    "Brier Score",
                    format!("{:.4}", acc.brier_score),
                    "Strictly proper mean squared error (lower is better)".to_string(),
                ),
                (
                    "AURC (Risk-Coverage Area)",
                    format!("{:.4}", acc.aurc),
                    "Mean error rate when the most confident items go first (lower is better)"
                        .to_string(),
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

        for (qid, q) in &report.questions {
            if q.labels.is_empty() {
                continue;
            }
            out.push_str(&format!("\n🏷️  Per-Label Metrics: {qid}\n"));
            let mut label_table = new_utf8_table(&[
                ("Label", Color::Cyan),
                ("Support", Color::White),
                ("Predicted", Color::White),
                ("Precision", Color::Green),
                ("Recall", Color::Green),
                ("F1", Color::Green),
            ]);
            for l in &q.labels {
                label_table.add_row(vec![
                    Cell::new(&l.label),
                    Cell::new(l.support),
                    Cell::new(l.predicted),
                    Cell::new(format!("{:.3}", l.precision)),
                    Cell::new(format!("{:.3}", l.recall)),
                    Cell::new(format!("{:.3}", l.f1)),
                ]);
            }
            out.push_str(&label_table.to_string());
            out.push('\n');
            if let (Some(m), Some(w)) = (q.macro_f1, q.weighted_f1) {
                out.push_str(&format!("Macro-F1: {m:.3} | Weighted F1: {w:.3}\n"));
            }
            if !q.confused_pairs.is_empty() {
                let pairs: Vec<String> = q
                    .confused_pairs
                    .iter()
                    .map(|c| format!("{} -> {} ({})", c.truth, c.predicted, c.count))
                    .collect();
                out.push_str(&format!("Most confused: {}\n", pairs.join(", ")));
            }
            if let Some(aurc) = q.aurc {
                let at = |c: f64| {
                    q.risk_coverage
                        .iter()
                        .find(|p| (p.coverage - c).abs() < 1e-9)
                        .map_or(0.0, |p| p.accuracy * 100.0)
                };
                out.push_str(&format!(
                    "Accuracy at 50% / 80% / 100% coverage: {:.1}% / {:.1}% / {:.1}% | AURC: {aurc:.4}\n",
                    at(0.5),
                    at(0.8),
                    at(1.0)
                ));
            }
        }
        out
    }

    /// Renders the report as Markdown, for a PR description or a model card.
    pub fn format_markdown(report: &SystemStatsReport) -> String {
        let cell = |s: &str| s.replace('|', "\\|");
        let mut md = format!(
            "# lemmata report: {}\n\n{} items and {} questions, timed over {} passes after {} warmup passes.\n\n## Summary\n\n| Metric | Value |\n|---|---|\n",
            report.checkpoint,
            report.dataset_items,
            report.questions.len(),
            report.benchmark_iterations,
            report.warmup_runs
        );
        if let Some(acc) = &report.accuracy {
            md.push_str(&format!(
                "| Accuracy | {:.2}% ({} labelled decisions) |\n| Soft accuracy | {:.2}% |\n| ECE | {:.4} |\n| Brier score | {:.4} |\n| AURC | {:.4} |\n",
                acc.overall_accuracy * 100.0,
                acc.labeled_sample_count,
                acc.soft_accuracy * 100.0,
                acc.expected_calibration_error,
                acc.brier_score,
                acc.aurc
            ));
        }
        md.push_str(&format!(
            "| Latency p50 / p95 / p99 | {:.2} / {:.2} / {:.2} ms |\n| Throughput | {:.1} items/s |\n| Escalation rate | {:.1}% |\n\n## Questions\n\n| Question | Type | Accuracy | Macro-F1 | AURC | Escalate |\n|---|---|---|---|---|---|\n",
            report.speed.latency_p50_ms,
            report.speed.latency_p95_ms,
            report.speed.latency_p99_ms,
            report.performance.throughput_items_per_sec,
            report.uncertainty.act_escalate_rate * 100.0
        ));
        let opt = |v: Option<f64>, pct: bool| match v {
            Some(x) if pct => format!("{:.1}%", x * 100.0),
            Some(x) => format!("{x:.3}"),
            None => "n/a".to_string(),
        };
        for (qid, q) in &report.questions {
            md.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {:.1}% |\n",
                cell(qid),
                q.question_type,
                opt(q.accuracy, true),
                opt(q.macro_f1, false),
                opt(q.aurc, false),
                q.act_escalate_rate * 100.0
            ));
        }
        for (qid, q) in &report.questions {
            if q.labels.is_empty() {
                continue;
            }
            md.push_str(&format!(
                "\n### `{}`\n\n| Label | Support | Precision | Recall | F1 |\n|---|---|---|---|---|\n",
                cell(qid)
            ));
            for l in &q.labels {
                md.push_str(&format!(
                    "| {} | {} | {:.3} | {:.3} | {:.3} |\n",
                    cell(&l.label),
                    l.support,
                    l.precision,
                    l.recall,
                    l.f1
                ));
            }
            if !q.confused_pairs.is_empty() {
                md.push_str("\nMost confused (true label -> predicted):\n\n");
                for c in &q.confused_pairs {
                    md.push_str(&format!(
                        "- {} -> {}: {}\n",
                        cell(&c.truth),
                        cell(&c.predicted),
                        c.count
                    ));
                }
            }
            if !q.risk_coverage.is_empty() {
                md.push_str("\n| Coverage | Accuracy |\n|---|---|\n");
                for p in &q.risk_coverage {
                    md.push_str(&format!(
                        "| {:.0}% | {:.1}% |\n",
                        p.coverage * 100.0,
                        p.accuracy * 100.0
                    ));
                }
            }
        }
        md
    }

    /// Renders the statistics report as formatted JSON string.
    pub fn format_json(report: &SystemStatsReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    /// Renders a single-line or compact summary string suitable for logs or CI/CD pipelines.
    pub fn format_compact(report: &SystemStatsReport) -> String {
        let acc_str = if let Some(acc) = &report.accuracy {
            format!(
                "ACC: {:.1}% (Soft: {:.1}%, ECE: {:.3}, Brier: {:.3}, AURC: {:.3})",
                acc.overall_accuracy * 100.0,
                acc.soft_accuracy * 100.0,
                acc.expected_calibration_error,
                acc.brier_score,
                acc.aurc
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
                ("accuracy", "aurc", format!("{:.4}", acc.aurc), "area"),
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
        for (qid, q) in &report.questions {
            let category = format!("question:{qid}");
            for (metric, value) in [
                ("macro_f1", q.macro_f1),
                ("weighted_f1", q.weighted_f1),
                ("aurc", q.aurc),
            ] {
                if let Some(v) = value {
                    let _ =
                        wtr.write_record([category.as_str(), metric, &format!("{v:.4}"), "ratio"]);
                }
            }
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

        let extension = destination
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase);
        let content = match extension.as_deref() {
            Some("csv") => Self::format_csv(report),
            Some("md") => Self::format_markdown(report),
            _ => Self::format_json(report)?,
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

/// Precision, recall, and F1 for every label that occurs as a true or a predicted label.
fn label_stats(truth_predicted: &[(String, String)]) -> Vec<LabelStats> {
    // (support, predicted, true positives) per label.
    let mut counts: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (truth, predicted) in truth_predicted {
        counts.entry(truth).or_default().0 += 1;
        counts.entry(predicted).or_default().1 += 1;
        if truth == predicted {
            counts.entry(truth).or_default().2 += 1;
        }
    }
    let ratio = |n: usize, d: usize| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    counts
        .into_iter()
        .map(|(label, (support, predicted, tp))| {
            let precision = ratio(tp, predicted);
            let recall = ratio(tp, support);
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            LabelStats {
                label: label.to_string(),
                support,
                predicted,
                precision,
                recall,
                f1,
            }
        })
        .collect()
}

/// Macro F1 over the labels that occur as true labels, and F1 weighted by their support.
fn f1_averages(labels: &[LabelStats]) -> (Option<f64>, Option<f64>) {
    let present: Vec<&LabelStats> = labels.iter().filter(|l| l.support > 0).collect();
    if present.is_empty() {
        return (None, None);
    }
    let total: usize = present.iter().map(|l| l.support).sum();
    let macro_f1 = present.iter().map(|l| l.f1).sum::<f64>() / present.len() as f64;
    let weighted = present.iter().map(|l| l.f1 * l.support as f64).sum::<f64>() / total as f64;
    (Some(macro_f1), Some(weighted))
}

/// The `k` most frequent mistakes, as (true label, predicted label) counts.
fn confused_pairs(truth_predicted: &[(String, String)], k: usize) -> Vec<ConfusedPair> {
    let mut counts: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for (truth, predicted) in truth_predicted {
        if truth != predicted {
            *counts.entry((truth, predicted)).or_default() += 1;
        }
    }
    let mut pairs: Vec<ConfusedPair> = counts
        .into_iter()
        .map(|((truth, predicted), count)| ConfusedPair {
            truth: truth.to_string(),
            predicted: predicted.to_string(),
            count,
        })
        .collect();
    pairs.sort_by_key(|p| std::cmp::Reverse(p.count));
    pairs.truncate(k);
    pairs
}

/// Rank items by confidence, most confident first. Returns the area under the risk-coverage
/// curve (the mean error rate over every coverage level from 1 item to all items, so 0 is
/// perfect) and the accuracy at 10%, 20%, ..., 100% coverage.
fn risk_coverage(confidence_correct: &[(f64, bool)]) -> (Option<f64>, Vec<CoveragePoint>) {
    if confidence_correct.is_empty() {
        return (None, Vec::new());
    }
    let mut ranked = confidence_correct.to_vec();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let n = ranked.len();
    let mut correct = 0;
    let mut risk_sum = 0.0;
    let mut accuracy_at = Vec::with_capacity(n);
    for (k, &(_, is_correct)) in ranked.iter().enumerate() {
        correct += usize::from(is_correct);
        let accuracy = correct as f64 / (k + 1) as f64;
        risk_sum += 1.0 - accuracy;
        accuracy_at.push(accuracy);
    }
    let points = (1..=10)
        .map(|tenth| {
            let coverage = tenth as f64 / 10.0;
            let k = ((coverage * n as f64).ceil() as usize).clamp(1, n);
            CoveragePoint {
                coverage,
                accuracy: accuracy_at[k - 1],
            }
        })
        .collect();
    (Some(risk_sum / n as f64), points)
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
            .with_warmup(1);

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

        let md = StatsEngine::format_markdown(&report);
        assert!(md.contains("## Summary"));
        assert!(md.contains("| Label | Support | Precision | Recall | F1 |"));
        assert!(report.questions["dept"].macro_f1.is_some());
        assert!(report.questions["dept"].aurc.is_some());
    }

    #[test]
    fn per_label_metrics_match_hand_counts() {
        let pairs: Vec<(String, String)> = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]
            .iter()
            .map(|(t, p)| (t.to_string(), p.to_string()))
            .collect();
        let labels = label_stats(&pairs);
        // a: support 2, predicted 1, 1 correct. b: support 2, predicted 3, 2 correct.
        assert_eq!((labels[0].support, labels[0].predicted), (2, 1));
        assert!((labels[0].f1 - 2.0 / 3.0).abs() < 1e-9);
        assert!((labels[1].f1 - 0.8).abs() < 1e-9);
        let (macro_f1, weighted) = f1_averages(&labels);
        assert!((macro_f1.unwrap() - (2.0 / 3.0 + 0.8) / 2.0).abs() < 1e-9);
        assert!((weighted.unwrap() - macro_f1.unwrap()).abs() < 1e-9);
        let confused = confused_pairs(&pairs, 5);
        assert_eq!(confused.len(), 1);
        assert_eq!(
            (confused[0].truth.as_str(), confused[0].predicted.as_str()),
            ("a", "b")
        );
    }

    #[test]
    fn risk_coverage_ranks_by_confidence() {
        // Ranked: correct, wrong, correct, correct. Accuracy at k = 1, 1/2, 2/3, 3/4.
        let (aurc, points) = risk_coverage(&[(0.7, true), (0.9, true), (0.6, true), (0.8, false)]);
        let expected = (0.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0;
        assert!((aurc.unwrap() - expected).abs() < 1e-9);
        assert_eq!(points.len(), 10);
        assert!(
            (points[0].accuracy - 1.0).abs() < 1e-9,
            "10% coverage is the top item"
        );
        assert!(
            (points[4].accuracy - 0.5).abs() < 1e-9,
            "50% coverage is the top 2"
        );
        assert!((points[9].accuracy - 0.75).abs() < 1e-9);
        assert_eq!(risk_coverage(&[]).0, None);
    }
}
