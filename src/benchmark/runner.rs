use anyhow::{Result, bail};
use comfy_table::{Cell, Color, Table};
use std::collections::BTreeMap;
use std::time::Instant;

use crate::datasource::DataItem;
use crate::engine::LayaClassifier;
use crate::model::Question;

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_items: usize,
    pub total_questions: usize,
    pub elapsed_secs: f64,
    pub throughput_items_per_sec: f64,
    pub throughput_questions_per_sec: f64,
    pub latency_min_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p90_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub latency_max_ms: f64,
    pub latency_avg_ms: f64,
}

pub struct BenchmarkRunner {
    classifier: LayaClassifier,
    warmup_runs: usize,
    silent: bool,
}

impl BenchmarkRunner {
    pub fn new(classifier: LayaClassifier) -> Self {
        Self {
            classifier,
            warmup_runs: 5,
            silent: false,
        }
    }

    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_runs = warmup;
        self
    }

    pub fn with_silent(mut self, silent: bool) -> Self {
        self.silent = silent;
        self
    }

    pub fn run(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        iterations: usize,
    ) -> Result<BenchmarkReport> {
        if items.is_empty() {
            bail!("No items to benchmark");
        }
        if questions.is_empty() {
            bail!("No questions provided for benchmark");
        }

        if !self.silent {
            println!(
                "🚀 Preparing benchmark: {} items, {} questions, {} iterations (warmup: {})",
                items.len(),
                questions.len(),
                iterations,
                self.warmup_runs
            );
        }

        // Warmup runs
        for _ in 0..self.warmup_runs {
            for item in items.iter().take(5) {
                let _ = self.classifier.classify(&item.state, questions);
            }
        }

        let mut latencies_ms = Vec::with_capacity(items.len() * iterations);
        let start_total = Instant::now();

        for _ in 0..iterations {
            for item in items {
                let t0 = Instant::now();
                let _ = self.classifier.classify(&item.state, questions)?;
                let duration = t0.elapsed().as_secs_f64() * 1000.0;
                latencies_ms.push(duration);
            }
        }

        let total_elapsed = start_total.elapsed().as_secs_f64().max(1e-9);
        let total_items = items.len() * iterations;
        let total_questions = total_items * questions.len();

        latencies_ms.sort_by(|a, b| a.total_cmp(b));

        let min = *latencies_ms.first().unwrap_or(&0.0);
        let max = *latencies_ms.last().unwrap_or(&0.0);
        let avg = latencies_ms.iter().sum::<f64>() / latencies_ms.len().max(1) as f64;

        let report = BenchmarkReport {
            total_items,
            total_questions,
            elapsed_secs: total_elapsed,
            throughput_items_per_sec: total_items as f64 / total_elapsed,
            throughput_questions_per_sec: total_questions as f64 / total_elapsed,
            latency_min_ms: min,
            latency_p50_ms: percentile(&latencies_ms, 0.50),
            latency_p90_ms: percentile(&latencies_ms, 0.90),
            latency_p95_ms: percentile(&latencies_ms, 0.95),
            latency_p99_ms: percentile(&latencies_ms, 0.99),
            latency_max_ms: max,
            latency_avg_ms: avg,
        };

        if !self.silent {
            Self::print_report(&report);
        }
        Ok(report)
    }

    pub fn print_report(report: &BenchmarkReport) {
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Metric").fg(Color::Cyan),
            Cell::new("Value").fg(Color::Green),
            Cell::new("Hardware Context / Reference").fg(Color::DarkGrey),
        ]);

        table.add_row(vec![
            Cell::new("Throughput (Items/sec)"),
            Cell::new(format!("{:.1} items/s", report.throughput_items_per_sec)),
            Cell::new("End-to-end classification ingestion speed"),
        ]);
        table.add_row(vec![
            Cell::new("Throughput (Decisions/sec)"),
            Cell::new(format!(
                "{:.1} decisions/s",
                report.throughput_questions_per_sec
            )),
            Cell::new("T4 baseline: 103-332 decisions/s batched"),
        ]);
        table.add_row(vec![
            Cell::new("Latency p50 (Median)"),
            Cell::new(format!("{:.2} ms", report.latency_p50_ms)),
            Cell::new("Laya GPU target: ~32.8 ms (multilingual) / 39.5 ms (English)"),
        ]);
        table.add_row(vec![
            Cell::new("Latency p90"),
            Cell::new(format!("{:.2} ms", report.latency_p90_ms)),
            Cell::new("90th percentile response time"),
        ]);
        table.add_row(vec![
            Cell::new("Latency p95"),
            Cell::new(format!("{:.2} ms", report.latency_p95_ms)),
            Cell::new("Tail latency threshold"),
        ]);
        table.add_row(vec![
            Cell::new("Latency p99"),
            Cell::new(format!("{:.2} ms", report.latency_p99_ms)),
            Cell::new("99th percentile worst-case"),
        ]);
        table.add_row(vec![
            Cell::new("Latency Min / Max"),
            Cell::new(format!(
                "{:.2} ms / {:.2} ms",
                report.latency_min_ms, report.latency_max_ms
            )),
            Cell::new(format!("Average: {:.2} ms", report.latency_avg_ms)),
        ]);
        table.add_row(vec![
            Cell::new("Total Elapsed Time"),
            Cell::new(format!("{:.3} s", report.elapsed_secs)),
            Cell::new(format!(
                "{} total decisions evaluated",
                report.total_questions
            )),
        ]);

        println!("\n{table}\n");
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
