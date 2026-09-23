use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::Result;
use comfy_table::{Cell, Color, Table};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::datasource::{DataItem, StreamSink, StreamingDataSource};
use crate::engine::attest::AttestationEngine;
use crate::engine::cache::{DecisionCache, SemanticCacheConfig};
use crate::engine::classifier::LayaClassifier;
use crate::engine::guardrail::GuardAction;
use crate::model::{DecisionResult, Question};

/// Configuration parameters for the streaming micro-batching engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPipelineConfig {
    /// Maximum number of items accumulated per micro-batch (e.g. 32 or 64).
    pub batch_size: usize,

    /// Maximum accumulation window timeout in milliseconds before triggering a flush.
    pub max_flush_interval_ms: u64,

    /// Confidence threshold below which items trigger supervisor / dead-letter escalation.
    pub escalate_threshold: f64,

    /// Whether semantic decision cache is enabled.
    pub enable_cache: bool,

    /// Whether cryptographic Merkle-chained attestation receipts are generated.
    pub enable_attest: bool,

    /// Signing key for attestation HMAC proofs.
    pub attest_secret: String,

    /// Maximum events to process before terminating (None = infinite stream).
    pub limit: Option<usize>,
}

impl Default for StreamPipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_flush_interval_ms: 50,
            escalate_threshold: 0.50,
            enable_cache: false,
            enable_attest: false,
            attest_secret: "lemma_stream_attest_2026".to_string(),
            limit: None,
        }
    }
}

/// Summary of an individual question's answer in a stream event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamAnswerSummary {
    pub prediction: String,
    pub confidence: f64,
    pub act_probability: Option<f64>,
    pub probabilities: BTreeMap<String, f64>,
}

/// A fully processed and enriched stream event emitted to egress sinks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRecord {
    pub item_id: String,
    pub source_uri: String,
    pub answers: BTreeMap<String, StreamAnswerSummary>,
    pub escalated: bool,
    pub escalation_reason: Option<String>,
    pub latency_ms: f64,
    pub receipt_digest: Option<String>,
    pub cached: bool,
    pub timestamp: String,
}

/// Real-time throughput and execution metrics for a streaming pipeline.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub total_ingested: usize,
    pub total_classified: usize,
    pub total_escalated: usize,
    pub total_cached: usize,
    pub total_attested: usize,
    pub batches_processed: usize,
    pub elapsed_duration_ms: f64,
    pub throughput_qps: f64,
    pub avg_item_latency_ms: f64,
    pub escalation_rate: f64,
    pub cache_hit_rate: f64,
}

/// Production Streaming Pipeline integrating micro-batching, Rayon parallelism,
/// cryptographic attestation receipts, and dual primary/escalation sinks.
pub struct StreamPipeline {
    classifier: LayaClassifier,
    questions: BTreeMap<String, Question>,
    primary_sink: StreamSink,
    escalate_sink: Option<StreamSink>,
    cache: Option<Arc<Mutex<DecisionCache>>>,
    attest_engine: Option<AttestationEngine>,
    config: StreamPipelineConfig,
}

impl StreamPipeline {
    pub fn new(
        classifier: LayaClassifier,
        questions: BTreeMap<String, Question>,
        primary_sink: StreamSink,
    ) -> Self {
        Self {
            classifier,
            questions,
            primary_sink,
            escalate_sink: None,
            cache: None,
            attest_engine: None,
            config: StreamPipelineConfig::default(),
        }
    }

    pub fn with_config(mut self, config: StreamPipelineConfig) -> Self {
        if config.enable_attest && self.attest_engine.is_none() {
            self.attest_engine = Some(AttestationEngine::new(&config.attest_secret));
        }
        if config.enable_cache && self.cache.is_none() {
            self.cache = Some(Arc::new(Mutex::new(DecisionCache::new(SemanticCacheConfig::default()))));
        }
        self.config = config;
        self
    }

    pub fn with_escalate_sink(mut self, sink: StreamSink) -> Self {
        self.escalate_sink = Some(sink);
        self
    }

    pub fn with_cache(mut self, cache: Arc<Mutex<DecisionCache>>) -> Self {
        self.cache = Some(cache);
        self.config.enable_cache = true;
        self
    }

    pub fn with_attest_engine(mut self, engine: AttestationEngine) -> Self {
        self.attest_engine = Some(engine);
        self.config.enable_attest = true;
        self
    }

    /// Executes the streaming pipeline against a `StreamingDataSource`.
    pub fn run(&mut self, source: &mut dyn StreamingDataSource) -> Result<StreamMetrics> {
        let start_time = Instant::now();
        let schema_hash = format!("{:x}", self.questions.len());

        let mut total_ingested = 0;
        let mut total_classified = 0;
        let mut total_escalated = 0;
        let mut total_cached = 0;
        let mut total_attested = 0;
        let mut batches_processed = 0;
        let mut cumulative_latency_ms = 0.0;

        let flush_timeout = Duration::from_millis(self.config.max_flush_interval_ms);

        while !source.is_exhausted() {
            if let Some(limit) = self.config.limit {
                if total_classified >= limit {
                    break;
                }
            }

            let remaining_budget = if let Some(limit) = self.config.limit {
                (limit - total_classified).min(self.config.batch_size)
            } else {
                self.config.batch_size
            };

            if remaining_budget == 0 {
                break;
            }

            let batch = source.next_batch(remaining_budget, flush_timeout)?;
            if batch.is_empty() {
                if source.is_exhausted() {
                    break;
                }
                // Avoid spin-locking on empty non-exhausted streams
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }

            batches_processed += 1;
            total_ingested += batch.len();

            let batch_start = Instant::now();

            // 1. Process items (leveraging DecisionCache and Rayon parallel evaluation)
            let processed_records: Vec<(StreamRecord, Option<DecisionResult>)> = if batch.len() > 1 {
                // Rayon parallel map for micro-batch
                batch
                    .par_iter()
                    .map(|item| self.process_single_item(item, &schema_hash))
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![self.process_single_item(&batch[0], &schema_hash)?]
            };

            let batch_duration_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
            cumulative_latency_ms += batch_duration_ms;

            // 2. Cryptographic Attestation & Cache Insertion on the sequential sink-emission step
            for (mut record, opt_decision) in processed_records {
                if record.cached {
                    total_cached += 1;
                }

                if record.escalated {
                    total_escalated += 1;
                }

                // If cache is enabled and we had a fresh decision with high confidence, insert into cache
                if let (Some(cache_arc), Some(decision)) = (&self.cache, &opt_decision) {
                    let min_conf = decision
                        .answers
                        .values()
                        .map(|a| a.confidence)
                        .fold(1.0, f64::min);
                    if !record.cached && min_conf >= 0.70 {
                        let text_val = match &batch.iter().find(|i| i.id == record.item_id) {
                            Some(item) => match &item.state {
                                serde_json::Value::String(s) => s.clone(),
                                other => serde_json::to_string(other).unwrap_or_default(),
                            },
                            None => String::new(),
                        };
                        if let Ok(mut c) = cache_arc.lock() {
                            c.insert(&text_val, &schema_hash, decision);
                        }
                    }
                }

                // Cryptographic Merkle chain attestation receipt
                if let (Some(attest), Some(decision)) = (&mut self.attest_engine, &opt_decision) {
                    if let Some(item) = batch.iter().find(|i| i.id == record.item_id) {
                        let action = if record.escalated {
                            GuardAction::Escalate
                        } else {
                            GuardAction::Allow
                        };
                        let receipt = attest.create_receipt(
                            item,
                            decision,
                            Some("STREAM_COMPLIANCE_2026"),
                            action,
                            &schema_hash,
                        )?;
                        record.receipt_digest = Some(receipt.receipt_digest);
                        total_attested += 1;
                    }
                }

                let record_json = serde_json::to_string(&record)?;

                // Egress Routing: Escalations diverted to escalate_sink if present
                if record.escalated {
                    if let Some(esc_sink) = &mut self.escalate_sink {
                        esc_sink.write_record(&record_json)?;
                    } else {
                        self.primary_sink.write_record(&record_json)?;
                    }
                } else {
                    self.primary_sink.write_record(&record_json)?;
                }

                total_classified += 1;
            }

            // Flush sinks
            self.primary_sink.flush()?;
            if let Some(esc_sink) = &mut self.escalate_sink {
                esc_sink.flush()?;
            }

            // Commit stream offset for the processed batch
            source.commit_offset(batch.len())?;
        }

        let total_elapsed = start_time.elapsed();
        let total_elapsed_ms = total_elapsed.as_secs_f64() * 1000.0;
        let throughput = if total_elapsed.as_secs_f64() > 0.0 {
            total_classified as f64 / total_elapsed.as_secs_f64()
        } else {
            0.0
        };

        let avg_latency = if total_classified > 0 {
            cumulative_latency_ms / total_classified as f64
        } else {
            0.0
        };

        let esc_rate = if total_classified > 0 {
            total_escalated as f64 / total_classified as f64
        } else {
            0.0
        };

        let cache_rate = if total_classified > 0 {
            total_cached as f64 / total_classified as f64
        } else {
            0.0
        };

        Ok(StreamMetrics {
            total_ingested,
            total_classified,
            total_escalated,
            total_cached,
            total_attested,
            batches_processed,
            elapsed_duration_ms: total_elapsed_ms,
            throughput_qps: throughput,
            avg_item_latency_ms: avg_latency,
            escalation_rate: esc_rate,
            cache_hit_rate: cache_rate,
        })
    }

    /// Evaluates a single DataItem through cache or LayaClassifier.
    fn process_single_item(
        &self,
        item: &DataItem,
        schema_hash: &str,
    ) -> Result<(StreamRecord, Option<DecisionResult>)> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let timestamp = format!("{}Z", now);

        let item_text = match &item.state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };

        // Cache lookup
        if let Some(cache_arc) = &self.cache {
            if let Ok(mut c) = cache_arc.lock() {
                match c.lookup(&item_text, schema_hash) {
                    crate::engine::cache::CacheLookupResult::ExactHit(hit)
                    | crate::engine::cache::CacheLookupResult::SemanticHit { result: hit, .. } => {
                        let (answers, escalated, reason) = self.build_answers_from_decision(&hit);
                        let record = StreamRecord {
                            item_id: item.id.clone(),
                            source_uri: item.source_uri.clone(),
                            answers,
                            escalated,
                            escalation_reason: reason,
                            latency_ms: 0.05, // Sub-millisecond cache hit
                            receipt_digest: None,
                            cached: true,
                            timestamp,
                        };
                        return Ok((record, Some(hit)));
                    }
                    crate::engine::cache::CacheLookupResult::Miss => {}
                }
            }
        }

        let item_start = Instant::now();
        let decision = self.classifier.classify(&item.state, &self.questions)?;
        let latency_ms = item_start.elapsed().as_secs_f64() * 1000.0;

        let (answers, escalated, reason) = self.build_answers_from_decision(&decision);

        let record = StreamRecord {
            item_id: item.id.clone(),
            source_uri: item.source_uri.clone(),
            answers,
            escalated,
            escalation_reason: reason,
            latency_ms,
            receipt_digest: None,
            cached: false,
            timestamp,
        };

        Ok((record, Some(decision)))
    }

    fn build_answers_from_decision(
        &self,
        decision: &DecisionResult,
    ) -> (BTreeMap<String, StreamAnswerSummary>, bool, Option<String>) {
        let mut answers = BTreeMap::new();
        let mut escalated = false;
        let mut reasons = Vec::new();

        for (qid, ans) in &decision.answers {
            let pred_str = if let Some(labels) = &ans.selected_labels {
                labels.join(", ")
            } else if let Some(c) = &ans.choice {
                c.clone()
            } else if let Some(s) = ans.score {
                format!("{:.2}", s)
            } else if let Some(n) = ans.noul {
                format!("true: {:.2}%", n * 100.0)
            } else {
                "unknown".to_string()
            };

            let act_prob = ans.act_probability.unwrap_or(0.0);
            let is_esc = ans.confidence < self.config.escalate_threshold
                || act_prob > self.config.escalate_threshold;

            if is_esc {
                escalated = true;
                reasons.push(format!(
                    "{}: conf={:.1}%, act={:.1}%",
                    qid,
                    ans.confidence * 100.0,
                    act_prob * 100.0
                ));
            }

            answers.insert(
                qid.clone(),
                StreamAnswerSummary {
                    prediction: pred_str,
                    confidence: ans.confidence,
                    act_probability: ans.act_probability,
                    probabilities: ans.probabilities.clone(),
                },
            );
        }

        let reason_str = if escalated {
            Some(reasons.join("; "))
        } else {
            None
        };

        (answers, escalated, reason_str)
    }

    /// Formats streaming performance metrics as a clean terminal table.
    pub fn format_table(metrics: &StreamMetrics) -> String {
        let mut table = Table::new();
        table.load_preset(comfy_table::presets::UTF8_FULL);
        table.set_header(vec![
            Cell::new("Stream Pipeline Metric").fg(Color::Cyan),
            Cell::new("Value").fg(Color::Yellow),
        ]);

        table.add_row(vec![
            Cell::new("Total Ingested Events"),
            Cell::new(format!("{}", metrics.total_ingested)),
        ]);
        table.add_row(vec![
            Cell::new("Total Classified"),
            Cell::new(format!("{}", metrics.total_classified)),
        ]);
        table.add_row(vec![
            Cell::new("Micro-Batches Processed"),
            Cell::new(format!("{}", metrics.batches_processed)),
        ]);
        table.add_row(vec![
            Cell::new("Escalated to Supervisor / DLQ"),
            Cell::new(format!(
                "{} ({:.1}%)",
                metrics.total_escalated,
                metrics.escalation_rate * 100.0
            )),
        ]);
        table.add_row(vec![
            Cell::new("Semantic Cache Hits"),
            Cell::new(format!(
                "{} ({:.1}%)",
                metrics.total_cached,
                metrics.cache_hit_rate * 100.0
            )),
        ]);
        table.add_row(vec![
            Cell::new("Attested Merkle Receipts"),
            Cell::new(format!("{}", metrics.total_attested)),
        ]);
        table.add_row(vec![
            Cell::new("Execution Duration"),
            Cell::new(format!("{:.2} ms", metrics.elapsed_duration_ms)),
        ]);
        table.add_row(vec![
            Cell::new("Stream Throughput"),
            Cell::new(format!("{:.1} events/sec", metrics.throughput_qps))
                .fg(Color::Green),
        ]);
        table.add_row(vec![
            Cell::new("Average Item Latency"),
            Cell::new(format!("{:.3} ms", metrics.avg_item_latency_ms)),
        ]);

        format!("\n⚡ Lemma Streaming Pipeline Results:\n{}", table)
    }

    /// Formats streaming performance metrics as JSON.
    pub fn format_json(metrics: &StreamMetrics) -> Result<String> {
        Ok(serde_json::to_string_pretty(metrics)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::KafkaSource;
    use crate::model::Checkpoint;

    #[test]
    fn test_stream_pipeline_execution() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "dept".into(),
            Question::new_choice("Route department", vec!["billing".into(), "technical".into()]),
        );

        let items = vec![
            DataItem::new("e1", serde_json::json!({"text": "urgent billing invoice refund"}), "kafka"),
            DataItem::new("e2", serde_json::json!({"text": "server crash null pointer exception"}), "kafka"),
            DataItem::new("e3", serde_json::json!({"text": "refund invoice billing"}), "kafka"),
        ];

        let mut source = KafkaSource::new("localhost:9092", "tickets", "test_group").with_items(items);

        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let sink = StreamSink::from_target("target/test_stream_out.jsonl").unwrap();

        let mut pipeline = StreamPipeline::new(classifier, questions, sink).with_config(StreamPipelineConfig {
            batch_size: 2,
            max_flush_interval_ms: 10,
            escalate_threshold: 0.60,
            enable_cache: true,
            enable_attest: true,
            attest_secret: "test_secret".into(),
            limit: None,
        });

        let metrics = pipeline.run(&mut source).unwrap();
        assert_eq!(metrics.total_classified, 3);
        assert_eq!(metrics.batches_processed, 2);
        assert_eq!(source.committed_offset(), 3);
        assert_eq!(metrics.total_attested, 3);
    }
}
