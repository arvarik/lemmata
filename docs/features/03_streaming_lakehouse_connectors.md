# Feature Specification: Pluggable Streaming & Lakehouse Connectors

## 1. Problem Statement & Motivation

In enterprise environments, data targeted for System 1 classification does not reside exclusively on local disk or in static files. High-throughput decision workloads originate from:
- Event streams: **Apache Kafka**, **AWS Kinesis**, **RabbitMQ**
- Object stores: **AWS S3**, **Google Cloud Storage (GCS)**, **Azure Blob Storage**
- Analytical Lakehouses: **Apache Iceberg**, **Delta Lake**, **Snowflake**, **DuckDB**

`lemma` was architected with an extensible `DataSource` trait:

```rust
pub trait DataSource: Send + Sync {
    fn name(&self) -> &str;
    fn read_items(&self) -> Result<Vec<DataItem>>;
}
```

This specification details the addition of production streaming and lakehouse connectors built directly on this abstraction.

## 2. Architecture & Ingestion Topology

```
   ┌────────────────────────────────────────────────────────┐
   │                  Inbound Data Sources                  │
   └───────┬────────────────────────┬──────────────────────┬┘
           │                        │                      │
           ▼                        ▼                      ▼
    [Kafka Topic]              [S3 Bucket]          [Iceberg / Parquet]
    (events/sec)             (batch archives)         (analytics query)
           │                        │                      │
           └────────────────────────┼──────────────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │ StreamingDataSource Trait │
                      │  (Backpressure & Batches) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  Lemma Batching Engine    │
                      │  (Laya Single-Turn Pass)  │
                      └─────────────┬─────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
   [Classified Stream]                               [Dead Letter / Escalate]
 (Kafka output / Webhook)                           (act_probability > threshold)
```

## 3. Streaming Trait Specification

To support infinite event streams without unbounded memory accumulation, `DataSource` is complemented by `StreamingDataSource`:

```rust
pub trait StreamingDataSource: Send + Sync {
    type ItemStream: futures_util::Stream<Item = Result<DataItem>> + Send + Unpin;

    fn name(&self) -> &str;
    fn stream(&self) -> Result<Self::ItemStream>;
}
```

### Connector Implementations

1. **`KafkaSource` (`--source kafka://broker:9092/support-tickets`):**
   - Uses `rdkafka` with consumer group coordination.
   - Batch commits offsets only after successful classification and delivery.
   - Routes low-confidence or high-entropy items directly to an escalation topic (`support-tickets-escalate`).

2. **`S3Source` (`--source s3://my-bucket/archive/2026/09/*.parquet`):**
   - Streamed chunk downloading using range requests.
   - Zero-copy column parsing with `arrow` and `parquet` crates.
   - Supports parallel bucket scanning across multiple S3 prefixes.

3. **`ParquetSource` (`--file dataset.parquet`):**
   - Direct columnar projection: loads only the required text and metadata columns, avoiding overhead on multimillion-row datasets.

## 4. CLI Interface Proposal

```bash
# Ingest live Kafka stream and publish classified events to an egress topic
lemma stream \
  --source "kafka://localhost:9092/inbound_emails?group_id=lemma_workers" \
  --schema schema/email_triage.json \
  --sink "kafka://localhost:9092/routed_emails" \
  --escalate-sink "kafka://localhost:9092/human_review_queue" \
  --batch-size 32 \
  --max-flush-interval-ms 50
```

## 5. Performance & Resource Guarantees

- **Sub-50ms End-to-End Latency:** Single forward pass in ~33ms, network parsing in ~5ms.
- **Backpressure Controlled:** Never exceeds configured concurrency worker threads.
- **Fail-Safe Dead-Lettering:** Unparseable payloads or model timeouts are quarantined with diagnostic error headers.
