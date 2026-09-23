# lemma ⚡

> **A high-performance Rust CLI and engine for Laya System 1 decision and classification models.**

`lemma` provides enterprise-grade data ingestion, classification, experimentation, evaluation, and benchmarking for the **Laya** decision model family developed by Convai Innovations.

---

## 🧠 About Laya Models

**Laya** is an open-weight family of non-autoregressive "System 1" decision models designed for instant, structured classification in a single forward pass (~33 ms). Rather than generating free-form text with traditional LLMs (which suffer from hallucinations, high latency, and parsing complexity), Laya takes an input state (raw text, email, ticket, or JSON) and returns mathematically calibrated answers across three typed primitives:

1. **`choice`**: Selects a label from discrete candidates with calibrated probability distributions.
2. **`score`**: Predicts an ordinal rubric value (e.g. 0 to 4).
3. **`noul`**: Outputs calibrated confidence for boolean assertions (true/false).

### Laya Model Family Checkpoints

| Checkpoint | Backbone Encoder | Parameters | Context Window | Head Budget | Best For |
|---|---|---|---|---|---|
| **`convaiinnovations/laya`** | ModernBERT-large | 421M | 512 tokens | 192 tokens | English classification, guardrails, email triage (~39.5ms) |
| **`convaiinnovations/laya-multilingual`** | mmBERT-base | 322M | 1024 tokens | 256 tokens | 100+ languages, 2.2x faster forward pass (~32.8ms) |
| **`convaiinnovations/laya-typed-decisions`** | ModernBERT-large | 421M | 1024 tokens | 256 tokens | Specialized enterprise workflows (0.766 acc, beats Jev teacher) |

---

## 🚀 Installation & Quickstart

### Prerequisites
- Rust 1.85+ (or installed via `cargo` / `mise`)
- Hugging Face CLI (`hf` or `huggingface-cli`) for model downloads

### Build
```bash
cargo build --release
cp target/release/lemma /usr/local/bin/ # or add target/release to PATH
```

---

## 🛠️ CLI Commands & Usage

### 1. Download Model Weights (`lemma download`)
Leverages the user's authenticated `hf` CLI (`hf download <repo> --local-dir <dir>`) to pull checkpoints:

```bash
# Preview download command with user's HF authentication
lemma download --checkpoint english --dry-run

# Download the multilingual checkpoint to local directory
lemma download --checkpoint multilingual --target-dir models/laya-multilingual

# Verify integrity of downloaded checkpoint files
lemma download --verify-only --target-dir models/laya
```

### 2. Ingest & Classify Data (`lemma classify`)
Classify inputs from the gitignored `data/` folder, a specific file, standard input stream, or inline text:

```bash
# Ingest entire data/ folder (JSON, JSONL, CSV) with schema
lemma classify --data-dir data --schema schema/questions.json

# Stream inputs from standard input (pipes & UNIX philosophy)
echo '{"text": "Production database crashed with OOM"}' | \
  lemma classify --stdin --labels "billing,technical,sales,other"

# Classify inline text with table output
lemma classify \
  --text "Hi, we were billed twice for invoice #4411. Please refund immediately." \
  --labels "billing,technical,sales,other" \
  --format table

# Output structured JSON for downstream pipelines
lemma classify --file data/customer_support.jsonl --schema schema/questions.json --format json
```

### 3. Feature Experimentation (`lemma experiment`)
Evaluate model behavior, prompt variations, and calibration:

```bash
# 1. Option Order Invariance Test: tests for positional bias in [MASK] markers
lemma experiment --mode order --file data/customer_support.jsonl --schema schema/questions.json --permutations 10

# 2. Temperature Calibration Sweep: find optimal ECE calibration temperature
lemma experiment --mode temp --file data/customer_support.jsonl --schema schema/questions.json

# 3. Checkpoint Comparison: benchmark accuracy, latency, and ECE across all 3 checkpoints
lemma experiment --mode checkpoints --file data/customer_support.jsonl --schema schema/questions.json
```

### 4. Evaluate & Test Accuracy (`lemma test`)
Compute statistical metrics against labeled ground truth data:

```bash
lemma test --file data/customer_support.jsonl --schema schema/questions.json
```
Computes:
- **Accuracy**: Argmax classification agreement.
- **Soft Accuracy**: Probability mass assigned to true targets.
- **Expected Calibration Error (ECE)**: 10-bin calibration gap.
- **Brier Score**: Strictly proper quadratic scoring rule.
- **Score MAE**: Mean absolute error for ordinal rubrics.
- **Escalate / Act Rate**: Percentage of decisions flagged by Laya's act/escalate head for human supervisor review.

### 5. Benchmark Performance (`lemma bench`)
Measure real-world throughput and latency percentiles:

```bash
lemma bench --data-dir data --schema schema/questions.json --iterations 50 --warmup 5
```
Reports:
- Throughput (Items/sec & Decisions/sec)
- Latency Percentiles (p50, p90, p95, p99, min, max, avg)
- Hardware reference comparisons (Tesla T4 baseline vs Apple Silicon)

### 6. Inspect Model Architecture (`lemma info`)
```bash
lemma info
```

---

## 🔌 Extensible Data Ingestion Architecture

`lemma` defines a pluggable `DataSource` trait in `src/datasource/traits.rs`:

```rust
pub trait DataSource: Send + Sync {
    fn name(&self) -> &str;
    fn read_items(&self) -> Result<Vec<DataItem>>;
}
```

Built-in implementations:
- `DirectorySource`: Recursively ingests files from directories (e.g. `data/`).
- `FileSource`: Parses `.json`, `.jsonl`, `.csv`, `.txt`, `.md`.
- `StdinSource`: Streams line-delimited items or JSONL from UNIX pipes.
- `MemorySource`: Zero-copy in-memory buffers for programmatic or test calls.

Custom connectors (Kafka, S3, Snowflake, Iceberg) can easily implement this trait.

---

## 📚 Future Feature Specifications

Explore in-depth engineering proposals in [`docs/features/`](docs/features/):

1. [**01. Hierarchical Coarse-to-Fine Routing**](docs/features/01_hierarchical_routing.md)  
   Solves the high-cardinality bottleneck (>20 labels) identified in Laya benchmarks by structuring decisions into a two-stage coarse-to-fine hierarchy.

2. [**02. Automated Temperature Calibration Tuner**](docs/features/02_temperature_calibration_tuner.md)  
   Automated Brent-optimization to refit domain-specific temperature parameters per `(question_type, option_count)` bucket, slashing Expected Calibration Error (ECE) from 0.466 to 0.081.

3. [**03. Pluggable Streaming & Lakehouse Connectors**](docs/features/03_streaming_lakehouse_connectors.md)  
   Connects `lemma` directly to Apache Kafka, AWS S3, Apache Iceberg, and Snowflake with backpressure and dead-letter routing.

4. [**04. Active Learning & Supervisor Escalation Pipeline**](docs/features/04_active_learning_and_distillation.md)  
   Harvests high-entropy items flagged by Laya's internal Act/Escalate head into human-in-the-loop review queues and generates training splits for continuous RLCD fine-tuning.

---

## 📄 License

Apache License 2.0.
