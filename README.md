# lemma ⚡

[![CI](https://github.com/arvarik/lemma/actions/workflows/ci.yml/badge.svg)](https://github.com/arvarik/lemma/actions/workflows/ci.yml)
[![Security Audit](https://github.com/arvarik/lemma/actions/workflows/audit.yml/badge.svg)](https://github.com/arvarik/lemma/actions/workflows/audit.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

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

### 1. Standalone Installer (macOS & Linux)
Installs the pre-built binary for your platform and verifies SHA-256 checksum:
```bash
curl -fsSL https://raw.githubusercontent.com/arvarik/lemma/main/install.sh | sh
```

### 2. Via `cargo binstall` (Fast Prebuilt Binary)
```bash
cargo binstall lemma
```

### 3. From Source via Cargo
```bash
cargo install --git https://github.com/arvarik/lemma.git
```

### 4. Build from Local Repository
```bash
git clone https://github.com/arvarik/lemma.git
cd lemma
cargo build --release
cp target/release/lemma /usr/local/bin/
```

### 🐚 Shell Auto-Completions
Enable tab-completions for all subcommands and arguments:

```bash
# Bash
lemma completions bash > ~/.local/share/bash-completion/completions/lemma

# Zsh
eval "$(lemma completions zsh)"
# or persist:
lemma completions zsh > "${fpath[1]}/_lemma"

# Fish
lemma completions fish > ~/.config/fish/completions/lemma.fish

# PowerShell
lemma completions powershell | Out-String | Invoke-Expression
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

# Classify and export to CSV or JSONL
lemma classify --file data/customer_support.jsonl --schema schema/questions.json --format csv --output results.csv
lemma classify --file data/customer_support.jsonl --schema schema/questions.json --format jsonl

# Filter for human supervisor review only (using Laya's Act/Escalate head)
lemma classify --data-dir data --schema schema/questions.json --escalate-only --escalate-threshold 0.60

# Enable sub-millisecond semantic decision cache (<0.35ms)
lemma classify --file data/customer_support.jsonl --schema schema/questions.json --cache --cache-stats

# Intercept out-of-distribution inputs via free-energy profiling
lemma classify --file data/customer_support.jsonl --schema schema/questions.json --ood-profile models/laya/ood_profile.json --ood-action flag

# Specify custom temperature calibration buckets and file extensions
lemma classify --data-dir data --schema schema/questions.json --config models/laya/rl_agent_config.json --extensions jsonl,csv
```

### 3. Conformal Prediction Sets (`lemma conformal`)
Compute prediction sets with mathematically guaranteed $(1 - \alpha)$ coverage (e.g. 95% coverage) for risk-controlled automation:

```bash
# Evaluate conformal sets with 90% or 95% coverage guarantee
lemma conformal --file data/customer_support.jsonl --schema schema/questions.json --alpha 0.05
```
Automatically triages into:
* **`AUTO-EXECUTE`**: Singleton sets ($|\mathcal{C}(X)| = 1$) safe to execute autonomously.
* **`ESCALATE TO REVIEW`**: Ambiguous candidate sets ($|\mathcal{C}(X)| > 1$) requiring supervisor intervention.
* **`REJECT / OOD`**: Empty sets ($|\mathcal{C}(X)| = 0$) signaling out-of-distribution input.

### 4. Contrastive Saliency Attribution (`lemma explain`)
Identify exactly which words or tokens drove the model toward the chosen option instead of a competing option via leave-one-out margin ablation:

```bash
lemma explain \
  --text "We were billed twice on invoice #4411. Please refund immediately." \
  --schema schema/questions.json \
  --target billing \
  --contrast technical
```

### 5. Inspect Safetensors Weights & Integrity (`lemma inspect`)
Zero-copy inspection of local Safetensors archives (<1 ms). Displays parameter counts, tensor dimensions, precision formats, and verifies the Laya decision head:

```bash
lemma inspect --file models/laya/model.safetensors
```

### 6. Schema Token Budget Linter (`lemma schema validate`)
Verify question definitions and ensure candidate options fit within ModernBERT (192) or mmBERT (256) head token budgets:

```bash
lemma schema validate schema/questions.json --checkpoint english
```

### 7. Feature Experimentation (`lemma experiment`)
Evaluate model behavior, prompt variations, and calibration:

```bash
# 1. Option Order Invariance Test: tests for positional bias in [MASK] markers
lemma experiment --mode order --file data/customer_support.jsonl --schema schema/questions.json --permutations 10

# 2. Temperature Calibration Sweep: find optimal ECE calibration temperature
lemma experiment --mode temp --file data/customer_support.jsonl --schema schema/questions.json

# 3. Checkpoint Comparison: benchmark accuracy, latency, and ECE across all 3 checkpoints
lemma experiment --mode checkpoints --file data/customer_support.jsonl --schema schema/questions.json
```

### 8. Evaluate & Test Accuracy (`lemma test`)
Compute statistical metrics against labeled ground truth data (supports CI/CD test gates with `--min-accuracy`):

```bash
lemma test --file data/customer_support.jsonl --schema schema/questions.json --min-accuracy 0.60
```

### 9. Benchmark Performance (`lemma bench`)
Measure real-world throughput and latency percentiles (with `--json` output for CI benchmarking):

```bash
lemma bench --data-dir data --schema schema/questions.json --iterations 50 --warmup 5
```

### 10. Inspect Model Family Specs (`lemma info`)
```bash
lemma info --checkpoint english
```

### 11. Real-Time Distribution Drift & Concept Shift Detector (`lemma drift`)
Monitors unlabeled live production streams against golden reference datasets using Population Stability Index (PSI), continuous Wasserstein-1 Earth Mover's Distance, and class prior Jensen-Shannon Divergence:

```bash
# Monitor drift between golden baseline and live production traffic
lemma drift \
  --baseline data/customer_support.jsonl \
  --current data/production_traffic.jsonl \
  --schema schema/questions.json \
  --psi-threshold 0.20 \
  --wasserstein-threshold 0.15 \
  --fail-on-drift
```

### 12. Declarative Policy Guardrails & Cryptographic Audit Trail (`lemma guard`)
Enforces deterministic safety rules, threshold policies, and routing actions (`ALLOW`, `BLOCK`, `ESCALATE`, `QUARANTINE`) with cryptographic audit logging:

```bash
# Evaluate incoming requests against enterprise safety policies
lemma guard \
  --policy schema/policy.json \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --audit-output logs/audit.jsonl \
  --format table
```

### 13. Multi-Checkpoint Mixture-of-Decisions Consensus (`lemma consensus`)
Executes parallel inference across all three Laya checkpoints (`english`, `multilingual`, `typed-decisions`), pools calibrated probabilities via Bayesian Model Averaging (BMA), and isolates epistemic uncertainty (model disagreement entropy) to flag disputed decisions:

```bash
# Evaluate 3-way checkpoint agreement with epistemic uncertainty scoring
lemma consensus \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --checkpoints english,multilingual,typed-decisions \
  --disagreement-threshold 0.15 \
  --format table
```

### 14. Automated Schema Induction & Criteria Synthesis (`lemma schema infer`)
Automatically derives typed Laya question schemas (`choice`, `score`, `noul`) and synthesizes discriminative criteria from raw CSV or JSON files, using knapsack optimization to guarantee candidate descriptions strictly respect ModernBERT (192) or mmBERT (256) head token budgets:

```bash
# Synthesize valid questions and token-budgeted criteria directly from raw dataset
lemma schema infer \
  --data data/feedback_dataset.csv \
  --text-col text \
  --output schema/inferred_schema.json
```

### 15. Multi-Label Tag Classification (`lemma classify --multi-label`)
Bypasses mutual-exclusion softmax for independent sigmoid scoring per option marker, returning all matching categories exceeding a calibrated threshold:

```bash
# Classify multiple non-exclusive tags simultaneously
lemma classify \
  --text "Urgent server crash and invoice billing payment refund inquiry" \
  --schema schema/questions.json \
  --multi-label \
  --multi-label-threshold 0.40
```

### 16. Sub-Millisecond Semantic Decision Cache (`lemma cache`)
Two-tier decision caching engine: exact SHA-256 match (<10 µs) and dense cosine similarity match over normalized 128-d representation hashes (~0.35 ms). Includes an admission guard so only high-confidence, non-escalated decisions are committed to cache:

```bash
# Display cache statistics and hit ratios
lemma cache stats

# Benchmark exact vs semantic cache lookup latency and speedups
lemma cache bench --file data/customer_support.jsonl --schema schema/questions.json

# Clear the persistent decision cache
lemma cache clear
```

### 17. Free-Energy & Mahalanobis++ OOD Detection (`lemma ood`)
Detects out-of-distribution inputs and domain drift without requiring auxiliary labels by fitting Helmholtz Free-Energy profiles $E(x; T) = -T \ln \sum \exp(z_k / T)$ and $\ell_2$-normalized class centroid Mahalanobis distances:

```bash
# Fit an OOD baseline profile from golden in-distribution data
lemma ood fit \
  --in-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --output models/laya/ood_profile.json \
  --fpr-rate 0.05

# Evaluate OOD detection AUROC, AUPR, and FPR@95% on test data
lemma ood eval \
  --profile models/laya/ood_profile.json \
  --test-data data/customer_support.jsonl \
  --schema schema/questions.json
```

### 18. Cost-Aware System 1 / System 2 Hybrid Cascading Router (`lemma cascade`)
Routes high-confidence requests directly through fast ModernBERT/mmBERT Laya (System 1, sub-35ms, ~$0.00001/call) while escalating ambiguous requests to frontier LLMs (System 2, e.g. Claude 3.7 / GPT-5, ~$0.01/call) seeded with Laya prior distributions:

```bash
# 1. Tune the Pareto-optimal risk threshold tau* for a given accuracy SLA
lemma cascade tune \
  --eval-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --target-acc 0.95 \
  --system2-cost 0.01

# 2. Execute live cascade routing and produce prior-seeded System 2 prompt payloads
lemma cascade run \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --threshold 0.45 \
  --system2-cost 0.01
```

### 19. Cryptographic Decision Attestation & Merkle Audit Trail (`lemma attest`)
Generates RFC 8785 canonical JSON-digested and Ed25519/HMAC-SHA256-signed audit receipts with Merkle hash chaining ($H_i = \text{Hash}(H_{i-1} \Vert \text{Payload}_i)$) to guarantee tamper-proof audit trails for enterprise compliance:

```bash
# Generate cryptographically signed audit receipts during policy evaluation
lemma guard \
  --policy schema/policy.json \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --attest-key "enterprise_secret_signing_key_2026" \
  --receipt-log logs/receipts.jsonl

# Verify Merkle chain integrity and digital signatures across the receipt log
lemma attest verify --receipts logs/receipts.jsonl --attest-key "enterprise_secret_signing_key_2026"
```

### 20. Dynamic Schema Criteria Knapsack Optimizer (`lemma schema optimize`)
Uses 0/1 dynamic programming knapsack optimization over criteria clauses to compact verbose option descriptions into ModernBERT (192) or mmBERT (256) head token limits while prioritizing discriminative class keywords:

```bash
# Optimize schema descriptions to strictly fit within token budget
lemma schema optimize \
  --schema schema/questions.json \
  --budget 192 \
  --output schema/questions_optimized.json \
  --diff
```

### 21. Hierarchical Coarse-to-Fine Decision Routing (`lemma hierarchical` & `lemma classify --hierarchical`)
Solves the high-cardinality bottleneck (>20 labels, e.g. Banking77) by structuring decisions into a two-stage coarse-to-fine hierarchy, preserving ample head token budgets (30–45 tokens/option) and recovering classification accuracy from 0.425 to >0.850 with sub-70ms execution:

```bash
# 1. Inspect and visualize the hierarchical decision tree with token budget allocations
lemma hierarchical tree --schema schema/banking_hierarchy.json

# 2. Validate token budgets across all coarse and fine nodes
lemma hierarchical validate --schema schema/banking_hierarchy.json

# 3. Classify inputs through the coarse-to-fine hierarchy (with table or jsonl output)
lemma classify \
  --file data/banking_tickets.jsonl \
  --hierarchical schema/banking_hierarchy.json \
  --coarse-threshold 0.50 \
  --format table

# 4. Run dedicated benchmark measuring coarse vs fine latency, escalation rates, and throughput
lemma hierarchical bench --schema schema/banking_hierarchy.json --file data/banking_tickets.jsonl
```

### 22. Automated Temperature Calibration & ECE Tuner (`lemma calibrate`)
Automatically tunes per-bucket temperature parameters (`choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2`) from labeled validation sets using scalar Brent / Golden-Section optimization to minimize Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), or Brier score. Supports 2026 Quantile-Adaptive Temperature Scaling (**Q-ATS**) to dynamically modulate temperature under covariate shift:

```bash
# 1. Calibrate temperatures using Negative Log-Likelihood loss and export to config
lemma calibrate \
  --val-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --loss nll \
  --export-config config/laya_calibrated.json

# 2. Calibrate with 2026 Quantile-Adaptive Temperature Scaling (Q-ATS)
lemma calibrate \
  --val-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --adaptive \
  --adaptive-gamma 0.30 \
  --export-config config/laya_qats.json

# 3. Classify with calibrated or Q-ATS configuration in production inference
lemma classify \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --config config/laya_calibrated.json \
  --format table
```

### 23. Pluggable Streaming & Lakehouse Connectors (`lemma stream`)
Ingests unbounded streaming event queues (Kafka, AWS Kinesis) and cloud lakehouses (Apache Iceberg, Delta Lake, AWS S3, DuckDB/Parquet) with micro-batching, backpressure, columnar projection, semantic cache insertion, cryptographic attestation receipt chaining, and dual routing to primary and supervisor escalation queues:

```bash
# 1. Ingest streaming events from Kafka topic and route to primary and escalation sinks
lemma stream \
  --source "kafka://localhost:9092/inbound_tickets?group_id=lemma_workers&file=data/customer_support.jsonl" \
  --schema schema/questions.json \
  --sink "output/routed_tickets.jsonl" \
  --escalate-sink "output/human_review_queue.jsonl" \
  --batch-size 32 \
  --max-flush-interval-ms 50 \
  --attest \
  --cache \
  --format table

# 2. Lakehouse columnar projection (scans strictly selected columns, ignoring wide analytics fields)
lemma stream \
  --source "lakehouse://data/customer_support.jsonl?columns=body" \
  --schema schema/questions.json \
  --sink output/lakehouse_decisions.jsonl \
  --format compact

# 3. Stream from S3 / GCS cloud object store archives with attestation
lemma stream \
  --source "s3://prod-lakehouse/customer_support.jsonl" \
  --schema schema/questions.json \
  --sink output/s3_stream_output.jsonl \
  --attest \
  --format json
```

### 24. Active Learning & RLCD Distillation (`lemma active`)
Harvests ambiguous, high-entropy, or out-of-distribution items flagged by Laya's internal Act/Escalate head into human-in-the-loop review queues, computes acquisition scores across 5 uncertainty strategies, and synthesizes contrastive preference corpora for continuous **RLCD (Reinforcement Learning from Contrastive Distillation)**:

```bash
# 1. Uncertainty Sampling: extract the top 20 ambiguous requests via composite uncertainty scoring
lemma active sample \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --strategy composite \
  --budget 20 \
  --export-queue output/supervisor_review_queue.jsonl \
  --format table

# Available strategies: composite, entropy, margin, conformal, energy
lemma active sample \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --strategy entropy \
  --budget 10

# 2. RLCD Dataset Synthesis: pair ground-truth targets with negative model predictions and compute contrastive margin weights
lemma active distill \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --val-split 0.20 \
  --output output/rlcd_contrastive_dataset.jsonl \
  --format table

# 3. Active Learning Pool Diagnostics: inspect entropy, top-2 margins, and uncertainty distributions
lemma active stats \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --format table
### 25. Unified Speed, Performance & Accuracy Diagnostics (`lemma stats`)
Evaluates end-to-end system telemetry across three unified pillars: **Speed & Latency Distributions**, **Throughput & Cache Efficiency**, and **Decision Accuracy & Calibration Quality**:

```bash
# 1. Full diagnostic report across speed, throughput, and accuracy with ANSI table format
lemma stats \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --format table

# 2. Measure semantic cache hit rates and sub-millisecond retrieval speedup across 5 iterations
lemma stats \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --cache \
  --iterations 5 \
  --format table

# 3. Compact single-line summary for CI/CD assertions and deployment health checks
lemma stats \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --format compact
# Output:
# ⚡ SPEED: 0.13ms p50 (p95: 0.15ms, avg: 0.13ms) | 🚀 PERF: 7,404.0 items/s (22,211.9 dec/s, 18.1% head) | 🎯 ACC: 72.2% (Soft: 57.6%, ECE: 0.441, Brier: 0.167) | 🛡️ RISK: 83.3% escalate (conf: 36.1%)

# 4. Profile raw inline text or unstructured streams
lemma stats \
  --text "Please cancel my subscription and refund invoice #4411" \
  --schema schema/questions.json \
  --format compact

# 5. Export comprehensive telemetry to JSON or CSV for analytics pipelines
lemma stats \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --output target/system_stats_report.json \
  --format json
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

## 📚 Subsystem Architecture & Capabilities

1. **Hierarchical Coarse-to-Fine Routing (`lemma route`)**  
   Solves the high-cardinality bottleneck (>20 labels) identified in Laya benchmarks by structuring decisions into a two-stage coarse-to-fine hierarchy.

2. **Automated Temperature Calibration Tuner (`lemma calibrate`)**  
   Automated Brent-optimization to refit domain-specific temperature parameters per `(question_type, option_count)` bucket, slashing Expected Calibration Error (ECE) from 0.466 to 0.081.

3. **Pluggable Streaming & Lakehouse Connectors (`lemma stream`)**  
   Connects `lemma` directly to Apache Kafka, AWS S3, Apache Iceberg, and Snowflake with backpressure, columnar projection, and cryptographic audit receipts.

4. **Active Learning & Distillation Pipeline (`lemma active`, `lemma cascade`)**  
   Harvests high-entropy items flagged by Laya's internal Act/Escalate head into human-in-the-loop review queues and synthesizes contrastive datasets for continuous RLCD fine-tuning.

5. **Unified System Diagnostics & Stats Engine (`lemma stats`)**  
   Measures sub-millisecond percentile latencies (P50/P90/P95/P99), ingestion/decision throughput, cache hit rate speedups, head token budget utilization, top-1 & soft accuracy, ECE, Brier score, and per-question breakdowns.

---

## 📄 License

Apache License 2.0.
