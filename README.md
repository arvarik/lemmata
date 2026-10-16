# lemma ⚡

[![CI](https://github.com/arvarik/lemma/actions/workflows/ci.yml/badge.svg)](https://github.com/arvarik/lemma/actions/workflows/ci.yml)
[![Security Audit](https://github.com/arvarik/lemma/actions/workflows/audit.yml/badge.svg)](https://github.com/arvarik/lemma/actions/workflows/audit.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)

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

Latency and accuracy figures above are Convai's own measurements on a Tesla T4, not lemma benchmarks. Convai also reports that the base checkpoints are close to chance on zero-shot typed decisions, and that the shipped checkpoints are over-confident until you recalibrate them on your own data.

### Inference backends

lemma runs the Laya weights natively on CPU (a Rust port of the reference code, built on [candle](https://github.com/huggingface/candle)). `tests/laya_parity_test.rs` checks the port against the upstream Python code.

| `--backend` (or `LEMMA_BACKEND`) | Behavior |
|---|---|
| `auto` (default) | Native inference when weights are in `models/laya`. Otherwise the mock, with a warning on stderr |
| `native` | Native inference. Fails when the weights are missing |
| `mock` | Keyword-rule simulator for tests and demos. It runs no model, and every result has a `mock:` model id |

Native inference supports `english`, `multilingual`, `typed-decisions`, and `route`. The NLI baselines (`deberta-v3`, `modernbert-zeroshot`, `eurobert`, `neobert`, `jev`, `modernbert-nli`) exist only in the mock.

---

## 🚀 Installation & Quickstart

> No release is published yet. Options 1 to 4 need a tagged release. Until then, use option 5 or 6.

### 1. Standalone Installer (macOS & Linux)
Installs the pre-built binary for your platform and verifies SHA-256 checksum:
```bash
curl -fsSL https://raw.githubusercontent.com/arvarik/lemma/main/install.sh | sh
```

### 2. Via `cargo binstall` (Fast Prebuilt Binary)
The crates.io name `lemma` belongs to an unrelated project, so point binstall at this repository:
```bash
cargo binstall --git https://github.com/arvarik/lemma lemma
```

### 3. Via GitHub Container Registry (GHCR Package)
```bash
docker run --rm ghcr.io/arvarik/lemma:latest classify --text "Refund my duplicate charge" --labels "billing,tech,sales"
```

### 4. Verify Release Provenance Attestation
```bash
gh attestation verify lemma-<target>.tar.gz --repo arvarik/lemma
```

### 5. From Source via Cargo
```bash
cargo install --git https://github.com/arvarik/lemma.git
```

### 6. Build from Local Repository
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

# Filter for human supervisor review only (escalation signal from normalized entropy, not Laya's act head)
lemma classify --data-dir data --schema schema/questions.json --escalate-only --escalate-threshold 0.60

# Reuse decisions for repeated inputs within one run (exact-match cache)
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

### 16. Decision Cache (`lemma cache`)
An in-memory cache for one run. The exact tier reuses a decision when the input text and schema repeat. The semantic tier (`--cache-semantic`, off by default) also reuses a decision for similar text, using cosine similarity over hashed 128-d word features. Those features do not come from the model, so a negated sentence can match the original. An admission guard caches only high-confidence, non-escalated decisions:

```bash
# Display cache statistics and hit ratios
lemma cache stats

# Benchmark exact vs semantic cache lookup latency and speedups
lemma cache bench --file data/customer_support.jsonl --schema schema/questions.json

# The cache is not persisted, so this only confirms there is nothing to clear
lemma cache clear
```

### 17. Free-Energy & Mahalanobis++ OOD Detection (`lemma ood`)
Detects out-of-distribution inputs and domain drift by fitting Helmholtz Free-Energy profiles $E(x; T) = -T \ln \sum \exp(z_k / T)$ and $\ell_2$-normalized class centroid Mahalanobis distances:

```bash
# Fit an OOD baseline profile from golden in-distribution data
lemma ood fit \
  --train-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --output models/laya/ood_profile.json \
  --fpr-rate 0.05

# Evaluate OOD detection FPR and TPR/Recall against test data
lemma ood eval \
  --profile models/laya/ood_profile.json \
  --id-data data/customer_support.jsonl \
  --ood-data data/banking_tickets.jsonl \
  --schema schema/questions.json
```

### 18. Cost-Aware System 1 / System 2 Hybrid Cascading Router (`lemma cascade`)
Routes high-confidence requests through fast ModernBERT/mmBERT Laya (System 1, sub-35ms) while escalating ambiguous requests to frontier LLMs (System 2) seeded with Laya prior distributions:

```bash
# 1. Tune the Pareto-optimal risk threshold tau* for a given accuracy SLA
lemma cascade tune \
  --val-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --target-accuracy 0.95 \
  --system2-cost 0.01

# 2. Execute live cascade routing
lemma cascade run \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --risk-threshold 0.45 \
  --system2-cost 0.01
```

### 19. Signed Decision Receipts & Hash Chain (`lemma attest`)
Each receipt holds a SHA-256 digest of its payload and an HMAC-SHA256 signature of that digest. Each receipt also stores the digest of the previous one ($H_i = \text{SHA256}(\text{Payload}_i)$, with $H_{i-1}$ inside the payload), so a changed, removed, or reordered receipt breaks the chain. HMAC is a shared-key scheme: anyone who can verify can also sign.

Read the key from the `LEMMA_ATTEST_KEY` environment variable. A key typed on the command line stays in your shell history:

```bash
export LEMMA_ATTEST_KEY="$(openssl rand -hex 32)"

# Generate signed receipts during policy evaluation
lemma guard \
  --policy schema/policy.json \
  --file data/customer_support.jsonl \
  --schema schema/questions.json \
  --receipt-log logs/receipts.jsonl

# Verify digests, signatures, and chain links across the receipt log
lemma attest verify --receipt-log logs/receipts.jsonl
```

### 20. Dynamic Schema Criteria Knapsack Optimizer (`lemma schema optimize`)
Uses 0/1 dynamic programming knapsack optimization over criteria clauses to compact verbose option descriptions into ModernBERT (192) or mmBERT (256) head token limits:

```bash
lemma schema optimize \
  --schema schema/questions.json \
  --budget 192 \
  --output schema/questions_optimized.json \
  --diff
```

### 21. Hierarchical Coarse-to-Fine Decision Routing (`lemma hierarchical` / `lemma route`)
Solves high-cardinality classification (>20 labels, e.g. Banking77) via a two-stage coarse-to-fine hierarchy:

```bash
# 1. Visualize the hierarchical decision tree with token budget allocations
lemma hierarchical tree --schema schema/banking_hierarchy.json

# 2. Validate token budgets across all coarse and fine nodes
lemma hierarchical validate --schema schema/banking_hierarchy.json

# 3. Classify inputs through the coarse-to-fine hierarchy (also via `lemma route run`)
lemma classify \
  --file data/banking_tickets.jsonl \
  --hierarchical schema/banking_hierarchy.json \
  --coarse-threshold 0.50 \
  --format table

# 4. Benchmark coarse vs fine latency, escalation rates, and throughput
lemma hierarchical bench --schema schema/banking_hierarchy.json --file data/banking_tickets.jsonl
```

### 22. Automated Temperature Calibration & ECE Tuner (`lemma calibrate`)
Tunes per-bucket temperature parameters (`choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2`) from labeled validation sets using Golden-Section optimization to minimize ECE, NLL, or Brier score, with optional Quantile-Adaptive Temperature Scaling (**Q-ATS**). Fitted temperatures stay in [0.5, 5.0]. A bucket with fewer than 30 labeled samples (`--min-bucket-samples`) keeps its prior temperature:

```bash
# 1. Calibrate temperatures using Negative Log-Likelihood loss and export to config
lemma calibrate \
  --val-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --loss nll \
  --export-config config/laya_calibrated.json

# 2. Calibrate with Quantile-Adaptive Temperature Scaling (Q-ATS)
lemma calibrate \
  --val-data data/customer_support.jsonl \
  --schema schema/questions.json \
  --adaptive \
  --adaptive-gamma 0.30 \
  --export-config config/laya_qats.json
```

### 23. Micro-Batch Streaming (`lemma stream`)
Classifies a stream of records in micro-batches, with optional caching and signed receipts. lemma has no Kafka, S3, or Iceberg client. A `kafka://` source replays the local file in `?file=` with Kafka-style offsets, a `kafka://` sink writes the local file in `?file=`, and `lakehouse://` projects columns from a local file. `s3://`, `gcs://`, and `blob://` URIs return an error:

```bash
# 1. Ingest streaming events from Kafka topic and route to primary and escalation sinks
lemma stream \
  --source "kafka://localhost:9092/inbound_tickets?group_id=lemma_workers&file=data/customer_support.jsonl" \
  --schema schema/questions.json \
  --sink "output/routed_tickets.jsonl" \
  --escalate-sink "output/human_review_queue.jsonl" \
  --batch-size 32 \
  --attest \
  --cache  # --attest reads the key from LEMMA_ATTEST_KEY

# 2. Lakehouse columnar projection
lemma stream \
  --source "lakehouse://data/customer_support.jsonl?columns=body" \
  --schema schema/questions.json \
  --sink output/lakehouse_decisions.jsonl \
  --format compact
```

### 24. Active Learning & RLCD Distillation (`lemma active`)
Harvests ambiguous or out-of-distribution items into human-in-the-loop review queues and synthesizes contrastive preference corpora for **RLCD**:

```bash
# 1. Uncertainty Sampling (strategies: composite, entropy, margin, conformal, energy)
lemma active sample \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --strategy composite \
  --budget 20 \
  --export-queue output/supervisor_review_queue.jsonl

# 2. RLCD Dataset Synthesis
lemma active distill \
  --data data/customer_support.jsonl \
  --schema schema/questions.json \
  --val-split 0.20 \
  --output output/rlcd_contrastive_dataset.jsonl

# 3. Active Learning Pool Diagnostics
lemma active stats --data data/customer_support.jsonl --schema schema/questions.json
```

### 25. Unified Speed, Performance & Accuracy Diagnostics (`lemma stats`)
Evaluates end-to-end system telemetry across **Speed & Latency Distributions**, **Throughput & Cache Efficiency**, and **Decision Accuracy & Calibration Quality**:

```bash
# Full diagnostic report or compact one-line summary
lemma stats --file data/customer_support.jsonl --schema schema/questions.json --format table
lemma stats --file data/customer_support.jsonl --schema schema/questions.json --format compact
```

### 26. Public Dataset & Web Ingestion (`lemma fetch` & `lemma scrape`)
Ingest datasets from Hugging Face Hub, Hacker News, Reddit, or Wikipedia, or scrape web pages into clean Markdown:

```bash
lemma fetch --hf fancyzhx/ag_news --limit 50 --auto-schema --output data/ag_news.jsonl
lemma scrape --url https://example.com/article --format jsonl
```

---

## 📄 License

Apache License 2.0.
