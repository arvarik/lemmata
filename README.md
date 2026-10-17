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
| **`convaiinnovations/laya-typed-decisions`** | ModernBERT-large | 421M | 1024 tokens | 256 tokens | Specialized enterprise workflows (0.766 acc after fine-tuning) |

Latency and accuracy figures above are Convai's own measurements on a Tesla T4, not lemma benchmarks. Convai also reports that the base checkpoints are close to chance on zero-shot typed decisions, and that the shipped checkpoints are over-confident until you recalibrate them on your own data.

### Inference backends

lemma runs every checkpoint's real weights natively with [candle](https://github.com/huggingface/candle). On macOS it runs models on the Metal GPU, and does CPU matrix math through Apple's Accelerate framework. Linux and Windows run on the CPU. `tests/native_probe_test.rs` fits a probe and classifies through it. `tests/native_robustness_test.rs` checks every installed model's answers on hostile inputs (empty, mask tokens, control characters, mixed scripts, very long text). `tests/native_consistency_test.rs` checks that answers do not change across repeated calls, batch and single calls, a question asked alone or with others, chunk strategies on short text, and (for NLI) option order. Parity tests compare each port with its Python reference: `tests/native_parity_test.rs` (Laya against the upstream code, NLI models against `transformers`), `tests/eurobert_parity_test.rs`, and `tests/neobert_parity_test.rs` (against the models' own `trust_remote_code` files).

| `--backend` (or `LEMMA_BACKEND`) | Behavior |
|---|---|
| `auto` (default) | Each checkpoint runs natively when lemma supports it and its weights are in `models/`. Every other checkpoint uses the mock, with one warning on stderr per checkpoint |
| `native` | Native inference only. Fails for a checkpoint without weights or native support |
| `mock` | Keyword-rule simulator for tests and demos. It runs no model |

Every mock result has a `mock:` model id, so simulated decisions cannot pass for model output.

| `--device` (or `LEMMA_DEVICE`) | Behavior |
|---|---|
| `auto` (default) | The Metal GPU when one is present (macOS), otherwise the CPU |
| `cpu` | The CPU |
| `metal` | The Metal GPU. Fails when none is present |

On the GPU, batch commands classify items in order, because calls from many threads only queue on one GPU. On the CPU they run in parallel. Latency for one item with 3 questions, and the peak memory of the process (`lemma --device <d> bench -m <ckpt> --iterations 5 --warmup 2` on `data/customer_support.jsonl`, Apple M5 Pro, peak memory from `/usr/bin/time -l`):

| Checkpoint | CPU p50 / p95 | CPU peak memory | Metal p50 / p95 | Metal peak memory |
|---|---|---|---|---|
| `english` | 306 / 332 ms | 2.6 GB | 95 / 101 ms | 1.6 GB |
| `multilingual` | 126 / 148 ms | 2.7 GB | 44 / 51 ms | 1.7 GB |
| `typed-decisions` | 309 / 335 ms | 2.6 GB | 95 / 101 ms | 1.6 GB |
| `route` | 310 / 335 ms | 2.6 GB | 95 / 101 ms | 1.6 GB |
| `modernbert-zeroshot` | 501 / 554 ms | 2.4 GB | 140 / 152 ms | 1.5 GB |
| `modernbert-nli` | 511 / 558 ms | 3.0 GB | 140 / 152 ms | 3.0 GB |
| `deberta-v3` | 1286 / 1384 ms | 2.9 GB | 230 / 243 ms | 1.8 GB |
| `eurobert` | 112 / 118 ms | 2.6 GB | 31 / 32 ms | 2.6 GB |
| `neobert` | 186 / 189 ms | 1.9 GB | 44 / 45 ms | 1.9 GB |

The robustness, consistency, and probe tests pass on both devices (`LEMMA_DEVICE=metal cargo test --release --test native_robustness_test --test native_consistency_test --test native_probe_test`).

| How lemma scores options | Checkpoints |
|---|---|
| Laya decision head | `english`, `multilingual`, `typed-decisions` |
| NLI zero-shot cross-encoder | `modernbert-zeroshot`, `modernbert-nli`, `deberta-v3` |
| Masked-LM label words | `eurobert`, `neobert` |

**Laya checkpoints with many options:** the decision head fits all option texts into 192 (English) or 256 tokens. Upstream shrinks every option evenly when they overflow, which cuts Banking77's 77 labels to 2 tokens each. lemma instead scores groups of options that fit, then scores the leading options together in a final round. Eliminated options rank below every finalist.

**NLI checkpoints** score each option as the hypothesis "This text is about {label}." (with the option's description in parentheses when it has one), paired with the text and cut to 512 tokens. All of a question's hypotheses run as one padded batch. Choice and score options use the entailment logit, which the zero-shot pipeline softmaxes across labels. A `noul` question uses its statement as the hypothesis.

**Masked-LM checkpoints** are pretrained encoders without a classification head, so lemma uses cloze scoring (Schick & Schütze 2021). The prompt is `{text}\nQuestion: {instructions}\nAnswer:` with one mask per label token. An option's score is the log-probability of its label tokens (their sum), minus the same score for the content-free input "N/A" (contextual calibration, Zhao et al. 2021). Without that subtraction, common words such as "other" win regardless of the text. The prompt does not list the options, because the model then copies whichever option has the right number of tokens. `noul` uses the words "no" and "yes", and `score` uses each level's description. Input is capped at 2048 tokens, because the full attention matrix at 8k tokens needs gigabytes.

Zero-shot accuracy, measured with `lemma test` on public data from `lemma fetch`: 1,000 seeded random items from the AG News and Banking77 test splits, the full SST-2 validation split (872), and 900 tweets from `cardiffnlp/tweet_sentiment_multilingual` (300 each in German, Spanish, and Arabic, 3 classes). Brackets give the 95% Wilson interval:

| Checkpoint | AG News, 1,000 | SST-2, 872 | Banking77, 1,000 | Multilingual sentiment, 900 |
|---|---|---|---|---|
| `english` | 92.3% (90–94) | 88.9% (87–91) | 58.3% (55–61) | 47.6% (44–51) |
| `multilingual` | 92.6% (91–94) | 82.2% (80–85) | 54.0% (51–57) | 49.9% (47–53) |
| `typed-decisions` | 92.5% (91–94) | 88.9% (87–91) | 57.1% (54–60) | 48.9% (46–52) |
| `route` | 89.3% (87–91) | 95.1% (93–96) | 71.0% (68–74) | 57.8% (55–61) |
| `modernbert-zeroshot` | 87.4% (85–89) | 94.7% (93–96) | 70.5% (68–73) | 47.6% (44–51) |
| `modernbert-nli` | 89.8% (88–92) | 96.0% (94–97) | 71.0% (68–74) | 48.4% (45–52) |
| `deberta-v3` | 89.3% (87–91) | 95.2% (94–96) | 67.8% (65–71) | 61.2% (58–64) |
| `eurobert` | 65.8% (63–69) | 78.7% (76–81) | 34.3% (31–37) | 46.4% (43–50) |
| `neobert` | 62.1% (59–65) | 78.9% (76–81) | 42.8% (40–46) | 37.9% (35–41) |

The repo's own sets are small, so treat these as smoke checks:

| Checkpoint | Golden (72 questions) | Support (18 questions) |
|---|---|---|
| `english` | 62.5% | 83.3% |
| `multilingual` | 68.1% | 66.7% |
| `typed-decisions` | 79.2% | 94.4% |
| `route` | 76.4% | 94.4% |
| `modernbert-zeroshot` | 52.8% | 83.3% |
| `modernbert-nli` | 59.7% | 77.8% |
| `deberta-v3` | 73.6% | 83.3% |
| `eurobert` | 52.8% | 55.6% |
| `neobert` | 66.7% | 44.4% |

What the table shows:
- Laya leads on topic classification and on the typed-decision benchmarks. Convai lists topic classification as an in-task family for Laya.
- The NLI models lead on sentiment and on many labels. Laya reaches 54-58% at 77 labels with lemma's tournament (36-38% with upstream's even shrink, against Convai's own 0.425).
- Multilingual tweet sentiment is hard for every model (chance is 33%). `deberta-v3` does best at 61%. For scale, XLM-T, a model trained on tweets and fully supervised on this benchmark, reports macro-F1 of 66.9 (Arabic), 67.9 (Spanish), and 77.3 (German) ([Barbieri et al. 2022](https://arxiv.org/abs/2104.12250)). Its own test splits are the source of these 900 items. A `deberta-v3` probe fitted on all 5,517 training tweets reaches 64.3% (61–67). More training data helps some models (`eurobert` 54.0% to 60.9% from 600 to 5,517 tweets) and not others (`multilingual` 59.0% to 59.1%).
- EuroBERT and NeoBERT are pretrained encoders without a classifier, so label-word scoring is their ceiling.

Choice options keep the order the schema lists them in, as Laya was trained. For more than 20 options, `route` picks an NLI model (`modernbert-nli`, then `deberta-v3`, then `modernbert-zeroshot`, whichever is installed). Calibrate on your own labelled data with `lemma calibrate`.

`route` uses the meta-router's pick when that checkpoint is installed. Otherwise it falls back to the script route across the installed Laya checkpoints. It does not follow the meta-router's long-context rules natively: the `modernbert-zeroshot` fine-tune was trained at 512 tokens, and EuroBERT is capped at 2048 tokens and has no classifier. For more than 20 options it picks an NLI model (see the accuracy tables).

`--chunk-strategy` works on real tokenizer tokens with native inference. `head-tail` keeps the first quarter of the token budget and the last three quarters (about 128 + 382 tokens of 510, as in Sun et al. 2019). `sliding-window` uses windows that overlap by a third and pools their logits. `input_tokens` reports the tokens the model read, and `latency_ms` excludes the one-time model load, which lemma prints to stderr.

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

### 17. Energy-Based OOD Detection (`lemma ood`)
Detects out-of-distribution inputs with the energy score of Liu et al. 2020, $E(x) = -\ln \sum_k \exp(z_k)$, on the raw option logits ($T = 1$). It also measures the distance to class centroids of hashed word features (not model embeddings). A profile covers one question. Its energy threshold is the in-distribution quantile that flags `--fpr-rate` of the in-distribution items, so `0.05` keeps 95% TPR. Profiles fitted before energy moved to raw logits are rejected, so refit them:

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

# 3. Also correct the model's preference for some labels (needs labelled data you will not test on)
lemma calibrate \
  -c eurobert \
  --val-data data/train_sample.jsonl \
  --schema schema/questions.json \
  --fit-label-bias \
  --export-config config/eurobert_bias.json
lemma test -c eurobert --file data/test.jsonl --schema schema/questions.json --config config/eurobert_bias.json
```

Zero-shot accuracy, then accuracy with `--fit-label-bias` fitted on 500 separate training items (600 for multilingual sentiment) and tested on the same sets as the accuracy table above:

| Checkpoint | AG News | SST-2 | Banking77 | Multilingual sentiment |
|---|---|---|---|---|
| `english` | 92.3% → 92.4% | 88.9% → 91.3% | 58.3% → 60.1% | 47.6% → 51.6% |
| `multilingual` | 92.6% → 92.6% | 82.2% → 84.3% | 54.0% → 55.2% | 49.9% → 55.0% |
| `typed-decisions` | 92.5% → 93.0% | 88.9% → 92.1% | 57.1% → 60.2% | 48.9% → 49.3% |
| `modernbert-zeroshot` | 87.4% → 88.0% | 94.7% → 95.0% | 70.5% → 73.6% | 47.6% → 49.9% |
| `modernbert-nli` | 89.8% → 89.9% | 96.0% → 96.2% | 71.0% → 74.3% | 48.4% → 55.1% |
| `deberta-v3` | 89.3% → 90.1% | 95.2% → 95.3% | 67.8% → 72.5% | 61.2% → 61.3% |
| `eurobert` | 65.8% → 71.3% | 78.7% → 78.7% | 34.3% → 39.8% | 46.4% → 41.9% |
| `neobert` | 62.1% → 76.5% | 78.9% → 85.3% | 42.8% → 47.7% | 37.9% → 40.9% |

The bias helps most where a model prefers some label words (NeoBERT on AG News: +14 points) and with many labels (Banking77: +1 to +6). It hurt EuroBERT on multilingual sentiment (-4.5). It is refused for `route`, which picks a different model per input. On multilingual sentiment a single bias lowered route from 57.8% to 54.0%.

`--fit-label-bias` fits one bias per choice label that maximizes the likelihood of the labelled answers (vector scaling with a bias term, Guo et al. 2017), and exports it as `cost_weights`. Zero-shot models favor some label words regardless of the text, and the bias removes that prior. Temperatures do not change which option wins, but the bias does.

### 22b. Supervised Linear Probes (`lemma probe fit`)
With labelled data, a probe learns one choice question directly. It fits a softmax regression on the checkpoint's mean-pooled final hidden states (standardized per dimension), and picks the L2 strength on a held-out fifth of the training data. `--probe` then replaces zero-shot scoring for that question in `classify` and `test`. Probes need real weights. They lift the base encoders the most (NeoBERT on Banking77: 42.8% zero-shot, 71.8% with a probe fitted on 500 items):

Accuracy (%) as zero-shot / `--fit-label-bias` / probe, each fitted on the same 500 training items (600 for multilingual sentiment) and tested on the sets of the accuracy table above. Bold marks the best of the three:

| Checkpoint | AG News | SST-2 | Banking77 | Multilingual sentiment |
|---|---|---|---|---|
| `english` | 92.3 / **92.4** / 91.5 | 88.9 / **91.3** / 91.3 | 58.3 / 60.1 / **67.5** | 47.6 / **51.6** / 46.4 |
| `multilingual` | **92.6** / 92.6 / 92.4 | 82.2 / 84.3 / **85.2** | 54.0 / 55.2 / **60.9** | 49.9 / 55.0 / **59.0** |
| `typed-decisions` | 92.5 / **93.0** / 92.4 | 88.9 / **92.1** / 91.9 | 57.1 / **60.2** / 60.0 | 48.9 / **49.3** / 46.2 |
| `modernbert-zeroshot` | 87.4 / 88.0 / **88.2** | 94.7 / **95.0** / 94.8 | 70.5 / 73.6 / **79.7** | 47.6 / 49.9 / **50.7** |
| `modernbert-nli` | 89.8 / **89.9** / 86.9 | 96.0 / 96.2 / **96.4** | 71.0 / 74.3 / **78.4** | 48.4 / 55.1 / **59.8** |
| `deberta-v3` | 89.3 / **90.1** / 89.8 | 95.2 / **95.3** / 95.0 | 67.8 / **72.5** / 70.8 | 61.2 / 61.3 / **61.7** |
| `eurobert` | 65.8 / 71.3 / **86.3** | 78.7 / 78.7 / **81.0** | 34.3 / 39.8 / **61.4** | 46.4 / 41.9 / **54.0** |
| `neobert` | 62.1 / 76.5 / **88.5** | 78.9 / 85.3 / **87.5** | 42.8 / 47.7 / **71.8** | 37.9 / 40.9 / **44.8** |

`lemma probe fit` fits two probes and keeps the one with the better held-out accuracy. A **plain** probe replaces zero-shot scoring, which suits the base encoders (EuroBERT, NeoBERT), whose embeddings are general. A **residual** probe adds a learned correction to the checkpoint's zero-shot logits, which suits the fine-tuned checkpoints (Laya, NLI), whose embeddings serve their own scoring heads. With a large L2 the residual probe reduces to zero-shot plus a per-label bias. Across these 32 cells the probe was best in 19, label bias in 12, and zero-shot in 1. `probe fit` reports the probe's and the zero-shot accuracy on the same held-out fifth, and warns when the probe loses.

```bash
# Fit on labelled training data (not on the data you test on)
lemma probe fit -c eurobert --data data/train.jsonl --schema schema/topics.json --output probes/eurobert-topics.json

# Use the probe
lemma test -c eurobert --file data/test.jsonl --schema schema/topics.json --probe probes/eurobert-topics.json
lemma classify -c eurobert --text "Stocks fell after the earnings report" --schema schema/topics.json --probe probes/eurobert-topics.json
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
  --receipt-log output/stream_receipts.jsonl \
  --cache  # --attest reads the key from LEMMA_ATTEST_KEY; verify with `lemma attest verify --receipt-log output/stream_receipts.jsonl`

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
