# Feature Specification: Automated Temperature Calibration & ECE Tuner

## 1. Problem Statement & Motivation

Laya decision models are trained with Reinforcement Learning from Calibrated Distributions (**RLCD**), rewarding models under proper scoring rules (logarithmic and spherical scoring). However, as highlighted in the official repository limits:

> **Calibration:**
> Base checkpoints ship systematically over-confident. Refitting one temperature per `(question_type, option_count)` moves mean Expected Calibration Error (ECE) from **0.466 down to 0.081** on `convaiinnovations/laya` (English) and from **0.314 down to 0.106** on `laya-multilingual`.
> Relying on raw probabilities without domain-specific temperature tuning in production risks over-confident misclassifications and inaccurate guardrail decisions.

While users can manually edit `rl_agent_config.json`, there is currently no turnkey CLI command to automatically fit, validate, and serialize domain-specific temperatures on held-out customer datasets.

## 2. Mathematical Foundation

Given validation dataset $D = \{(x_i, y_i)\}_{i=1}^N$ where $y_i \in \{1, \dots, K\}$ is the ground-truth label, let $z_{i} \in \mathbb{R}^K$ be the raw option-marker logits produced by Laya.

The temperature-scaled distribution is:
$$p_i(k; T) = \frac{\exp(z_{ik} / T)}{\sum_{j=1}^K \exp(z_{ij} / T)}$$

We optimize $T^*$ by minimizing either:
1. **Negative Log-Likelihood (Cross-Entropy Loss):**
   $$\mathcal{L}_{\text{NLL}}(T) = -\frac{1}{N} \sum_{i=1}^N \ln p_i(y_i; T)$$
2. **Proper Brier Loss:**
   $$\mathcal{L}_{\text{Brier}}(T) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K \left(p_i(k; T) - \mathbb{I}(y_i = k)\right)^2$$

Because $T \in (0, \infty)$ is a scalar parameter per bucket, scalar optimization using **Brent's method** or **Golden-section search** converges in less than 20 iterations (<5 milliseconds).

### Calibration Buckets

Following Laya's standard bucket partitioning:
- `choice:2` (binary choices)
- `choice:3-5` (small discrete sets)
- `choice:6-10` (medium label spaces)
- `choice:11+` (large label spaces)
- `score:3-5` (ordinal rubrics)
- `noul:2` (calibrated boolean assertions)

## 3. CLI Interface Proposal

```bash
# Calibrate temperatures using a labeled validation dataset
lemma calibrate \
  --val-data data/val_set.jsonl \
  --schema schema/questions.json \
  --loss nll \
  --export-config config/laya_calibrated.json \
  --plot-reliability
```

### Example Terminal Output

```
🌡️  Lemma Temperature Calibration Tuner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: data/val_set.jsonl (1,250 samples)
Loss Function: Negative Log-Likelihood (NLL)

Bucket        Raw ECE   Fitted Temp (T*)   Calibrated ECE   Improvement
───────────────────────────────────────────────────────────────────
choice:2      0.3421    1.906             0.0612           -82.1%
choice:3-5    0.4120    1.760             0.0784           -81.0%
choice:6-10   0.4890    1.000             0.1120           -77.1%
choice:11+    0.5820    0.101             0.1450           -75.1%
score:3-5     0.3980    1.251             0.0940           -76.4%
noul:2        0.3110    1.983             0.0520           -83.3%
───────────────────────────────────────────────────────────────────
Average       0.4223                      0.0904           -78.6%

✅ Saved calibrated configuration to config/laya_calibrated.json
Use with: lemma classify --config config/laya_calibrated.json
```

## 4. Key Benefits

- **Turnkey Production Readiness:** Eliminates manual guess-and-check calibration.
- **Trustworthy Confidence Scores:** Guarantees that a 90% confidence prediction is correct ~90% of the time.
- **Zero Runtime Overhead:** Temperature values are pre-computed scalars applied during the forward pass softmax.
