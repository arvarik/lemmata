# Feature Specification: Active Learning & Supervisor Escalation Pipeline

## 1. Problem Statement & Motivation

A signature capability of the Laya model architecture is its built-in **Act/Escalate head**.

Unlike conventional language models that output only text without calibrated awareness of uncertainty, Laya's decision head simultaneously generates:
1. **Option logits:** Calibrated probabilities over the candidate answers.
2. **Act logits:** A dedicated scalar probability determining whether the system should execute autonomously or escalate to human review.

The Act head is conditioned on the pooled sequence representation and an information-theoretic summary of the answer distribution (top-2 probability gap, normalized entropy, and option cardinality $K$).

In production environments, organizations need a closed feedback loop:
- High-confidence items execute automatically.
- Ambiguous or escalated items are dispatched to human supervisors.
- Human review decisions are collected to continuously fine-tune the model using Laya's RLCD reinforcement learning pipeline.

## 2. System Architecture

```
                                  Incoming Request
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │      Lemma Engine      │
                            │  (Laya Classification) │
                            └────────────┬───────────┘
                                         │
                         Decision + Act Probability P(act)
                                         │
                         Is P(act) > Escalate Threshold?
                                         │
                     ┌───────────────────┴───────────────────┐
                     │ NO                                    │ YES
                     ▼                                       ▼
        ┌─────────────────────────┐             ┌─────────────────────────┐
        │  Automated Execution    │             │   Human-in-the-Loop     │
        │  (Fast-path sub-35ms)   │             │   Supervisor Queue      │
        └─────────────────────────┘             └────────────┬────────────┘
                                                             │
                                                    Human Supervisor
                                                    Labels True Action
                                                             │
                                                             ▼
                                                ┌─────────────────────────┐
                                                │   RLCD Fine-Tuning Split│
                                                │ (Active Learning Corpus)│
                                                └────────────┬────────────┘
                                                             │
                                                             ▼
                                                ┌─────────────────────────┐
                                                │ Periodic Checkpoint     │
                                                │ Fine-Tuning & Deploy    │
                                                └─────────────────────────┘
```

## 3. CLI Interface Proposal

### 1. Run Classification with Automated Escalation Routing

```bash
lemma classify \
  --data-dir data/incoming_tickets \
  --schema schema/questions.json \
  --escalate-threshold 0.65 \
  --escalate-output data/quarantine/review_queue.jsonl \
  --auto-output data/processed/executed.jsonl
```

### 2. Export Active Learning Training Split

Once human supervisors complete their reviews in `review_queue.jsonl`:

```bash
# Prepare training dataset formatted for Laya's RLCD fine-tuning notebook
lemma export-training \
  --reviewed-data data/quarantine/reviewed.jsonl \
  --output data/finetune/rlcd_dataset.jsonl \
  --split-ratio 0.8
```

## 4. Key Benefits

- **Risk Mitigation:** Prevents catastrophic mistakes on out-of-distribution inputs (e.g. legal threats, severe security incidents).
- **Reduced Human Labor:** Only 5–15% of ambiguous cases require human attention; the remaining 85–95% are automated safely.
- **Continuous Domain Specialization:** Harvests the exact hard edge-cases where the model is uncertain, maximizing the sample efficiency of subsequent fine-tuning rounds.
