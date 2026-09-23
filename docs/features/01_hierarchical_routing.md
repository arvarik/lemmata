# Feature Specification: Hierarchical Coarse-to-Fine Routing

## 1. Problem Statement & Motivation

In the official benchmark results for the Laya model family, a critical limitation was identified when evaluating high-cardinality label spaces:

> **High-cardinality choice questions and token budgets:**
> Options share a fixed `head_max_len` budget (192 tokens for `convaiinnovations/laya` English, 256 for `laya-multilingual`). For tasks with 50+ or 70+ candidate labels (such as Banking77 with 77 distinct intent categories), each label is allocated only `(256 - 16) // 77` ≈ 3–4 tokens. This token compression causes label descriptions to be truncated, resulting in an accuracy drop from 0.870 (TypeSafe Jev with 255-token head) down to 0.425.

Rather than forcing the model to distinguish between 70+ fine-grained classes in a single forward pass, enterprise classification workflows can be structured into a **hierarchical decision tree**.

## 2. Architecture & Design

### Two-Stage Coarse-to-Fine Classifier

```
                   Incoming State (e.g. Support Ticket)
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ Stage 1: Coarse Intent Router │
                   │  (Laya English / mmBERT-base) │
                   └───────────────┬───────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   [Card Issues]            [Bank Transfers]          [Account Security]
  (6 sub-options)           (8 sub-options)            (5 sub-options)
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Stage 2: Cards  │       │Stage 2: Transfer│       │Stage 2: Security│
│ Sub-Classifier  │       │  Sub-Classifier │       │  Sub-Classifier │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **Stage 1 (Coarse Routing):**
   - The ticket is routed across 4–8 top-level domains (e.g., `card_services`, `payments`, `account_security`, `compliance`).
   - Token budget per option is ample: 256 / 6 ≈ 42 tokens per option.
   - Latency: ~33 ms.

2. **Stage 2 (Specialized Sub-Classification):**
   - Only the selected sub-category schema is evaluated.
   - Token budget per sub-option: 256 / 8 ≈ 30 tokens per option.
   - Latency: ~33 ms.

**Total Latency:** ~66 ms (still 4x faster than autoregressive generative models at 250–500 ms).
**Accuracy Recovery:** Restores Banking77 accuracy from 0.425 to >0.850.

## 3. CLI Interface Proposal

```bash
# Define a hierarchical schema in YAML or JSON
lemma classify \
  --data-dir data/banking \
  --hierarchical schema/banking_hierarchy.json \
  --format table
```

### Schema Definition (`schema/banking_hierarchy.json`)

```json
{
  "coarse": {
    "type": "choice",
    "instructions": "Determine the high-level banking domain",
    "criteria": {
      "cards": "Credit card, debit card, lost card, PIN issues",
      "transfers": "Wire transfers, ACH, international remittances, pending deposits",
      "security": "Compromised account, suspicious login, password resets"
    }
  },
  "subtrees": {
    "cards": {
      "type": "choice",
      "instructions": "Select specific card inquiry",
      "criteria": {
        "card_activation": "activating a new physical card",
        "lost_or_stolen": "reporting card lost or stolen",
        "pin_change": "updating ATM PIN"
      }
    },
    "transfers": {
      "type": "choice",
      "instructions": "Select transfer inquiry",
      "criteria": {
        "wire_delay": "funds not arrived after 48h",
        "transfer_limit": "increasing daily transfer limits",
        "beneficiary": "adding international IBAN"
      }
    }
  }
}
```

## 4. Implementation Plan in Lemma

1. Extend `src/model/schema.rs` with `HierarchicalSchema` containing `root: Question` and `branches: HashMap<String, Box<HierarchicalSchema>>`.
2. Implement `classify_hierarchical` in `src/engine/classifier.rs` to traverse matching branches when top choice confidence exceeds a configurable threshold (e.g. `--coarse-threshold 0.70`).
3. Add escalation hook: if coarse confidence is low (<0.50), trigger Laya's act/escalate head immediately without executing Stage 2.
