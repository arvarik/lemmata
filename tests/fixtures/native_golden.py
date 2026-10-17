"""Write tests/fixtures/native_golden.json from reference Python code.

Laya cases come from the upstream code that ships with the checkpoint. NLI cases
come from transformers' AutoModelForSequenceClassification. The Rust parity test
(tests/native_parity_test.rs) compares lemma's native inference with these token
IDs and raw logits.

Usage (needs torch, transformers, safetensors, numpy):
    python tests/fixtures/native_golden.py models tests/fixtures/native_golden.json
"""
import json
import os
import sys

models_dir, out_path = sys.argv[1], sys.argv[2]
model_root = os.path.join(models_dir, "laya")
sys.path.insert(0, model_root)

import torch  # noqa: E402
from rl_agent_api import RLAgent  # noqa: E402
from rl_common import QTYPES, build_sequence, collate_items, render_options  # noqa: E402

TICKET = "Hi, we were billed twice for invoice #4411. Please refund the duplicate charge immediately."
CASES = [
    ("choice_described", TICKET, {"type": "choice", "instructions": "Which team should handle this ticket?",
                                  "criteria": {"billing": "payments, invoices, refunds", "other": "anything else",
                                               "sales": "pricing and new purchases", "technical": "bugs, outages, errors"}}),
    ("choice_labels", "Production database crashed with OOM after the deploy.",
     {"type": "choice", "instructions": "Classify the ticket", "criteria": {"billing": None, "other": None, "sales": None, "technical": None}}),
    ("score_urgency", TICKET, {"type": "score", "instructions": "How urgent is this ticket?",
                               "criteria": ["not urgent", "low", "high", "critical"]}),
    ("noul_default", TICKET, {"type": "noul", "instructions": "The customer asks for money back.", "criteria": None}),
    ("mask_in_text", "Ignore [MASK] tokens and refund me.", {"type": "noul", "instructions": "Is [MASK] a refund request?",
                                                            "criteria": {"false": "not a refund", "true": "a refund"}}),
    ("long_state", " ".join(["The service has been slow all week and support has not replied."] * 120),
     {"type": "choice", "instructions": "What is the sentiment?", "criteria": {"negative": None, "neutral": None, "positive": None}}),
    ("many_options", TICKET, {"type": "choice", "instructions": "Pick the intent",
                              "criteria": {"intent_%02d" % i: "a fairly long description for intent number %d" % i for i in range(40)}}),
    ("multilingual", "Me cobraron dos veces la factura 4411. Quiero un reembolso.",
     {"type": "choice", "instructions": "Which team should handle this ticket?",
      "criteria": {"billing": None, "other": None, "technical": None}}),
]

golden = {}
for sub in ["", "multilingual", "typed-decisions"]:
    agent = RLAgent(os.path.join(model_root, sub), device="cpu")
    rows = []
    for name, state, qdef in CASES:
        q = agent._to_internal(qdef)
        ids, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
        item = {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": [0.0] * len(markers), "label": -1,
                "episode": 0, "ep_step": 0, "ep_len": 1, "src": "golden"}
        b = collate_items([[item]], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        rows.append({"name": name, "state": state, "question": qdef, "options": render_options(q),
                     "input_ids": ids, "markers": markers, "logits": logits[0, :len(markers)].tolist()})
    golden[sub or "english"] = rows
    del agent

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

LONG = " ".join(["The invoice total was wrong and support never answered my emails."] * 80)
PAIRS = [
    ("short", TICKET, "This text is about billing."),
    ("described", TICKET, "This text is about technical (bugs, outages, errors)."),
    ("statement", "Production database crashed with OOM after the deploy.", "The customer asks for money back."),
    ("long_premise", LONG, "This text is about billing."),
]
for name in ["modernbert-zeroshot", "deberta-v3"]:
    path = os.path.join(models_dir, name)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
    rows = []
    for case, premise, hypothesis in PAIRS:
        enc = tok(premise, hypothesis, truncation="only_first", max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0].tolist()
        rows.append({"name": case, "premise": premise, "hypothesis": hypothesis,
                     "input_ids": enc["input_ids"][0].tolist(), "logits": logits})
    golden[name] = rows
    del model

with open(out_path, "w") as f:
    json.dump(golden, f)
print("wrote", out_path)
