"""Write tests/fixtures/neobert_golden.json from the NeoBERT reference code (model.py).

The Rust parity test (tests/neobert_parity_test.rs) compares lemmata's NeoBERT port with
these token IDs and masked-LM logits: the top 20 tokens at each [MASK], plus a few fixed
tokens.

Usage (needs torch, transformers, safetensors; model.py imports xformers.ops.SwiGLU):
    python tests/fixtures/neobert_golden.py models/neobert tests/fixtures/neobert_golden.json
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_dir, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32).eval()
# from_pretrained only warns about missing keys, so check every weight loads by name.
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
# transformers 5 builds the model on the meta device, so the non-persistent freqs_cis buffer
# comes back uninitialized. Recompute it exactly as NeoBERT.__init__ does.
cfg = model.config
remote = sys.modules[type(model).__module__]
model.model.freqs_cis = remote.precompute_freqs_cis(cfg.hidden_size // cfg.num_attention_heads, cfg.max_length)

PARAGRAPH = " ".join([
    "Our team migrated the billing service to a new cluster last weekend.",
    "Since then, several customers have reported that their monthly invoices show duplicate charges.",
    "The support queue grew from forty tickets a day to more than three hundred.",
    "Engineers traced the problem to a retry loop that resubmitted payment requests after a timeout.",
    "The loop did not check whether the first request had already succeeded.",
    "We disabled the retry logic on Tuesday morning and started a script to find every affected account.",
    "Finance approved refunds for all duplicate charges, and the first batch went out on Wednesday.",
    "Customers who paid by bank transfer will need to wait longer, because those refunds are processed by hand.",
    "We also added an alert that fires when the payment error rate rises above two percent.",
    "The incident review is scheduled for Friday, and every team that touched the migration will attend.",
    "Until the review is complete, all deployments to the billing cluster are frozen.",
    "If a customer writes in about a double charge, agents should apologize, confirm the amount, and promise a refund within five business days.",
    "Agents should not offer store credit instead of a refund, because finance cannot reconcile credits against the duplicate payments.",
    "Please escalate any ticket where the customer was charged three or more times.",
    "Those accounts may have hit a second bug in the invoice generator, and the engineers want to inspect them directly.",
    "We will send a written apology to every affected customer once all refunds have cleared.",
    "The apology will include a discount code for next month, approved by the head of customer success.",
    "Thank you all for your patience during a stressful week.",
    "In short, the root cause was a faulty retry loop, and the fix is to issue a full [MASK] to every customer.",
])
CASES = [
    ("short_cloze", "The capital of France is [MASK]."),
    ("support_ticket", "Hi, I was charged twice for my subscription this month, so I would like a [MASK] for the extra payment."),
    ("long_paragraph", PARAGRAPH),
    ("two_masks", "The [MASK] barked at the mailman, and then it ran back into the [MASK]."),
]
FIXED_TOKENS = ["paris", "refund", "the", ".", "[UNK]"]
fixed_ids = tok.convert_tokens_to_ids(FIXED_TOKENS)

rows = []
for name, text in CASES:
    ids = tok(text)["input_ids"]
    positions = [i for i, t in enumerate(ids) if t == tok.mask_token_id]
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        # model.py passes attention_mask straight to SDPA, so it needs one even without padding.
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]
    masks = []
    for p in positions:
        top = torch.topk(logits[p], 20)
        masks.append({
            "position": p,
            "top_ids": top.indices.tolist(),
            "top_logits": top.values.tolist(),
            "fixed_ids": fixed_ids,
            "fixed_logits": logits[p, fixed_ids].tolist(),
        })
    print(name, len(ids), "tokens, top-1 at masks:", [tok.convert_ids_to_tokens(m["top_ids"][0]) for m in masks])
    rows.append({"name": name, "text": text, "input_ids": ids, "masks": masks})

with open(out_path, "w") as f:
    json.dump({"mask_token_id": tok.mask_token_id, "cases": rows}, f)
print("wrote", out_path)
