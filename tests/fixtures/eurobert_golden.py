"""Write tests/fixtures/eurobert_golden.json from the EuroBERT reference code.

The model comes from transformers' AutoModelForMaskedLM with the remote code in
models/eurobert (modeling_eurobert.py). The Rust parity test
(tests/eurobert_parity_test.rs) compares lemma's EuroBertMlm with these token IDs
and raw logits.

Usage (needs torch, transformers 4.48, safetensors, numpy; the remote code does
not load with transformers 5):
    python tests/fixtures/eurobert_golden.py models/eurobert tests/fixtures/eurobert_golden.json
"""
import json
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_dir, out_path = sys.argv[1], sys.argv[2]
TOP_K = 20
# Low, middle and high vocabulary rows, including the last one and the EOS token.
FIXED_IDS = [13, 279, 50000, 128001, 128255]

PARAGRAPH = (
    "Paris has been the capital of France for most of the last thousand years. The city grew on "
    "both banks of a wide river, and its oldest district sits on a small island in the middle of "
    "the water. During the Middle Ages, Paris became a centre of learning, and students from across "
    "Europe came to study at its university. Builders raised great churches, bridges and markets, "
    "and the city walls moved outward again and again as the population increased. In the "
    "seventeenth century, the kings moved their court to Versailles, but Paris remained the heart "
    "of trade, printing and political debate. The revolution of 1789 began in its streets, and "
    "the city saw many uprisings during the century that followed. In the 1850s and 1860s, the "
    "prefect Haussmann cut long avenues through the crowded medieval quarters, added parks and "
    "sewers, and gave the city the stone facades that visitors still recognise today. The Eiffel "
    "Tower was built for the world fair of 1889, and at first many artists said that it was ugly. "
    "Today the city has more than two million residents, and its metropolitan area holds about a "
    "fifth of the national population. Millions of tourists visit its museums every year, and the "
    "Louvre is the most visited art museum in the world. Boats carry visitors past the cathedral "
    "of Notre-Dame, which was badly damaged by a fire in 2019 and reopened in 2024. The river that "
    "flows through the centre of Paris is called the <|mask|>."
)
CASES = [
    ("english_cloze", "The capital of France is <|mask|>."),
    ("french", "Le chat dort tranquillement sur le <|mask|> du salon."),
    ("long_paragraph", PARAGRAPH),
    ("two_masks", "She opened the <|mask|> and poured a cup of hot <|mask|>."),
]

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMaskedLM.from_pretrained(
    model_dir, trust_remote_code=True, torch_dtype=torch.float32
).eval()

rows = []
for name, text in CASES:
    ids = tok(text)["input_ids"]
    masks = [i for i, t in enumerate(ids) if t == tok.mask_token_id]
    assert masks, name
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    preds = []
    for p in masks:
        top = logits[p].topk(TOP_K)
        preds.append({
            "position": p,
            "top_ids": top.indices.tolist(),
            "top_logits": top.values.tolist(),
            "fixed_ids": FIXED_IDS,
            "fixed_logits": logits[p, FIXED_IDS].tolist(),
        })
        print(name, len(ids), "tokens, mask", p, "->", tok.convert_ids_to_tokens(top.indices[:3].tolist()))
    rows.append({"name": name, "text": text, "input_ids": ids, "mask_positions": masks, "predictions": preds})

with open(out_path, "w") as f:
    json.dump({"mask_token_id": tok.mask_token_id, "cases": rows}, f)
print("wrote", out_path)
