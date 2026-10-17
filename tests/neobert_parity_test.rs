//! The NeoBERT port must match the Python reference code.
//!
//! `tests/fixtures/neobert_golden.json` comes from `tests/fixtures/neobert_golden.py`.
//! The test needs the weights in `models/neobert`, so it skips on CI.

use std::path::Path;

use lemma::model::neobert::NeoBertMlm;
use tokenizers::Tokenizer;

#[test]
fn neobert_mlm_matches_python_reference() {
    let dir = Path::new("models/neobert");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping neobert: no weights in {}", dir.display());
        return;
    }
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/neobert_golden.json")).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let model = NeoBertMlm::load(dir).unwrap();
    let mask_id = golden["mask_token_id"].as_u64().unwrap() as u32;
    assert_eq!(tokenizer.token_to_id("[MASK]"), Some(mask_id));

    let mut max_diff = 0.0f32;
    for case in golden["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ids = tokenizer
            .encode(case["text"].as_str().unwrap(), true)
            .unwrap()
            .get_ids()
            .to_vec();
        let expected_ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        assert_eq!(ids, expected_ids, "{name}: input ids");

        let masks = case["masks"].as_array().unwrap();
        let positions: Vec<usize> = masks
            .iter()
            .map(|m| m["position"].as_u64().unwrap() as usize)
            .collect();
        assert!(
            positions.iter().all(|&p| ids[p] == mask_id),
            "{name}: mask positions"
        );
        let rows = model.mlm_logits_at(&ids, &positions).unwrap();
        assert_eq!(rows.len(), positions.len(), "{name}: row count");

        for (row, m) in rows.iter().zip(masks) {
            for (ids_key, logits_key) in [("top_ids", "top_logits"), ("fixed_ids", "fixed_logits")]
            {
                let want_ids: Vec<usize> = serde_json::from_value(m[ids_key].clone()).unwrap();
                let want: Vec<f32> = serde_json::from_value(m[logits_key].clone()).unwrap();
                for (&id, &w) in want_ids.iter().zip(&want) {
                    let diff = (row[id] - w).abs();
                    max_diff = max_diff.max(diff);
                    assert!(diff < 1e-3, "{name}: token {id} logit {} != {w}", row[id]);
                }
            }
        }
    }
    eprintln!("neobert: max logit difference {max_diff:.2e}");
}
