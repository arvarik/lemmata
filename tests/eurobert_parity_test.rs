//! EuroBERT masked-LM inference must match transformers.
//!
//! `tests/fixtures/eurobert_golden.json` comes from `tests/fixtures/eurobert_golden.py`.
//! The test needs the weights in `models/eurobert`, so it skips on CI.

use std::path::Path;

use lemma::model::eurobert::EuroBertMlm;
use tokenizers::Tokenizer;

#[test]
fn eurobert_mlm_matches_transformers() {
    let dir = Path::new("models/eurobert");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping eurobert: no weights in {}", dir.display());
        return;
    }
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/eurobert_golden.json")).unwrap();
    let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    // tokenizer.json sets BatchLongest padding. The HF Python tokenizer pads only when asked.
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).unwrap();
    let model = EuroBertMlm::load(dir).unwrap();

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

        let positions: Vec<usize> = serde_json::from_value(case["mask_positions"].clone()).unwrap();
        let rows = model.mlm_logits_at(&ids, &positions).unwrap();
        assert_eq!(rows.len(), positions.len(), "{name}: row count");
        for (row, pred) in rows.iter().zip(case["predictions"].as_array().unwrap()) {
            let mut check = |id_key: &str, logit_key: &str| {
                let ids: Vec<usize> = serde_json::from_value(pred[id_key].clone()).unwrap();
                let want: Vec<f32> = serde_json::from_value(pred[logit_key].clone()).unwrap();
                for (&id, &w) in ids.iter().zip(&want) {
                    let diff = (row[id] - w).abs();
                    max_diff = max_diff.max(diff);
                    assert!(diff < 1e-3, "{name}: token {id} logit {} != {w}", row[id]);
                }
            };
            check("top_ids", "top_logits");
            check("fixed_ids", "fixed_logits");
            // No other token may outscore the reference's best one.
            let best = row.iter().copied().fold(f32::MIN, f32::max);
            let want_best = pred["top_logits"][0].as_f64().unwrap() as f32;
            assert!(
                best < want_best + 1e-3,
                "{name}: max logit {best} > {want_best}"
            );
        }
    }
    eprintln!("eurobert: max logit difference {max_diff:.2e}");
}
