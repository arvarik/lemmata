//! Native Laya inference must match the upstream Python reference.
//!
//! `tests/fixtures/laya_golden.json` comes from `tests/fixtures/laya_golden.py`.
//! The test needs the weights in `models/laya`, so it skips on CI.

use lemma::model::native::LayaModel;
use lemma::model::{Checkpoint, Question};

#[test]
fn native_inference_matches_python_reference() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/laya_golden.json")).unwrap();
    for (name, ckpt) in [
        ("english", Checkpoint::English),
        ("multilingual", Checkpoint::Multilingual),
        ("typed-decisions", Checkpoint::TypedDecisions),
    ] {
        let dir = ckpt.local_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping {name}: no weights in {}", dir.display());
            continue;
        }
        let model = LayaModel::load(&dir).unwrap();
        let mut max_diff = 0.0f64;
        for case in golden[name].as_array().unwrap() {
            let case_name = case["name"].as_str().unwrap();
            let q: Question = serde_json::from_value(case["question"].clone()).unwrap();
            let options = q.render_options();
            let expected_options: Vec<String> =
                serde_json::from_value(case["options"].clone()).unwrap();
            assert_eq!(options, expected_options, "{name}/{case_name}: options");

            let state = case["state"].as_str().unwrap();
            let (ids, markers) = model
                .build_sequence(state, q.qtype, &q.instructions, &options)
                .unwrap();
            let expected_ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let expected_markers: Vec<usize> =
                serde_json::from_value(case["markers"].clone()).unwrap();
            assert_eq!(ids, expected_ids, "{name}/{case_name}: input ids");
            assert_eq!(markers, expected_markers, "{name}/{case_name}: markers");

            let logits = model
                .option_logits(state, q.qtype, &q.instructions, &options)
                .unwrap();
            let expected: Vec<f64> = serde_json::from_value(case["logits"].clone()).unwrap();
            assert_eq!(
                logits.len(),
                expected.len(),
                "{name}/{case_name}: option count"
            );
            for (got, want) in logits.iter().zip(&expected) {
                max_diff = max_diff.max((got - want).abs());
                assert!(
                    (got - want).abs() < 1e-3,
                    "{name}/{case_name}: logits {logits:?} != {expected:?}"
                );
            }
        }
        eprintln!("{name}: max logit difference {max_diff:.2e}");
    }
}
