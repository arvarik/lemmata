//! Native inference must match the Python reference code.
//!
//! `tests/fixtures/native_golden.json` comes from `tests/fixtures/native_golden.py`.
//! The tests need the weights in `models/`, so they skip on CI.

use lemma::model::ChunkStrategy;
use lemma::model::native::{LayaModel, NativeModel, NliModel, ScoreRequest};
use lemma::model::{Checkpoint, Question};

#[test]
fn native_inference_matches_python_reference() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/native_golden.json")).unwrap();
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
        // Every case as one padded batch (mixed question types and lengths) must match too.
        let cases = golden[name].as_array().unwrap();
        let questions: Vec<Question> = cases
            .iter()
            .map(|c| serde_json::from_value(c["question"].clone()).unwrap())
            .collect();
        let options: Vec<Vec<String>> = questions.iter().map(|q| q.render_options()).collect();
        let reqs: Vec<ScoreRequest> = cases
            .iter()
            .zip(&questions)
            .zip(&options)
            .map(|((c, q), o)| ScoreRequest {
                state: c["state"].as_str().unwrap(),
                question: q,
                options: o,
                strategy: ChunkStrategy::TruncatePrefix,
                hypothesis_template: None,
                multi_label: false,
            })
            .collect();
        let batched = NativeModel::Laya(Box::new(model))
            .score_batch(&reqs)
            .unwrap();
        for (scored, case) in batched.iter().zip(cases) {
            let expected: Vec<f64> = serde_json::from_value(case["logits"].clone()).unwrap();
            for (got, want) in scored.logits.iter().zip(&expected) {
                max_diff = max_diff.max((got - want).abs());
                assert!((got - want).abs() < 1e-3, "{name}: batched {got} != {want}");
            }
        }
        eprintln!("{name}: max logit difference {max_diff:.2e}");
    }
}

#[test]
fn native_nli_matches_transformers() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/native_golden.json")).unwrap();
    for (name, ckpt) in [
        ("modernbert-zeroshot", Checkpoint::ModernBertZeroShot),
        ("modernbert-nli", Checkpoint::ModernBertNli),
        ("deberta-v3", Checkpoint::DebertaV3ZeroShot),
    ] {
        let dir = ckpt.local_dir();
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipping {name}: no weights in {}", dir.display());
            continue;
        }
        let model = NliModel::load(&dir).unwrap();
        let mut max_diff = 0.0f64;
        let mut batch = Vec::new();
        for case in golden[name].as_array().unwrap() {
            let case_name = case["name"].as_str().unwrap();
            let ids = model
                .pair_ids(
                    case["premise"].as_str().unwrap(),
                    case["hypothesis"].as_str().unwrap(),
                )
                .unwrap();
            let expected_ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            assert_eq!(ids, expected_ids, "{name}/{case_name}: input ids");
            batch.push(ids.clone());

            let logits = model.forward(&ids).unwrap();
            let expected: Vec<f64> = serde_json::from_value(case["logits"].clone()).unwrap();
            assert_eq!(
                logits.len(),
                expected.len(),
                "{name}/{case_name}: label count"
            );
            for (got, want) in logits.iter().zip(&expected) {
                max_diff = max_diff.max((got - want).abs());
                assert!(
                    (got - want).abs() < 1e-3,
                    "{name}/{case_name}: logits {logits:?} != {expected:?}"
                );
            }
        }
        // One padded batch of every pair must give each pair's unpadded result.
        for (row, case) in model
            .forward_batch(&batch)
            .unwrap()
            .iter()
            .zip(golden[name].as_array().unwrap())
        {
            let expected: Vec<f64> = serde_json::from_value(case["logits"].clone()).unwrap();
            for (got, want) in row.iter().zip(&expected) {
                max_diff = max_diff.max((got - want).abs());
                assert!(
                    (got - want).abs() < 1e-3,
                    "{name}: batched {row:?} != {expected:?}"
                );
            }
        }
        eprintln!("{name}: max logit difference {max_diff:.2e}");
    }
}

#[test]
fn route_sends_many_labels_to_an_installed_nli_model() {
    use lemma::model::native::{Backend, native_available, set_backend};
    let nli = [
        Checkpoint::ModernBertNli,
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::ModernBertZeroShot,
    ];
    let Some(&expected) = nli.iter().find(|&&c| native_available(c)) else {
        eprintln!("skipping: no NLI weights installed");
        return;
    };
    set_backend(Backend::Auto);
    let labels: Vec<String> = (0..30).map(|i| format!("intent_{i}")).collect();
    let mut questions = std::collections::BTreeMap::new();
    questions.insert(
        "intent".to_string(),
        Question::new_choice("Which intent?", labels),
    );
    let clf = lemma::engine::LayaClassifier::with_checkpoint(Checkpoint::AutoRoute);
    let (picked, reason) = clf.resolve_checkpoint_for_questions("How do I top up?", &questions);
    assert_eq!((picked, reason), (expected, "high_cardinality_nli"));
}
