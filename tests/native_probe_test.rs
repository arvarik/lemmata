//! A probe fits on labelled items and then answers its question through `classify`.
//!
//! Needs weights in `models/`, so it skips on CI.

use std::sync::Arc;

use lemmata::engine::{ClassifierConfig, LayaClassifier, Probe, fit_probe};
use lemmata::model::native::{Backend, native_available, set_backend};
use lemmata::model::{Checkpoint, ChunkStrategy};

#[test]
fn probe_fits_and_replaces_zero_shot_scoring() {
    let ckpt = Checkpoint::English;
    if !native_available(ckpt) {
        eprintln!("skipping: {ckpt} is not installed");
        return;
    }
    set_backend(Backend::Native);
    let questions =
        lemmata::load_questions_from_file("schema/system1_benchmark_questions.json".as_ref())
            .unwrap();
    let items = lemmata::resolve_data_source(
        "data".as_ref(),
        Some("data/system1_golden_benchmark.jsonl".as_ref()),
        false,
        None,
    )
    .unwrap()
    .read_items()
    .unwrap();
    let q = &questions["department"];

    let probe = fit_probe(ckpt, &items, "department", q, ChunkStrategy::TruncatePrefix).unwrap();
    assert_eq!(
        probe.labels,
        ["billing", "technical", "security", "account"]
    );
    assert_eq!(probe.train_samples, items.len());
    assert_eq!(probe.weights.len(), probe.labels.len() * probe.mean.len());

    // Round-trip through JSON, as `lemmata probe fit` and `--probe` do.
    let path = std::env::temp_dir().join(format!("lemmata_probe_{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_string(&probe).unwrap()).unwrap();
    let loaded = Probe::load(&path).unwrap();

    let mut cfg = ClassifierConfig::with_checkpoint(ckpt);
    cfg.probe = Some(Arc::new(loaded));
    let clf = LayaClassifier::new(cfg);
    let mut correct = 0;
    for item in &items {
        let res = clf.classify(&item.state, &questions).unwrap();
        let ans = &res.answers["department"];
        let sum: f64 = ans.probabilities.values().sum();
        assert!((sum - 1.0).abs() < 1e-3, "probabilities sum to {sum}");
        let truth = item.ground_truth.as_ref().unwrap()["department"]
            .as_str()
            .unwrap();
        correct += usize::from(ans.choice.as_deref() == Some(truth));
    }
    // The probe saw these items, so it should fit them almost perfectly.
    assert!(
        correct * 10 >= items.len() * 9,
        "{correct}/{} on its own training items",
        items.len()
    );
    let _ = std::fs::remove_file(path);
}
