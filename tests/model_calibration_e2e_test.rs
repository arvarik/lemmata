use std::path::Path;

use lemmata::datasource::{DataSource, FileSource};
use lemmata::engine::classifier::{ClassifierConfig, LayaClassifier};
use lemmata::engine::model_calibration::{
    MODEL_CALIBRATION_FILENAME, apply_persisted_calibration, calibrate_and_persist_checkpoints,
    calibration_path_for, load_checkpoint_calibration_from, save_checkpoint_calibration_at,
};
use lemmata::load_questions_from_file;
use lemmata::model::Checkpoint;

#[test]
fn test_calibration_path_for_heterogeneous_checkpoints() {
    assert_eq!(
        calibration_path_for(Checkpoint::DebertaV3ZeroShot),
        Path::new("models/deberta-v3/lemmata_calibration.json")
    );
    assert_eq!(
        calibration_path_for(Checkpoint::ModernBertZeroShot),
        Path::new("models/modernbert-zeroshot/lemmata_calibration.json")
    );
    assert_eq!(
        calibration_path_for(Checkpoint::EuroBert),
        Path::new("models/eurobert/lemmata_calibration.json")
    );
    assert_eq!(
        calibration_path_for(Checkpoint::NeoBert),
        Path::new("models/neobert/lemmata_calibration.json")
    );
}

#[test]
fn test_calibrate_and_persist_checkpoints_e2e_with_qats() {
    let items = FileSource::new(Path::new("data/system1_golden_benchmark.jsonl"))
        .read_items()
        .expect("Failed to read golden benchmark items");
    let questions = load_questions_from_file(Path::new("schema/system1_benchmark_questions.json"))
        .expect("Failed to load benchmark questions");

    let checkpoints = [
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::ModernBertZeroShot,
        Checkpoint::EuroBert,
        Checkpoint::NeoBert,
    ];

    let temp_dir =
        std::env::temp_dir().join(format!("lemmata_model_cal_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);

    let reports =
        // The 24-row benchmark is below the default sample minimum, so opt in to fitting it.
        calibrate_and_persist_checkpoints(&checkpoints, &items, &questions, true, 2, Some(&temp_dir))
            .expect("calibrate_and_persist_checkpoints failed");

    assert_eq!(reports.len(), 4);

    for ckpt in checkpoints {
        let report = reports
            .get(ckpt.repo_id())
            .unwrap_or_else(|| panic!("Missing calibration report for {}", ckpt.repo_id()));

        assert!(
            report.average_calibrated_ece <= report.average_raw_ece + 1e-9,
            "Expected non-increasing average ECE for {}: raw={:.4}, cal={:.4}",
            ckpt.repo_id(),
            report.average_raw_ece,
            report.average_calibrated_ece
        );

        for bucket in &report.buckets {
            assert!(
                bucket.calibrated_ece <= bucket.raw_ece + 1e-9,
                "Expected non-increasing bucket ECE for {} [{}]: raw={:.4}, cal={:.4}",
                ckpt.repo_id(),
                bucket.bucket,
                bucket.raw_ece,
                bucket.calibrated_ece
            );
            assert!(
                bucket.calibrated_nll <= bucket.raw_nll + 1e-9,
                "Expected non-increasing bucket NLL for {} [{}]: raw={:.4}, cal={:.4}",
                ckpt.repo_id(),
                bucket.bucket,
                bucket.raw_nll,
                bucket.calibrated_nll
            );
        }

        let ckpt_dir = temp_dir.join(ckpt.short_name());
        let expected_file = ckpt_dir.join(MODEL_CALIBRATION_FILENAME);
        assert!(
            expected_file.exists(),
            "Expected persisted calibration file at {}",
            expected_file.display()
        );

        let mut loaded = load_checkpoint_calibration_from(&ckpt_dir)
            .expect("Failed to roundtrip load persisted calibration");
        assert_eq!(loaded.checkpoint, ckpt.repo_id());

        let qats = loaded
            .qats
            .as_ref()
            .expect("Expected Q-ATS profile to be present when enable_qats=true");
        assert!(qats.enabled);
        assert!((qats.gamma - 0.30).abs() < 1e-9);
        assert!(!qats.class_centroids.is_empty());

        // Also verify cost_weights and temperature propagation through save/load/apply
        loaded.temperature = 1.15;
        loaded.cost_weights.insert("security".to_string(), 3.0);
        let saved_path = save_checkpoint_calibration_at(&ckpt_dir, &loaded)
            .expect("save_checkpoint_calibration_at failed");
        assert_eq!(saved_path, expected_file);

        let reloaded = load_checkpoint_calibration_from(&ckpt_dir)
            .expect("Failed to reload updated calibration");

        let mut config = ClassifierConfig::with_checkpoint(ckpt);
        apply_persisted_calibration(&mut config, &reloaded);

        assert!((config.temperature - 1.15).abs() < 1e-9);
        assert_eq!(config.cost_weights.get("security"), Some(&3.0));
        assert!(config.qats_profile.is_some());
        for (bucket_key, &temp_val) in &reloaded.temperature_by_options {
            assert_eq!(
                config.temperature_by_options.get(bucket_key).copied(),
                Some(temp_val)
            );
        }

        // Verify calibrated classifier runs end-to-end with Q-ATS enabled
        let classifier = LayaClassifier::new(config);
        let decision = classifier
            .classify(&items[0].state, &questions)
            .expect("Classification with persisted calibration failed");
        assert_eq!(decision.answers.len(), questions.len());
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}
