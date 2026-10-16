use lemma::datasource::{DataItem, FileSource};
use lemma::engine::slicing::{SliceBy, compute_arena_slice_matrix_with_options};
use lemma::engine::{ClassifierConfig, ExperimentRunner, LayaClassifier};
use lemma::model::{Checkpoint, ChunkStrategy, Downloader, Question, QuestionType};
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn test_auto_best_checkpoint_consistency_between_classify_and_option_logits() {
    let q = Question::new_choice(
        "Classify department",
        vec!["billing".into(), "technical".into(), "security".into()],
    );
    let options = q.render_options();
    let state_text =
        "We were billed twice on our monthly invoice; please refund the duplicate charge.";

    let auto_clf = LayaClassifier::with_auto_best();
    let deb_clf = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot);
    let en_clf = LayaClassifier::with_checkpoint(Checkpoint::English);

    // Under auto-best, a short 3-way Choice question routes to DebertaV3ZeroShot
    let mut questions = BTreeMap::new();
    questions.insert("department".to_string(), q.clone());
    let (resolved, reason) = auto_clf.resolve_checkpoint_for_questions(state_text, &questions);
    assert_eq!(resolved, Checkpoint::DebertaV3ZeroShot);
    assert_eq!(reason, "short_nli_disentangled_attn");

    let auto_logits = auto_clf.compute_option_logits(state_text, &q, &options);
    let deb_logits = deb_clf.compute_option_logits(state_text, &q, &options);
    let en_logits = en_clf.compute_option_logits(state_text, &q, &options);
    let explicit_ckpt_logits = auto_clf.compute_option_logits_for_checkpoint(
        state_text,
        &q,
        &options,
        Checkpoint::DebertaV3ZeroShot,
    );

    assert_eq!(auto_logits, deb_logits);
    assert_eq!(auto_logits, explicit_ckpt_logits);
    assert_ne!(auto_logits, en_logits);

    let res = auto_clf
        .classify(
            &serde_json::Value::String(state_text.to_string()),
            &questions,
        )
        .unwrap();
    assert_eq!(res.model, Checkpoint::DebertaV3ZeroShot.repo_id());
    assert_eq!(res.answers["department"].choice.as_deref(), Some("billing"));
}

#[test]
fn test_truncate_prefix_multibyte_cjk_without_whitespace_does_not_panic() {
    // 60 repetitions of 22 3-byte CJK characters (>1300 tokens, 0 whitespace chars).
    // Byte index 512 * 4 = 2048 is not divisible by 3 (2048 % 3 == 2), which previously panicked.
    let long_cjk = "请立即退还重复收取的发票账单费用并处理严重故障".repeat(60);
    let q = Question::new_choice(
        "Department",
        vec!["billing".into(), "technical".into(), "security".into()],
    );

    let clf = LayaClassifier::with_checkpoint(Checkpoint::English)
        .with_chunk_strategy(ChunkStrategy::TruncatePrefix);
    let ans = clf.classify_single(&long_cjk, &q).unwrap();
    assert!(ans.choice.is_some());
    assert!(ans.confidence > 0.0);
}

#[test]
fn test_explicit_temperature_overrides_default_bucket_temperatures() {
    let q = Question::new_choice(
        "Route ticket",
        vec!["billing".into(), "technical".into(), "security".into()],
    );
    let state = "We need a refund for a duplicate charge on our April invoice.";

    // 1. Builder .with_temperature()
    let sharp_clf = LayaClassifier::new(
        ClassifierConfig::with_checkpoint(Checkpoint::English).with_temperature(0.25),
    );
    let smooth_clf = LayaClassifier::new(
        ClassifierConfig::with_checkpoint(Checkpoint::English).with_temperature(3.0),
    );

    let sharp_ans = sharp_clf.classify_single(state, &q).unwrap();
    let smooth_ans = smooth_clf.classify_single(state, &q).unwrap();

    assert!(
        sharp_ans.probabilities["billing"] > smooth_ans.probabilities["billing"],
        "Expected T=0.25 billing prob ({}) > T=3.0 billing prob ({})",
        sharp_ans.probabilities["billing"],
        smooth_ans.probabilities["billing"]
    );
    assert!(sharp_ans.confidence > smooth_ans.confidence);

    // 2. Direct struct field assignment (cfg.temperature = 0.25) without clearing default temperature_by_options
    let mut direct_sharp_cfg = ClassifierConfig::with_checkpoint(Checkpoint::English);
    direct_sharp_cfg.temperature = 0.25;
    let mut direct_smooth_cfg = ClassifierConfig::with_checkpoint(Checkpoint::English);
    direct_smooth_cfg.temperature = 3.0;

    let direct_sharp = LayaClassifier::new(direct_sharp_cfg)
        .classify_single(state, &q)
        .unwrap();
    let direct_smooth = LayaClassifier::new(direct_smooth_cfg)
        .classify_single(state, &q)
        .unwrap();
    assert!(direct_sharp.probabilities["billing"] > direct_smooth.probabilities["billing"]);
}

#[test]
fn test_mlm_verbalizer_contrast_scoring_works_without_custom_hypothesis_template() {
    let q = Question {
        qtype: QuestionType::Choice,
        instructions: "Classify support ticket department".into(),
        criteria: Some(serde_json::json!({
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, crashes",
            "security": "phishing, unauthorized, credentials",
            "account": "login, username, sso, permissions"
        })),
        hypothesis_template: None,
    };

    let contrast_state = "To clarify: we were NOT charged twice and there is no invoice issue. Instead, our production service is crashing with critical database exception errors!";

    for &mlm_ckpt in &[Checkpoint::EuroBert, Checkpoint::NeoBert] {
        let clf = LayaClassifier::with_checkpoint(mlm_ckpt);
        assert!(clf.config().hypothesis_template.is_none());
        let ans = clf.classify_single(contrast_state, &q).unwrap();
        assert_eq!(
            ans.choice.as_deref(),
            Some("technical"),
            "Expected {mlm_ckpt:?} to resolve post-contrast technical clause without requiring custom hypothesis_template"
        );
    }
}

#[test]
fn test_downloader_verify_checkpoint_no_false_positive_on_missing_dir() {
    let dl = Downloader {
        cli_binary: "hf".into(),
    };

    let rep = dl
        .verify_checkpoint(Checkpoint::EuroBert, Path::new("/tmp/nonexistent_laya_dir"))
        .unwrap();
    assert!(!rep.is_valid);
    assert!(!rep.missing.is_empty());

    // Even if a sibling root directory has config.json + model.safetensors,
    // verifying EuroBert when the target subdir does not exist must not report valid.
    let tmp_root = std::env::temp_dir().join("lemma_verify_ckpt_bugfix_test");
    let _ = std::fs::remove_dir_all(&tmp_root);
    std::fs::create_dir_all(&tmp_root).unwrap();
    std::fs::write(tmp_root.join("config.json"), "{}").unwrap();
    std::fs::write(tmp_root.join("model.safetensors"), "stub").unwrap();
    std::fs::write(tmp_root.join("tokenizer.json"), "{}").unwrap();

    let non_existent_sub = tmp_root.join("missing_eurobert");
    let rep_missing = dl
        .verify_checkpoint(Checkpoint::EuroBert, &non_existent_sub)
        .unwrap();
    assert!(!rep_missing.is_valid);

    let _ = std::fs::remove_dir_all(&tmp_root);
}

#[test]
fn test_arena_and_slice_matrix_with_options_and_checkpoint_temperature_sweep() {
    let items = FileSource::parse_file(Path::new("data/system1_golden_benchmark.jsonl")).unwrap();
    let schema_str = std::fs::read_to_string("schema/system1_benchmark_questions.json").unwrap();
    let questions: BTreeMap<String, Question> = serde_json::from_str(&schema_str).unwrap();

    let long_items: Vec<DataItem> = items
        .into_iter()
        .filter(|i| i.id.starts_with("long_") || i.id == "en_01" || i.id == "en_02")
        .collect();

    let checkpoints = [
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::ModernBertZeroShot,
    ];

    let arena = ExperimentRunner::run_arena_with_options(
        &long_items,
        &questions,
        &checkpoints,
        3,
        ChunkStrategy::HeadTail,
        Some("This ticket is about {label} ({description})."),
        4,
    )
    .unwrap();
    assert_eq!(arena.models.len(), 2);
    assert_eq!(arena.sample_count, long_items.len());
    // With HeadTail chunking, 512-token DeBERTa-v3-ZS sees the incident at the tail of long_01..04
    assert!(arena.models[0].accuracy.unwrap_or(0.0) >= 0.70);

    let (slices, rows) = compute_arena_slice_matrix_with_options(
        &long_items,
        &questions,
        &checkpoints,
        &SliceBy::AutoPrefix,
        ChunkStrategy::HeadTail,
        Some("This ticket is about {label} ({description})."),
        4,
    )
    .unwrap();
    assert!(slices.contains(&"long".to_string()));
    assert_eq!(rows.len(), 2);
    assert!(rows[0].slice_accuracies["long"] >= 0.60);

    ExperimentRunner::sweep_temperature_calibration_for_checkpoint(
        Checkpoint::DebertaV3ZeroShot,
        &long_items,
        &questions,
        &[0.5, 1.0, 2.0],
    )
    .unwrap();
}
