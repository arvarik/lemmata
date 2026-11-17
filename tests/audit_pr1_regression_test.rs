use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use lemmata::datasource::{DataItem, FileSource};
use lemmata::engine::cascade::fit_samples;
use lemmata::engine::{
    AttestationEngine, AttributionEngine, CalibrationLoss, CalibrationTuner, CascadeConfig,
    CascadeRouter, ClassifierConfig, ConformalEngine, DecisionCache, Evaluator, GuardAction,
    HierarchicalRouter, LayaClassifier, SchemaSynthesizer, SliceBy, StatsEngine,
    extract_item_slice, fingerprint,
};
use lemmata::model::{
    Checkpoint, ChunkStrategy, HierarchicalNode, HierarchicalSchema, Question, QuestionType,
    chunk_text_for_window, inspect_safetensors, route_auto_best, safetensors_header_sha256,
};

#[test]
fn test_state_text_unwraps_single_key_text_without_breaking_structured_json() {
    let wrapped = DataItem::new(
        "csv_1",
        serde_json::json!({ "text": "Refund duplicate invoice charge immediately" }),
        "test.csv",
    );
    assert_eq!(
        wrapped.state_text(),
        "Refund duplicate invoice charge immediately"
    );

    let q_map = BTreeMap::from([(
        "dept".to_string(),
        Question::new_choice(
            "Classify ticket",
            vec!["billing".into(), "technical".into(), "sales".into()],
        ),
    )]);
    let route_wrapped = route_auto_best(&wrapped.state_text(), &q_map);
    assert_ne!(
        route_wrapped.reason, "structured_json_payload",
        "Plain text wrapped in single-key {{\"text\": ...}} must not route as structured JSON"
    );
    assert_eq!(route_wrapped.selected, Checkpoint::DebertaV3ZeroShot);

    // Multi-field structured JSON objects still serialize as JSON and route to TypedDecisions
    let structured = DataItem::new(
        "json_1",
        serde_json::json!({ "event": "charge_failed", "amount": 499 }),
        "events.jsonl",
    );
    assert!(structured.state_text().starts_with('{'));
    let route_struct = route_auto_best(&structured.state_text(), &q_map);
    assert_eq!(route_struct.selected, Checkpoint::TypedDecisions);
    assert_eq!(route_struct.reason, "structured_json_payload");
}

#[test]
fn test_csv_bom_camelcase_gt_and_metadata_slicing() {
    let tmp = std::env::temp_dir().join(format!("lemmata_pr1_bom_{}.csv", std::process::id()));
    let csv_body = "\u{feff}id,text,GT_ticketType\nrow_1,Duplicate invoice charge,billing\n";
    std::fs::write(&tmp, csv_body).unwrap();

    let items = FileSource::parse_file(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].id, "row_1");
    assert_eq!(items[0].state_text(), "Duplicate invoice charge");
    let gt = items[0].ground_truth.as_ref().unwrap();
    assert_eq!(
        gt.get("ticketType").and_then(|v| v.as_str()),
        Some("billing"),
        "Expected camelCase question ID to be preserved after stripping GT_ prefix"
    );

    let item_with_meta = items[0].clone().with_metadata(BTreeMap::from([(
        "region".to_string(),
        serde_json::json!("eu-west"),
    )]));
    assert_eq!(
        extract_item_slice(&item_with_meta, &SliceBy::Field("region".into())),
        "eu-west"
    );
}

#[test]
fn test_conformal_symmetric_aps_preserves_singleton_efficiency() {
    let q = Question::new_choice(
        "Classify department",
        vec![
            "billing: duplicate charge, invoice, refund".into(),
            "technical: server crash, api outage, exception".into(),
            "security: phishing, unauthorized login, mfa breach".into(),
        ],
    );
    let mut items = Vec::new();
    for i in 0..24 {
        let (txt, lbl) = match i % 3 {
            0 => (
                "Duplicate charge on our monthly invoice, please refund payment",
                "billing",
            ),
            1 => (
                "Production API server crash and unhandled exception outage",
                "technical",
            ),
            _ => (
                "Unauthorized login security breach compromised MFA credentials",
                "security",
            ),
        };
        items.push(
            DataItem::new(
                format!("cal_{i}"),
                serde_json::json!({ "text": txt }),
                "cal",
            )
            .with_ground_truth(BTreeMap::from([(
                "dept".to_string(),
                serde_json::json!(lbl),
            )])),
        );
    }

    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let (report, pred_items) = ConformalEngine::evaluate(&clf, &items, "dept", &q, 0.10).unwrap();
    assert_eq!(report.empty_set_rate_pct, 0.0);
    assert!(
        report.empirical_coverage_pct >= 90.0,
        "Expected >= 90% coverage, got {}",
        report.empirical_coverage_pct
    );
    assert!(
        report.singleton_rate_pct >= 90.0,
        "Expected high singleton efficiency on clean separable items, got {}%",
        report.singleton_rate_pct
    );
    assert!(pred_items.iter().all(|p| p.set_size >= 1));
}

#[test]
fn test_calibration_nli_priors_do_not_inject_laya_english_temperatures() {
    let items = FileSource::parse_file(Path::new("data/customer_support.jsonl")).unwrap();
    let questions = lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();

    let tuner = CalibrationTuner::new(Checkpoint::DebertaV3ZeroShot)
        .with_loss(CalibrationLoss::Nll)
        .with_min_bucket_samples(2);
    let (_report, cfg) = tuner.fit(&items, &questions, "cs_nli").unwrap();

    // Unseen buckets on NLI checkpoints must remain at their 1.0 default, not Laya English's 1.906 / 0.5!
    assert_eq!(cfg.temperature_by_options.get("choice:2"), Some(&1.0));
    assert_eq!(cfg.temperature_by_options.get("choice:11+"), Some(&1.0));
}

#[test]
fn test_multi_label_array_and_integer_noul_evaluation() {
    let questions = BTreeMap::from([
        (
            "tags".to_string(),
            Question::new_choice(
                "Select relevant tags",
                vec!["billing".into(), "security".into(), "sales".into()],
            ),
        ),
        (
            "urgent".to_string(),
            Question::new_noul("Is this request urgent?"),
        ),
    ]);

    let clf = LayaClassifier::new(ClassifierConfig {
        checkpoint: Checkpoint::English,
        multi_label: true,
        multi_label_threshold: 0.45,
        ..Default::default()
    });
    let sample_state = serde_json::json!("Urgent duplicate billing invoice refund needed asap!");
    let pred = clf.classify(&sample_state, &questions).unwrap();
    let expected_tags = pred.answers["tags"].selected_labels.clone().unwrap();
    let expected_noul_int = if pred.answers["urgent"].noul.unwrap() >= 0.5 {
        1
    } else {
        0
    };

    let item = DataItem::new("ml_1", sample_state, "test").with_ground_truth(BTreeMap::from([
        ("tags".into(), serde_json::json!(expected_tags)),
        ("urgent".into(), serde_json::json!(expected_noul_int)),
    ]));

    let eval_metrics = Evaluator::new(LayaClassifier::new(clf.config().clone()))
        .evaluate(std::slice::from_ref(&item), &questions)
        .unwrap();
    assert_eq!(eval_metrics.accuracy, 1.0);

    let stats_report = StatsEngine::new(clf)
        .with_iterations(1)
        .with_warmup(0)
        .evaluate(&[item], &questions)
        .unwrap();
    assert_eq!(stats_report.accuracy.unwrap().overall_accuracy, 1.0);
}

#[test]
fn test_taxonomy_no_sibling_prefix_collision_and_orphan_branch_validation() {
    let mut branches = BTreeMap::new();
    branches.insert(
        "card_delivery".to_string(),
        HierarchicalNode::Leaf(Question::new_choice(
            "Delivery issue",
            vec!["delayed".into(), "wrong_address".into()],
        )),
    );
    let schema = HierarchicalSchema {
        coarse: Question::new_choice(
            "Select banking category",
            vec![
                "card: general card usage and inquiries".into(),
                "card_delivery: shipping and postal delivery".into(),
            ],
        ),
        branches,
    };
    assert!(schema.validate(192).is_ok());

    // Root leaf "card" must NOT be routed into sibling branch "card_delivery"
    let router =
        HierarchicalRouter::new(schema, LayaClassifier::with_checkpoint(Checkpoint::English))
            .with_beam(1)
            .with_path_threshold(0.01);
    let res = router
        .route_text("General card usage inquiry", Some("t1".into()))
        .unwrap();
    if res.path.first().map(String::as_str) == Some("card") {
        assert_eq!(
            res.path,
            vec!["card"],
            "Root leaf 'card' must stay terminal instead of matching prefix of 'card_delivery'"
        );
    }

    // Orphan branch key must fail validation
    let mut bad_branches = BTreeMap::new();
    bad_branches.insert(
        "nonexistent_branch".to_string(),
        HierarchicalNode::Leaf(Question::new_choice("Sub", vec!["a".into(), "b".into()])),
    );
    let bad_schema = HierarchicalSchema {
        coarse: Question::new_choice("Root", vec!["billing".into(), "tech".into()]),
        branches: bad_branches,
    };
    assert!(bad_schema.validate(192).is_err());
}

#[test]
fn test_defer_ieee754_boundary_and_empty_samples_safety() {
    let mut probs = BTreeMap::new();
    probs.insert("billing".to_string(), 0.85);
    probs.insert("other".to_string(), 0.15);
    let res = lemmata::model::DecisionResult {
        model: "mock".into(),
        answers: BTreeMap::from([(
            "q".into(),
            lemmata::model::DecisionAnswer {
                qtype: QuestionType::Choice,
                choice: Some("billing".into()),
                selected_labels: None,
                score: None,
                noul: None,
                confidence: 0.85,
                probabilities: probs,
                act_probability: Some(0.15),
                logits: Some(vec![1.7346, 0.0]),
            },
        )]),
        input_tokens: 5,
        latency_ms: 0.1,
    };
    assert_eq!(CascadeRouter::risk(&res), 0.15);
    let router = CascadeRouter::new(CascadeConfig {
        risk_threshold: 0.15,
        system2_cost_per_call: 0.01,
    });
    let item = DataItem::new("b1", serde_json::json!("test"), "src");
    let dec = router.route_with_result(&item, res);
    assert_eq!(dec.tier, lemmata::engine::SystemTier::System1Fast);

    // Empty samples do not produce NaN coverage
    let empty_fit = fit_samples(&[], 0.95, 0.95, 0.01, "mock");
    assert!(empty_fit.points.iter().all(|p| p.coverage.is_finite()));
}

#[test]
fn test_explain_supports_score_questions_via_logits() {
    let q = Question::new_score(
        "Rate urgency from 0 to 3",
        vec![
            "routine".into(),
            "moderate".into(),
            "high".into(),
            "critical production outage".into(),
        ],
    );
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let engine = AttributionEngine::new(&clf);
    let rep = engine
        .explain(
            "CRITICAL production outage crashing checkout API",
            "urgency",
            &q,
            "3",
            "0",
        )
        .unwrap();
    assert!(
        rep.baseline_margin.abs() > 0.0
            || rep.attributions.iter().any(|a| a.delta_margin.abs() > 0.0),
        "Score question explanation should produce non-zero logit attributions"
    );
}

#[test]
fn test_safetensors_bounds_overflow_and_header_sha256() {
    let tmp =
        std::env::temp_dir().join(format!("lemmata_pr1_st_{}.safetensors", std::process::id()));
    let write_st = |header: &str| {
        let hb = header.as_bytes();
        let mut buf = Vec::from((hb.len() as u64).to_le_bytes());
        buf.extend_from_slice(hb);
        std::fs::write(&tmp, buf).unwrap();
    };

    // Valid minimal safetensors
    write_st(r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,0]}}"#);
    let sha1 = safetensors_header_sha256(&tmp).unwrap();
    let sha2 = safetensors_header_sha256(&tmp).unwrap();
    assert_eq!(sha1.len(), 64);
    assert_eq!(sha1, sha2);

    // Out-of-bounds data_offsets rejected
    write_st(r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,64]}}"#);
    assert!(inspect_safetensors(&tmp).is_err());

    // Inverted data_offsets rejected
    write_st(r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[16,8]}}"#);
    assert!(inspect_safetensors(&tmp).is_err());

    // Shape u64 overflow rejected
    write_st(r#"{"w":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#);
    assert!(inspect_safetensors(&tmp).is_err());

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_attest_exact_float_bits_and_multi_label_selected_labels() {
    let mut engine = AttestationEngine::new("attest_secret_key");
    let item = DataItem::new("a1", serde_json::json!({ "text": "Refund invoice" }), "src");
    let decision = lemmata::model::DecisionResult {
        model: "convaiinnovations/laya".into(),
        answers: BTreeMap::from([(
            "tags".into(),
            lemmata::model::DecisionAnswer {
                qtype: QuestionType::Choice,
                choice: Some("billing".into()),
                selected_labels: Some(vec!["billing".into(), "urgent".into()]),
                score: None,
                noul: None,
                confidence: 0.9123456,
                probabilities: BTreeMap::new(),
                act_probability: Some(0.0876543),
                logits: None,
            },
        )]),
        input_tokens: 4,
        latency_ms: 1.0,
    };

    let rec = engine
        .create_receipt(&item, &decision, None, GuardAction::Allow, "schema_sha")
        .unwrap();
    assert_eq!(
        rec.selected_labels.get("tags"),
        Some(&vec!["billing".to_string(), "urgent".to_string()])
    );

    // Sub-1e-5 float perturbation is now caught by exact IEEE-754 bit canonicalization
    let mut tampered = rec.clone();
    *tampered.confidences.get_mut("tags").unwrap() += 1e-6;
    let ver = AttestationEngine::verify_ledger(&[tampered], "attest_secret_key");
    assert!(!ver.chain_intact);
    assert_eq!(ver.invalid_receipts, 1);
}

#[test]
fn test_schema_infer_json_array_and_cjk_chunking_strategies() {
    let tmp = std::env::temp_dir().join(format!("lemmata_pr1_infer_{}.json", std::process::id()));
    let json_arr = serde_json::json!([
        {"id": 1, "state": "Invoice overcharge refund", "ground_truth": {"rating": 1}},
        {"id": 2, "state": "Minor UI question", "ground_truth": {"rating": 3}},
        {"id": 3, "state": "Great product onboarding", "ground_truth": {"rating": 5}},
        {"id": 4, "state": "Billing receipt request", "ground_truth": {"rating": 1}},
        {"id": 5, "state": "Feature request question", "ground_truth": {"rating": 3}},
        {"id": 6, "state": "Smooth renewal experience", "ground_truth": {"rating": 5}}
    ]);
    std::fs::write(&tmp, serde_json::to_string_pretty(&json_arr).unwrap()).unwrap();

    let res = SchemaSynthesizer::new(Checkpoint::English)
        .infer_from_file(&tmp)
        .unwrap();
    let _ = std::fs::remove_file(&tmp);

    let q_rating = &res.schema["rating"];
    assert_eq!(q_rating.qtype, QuestionType::Score);
    // 1..=5 star ratings produce 6 ordinal 0-indexed levels (0..=5) so index 5 is valid
    assert_eq!(q_rating.render_options().len(), 6);

    // Long unspaced CJK text respects HeadTail and SlidingWindow instead of always prefix-cutting
    let long_cjk = "请立即检查生产环境服务器严重故障并退还重复收取的账单费用。".repeat(30);
    let ht = chunk_text_for_window(&long_cjk, 64, ChunkStrategy::HeadTail);
    assert_eq!(ht.len(), 1);
    assert!(ht[0].contains(" ... "));
    let sw = chunk_text_for_window(&long_cjk, 64, ChunkStrategy::SlidingWindow);
    assert!(sw.len() >= 2);

    // ModernBertNli is included in downloadable_checkpoints, and deberta-nli alias is gone
    assert!(Checkpoint::downloadable_checkpoints().contains(&Checkpoint::ModernBertNli));
    assert!(Checkpoint::from_str("deberta-nli").is_err());

    // AutoRoute cache fingerprint succeeds cleanly
    let router_clf = LayaClassifier::with_checkpoint(Checkpoint::AutoRoute);
    assert_eq!(fingerprint(&router_clf).unwrap().len(), 64);
    let _ = DecisionCache::in_memory();
}
