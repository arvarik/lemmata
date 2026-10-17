use lemmata::benchmark::BenchmarkRunner;
use lemmata::datasource::{DataItem, DataSource, MemorySource};
use lemmata::engine::{
    AttestationEngine, AttributionEngine, CalibratedAgentConfig, CalibrationLoss, CalibrationTuner,
    CascadeConfig, CascadeRouter, ClassifierConfig, ConformalEngine, ConsensusEngine,
    ConsensusStatus, DecisionCache, Evaluator, GuardAction, GuardrailEngine, GuardrailPolicy,
    HierarchicalRouter, LayaClassifier, OodDetector, PolicyRule, SchemaKnapsackOptimizer,
    SchemaSynthesizer, StatsEngine, SystemStatsReport, SystemTier,
};
use lemmata::model::{Checkpoint, HierarchicalSchema, Question, QuestionType, inspect_safetensors};
use lemmata::resolve_data_source_filtered;
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn test_laya_question_schemas() {
    let q_choice = Question::new_choice(
        "Route this ticket to department",
        vec!["billing".into(), "technical".into(), "sales".into()],
    );
    assert_eq!(q_choice.qtype, QuestionType::Choice);
    assert_eq!(q_choice.render_options().len(), 3);

    let q_score = Question::new_score(
        "Rate urgency level",
        vec!["low".into(), "med".into(), "high".into()],
    );
    assert_eq!(q_score.qtype, QuestionType::Score);
    assert_eq!(q_score.render_options().len(), 3);

    let q_noul = Question::new_noul("Is customer churning?");
    assert_eq!(q_noul.qtype, QuestionType::Noul);
    assert_eq!(q_noul.render_options().len(), 2);
}

#[test]
fn test_schema_validation() {
    let q_valid = Question::new_choice("Department", vec!["billing".into(), "technical".into()]);
    let res = q_valid.validate("dept", 192);
    assert!(res.is_ok());

    let q_invalid = Question::new_choice("Single", vec!["only_one".into()]);
    let err = q_invalid.validate("single", 192);
    assert!(err.is_err());
}

#[test]
fn test_memory_source_and_classification() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Pick department", vec!["billing".into(), "sales".into()]),
    );
    questions.insert("churn".into(), Question::new_noul("Is churn likely?"));

    let items = vec![
        DataItem::new(
            "item_1",
            serde_json::json!({ "text": "Please refund my invoice payment billing charge" }),
            "test_mem",
        ),
        DataItem::new(
            "item_2",
            serde_json::json!({ "text": "I am canceling my plan and leaving sales contracts" }),
            "test_mem",
        ),
    ];

    let source = MemorySource::new("test", items);
    let read_items = source.read_items().unwrap();
    assert_eq!(read_items.len(), 2);

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let results = classifier.classify_batch(&read_items, &questions).unwrap();

    assert_eq!(results.len(), 2);
    let (id, res) = &results[0];
    assert_eq!(id, "item_1");
    assert!(res.answers.contains_key("dept"));
    assert!(res.answers.contains_key("churn"));
    assert_eq!(res.answers["dept"].choice.as_deref(), Some("billing"));
}

#[test]
fn test_evaluator_with_ground_truth() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "department".into(),
        Question::new_choice(
            "Department triage",
            vec!["billing".into(), "technical".into()],
        ),
    );

    let mut gt1 = BTreeMap::new();
    gt1.insert(
        "department".into(),
        serde_json::Value::String("billing".into()),
    );

    let mut gt2 = BTreeMap::new();
    gt2.insert(
        "department".into(),
        serde_json::Value::String("technical".into()),
    );

    let items = vec![
        DataItem::new(
            "t1",
            serde_json::json!({ "text": "Invoice duplicate payment billing issue" }),
            "test",
        )
        .with_ground_truth(gt1),
        DataItem::new(
            "t2",
            serde_json::json!({ "text": "System error stack trace technical bug crash" }),
            "test",
        )
        .with_ground_truth(gt2),
    ];

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let evaluator = Evaluator::new(classifier);
    let metrics = evaluator.evaluate(&items, &questions).unwrap();

    assert_eq!(metrics.sample_count, 2);
    assert_eq!(metrics.accuracy, 1.0);
    assert!(metrics.ece >= 0.0);
    assert!(metrics.brier_score >= 0.0);
}

#[test]
fn test_benchmark_runner() {
    let items = vec![
        DataItem::new(
            "b1",
            serde_json::json!({ "text": "Fast path benchmark query 1" }),
            "test",
        ),
        DataItem::new(
            "b2",
            serde_json::json!({ "text": "Fast path benchmark query 2" }),
            "test",
        ),
    ];

    let mut questions = BTreeMap::new();
    questions.insert("q1".into(), Question::new_noul("Is active?"));

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::Multilingual);
    let runner = BenchmarkRunner::new(classifier).with_warmup(1);
    let report = runner.run(&items, &questions, 2).unwrap();

    assert_eq!(report.total_items, 4);
    assert!(report.throughput_items_per_sec > 0.0);
    assert!(report.latency_p50_ms >= 0.0);
}

#[test]
fn test_conformal_prediction_engine() {
    let mut questions: BTreeMap<String, Question> = BTreeMap::new();
    let q = Question::new_choice("Dept", vec!["billing".into(), "technical".into()]);
    questions.insert("dept".to_string(), q.clone());

    let mut gt1 = BTreeMap::new();
    gt1.insert("dept".into(), serde_json::Value::String("billing".into()));

    let items = vec![
        DataItem::new(
            "c1",
            serde_json::json!({ "text": "payment billing invoice" }),
            "test",
        )
        .with_ground_truth(gt1),
    ];

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let (report, prediction_items) =
        ConformalEngine::evaluate(&classifier, &items, "dept", &q, 0.10).unwrap();

    assert_eq!(report.total_evaluated, 1);
    assert!(!prediction_items.is_empty());
    assert!(!prediction_items[0].prediction_set.is_empty());
}

#[test]
fn test_attribution_explain_engine() {
    let q = Question::new_choice("Dept", vec!["billing".into(), "technical".into()]);
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let engine = AttributionEngine::new(&classifier);

    let report = engine
        .explain(
            "We need a refund on invoice charges",
            "dept",
            &q,
            "billing",
            "technical",
        )
        .unwrap();

    assert_eq!(report.target_choice, "billing");
    assert_eq!(report.contrast_choice, "technical");
    assert!(!report.attributions.is_empty());
}

#[test]
fn test_safetensors_inspector_on_local_weights() {
    let path = Path::new("models/laya/model.safetensors");
    if path.exists() {
        let rep = inspect_safetensors(path).unwrap();
        assert!(rep.total_tensors > 100);
        assert!(rep.total_parameters > 300_000_000);
        assert!(rep.is_laya_compatible);
        assert!(rep.has_act_head);
    }
}

#[test]
fn test_guardrail_policy_engine() {
    let policy = GuardrailPolicy {
        name: "test_policy".into(),
        version: "1.0".into(),
        default_action: GuardAction::Allow,
        rules: vec![
            PolicyRule {
                id: "RULE_BILLING_URGENT".into(),
                description: Some("Escalate urgent billing inquiries".into()),
                condition: "dept.choice == \"billing\" AND urgency.score >= 0.5".into(),
                action: GuardAction::Escalate,
                target: Some("finance_team".into()),
                reason: Some("Urgent billing issue".into()),
            },
            PolicyRule {
                id: "RULE_OOD".into(),
                description: Some("Block highly uncertain inputs".into()),
                condition: "act_probability > 0.95".into(),
                action: GuardAction::Block,
                target: None,
                reason: Some("Uncertainty threshold exceeded".into()),
            },
        ],
    };

    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Triage", vec!["billing".into(), "technical".into()]),
    );
    questions.insert(
        "urgency".into(),
        Question::new_score("Rate", vec!["low".into(), "high".into()]),
    );

    let item = DataItem::new(
        "item_urgent_bill",
        serde_json::json!({ "text": "Immediate urgent invoice payment billing refund required today" }),
        "test",
    );

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let decision = classifier.classify(&item.state, &questions).unwrap();
    let engine = GuardrailEngine::new(policy);
    let record = engine.evaluate(&item, &decision);

    assert_eq!(record.item_id, "item_urgent_bill");
    assert_eq!(record.final_action, GuardAction::Escalate);
    assert_eq!(record.matched_rule.as_deref(), Some("RULE_BILLING_URGENT"));
    assert_eq!(record.target.as_deref(), Some("finance_team"));

    // Verify parenthesized clause in schema/policy.json (RULE_VIP_CHURN_ESCALATION)
    let schema_policy = GuardrailPolicy::from_file(Path::new("schema/policy.json")).unwrap();
    let schema_questions =
        lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();
    let schema_engine = GuardrailEngine::new(schema_policy);
    let mut churn_dec = classifier.classify(&item.state, &schema_questions).unwrap();
    for ans in churn_dec.answers.values_mut() {
        ans.act_probability = Some(0.10);
        ans.confidence = 0.90;
    }
    if let Some(dept) = churn_dec.answers.get_mut("department") {
        dept.choice = Some("billing".into());
    }
    if let Some(churn) = churn_dec.answers.get_mut("churn_risk") {
        churn.noul = Some(0.10);
    }
    if let Some(urg) = churn_dec.answers.get_mut("urgency") {
        urg.score = Some(2.0);
    }
    let churn_rec = schema_engine.evaluate(&item, &churn_dec);
    assert_eq!(
        churn_rec.matched_rule.as_deref(),
        Some("RULE_VIP_CHURN_ESCALATION")
    );
    assert_eq!(churn_rec.final_action, GuardAction::Escalate);
}

#[test]
fn test_consensus_ensemble_engine() {
    let items = vec![
        DataItem::new(
            "ens_1",
            serde_json::json!({ "text": "Please refund duplicate invoice payment billing" }),
            "test",
        ),
        DataItem::new(
            "ens_2",
            serde_json::json!({ "text": "Production server outage crash error" }),
            "test",
        ),
    ];

    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
    );

    let engine = ConsensusEngine::new(vec![
        Checkpoint::English,
        Checkpoint::Multilingual,
        Checkpoint::TypedDecisions,
    ])
    .with_threshold(0.20);

    let results = engine.evaluate_batch(&items, &questions).unwrap();
    assert_eq!(results.len(), 2);

    for res in &results {
        assert!(res.questions.contains_key("dept"));
        let q_cons = &res.questions["dept"];
        assert!(q_cons.consensus_confidence > 0.0);
        assert_eq!(q_cons.checkpoint_votes.len(), 3);
        assert!(matches!(
            res.overall_status,
            ConsensusStatus::Agreed | ConsensusStatus::Disputed
        ));
    }
}

#[test]
fn test_schema_synthesizer_induction() {
    let synth = SchemaSynthesizer::new(Checkpoint::English).with_text_col("text");

    let csv_path = Path::new("data/feedback_dataset.csv");
    let result = synth.infer_from_file(csv_path).unwrap();

    assert!(result.total_questions >= 1);
    assert!(result.total_samples_analyzed > 0);

    // Verify questions fit within ModernBERT English 192 head budget
    for (qid, (tokens, budget)) in &result.token_budgets {
        assert!(
            *tokens <= *budget,
            "Question '{}' tokens ({}) exceeded budget ({})",
            qid,
            tokens,
            budget
        );
    }
}

#[test]
fn test_multi_label_classification() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "tags".into(),
        Question::new_choice(
            "Select all relevant tags",
            vec![
                "billing".into(),
                "urgent".into(),
                "refund".into(),
                "sales".into(),
            ],
        ),
    );

    let item = DataItem::new(
        "multi_1",
        serde_json::json!({ "text": "Urgent immediate billing payment refund issue" }),
        "test",
    );

    let cfg = ClassifierConfig {
        checkpoint: Checkpoint::English,
        multi_label: true,
        multi_label_threshold: 0.35,
        ..Default::default()
    };
    let classifier = LayaClassifier::new(cfg);

    let res = classifier.classify(&item.state, &questions).unwrap();
    let ans = &res.answers["tags"];

    assert!(ans.selected_labels.is_some());
    let selected = ans.selected_labels.as_ref().unwrap();
    // At least billing and refund or urgent should be selected
    assert!(!selected.is_empty());
    assert!(selected.contains(&"billing".to_string()) || selected.contains(&"refund".to_string()));
}

#[test]
fn test_decision_cache_integration() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
    );

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let mut cache = DecisionCache::in_memory();

    let original_text = "Please refund duplicate charge on invoice";
    let state = serde_json::Value::String(original_text.into());
    let schema_hash = "dept_schema";

    // 1. Initial lookup is a miss
    assert!(cache.lookup(original_text, schema_hash).is_none());

    // 2. Classify and insert into cache
    let decision = classifier.classify(&state, &questions).unwrap();
    cache.insert(original_text, schema_hash, &decision);

    // 3. The exact text hits and runs no model
    let hit = cache
        .lookup(original_text, schema_hash)
        .expect("exact cache hit");
    assert_eq!(hit.answers["dept"].choice, decision.answers["dept"].choice);
    assert_eq!(hit.latency_ms, 0.0);

    // 4. A paraphrase misses: the cache never guesses from similar text
    let paraphrased_text = "Can I get a refund for a duplicate invoice charge?";
    assert!(cache.lookup(paraphrased_text, schema_hash).is_none());

    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 2);
}

#[test]
fn test_ood_detection_profile_and_quarantine() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
    );

    let id_dataset = vec![
        DataItem::new(
            "b1",
            serde_json::json!({ "text": "Invoice refund billing payment" }),
            "id",
        ),
        DataItem::new(
            "b2",
            serde_json::json!({ "text": "Credit card billing charge invoice" }),
            "id",
        ),
        DataItem::new(
            "t1",
            serde_json::json!({ "text": "Server outage crash stack trace bug" }),
            "id",
        ),
        DataItem::new(
            "t2",
            serde_json::json!({ "text": "System error exception stack bug crash" }),
            "id",
        ),
    ];

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let profile = OodDetector::fit(
        "test_ood_profile",
        &id_dataset,
        &classifier,
        &questions,
        "dept",
        0.05,
    )
    .unwrap();

    assert_eq!(profile.target_classes.len(), 2);
    let detector = OodDetector::new(profile).unwrap();

    // In-distribution test sample
    let id_item = DataItem::new(
        "id_sample",
        serde_json::json!({ "text": "Invoice billing payment refund" }),
        "test",
    );
    let id_text = "Invoice billing payment refund";
    let id_eval = detector
        .evaluate(&id_item.id, id_text, &classifier, &questions)
        .unwrap();
    assert_eq!(id_eval.action, GuardAction::Allow);

    // Out-of-distribution anomaly (random gibberish)
    let ood_text = "9283740918237409812374091823740918237409812374098123740981237409";
    let ood_eval = detector
        .evaluate("ood_sample", ood_text, &classifier, &questions)
        .unwrap();
    assert!(ood_eval.mahalanobis_distance > id_eval.mahalanobis_distance);
}

#[test]
fn test_cost_aware_cascade_pareto_tuning() {
    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
    );

    let items = vec![
        DataItem::new(
            "s1",
            serde_json::json!({ "text": "Invoice refund payment billing charge" }),
            "test",
        )
        .with_ground_truth(BTreeMap::from([(
            "dept".into(),
            serde_json::json!("billing"),
        )])),
        DataItem::new(
            "s2",
            serde_json::json!({ "text": "Server crash bug exception stack trace" }),
            "test",
        )
        .with_ground_truth(BTreeMap::from([(
            "default".into(),
            serde_json::json!("technical"),
        )])),
    ];

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let fit = CascadeRouter::fit(&items, &questions, &classifier, 0.95, 0.95, 0.005).unwrap();
    // Two labelled items cannot show 95% accuracy at 95% confidence.
    assert_eq!(fit.labelled_items, 2);
    assert!(fit.policy.is_none());
    assert!(fit.min_items_needed > 2);

    let router = CascadeRouter::new(CascadeConfig::default());
    let decision = router.route(&items[0], &questions, &classifier).unwrap();

    assert_eq!(decision.tier, SystemTier::System1Fast);
    assert_eq!(decision.cost_usd, 0.0);
    assert!(decision.risk < 0.45);
}

#[test]
fn test_cryptographic_attestation_and_merkle_chain() {
    let mut engine = AttestationEngine::new("enterprise_hmac_secret_2026");

    let item1 = DataItem::new(
        "item_1",
        serde_json::json!({ "text": "Urgent invoice refund" }),
        "test",
    );
    let item2 = DataItem::new(
        "item_2",
        serde_json::json!({ "text": "Account login compromise" }),
        "test",
    );

    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
    );
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let decision = classifier.classify(&item1.state, &questions).unwrap();

    let rec1 = engine
        .create_receipt(
            &item1,
            &decision,
            Some("RULE_REFUND"),
            GuardAction::Allow,
            "schema_v1",
        )
        .unwrap();

    let rec2 = engine
        .create_receipt(
            &item2,
            &decision,
            Some("RULE_SECURITY"),
            GuardAction::Escalate,
            "schema_v1",
        )
        .unwrap();

    assert_eq!(rec1.parent_hash, AttestationEngine::GENESIS_HASH);
    assert_eq!(rec2.parent_hash, rec1.receipt_digest);

    // Verify valid ledger
    let ledger = vec![rec1.clone(), rec2.clone()];
    let report = AttestationEngine::verify_ledger(&ledger, "enterprise_hmac_secret_2026");
    assert!(report.chain_intact);
    assert_eq!(report.valid_receipts, 2);
    assert_eq!(report.invalid_receipts, 0);

    // Verify tampered parent hash breaks chain
    let mut tampered_rec2 = rec2.clone();
    tampered_rec2.parent_hash = "corrupted_hash".to_string();
    let broken_report = AttestationEngine::verify_ledger(
        &[rec1.clone(), tampered_rec2],
        "enterprise_hmac_secret_2026",
    );
    assert!(!broken_report.chain_intact);

    // Verify tampered confidence breaks cryptographic digest verification
    let mut tampered_conf_rec = rec1.clone();
    tampered_conf_rec.confidences.insert("dept".into(), 0.01);
    let conf_report =
        AttestationEngine::verify_ledger(&[tampered_conf_rec], "enterprise_hmac_secret_2026");
    assert!(!conf_report.chain_intact && conf_report.invalid_receipts > 0);
}

#[test]
fn test_schema_knapsack_optimizer() {
    let mut questions = BTreeMap::new();

    // Create a verbose question that exceeds the 192-token head budget
    let mut criteria_map = serde_json::Map::new();
    criteria_map.insert(
        "billing".into(),
        serde_json::json!("In the event that the customer wishes to inquire regarding an invoice, duplicate billing charges, credit card payment disputes, refund inquiries, subscription adjustments, payment receipts, accounting ledgers, or currency reconciliation."),
    );
    criteria_map.insert(
        "technical".into(),
        serde_json::json!("When a user experiences server crash errors, production stack trace exceptions, database timeout failures, application bug reports, network connectivity disruption, or API gateway failure responses."),
    );
    criteria_map.insert(
        "sales".into(),
        serde_json::json!("Inquiries concerning enterprise contracts, annual enterprise pricing, commercial quotes, software sales demos, volume licensing discussions, and account procurement."),
    );

    questions.insert(
        "dept".into(),
        Question {
            qtype: QuestionType::Choice,
            instructions: "Classify incoming support inquiry to appropriate enterprise domain"
                .into(),
            criteria: Some(serde_json::Value::Object(criteria_map)),
            hypothesis_template: None,
        },
    );

    let orig_tokens: usize = questions["dept"]
        .render_options()
        .iter()
        .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
        .sum();

    // Optimize to fit into 50 token budget
    let result = SchemaKnapsackOptimizer::optimize(&questions, 50);

    assert!(result.original_tokens >= orig_tokens);
    assert!(result.optimized_tokens < result.original_tokens);
    assert!(result.pruned_clauses >= 1);

    // Optimized criteria still contains keywords
    let opt_q = &result.optimized_schema["dept"];
    let opt_opts = opt_q.render_options();
    assert_eq!(opt_opts.len(), 3);
}

#[test]
fn test_filtered_data_source_and_temperature_config() {
    // Test filtered directory ingestion
    let source =
        resolve_data_source_filtered(Path::new("data"), None, false, None, Some("jsonl"), true)
            .unwrap();

    let items = source.read_items().unwrap();
    // customer_support.jsonl contains 6 items
    assert!(items.len() >= 6);

    // Test ClassifierConfig with custom temperature_by_options
    let mut cfg = ClassifierConfig::default();
    cfg.temperature_by_options.insert("choice:2".into(), 2.15);
    cfg.temperature_by_options.insert("choice:3-5".into(), 1.85);

    let classifier = LayaClassifier::new(cfg);
    assert_eq!(classifier.config().temperature_by_options["choice:2"], 2.15);
}

#[test]
fn test_hierarchical_schema_parsing_and_validation() {
    let schema_path = Path::new("schema/banking_hierarchy.json");
    let schema = HierarchicalSchema::load_from_file(schema_path).unwrap();

    assert_eq!(
        schema.coarse.instructions,
        "Determine the high-level banking domain"
    );
    assert_eq!(schema.branches.len(), 2);
    assert_eq!(schema.total_leaf_count(), 7);
    assert_eq!(schema.max_depth(), 2);

    // Validate token budget against ModernBERT (192) and mmBERT (256)
    let rep_192 = schema.validate(192).unwrap();
    assert!(rep_192.valid);
    assert!(rep_192.root_tokens < 192);
    assert!(rep_192.max_branch_tokens < 192);
    assert_eq!(rep_192.total_leaf_classes, 7);

    let tree_repr = schema.render_ascii_tree(192);
    assert!(tree_repr.contains("Root: Determine the high-level banking domain"));
    assert!(tree_repr.contains("cards:"));
    assert!(tree_repr.contains("transfers:"));
    assert!(tree_repr.contains("Total Leaf Classes: 7"));
}

#[test]
fn test_hierarchical_routing_execution() {
    let schema_path = Path::new("schema/banking_hierarchy.json");
    let schema = HierarchicalSchema::load_from_file(schema_path).unwrap();
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let router = HierarchicalRouter::new(schema, classifier)
        .with_beam(2)
        .with_path_threshold(0.10);

    // 1. Card stolen ticket -> cards -> lost_or_stolen
    let card_ticket = "My debit card was stolen while traveling in London, please cancel it immediately to prevent fraud.";
    let res_card = router.route_text(card_ticket, Some("t1".into())).unwrap();
    assert_eq!(res_card.path, vec!["cards", "lost_or_stolen"]);
    assert_eq!(res_card.formatted_path, "cards -> lost_or_stolen");
    assert_eq!(res_card.final_choice, "lost_or_stolen");
    assert_eq!(res_card.status, "COMPLETED");
    assert!(!res_card.escalated);
    assert_eq!(res_card.steps.len(), 2);
    assert!(res_card.compound_confidence > 0.0);

    // 2. Wire transfer ticket -> transfers -> wire_delay
    let transfer_ticket = "I sent an international wire transfer 3 days ago but the recipient says funds have not arrived yet.";
    let res_transfer = router
        .route_text(transfer_ticket, Some("t2".into()))
        .unwrap();
    assert_eq!(res_transfer.path, vec!["transfers", "wire_delay"]);
    assert_eq!(res_transfer.final_choice, "wire_delay");
    assert_eq!(res_transfer.status, "COMPLETED");

    // 3. Security ticket -> security (terminal leaf at root)
    let security_ticket = "I noticed an unauthorized login attempt from an unknown device in an unfamiliar country. My password might be compromised.";
    let res_sec = router
        .route_text(security_ticket, Some("t3".into()))
        .unwrap();
    assert_eq!(res_sec.path, vec!["security"]);
    assert_eq!(res_sec.final_choice, "security");
    assert_eq!(res_sec.steps.len(), 1);
    assert_eq!(res_sec.status, "COMPLETED");
}

#[test]
fn test_hierarchical_routing_coarse_escalation_gate() {
    let schema_path = Path::new("schema/banking_hierarchy.json");
    let schema = HierarchicalSchema::load_from_file(schema_path).unwrap();
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    // A high path threshold (0.95) escalates ambiguous or out-of-domain inputs
    let router = HierarchicalRouter::new(schema, classifier).with_path_threshold(0.95);

    let irrelevant_ticket =
        "Can you tell me what the current weather forecast is for tomorrow morning?";
    let res = router
        .route_text(irrelevant_ticket, Some("t_irr".into()))
        .unwrap();

    assert!(res.escalated);
    assert_eq!(res.status, "ESCALATED");
    assert!(res.compound_confidence < 0.95);
    assert!(res.escalation_reason.is_some());
}

#[test]
fn test_hierarchical_batch_and_benchmark() {
    let schema_path = Path::new("schema/banking_hierarchy.json");
    let schema = HierarchicalSchema::load_from_file(schema_path).unwrap();
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let router = HierarchicalRouter::new(schema, classifier).with_path_threshold(0.40);

    let source = resolve_data_source_filtered(
        Path::new("data"),
        Some(Path::new("data/banking_tickets.jsonl")),
        false,
        None,
        None,
        true,
    )
    .unwrap();

    let items = source.read_items().unwrap();
    assert_eq!(items.len(), 8);

    let results = router.classify_items(&items).unwrap();
    assert_eq!(results.len(), 8);

    // Verify formatters
    let table = HierarchicalRouter::format_table(&results);
    assert!(table.contains("Input Snippet"));
    assert!(table.contains("Coarse Intent"));
    assert!(table.contains("Branch Intent"));

    let csv = HierarchicalRouter::format_csv(&results);
    assert!(csv.contains("coarse_intent,coarse_conf,fine_intent"));

    let jsonl = HierarchicalRouter::format_jsonl(&results).unwrap();
    assert_eq!(jsonl.lines().count(), 8);

    // Run benchmark
    let bench = router.benchmark(&items, 5).unwrap();
    assert_eq!(bench.total_samples, 40);
    assert!(bench.avg_total_latency_ms >= 0.0);
    assert!(bench.throughput_qps > 0.0);
}

#[test]
fn test_calibration_tuner_fit_and_metrics() {
    let source = resolve_data_source_filtered(
        Path::new("data"),
        Some(Path::new("data/customer_support.jsonl")),
        false,
        None,
        None,
        true,
    )
    .unwrap();

    let items = source.read_items().unwrap();
    let questions = lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();

    // The 18-row fixture is below the default 30-sample minimum, so opt in to fitting it.
    let tuner = CalibrationTuner::new(Checkpoint::English)
        .with_loss(CalibrationLoss::Nll)
        .with_bins(10)
        .with_min_bucket_samples(2);

    let (report, config) = tuner.fit(&items, &questions, "customer_support").unwrap();

    assert_eq!(report.dataset_name, "customer_support");
    assert_eq!(report.total_samples, 18);
    assert_eq!(report.buckets.len(), 3);

    // Each bucket must have improved ECE or maintained excellent calibration
    for b in &report.buckets {
        assert!(b.sample_count > 0);
        assert!(b.optimal_temperature > 0.0);
        assert!(b.calibrated_ece <= b.raw_ece + 1e-4);
    }

    assert!(report.average_calibrated_ece < report.average_raw_ece);
    assert!(report.overall_improvement_pct > 0.0);

    // Check rendered outputs
    let table = CalibrationTuner::format_table(&report, None);
    assert!(table.contains("Lemmata Temperature Calibration Tuner"));
    assert!(table.contains("choice:3-5"));
    assert!(table.contains("noul:2"));
    assert!(table.contains("score:3-5"));

    let json_output = CalibrationTuner::format_json(&report).unwrap();
    assert!(json_output.contains("average_calibrated_ece"));

    // Check exported config fields
    assert_eq!(config.checkpoint, "convaiinnovations/laya");
    assert_eq!(config.loss_function, "nll");
    assert!(config.temperature_by_options.contains_key("choice:3-5"));
    assert!(config.temperature_by_options.contains_key("noul:2"));
    assert!(config.temperature_by_options.contains_key("score:3-5"));
    assert!(config.temperature_by_options.contains_key("choice:11+")); // filled default
}

#[test]
fn test_qats_profile_and_adaptive_classification() {
    let source = resolve_data_source_filtered(
        Path::new("data"),
        Some(Path::new("data/customer_support.jsonl")),
        false,
        None,
        None,
        true,
    )
    .unwrap();

    let items = source.read_items().unwrap();
    let questions = lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();

    let tuner = CalibrationTuner::new(Checkpoint::English)
        .with_loss(CalibrationLoss::Nll)
        .with_bins(10)
        .with_min_bucket_samples(2)
        .with_qats(true, 0.35);

    let (report, config) = tuner.fit(&items, &questions, "customer_support").unwrap();

    let qats = config.qats.expect("Q-ATS profile must be generated");
    assert!(qats.enabled);
    assert_eq!(qats.gamma, 0.35);
    assert!(qats.mean_distance > 0.0);
    assert!(qats.std_distance > 0.0);
    assert!(!qats.class_centroids.is_empty());

    // Table should display Q-ATS profile section
    let table = CalibrationTuner::format_table(&report, Some(Path::new("config/laya_qats.json")));
    assert!(table.contains("2026 Q-ATS Profile"));
    assert!(table.contains("0.35"));

    // Use with classifier
    let clf_cfg = ClassifierConfig {
        temperature_by_options: config.temperature_by_options,
        qats_profile: Some(qats),
        ..Default::default()
    };

    let classifier = LayaClassifier::new(clf_cfg);
    for item in items.iter().take(3) {
        let res = classifier.classify(&item.state, &questions).unwrap();
        assert_eq!(res.answers.len(), 3);
        for (_qid, ans) in res.answers {
            let total_prob: f64 = ans.probabilities.values().sum();
            assert!((total_prob - 1.0).abs() < 1e-3);
        }
    }
}

#[test]
fn test_calibrated_config_roundtrip_and_losses() {
    let source = resolve_data_source_filtered(
        Path::new("data"),
        Some(Path::new("data/customer_support.jsonl")),
        false,
        None,
        None,
        true,
    )
    .unwrap();

    let items = source.read_items().unwrap();
    let questions = lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();

    // Test Brier Loss
    let tuner_brier = CalibrationTuner::new(Checkpoint::English)
        .with_loss(CalibrationLoss::Brier)
        .with_min_bucket_samples(2);
    let (report_brier, _) = tuner_brier
        .fit(&items, &questions, "customer_support")
        .unwrap();
    assert_eq!(report_brier.loss_function, CalibrationLoss::Brier);
    assert!(report_brier.average_calibrated_ece < report_brier.average_raw_ece);

    // Test ECE Loss
    let tuner_ece = CalibrationTuner::new(Checkpoint::English)
        .with_loss(CalibrationLoss::Ece)
        .with_min_bucket_samples(2);
    let (report_ece, config_ece) = tuner_ece
        .fit(&items, &questions, "customer_support")
        .unwrap();
    assert_eq!(report_ece.loss_function, CalibrationLoss::Ece);
    assert!(report_ece.average_calibrated_ece < report_ece.average_raw_ece);
    assert!(report_ece.average_calibrated_ece < 0.10);

    // Test JSON roundtrip of CalibratedAgentConfig
    let serialized = serde_json::to_string(&config_ece).unwrap();
    let deserialized: CalibratedAgentConfig = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.checkpoint, config_ece.checkpoint);
    assert_eq!(deserialized.loss_function, "ece");
    assert_eq!(
        deserialized.temperature_by_options.len(),
        config_ece.temperature_by_options.len()
    );
}

#[test]
fn test_stats_engine_speed_performance_accuracy() {
    let source = lemmata::resolve_data_source(
        Path::new("data"),
        Some(Path::new("data/customer_support.jsonl")),
        false,
        None,
    )
    .unwrap();
    let items = source.read_items().unwrap();
    let questions = lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);

    let stats_engine = StatsEngine::new(classifier)
        .with_iterations(3)
        .with_warmup(2);

    let report: SystemStatsReport = stats_engine.evaluate(&items, &questions).unwrap();

    // 1. Metadata Verification
    assert_eq!(report.dataset_items, items.len());
    assert_eq!(report.benchmark_iterations, 3);
    assert_eq!(report.warmup_runs, 2);
    assert_eq!(report.questions.len(), questions.len());
    assert_eq!(report.head_budget, 192);

    // 2. Speed & Latency Verification
    assert!(report.speed.total_elapsed_secs > 0.0);
    assert!(report.speed.latency_avg_ms > 0.0);
    assert!(report.speed.latency_min_ms > 0.0);
    assert!(report.speed.latency_max_ms >= report.speed.latency_min_ms);
    assert!(report.speed.latency_p50_ms >= report.speed.latency_min_ms);
    assert!(report.speed.latency_p90_ms >= report.speed.latency_p50_ms - 1e-6);
    assert!(report.speed.latency_p95_ms >= report.speed.latency_p90_ms - 1e-6);
    assert!(report.speed.latency_p99_ms >= report.speed.latency_p95_ms - 1e-6);
    assert!(report.speed.latency_max_ms >= report.speed.latency_p99_ms - 1e-6);

    // 3. Performance & Throughput Verification
    assert!(report.performance.throughput_items_per_sec > 0.0);
    assert!(report.performance.throughput_decisions_per_sec > 0.0);
    assert_eq!(report.performance.total_items_processed, items.len() * 3);
    assert_eq!(
        report.performance.total_decisions_evaluated,
        items.len() * 3 * questions.len()
    );
    assert!(report.performance.avg_head_token_utilization_pct > 0.0);
    assert!(report.performance.avg_head_token_utilization_pct < 100.0);

    // 4. Accuracy & Calibration Verification
    assert!(report.accuracy.is_some());
    let acc = report.accuracy.as_ref().unwrap();
    assert!(acc.overall_accuracy >= 0.0 && acc.overall_accuracy <= 1.0);
    assert!(acc.soft_accuracy >= 0.0 && acc.soft_accuracy <= 1.0);
    assert!(acc.expected_calibration_error >= 0.0 && acc.expected_calibration_error <= 1.0);
    assert!(acc.brier_score >= 0.0);
    assert!(acc.labeled_sample_count > 0);

    // 5. Per-Question Breakdown Verification
    for (qid, q_stat) in &report.questions {
        assert!(questions.contains_key(qid));
        assert!(q_stat.evaluated_samples > 0);
        assert!(q_stat.mean_confidence >= 0.0 && q_stat.mean_confidence <= 1.0);
        assert!(q_stat.mean_top2_margin >= 0.0 && q_stat.mean_top2_margin <= 1.0);
        assert!(q_stat.mean_entropy >= 0.0 && q_stat.mean_entropy <= 1.0);
        assert!(q_stat.act_escalate_rate >= 0.0 && q_stat.act_escalate_rate <= 1.0);
        assert!(q_stat.accuracy.is_some());
    }

    // 6. Formatters Verification
    let table_str = StatsEngine::format_table(&report);
    assert!(table_str.contains("Speed & Latency Distribution"));
    assert!(table_str.contains("System Throughput & Efficiency"));
    assert!(table_str.contains("Decision Accuracy, Calibration & Safety"));
    assert!(table_str.contains("Per-Question Schema Breakdown"));

    let compact_str = StatsEngine::format_compact(&report);
    assert!(compact_str.contains("SPEED:"));
    assert!(compact_str.contains("PERF:"));
    assert!(compact_str.contains("ACC:"));

    let json_str = StatsEngine::format_json(&report).unwrap();
    let deserialized: SystemStatsReport = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.checkpoint, report.checkpoint);

    let csv_str = StatsEngine::format_csv(&report);
    assert!(csv_str.contains("latency_p50"));
    assert!(csv_str.contains("throughput_items"));
    assert!(csv_str.contains("overall_accuracy"));
}

#[test]
fn test_stats_engine_export_file() {
    let source = lemmata::resolve_data_source(
        Path::new("data"),
        Some(Path::new("data/customer_support.jsonl")),
        false,
        None,
    )
    .unwrap();
    let items = source.read_items().unwrap();
    let questions = lemmata::load_questions_from_file(Path::new("schema/questions.json")).unwrap();
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);

    let stats_engine = StatsEngine::new(classifier)
        .with_iterations(2)
        .with_warmup(1);

    let report = stats_engine.evaluate(&items, &questions).unwrap();

    let json_path = Path::new("target/integration_stats_export.json");
    if json_path.exists() {
        let _ = std::fs::remove_file(json_path);
    }
    StatsEngine::export_report(&report, json_path).unwrap();
    assert!(json_path.exists());
    let json_content = std::fs::read_to_string(json_path).unwrap();
    let decoded: SystemStatsReport = serde_json::from_str(&json_content).unwrap();
    assert_eq!(decoded.dataset_items, items.len());

    let csv_path = Path::new("target/integration_stats_export.csv");
    if csv_path.exists() {
        let _ = std::fs::remove_file(csv_path);
    }
    StatsEngine::export_report(&report, csv_path).unwrap();
    assert!(csv_path.exists());
    let csv_content = std::fs::read_to_string(csv_path).unwrap();
    assert!(csv_content.contains("latency_p50"));
}

#[test]
fn test_web_scraper_html_extraction_and_classification() {
    unsafe {
        std::env::set_var("LEMMATA_OFFLINE", "1");
    }

    let scraper =
        lemmata::datasource::WebScraper::new(lemmata::datasource::ScrapeOptions::default());
    let doc = scraper.scrape("https://example.com/ai-article").unwrap();

    assert_eq!(doc.title, "Fast Non-Autoregressive Decision Encoders");
    assert!(doc.word_count > 20);
    assert!(!doc.is_likely_spa);
    assert!(
        doc.markdown
            .contains("# Fast Non-Autoregressive Decision Encoders")
    );
    assert!(!doc.markdown.contains("analytics tracking code"));

    // Classify using Laya
    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let mut criteria = serde_json::Map::new();
    criteria.insert(
        "encoders".into(),
        serde_json::json!("ModernBERT, non-autoregressive, classification models"),
    );
    criteria.insert(
        "cooking".into(),
        serde_json::json!("culinary, kitchen, recipes, ingredients"),
    );
    let q = Question {
        qtype: QuestionType::Choice,
        instructions: "Is this article about machine learning encoders or culinary cooking?".into(),
        criteria: Some(serde_json::Value::Object(criteria)),
        hypothesis_template: None,
    };
    let mut questions = std::collections::BTreeMap::new();
    questions.insert("topic".to_string(), q);

    let res = classifier
        .classify(&serde_json::json!({ "text": doc.markdown }), &questions)
        .unwrap();
    let ans = res.answers.get("topic").unwrap();
    assert_eq!(ans.choice.as_deref(), Some("encoders"));
    assert!(ans.confidence > 0.5);
}

#[test]
fn test_web_source_parallel_read_items() {
    unsafe {
        std::env::set_var("LEMMATA_OFFLINE", "1");
    }

    let urls = vec![
        "https://example.com/page1".to_string(),
        "https://example.com/page2".to_string(),
    ];
    let web_source =
        lemmata::datasource::WebSource::new(urls, lemmata::datasource::ScrapeOptions::default());
    let items = web_source.read_items().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].id, "https://example.com/page1");
    assert!(items[0].state.get("title").is_some());
}

#[test]
fn test_schema_hash_isolation_and_exact_keys() {
    use lemmata::create_inline_question;
    use lemmata::engine::{DecisionCache, compute_schema_hash};

    let schema_support =
        create_inline_question("choice", "Department triage", "billing,sales,technical").unwrap();
    let schema_sentiment =
        create_inline_question("choice", "Sentiment analysis", "negative,positive,neutral")
            .unwrap();

    assert_eq!(schema_support.len(), 1);
    assert_eq!(schema_sentiment.len(), 1);

    let hash_support = compute_schema_hash(&schema_support);
    let hash_sentiment = compute_schema_hash(&schema_sentiment);

    assert_ne!(hash_support, hash_sentiment);

    let key_support = DecisionCache::key("hello world", &hash_support);
    let key_sentiment = DecisionCache::key("hello world", &hash_sentiment);

    assert_ne!(key_support, key_sentiment);
}

#[test]
fn test_html_extractor_edge_cases_and_consecutive_lines() {
    use lemmata::datasource::HtmlExtractor;

    // Consecutive lines should have newline between them
    let html = "<p>First paragraph.</p><p>Second paragraph.</p>";
    let doc = HtmlExtractor::extract(html, "https://example.com");
    assert!(
        doc.markdown
            .contains("First paragraph.\n\nSecond paragraph.")
    );

    // Entities next to tags
    let html_entity = "<div>AT&T <br> 5 &lt; 10</div>";
    let doc_entity = HtmlExtractor::extract(html_entity, "https://example.com");
    assert!(doc_entity.markdown.contains("AT&T"));
    assert!(doc_entity.markdown.contains("5 < 10"));
}

#[test]
fn test_shell_completions_generation() {
    use clap::CommandFactory;
    use clap_complete::Shell;

    let mut cmd = lemmata::cli::Cli::command();
    for shell in [Shell::Bash, Shell::Zsh, Shell::Fish, Shell::PowerShell] {
        let mut buf = Vec::new();
        clap_complete::generate(shell, &mut cmd, "lemmata", &mut buf);
        assert!(!buf.is_empty());
        let content = String::from_utf8_lossy(&buf);
        assert!(content.contains("lemmata"));
    }
}
