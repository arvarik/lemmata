use std::collections::BTreeMap;
use std::path::Path;
use lemma::benchmark::BenchmarkRunner;
use lemma::datasource::{DataItem, DataSource, MemorySource};
use lemma::engine::{
    AttributionEngine, ConformalEngine, Evaluator, ExperimentRunner, LayaClassifier,
};
use lemma::model::{Checkpoint, Question, QuestionType, inspect_safetensors};

#[test]
fn test_laya_question_schemas() {
    let q_choice = Question::new_choice(
        "Route this ticket to department",
        vec!["billing".into(), "technical".into(), "sales".into()],
    );
    assert_eq!(q_choice.qtype, QuestionType::Choice);
    assert_eq!(q_choice.render_options().len(), 3);

    let q_score = Question::new_score("Rate urgency level", vec!["low".into(), "med".into(), "high".into()]);
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
    gt1.insert("department".into(), serde_json::Value::String("billing".into()));

    let mut gt2 = BTreeMap::new();
    gt2.insert("department".into(), serde_json::Value::String("technical".into()));

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
fn test_experiment_option_ordering() {
    let items = vec![DataItem::new(
        "exp_1",
        serde_json::json!({ "text": "Can I get a refund on billing invoice" }),
        "test",
    )];

    let q = Question::new_choice(
        "Department triage",
        vec!["billing".into(), "technical".into(), "sales".into()],
    );

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let res = ExperimentRunner::test_option_order_invariance(&classifier, &items, "dept", &q, 3);
    assert!(res.is_ok());
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
        DataItem::new("c1", serde_json::json!({ "text": "payment billing invoice" }), "test")
            .with_ground_truth(gt1),
    ];

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
    let (report, prediction_items) = ConformalEngine::evaluate(&classifier, &items, "dept", &q, 0.10).unwrap();

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
        .explain("We need a refund on invoice charges", "dept", &q, "billing", "technical")
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
