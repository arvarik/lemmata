use lemma::datasource::{DataItem, FileSource};
use lemma::engine::{ConsensusEngine, ExperimentRunner, LayaClassifier};
use lemma::model::{
    Checkpoint, DecisionAnswer, DecisionModel, DecisionResult, Downloader, Question,
    clean_email_body, email_questions, estimate_tokens, inspect_safetensors,
};
use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

struct CustomMockSystem1Model;

impl DecisionModel for CustomMockSystem1Model {
    fn model_id(&self) -> &str {
        "custom/system1-mock"
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint::English
    }

    fn classify(
        &self,
        _state: &serde_json::Value,
        questions: &BTreeMap<String, Question>,
    ) -> anyhow::Result<DecisionResult> {
        let mut answers = BTreeMap::new();
        for (qid, q) in questions {
            answers.insert(
                qid.clone(),
                DecisionAnswer {
                    qtype: q.qtype,
                    choice: Some("billing".into()),
                    selected_labels: None,
                    score: None,
                    noul: Some(0.91),
                    confidence: 0.92,
                    probabilities: BTreeMap::from([
                        ("billing".into(), 0.92),
                        ("other".into(), 0.08),
                    ]),
                    act_probability: Some(0.08),
                },
            );
        }
        Ok(DecisionResult {
            model: self.model_id().to_string(),
            answers,
            input_tokens: 16,
            latency_ms: 0.12,
        })
    }
}

#[test]
fn test_decision_model_trait_and_all_system1_checkpoints() {
    let models: Vec<Box<dyn DecisionModel>> = vec![
        Box::new(LayaClassifier::with_checkpoint(Checkpoint::English)),
        Box::new(LayaClassifier::with_checkpoint(Checkpoint::Multilingual)),
        Box::new(LayaClassifier::with_checkpoint(Checkpoint::TypedDecisions)),
        Box::new(LayaClassifier::with_checkpoint(Checkpoint::AutoRoute)),
        Box::new(LayaClassifier::with_checkpoint(Checkpoint::Jev)),
        Box::new(LayaClassifier::with_checkpoint(Checkpoint::ModernBertNli)),
        Box::new(CustomMockSystem1Model),
    ];

    let q = Question::new_choice(
        "Route support ticket",
        vec!["billing".into(), "technical".into(), "sales".into()],
    );

    for m in &models {
        let ans = m
            .classify_single("Please refund duplicate charge on my March invoice", &q)
            .unwrap();
        assert_eq!(ans.choice.as_deref(), Some("billing"));
        assert!(ans.confidence > 0.0);
    }

    assert_eq!(
        Checkpoint::from_str("route").unwrap(),
        Checkpoint::AutoRoute
    );
    assert_eq!(Checkpoint::from_str("jev").unwrap(), Checkpoint::Jev);
    assert_eq!(
        Checkpoint::from_str("modernbert-nli").unwrap(),
        Checkpoint::ModernBertNli
    );
    assert_eq!(
        Checkpoint::TypedDecisions.default_temperatures()[0],
        1.0148024559020996
    );
    assert!(
        Checkpoint::Multilingual
            .default_temperature_by_options()
            .is_empty()
    );
}

#[test]
fn test_script_router_and_autoroute_multilingual_dispatch() {
    assert_eq!(
        Checkpoint::detect_script_route("Please issue an invoice refund"),
        (Checkpoint::English, "ascii_latin")
    );
    assert_eq!(
        Checkpoint::detect_script_route("请退还我的发票账单费用"),
        (Checkpoint::Multilingual, "cjk")
    );
    assert_eq!(
        Checkpoint::detect_script_route("कृपया मेरे चालान का तुरंत रिफंड करें"),
        (Checkpoint::Multilingual, "devanagari")
    );
    assert_eq!(
        Checkpoint::detect_script_route("Пожалуйста, оформите возврат по счету"),
        (Checkpoint::Multilingual, "cyrillic")
    );
    assert_eq!(
        Checkpoint::detect_script_route("សូមពិនិត្យវិក្កយបត្របន្ទាន់"),
        (Checkpoint::Multilingual, "khmer")
    );

    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice(
            "Department",
            vec!["billing".into(), "technical".into(), "sales".into()],
        ),
    );

    let router_clf = LayaClassifier::with_checkpoint(Checkpoint::AutoRoute);
    let eng_clf = LayaClassifier::with_checkpoint(Checkpoint::English);

    let cjk_state = serde_json::json!("系统出现严重故障和崩溃，请立即修复错误");
    let res_routed = router_clf.classify(&cjk_state, &questions).unwrap();
    let res_eng = eng_clf.classify(&cjk_state, &questions).unwrap();

    assert_eq!(res_routed.model, "mock:convaiinnovations/laya-multilingual");
    assert_eq!(
        res_routed.answers["dept"].choice.as_deref(),
        Some("technical")
    );
    assert!(res_routed.answers["dept"].confidence > res_eng.answers["dept"].confidence);
}

#[test]
fn test_clean_email_body_and_email_questions_parity() {
    let raw_email = "Hi Billing Team,\r\n\r\n\
        We were double charged $499 on our latest invoice. Can you please refund this?\r\n\
        > Previous quoted line that should be stripped\r\n\r\n\
        Best regards,\r\n\
        Alice\r\n\
        Sent from my iPhone\r\n\r\n\
        CONFIDENTIALITY NOTICE: This message is confidential and intended solely for the addressee.\r\n\r\n\
        On Mon, Oct 12, 2026 at 10:00 AM Support <support@acme.com> wrote:\r\n\
        > Old thread content here";

    let cleaned = clean_email_body(raw_email, 3000);
    assert!(cleaned.contains("double charged $499"));
    assert!(!cleaned.contains("Previous quoted line"));
    assert!(!cleaned.contains("Best regards"));
    assert!(!cleaned.contains("Sent from my iPhone"));
    assert!(!cleaned.contains("CONFIDENTIALITY NOTICE"));
    assert!(!cleaned.contains("Old thread content"));

    let eqs = email_questions();
    assert_eq!(eqs.len(), 6);
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let res = clf
        .classify(&serde_json::Value::String(cleaned), &eqs)
        .unwrap();
    assert_eq!(res.answers["category"].choice.as_deref(), Some("billing"));
    assert!(res.answers["is_spam"].noul.unwrap() < 0.30);
    assert!(res.answers["is_phishing"].noul.unwrap() < 0.30);
    assert!(res.answers["needs_reply"].noul.unwrap() > 0.70);
}

#[test]
fn test_noul_calibration_fixes_constant_05252_bug() {
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let q_churn =
        Question::new_noul("Does the user threaten to cancel, switch to a competitor, or leave?");

    let ans_true = clf
        .classify_single(
            "Your app keeps crashing. I am canceling my subscription and switching to your competitor today.",
            &q_churn,
        )
        .unwrap();
    let ans_false = clf
        .classify_single(
            "Just wanted to say the new dark mode UI looks very sleek and modern. Nice job team!",
            &q_churn,
        )
        .unwrap();

    assert!(
        ans_true.noul.unwrap() > 0.70,
        "Expected noul > 0.70 on true churn case, got {}",
        ans_true.noul.unwrap()
    );
    assert!(
        ans_false.noul.unwrap() < 0.30,
        "Expected noul < 0.30 on false churn case, got {}",
        ans_false.noul.unwrap()
    );
}

#[test]
fn test_multibyte_utf8_labels_and_cjk_token_estimation() {
    let clf = LayaClassifier::with_checkpoint(Checkpoint::Multilingual);
    let q = Question::new_choice(
        "Classify language label",
        vec![
            "日本語".into(),
            "हिन्दी".into(),
            "ខ្មែរ".into(),
            "Русский".into(),
        ],
    );

    let ans = clf.classify_single("これは日本語のテストです", &q).unwrap();
    assert_eq!(ans.choice.as_deref(), Some("日本語"));

    let cjk_tokens = estimate_tokens("请退还我的发票账单费用并取消订阅");
    assert!(cjk_tokens >= 15);
}

#[test]
fn test_consensus_epistemic_uncertainty_and_checkpoint_verification() {
    let items = vec![DataItem::new(
        "m1",
        serde_json::json!({ "text": "请立即退还重复收取的发票账单费用" }),
        "test",
    )];
    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::new_choice("Department", vec!["billing".into(), "technical".into()]),
    );

    let engine = ConsensusEngine::new(vec![
        Checkpoint::English,
        Checkpoint::Multilingual,
        Checkpoint::TypedDecisions,
    ]);
    let res = engine.evaluate_batch(&items, &questions).unwrap();
    assert!(res[0].questions["dept"].epistemic_uncertainty > 0.0);

    let q_choice = &questions["dept"];
    ExperimentRunner::test_option_order_invariance(
        &LayaClassifier::with_checkpoint(Checkpoint::English),
        &items,
        "dept",
        q_choice,
        3,
    )
    .unwrap();

    let dl = Downloader {
        cli_binary: "hf".into(),
    };
    let laya_dir = Path::new("models/laya");
    if laya_dir.exists() {
        for ckpt in Checkpoint::laya_checkpoints() {
            let rep = dl.verify_checkpoint(*ckpt, laya_dir).unwrap();
            assert!(rep.is_valid, "Checkpoint {:?} should verify cleanly", ckpt);
        }
        let st_rep = inspect_safetensors(laya_dir).unwrap();
        assert!(st_rep.is_laya_compatible);
    }
}

#[test]
fn test_multi_model_system1_golden_benchmark_and_local_checkpoints() {
    let items = FileSource::parse_file(Path::new("data/system1_golden_benchmark.jsonl"))
        .expect("Failed to load golden benchmark dataset");
    assert_eq!(items.len(), 24);

    let schema_str = std::fs::read_to_string("schema/system1_benchmark_questions.json")
        .expect("Failed to read benchmark schema");
    let questions: BTreeMap<String, Question> =
        serde_json::from_str(&schema_str).expect("Failed to parse benchmark schema");
    assert_eq!(questions.len(), 3);

    let mb_zs = LayaClassifier::with_checkpoint(Checkpoint::ModernBertZeroShot);
    let deb_zs = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot);
    let euro = LayaClassifier::with_checkpoint(Checkpoint::EuroBert);
    let neo = LayaClassifier::with_checkpoint(Checkpoint::NeoBert);
    let router = LayaClassifier::with_checkpoint(Checkpoint::AutoRoute);

    // 1. Long-context (>550 tokens) telemetry: 8k ModernBERT-ZS succeeds where 512-token DeBERTa-v3-ZS truncates
    let long_01 = items.iter().find(|i| i.id == "long_01").unwrap();
    let res_mb_long = mb_zs.classify(&long_01.state, &questions).unwrap();
    let res_deb_long = deb_zs.classify(&long_01.state, &questions).unwrap();
    assert_eq!(
        res_mb_long.answers["department"].choice.as_deref(),
        Some("technical")
    );
    assert_ne!(
        res_deb_long.answers["department"].choice.as_deref(),
        Some("technical")
    );

    // 2. NLI negation trap: NLI-trained DeBERTa-v3-ZS & ModernBERT-ZS invert negated clause while raw MLM NeoBERT is fooled
    let neg_02 = items.iter().find(|i| i.id == "neg_02").unwrap();
    let res_deb_neg = deb_zs.classify(&neg_02.state, &questions).unwrap();
    let res_neo_neg = neo.classify(&neg_02.state, &questions).unwrap();
    assert_eq!(
        res_deb_neg.answers["department"].choice.as_deref(),
        Some("security")
    );
    assert_eq!(
        res_neo_neg.answers["department"].choice.as_deref(),
        Some("technical")
    );

    // 3. European vs CJK multilingual: EuroBERT handles Russian (multi_06), Laya AutoRoute handles both Russian and Chinese (multi_05)
    let multi_ru = items.iter().find(|i| i.id == "multi_06").unwrap();
    let multi_zh = items.iter().find(|i| i.id == "multi_05").unwrap();
    assert_eq!(
        euro.classify(&multi_ru.state, &questions).unwrap().answers["department"]
            .choice
            .as_deref(),
        Some("billing")
    );
    assert_eq!(
        router
            .classify(&multi_zh.state, &questions)
            .unwrap()
            .answers["department"]
            .choice
            .as_deref(),
        Some("technical")
    );

    // 4. Verify local checkpoints and safetensors headers when downloaded on disk
    let dl = Downloader {
        cli_binary: "hf".into(),
    };
    for ckpt in [
        Checkpoint::ModernBertZeroShot,
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::EuroBert,
        Checkpoint::NeoBert,
    ] {
        let dir = ckpt.local_dir();
        if dir.exists() {
            let rep = dl.verify_checkpoint(ckpt, &dir).unwrap();
            assert!(rep.is_valid, "Checkpoint {:?} should verify cleanly", ckpt);
            let st = inspect_safetensors(&dir).unwrap();
            assert!(st.has_classifier_head && st.total_parameters > 200_000_000);
        }
    }
}
