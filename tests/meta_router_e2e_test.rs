use std::collections::BTreeMap;

use lemma::datasource::DataItem;
use lemma::model::meta_router::{MetaRouteDecision, route_auto_best, summarize_route_distribution};
use lemma::model::{Checkpoint, Question, email_questions};

fn pure_choice_schema(k: usize) -> BTreeMap<String, Question> {
    let labels: Vec<String> = (0..k).map(|i| format!("class_{i}")).collect();
    BTreeMap::from([(
        "topic".to_string(),
        Question::new_choice("Classify the primary topic", labels),
    )])
}

#[test]
fn test_meta_router_cjk_and_multilingual_short() {
    let questions = pure_choice_schema(3);
    let cjk_text = "订单无法支付，请尽快处理退款请求。";
    let dec = route_auto_best(cjk_text, &questions);
    assert_eq!(dec.selected, Checkpoint::Multilingual);
    assert_eq!(dec.reason, "cjk");
    assert_eq!(dec.script, "cjk");
    assert_eq!(dec.max_label_cardinality, 3);
}

#[test]
fn test_meta_router_euro_and_russian_long_context() {
    let questions = pure_choice_schema(4);

    // > 1,024 estimated tokens in Cyrillic without typed heads -> EuroBert
    let ru_long =
        "Система зафиксировала критическую ошибку базы данных при обработке платежей. ".repeat(120);
    let dec_ru = route_auto_best(&ru_long, &questions);
    assert!(dec_ru.estimated_tokens > 1024);
    assert_eq!(dec_ru.selected, Checkpoint::EuroBert);
    assert_eq!(dec_ru.reason, "euro_long_context_8k");
    assert_eq!(dec_ru.script, "cyrillic");

    // > 1,024 estimated tokens in European Latin Extended without typed heads -> EuroBert
    let fr_long =
        "Bonjour, nous avons besoin d'un remboursement immédiat pour la facture erronée merci. "
            .repeat(110);
    let dec_fr = route_auto_best(&fr_long, &questions);
    assert!(dec_fr.estimated_tokens > 1024);
    assert_eq!(dec_fr.selected, Checkpoint::EuroBert);
    assert_eq!(dec_fr.reason, "euro_long_context_8k");
    assert_eq!(dec_fr.script, "latin_extended");

    // Same long European text WITH typed heads (Score/Noul) -> falls back to Multilingual RLCD
    let typed_questions = email_questions();
    let dec_fr_typed = route_auto_best(&fr_long, &typed_questions);
    assert_eq!(dec_fr_typed.selected, Checkpoint::Multilingual);
    assert_eq!(dec_fr_typed.reason, "latin_extended");
}

#[test]
fn test_meta_router_english_long_and_medium_contexts() {
    let choice_q = pure_choice_schema(4);
    let typed_q = email_questions();

    // > 1,500 tokens English log -> ModernBertZeroShot (8k RoPE context)
    let long_log =
        "ERROR [shard-04] connection pool exhausted while awaiting upstream ack ".repeat(160);
    let dec_long = route_auto_best(&long_log, &typed_q);
    assert!(dec_long.estimated_tokens > 1500);
    assert_eq!(dec_long.selected, Checkpoint::ModernBertZeroShot);
    assert_eq!(dec_long.reason, "long_context_8k_nli");
    assert_eq!(dec_long.script, "ascii_latin");

    // ~600 tokens (>512, <=1024) with multi-head typed questions -> TypedDecisions (1k context)
    let medium_log = "WARN retry attempt failed on payment gateway timeout segment ".repeat(55);
    let dec_med_typed = route_auto_best(&medium_log, &typed_q);
    assert!(dec_med_typed.estimated_tokens > 512 && dec_med_typed.estimated_tokens <= 1024);
    assert_eq!(dec_med_typed.selected, Checkpoint::TypedDecisions);
    assert_eq!(dec_med_typed.reason, "medium_long_typed_1k");

    // ~600 tokens (>512, <=1024) with pure Choice questions -> ModernBertZeroShot
    let dec_med_choice = route_auto_best(&medium_log, &choice_q);
    assert_eq!(dec_med_choice.selected, Checkpoint::ModernBertZeroShot);
    assert_eq!(dec_med_choice.reason, "medium_long_8k_nli");
}

#[test]
fn test_meta_router_structured_json_cardinality_and_short_nli() {
    // Structured JSON payload -> TypedDecisions
    let json_state = r#"{"event": "charge_failed", "amount_cents": 49900, "retry_count": 3}"#;
    let choice_q = pure_choice_schema(3);
    let dec_json = route_auto_best(json_state, &choice_q);
    assert_eq!(dec_json.selected, Checkpoint::TypedDecisions);
    assert_eq!(dec_json.reason, "structured_json_payload");

    // High cardinality (K=12 > 8) short English text -> English single-pass joint head
    let high_k_q = pure_choice_schema(12);
    let short_en = "Customer requests an enterprise seat downgrade before the annual renewal date.";
    let dec_high_k = route_auto_best(short_en, &high_k_q);
    assert_eq!(dec_high_k.selected, Checkpoint::English);
    assert_eq!(dec_high_k.reason, "high_cardinality_single_pass");
    assert_eq!(dec_high_k.max_label_cardinality, 12);

    // Short pure-Choice NLI prompt (K <= 5, no typed heads) -> DebertaV3ZeroShot
    let dec_short_nli = route_auto_best(short_en, &choice_q);
    assert_eq!(dec_short_nli.selected, Checkpoint::DebertaV3ZeroShot);
    assert_eq!(dec_short_nli.reason, "short_nli_disentangled_attn");
    assert_eq!(dec_short_nli.max_label_cardinality, 3);

    // Short multi-head (Choice + Score + Noul) English prompt -> English RLCD
    let typed_q = email_questions();
    let dec_multi_head = route_auto_best(short_en, &typed_q);
    assert_eq!(dec_multi_head.selected, Checkpoint::English);
    assert_eq!(dec_multi_head.reason, "multi_head_english_rlcd");

    // Verify serde roundtrip of MetaRouteDecision
    let encoded = serde_json::to_string(&dec_short_nli).unwrap();
    let decoded: MetaRouteDecision =
        serde_json::from_str(Box::leak(encoded.into_boxed_str())).unwrap();
    assert_eq!(decoded, dec_short_nli);
}

#[test]
fn test_summarize_route_distribution_across_heterogeneous_dataset() {
    let choice_q = pure_choice_schema(4);
    let ru_long =
        "Система зафиксировала критическую ошибку базы данных при обработке платежей. ".repeat(120);
    let long_en =
        "ERROR [shard-04] connection pool exhausted while awaiting upstream ack ".repeat(160);

    let items = vec![
        DataItem::new(
            "item_cjk",
            serde_json::json!("订单无法支付，请处理退款。"),
            "test",
        ),
        DataItem::new("item_ru_long", serde_json::json!(ru_long), "test"),
        DataItem::new("item_en_long", serde_json::json!(long_en), "test"),
        DataItem::new(
            "item_json",
            serde_json::json!({"ticket_id": 99, "status": "escalated"}),
            "test",
        ),
        DataItem::new(
            "item_short_1",
            serde_json::json!("Invoice #1042 was charged twice on my corporate card."),
            "test",
        ),
        DataItem::new(
            "item_short_2",
            serde_json::json!("Production API returns HTTP 502 bad gateway on /v1/classify."),
            "test",
        ),
    ];

    let dist = summarize_route_distribution(&items, &choice_q);
    assert_eq!(dist.values().sum::<usize>(), items.len());
    assert_eq!(dist.get(Checkpoint::Multilingual.repo_id()), Some(&1));
    assert_eq!(dist.get(Checkpoint::EuroBert.repo_id()), Some(&1));
    assert_eq!(dist.get(Checkpoint::ModernBertZeroShot.repo_id()), Some(&1));
    assert_eq!(dist.get(Checkpoint::TypedDecisions.repo_id()), Some(&1));
    assert_eq!(dist.get(Checkpoint::DebertaV3ZeroShot.repo_id()), Some(&2));
}
