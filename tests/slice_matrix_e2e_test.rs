use std::path::Path;
use std::str::FromStr;

use lemma::datasource::{DataSource, FileSource};
use lemma::engine::LayaClassifier;
use lemma::engine::slicing::{
    SliceBy, compute_arena_slice_matrix, evaluate_slices, extract_item_slice,
};
use lemma::load_questions_from_file;
use lemma::model::Checkpoint;

#[test]
fn test_slice_by_parsing_and_item_extraction() {
    assert_eq!(SliceBy::from_str("auto").unwrap(), SliceBy::AutoPrefix);
    assert_eq!(SliceBy::from_str("prefix").unwrap(), SliceBy::AutoPrefix);
    assert_eq!(SliceBy::from_str("script").unwrap(), SliceBy::Script);
    assert_eq!(SliceBy::from_str("length").unwrap(), SliceBy::LengthBucket);
    assert_eq!(
        SliceBy::from_str("field:event_type").unwrap(),
        SliceBy::Field("event_type".to_string())
    );
    assert_eq!(
        SliceBy::from_str("department").unwrap(),
        SliceBy::Field("department".to_string())
    );

    let items = FileSource::new(Path::new("data/system1_golden_benchmark.jsonl"))
        .read_items()
        .expect("Failed to read golden benchmark items");
    assert_eq!(items.len(), 24);

    // AutoPrefix should discover all 5 diagnostic slices: en, neg, multi, long, struct
    let mut prefixes: Vec<String> = items
        .iter()
        .map(|it| extract_item_slice(it, &SliceBy::AutoPrefix))
        .collect();
    prefixes.sort();
    prefixes.dedup();
    assert_eq!(prefixes, vec!["en", "long", "multi", "neg", "struct"]);

    // Script should detect ascii_latin, latin_extended, cjk / hiragana_katakana, cyrillic
    let mut scripts: Vec<String> = items
        .iter()
        .map(|it| extract_item_slice(it, &SliceBy::Script))
        .collect();
    scripts.sort();
    scripts.dedup();
    assert!(scripts.contains(&"ascii_latin".to_string()));
    assert!(scripts.contains(&"latin_extended".to_string()));
    assert!(scripts.contains(&"cjk".to_string()));
    assert!(scripts.contains(&"cyrillic".to_string()));

    // LengthBucket should separate short (<=128t) items from long (513-1024t) items
    let mut buckets: Vec<String> = items
        .iter()
        .map(|it| extract_item_slice(it, &SliceBy::LengthBucket))
        .collect();
    buckets.sort();
    buckets.dedup();
    assert!(buckets.contains(&"short (<=128t)".to_string()));
    assert!(buckets.contains(&"long (513-1024t)".to_string()));

    // Field("event_type") should extract structured state fields and fallback to "unknown"
    let struct_item = items.iter().find(|it| it.id == "struct_01").unwrap();
    assert_eq!(
        extract_item_slice(struct_item, &SliceBy::Field("event_type".to_string())),
        "billing_webhook"
    );
    let en_item = items.iter().find(|it| it.id == "en_01").unwrap();
    assert_eq!(
        extract_item_slice(en_item, &SliceBy::Field("event_type".to_string())),
        "unknown"
    );
}

#[test]
fn test_evaluate_slices_and_arena_slice_matrix_on_golden_benchmark() {
    let items = FileSource::new(Path::new("data/system1_golden_benchmark.jsonl"))
        .read_items()
        .expect("Failed to read golden benchmark items");
    let questions = load_questions_from_file(Path::new("schema/system1_benchmark_questions.json"))
        .expect("Failed to load benchmark questions");

    // 1. Single-checkpoint slice evaluation with AutoRoute across AutoPrefix
    let router_cls = LayaClassifier::with_checkpoint(Checkpoint::AutoRoute);
    let entries = evaluate_slices(&router_cls, &items, &questions, &SliceBy::AutoPrefix)
        .expect("evaluate_slices failed");
    assert_eq!(entries.len(), 5);
    for entry in &entries {
        assert!(entry.sample_count > 0);
        assert!(
            entry.accuracy >= 0.90,
            "Expected high accuracy for AutoRoute on slice '{}', got {:.2}%",
            entry.slice,
            entry.accuracy * 100.0
        );
    }

    // 2. Multi-checkpoint Arena Slice Matrix across AutoPrefix, Script, and LengthBucket
    let checkpoints = [
        Checkpoint::AutoRoute,
        Checkpoint::English,
        Checkpoint::Multilingual,
        Checkpoint::TypedDecisions,
    ];
    let (slices, rows) =
        compute_arena_slice_matrix(&items, &questions, &checkpoints, &SliceBy::AutoPrefix)
            .expect("compute_arena_slice_matrix failed");
    assert_eq!(slices, vec!["en", "long", "multi", "neg", "struct"]);
    assert_eq!(rows.len(), 4);

    // AutoRoute should outperform English on the `long` slice (512-token context truncation on English)
    let route_row = &rows[0];
    let en_row = &rows[1];
    assert!(
        route_row.slice_accuracies["long"] > en_row.slice_accuracies["long"],
        "Expected AutoRoute ({}) > English ({}) on long slice",
        route_row.slice_accuracies["long"],
        en_row.slice_accuracies["long"]
    );

    // Script & LengthBucket matrices should also compute cleanly
    let (script_slices, script_rows) =
        compute_arena_slice_matrix(&items, &questions, &checkpoints, &SliceBy::Script)
            .expect("script matrix failed");
    assert!(script_slices.len() >= 4);
    assert_eq!(script_rows.len(), 4);

    let (len_slices, len_rows) =
        compute_arena_slice_matrix(&items, &questions, &checkpoints, &SliceBy::LengthBucket)
            .expect("length matrix failed");
    assert_eq!(len_slices.len(), 2);
    assert_eq!(len_rows.len(), 4);
}
