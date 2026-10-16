use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use lemma::datasource::FileSource;
use lemma::engine::LayaClassifier;
use lemma::model::{
    Checkpoint, ChunkStrategy, Question, chunk_text_for_window, estimate_tokens, pool_window_logits,
};

#[test]
fn test_chunk_strategy_from_str_and_display() {
    for alias in ["truncate", "prefix", "truncate-prefix", "truncate_prefix"] {
        assert_eq!(
            ChunkStrategy::from_str(alias).unwrap(),
            ChunkStrategy::TruncatePrefix
        );
    }
    for alias in ["head-tail", "head_tail", "headtail"] {
        assert_eq!(
            ChunkStrategy::from_str(alias).unwrap(),
            ChunkStrategy::HeadTail
        );
    }
    for alias in ["sliding", "sliding-window", "sliding_window", "window"] {
        assert_eq!(
            ChunkStrategy::from_str(alias).unwrap(),
            ChunkStrategy::SlidingWindow
        );
    }
    assert!(ChunkStrategy::from_str("unknown-strategy").is_err());
    assert_eq!(ChunkStrategy::default(), ChunkStrategy::TruncatePrefix);
    assert_eq!(ChunkStrategy::HeadTail.to_string(), "head-tail");
}

#[test]
fn test_chunk_text_for_window_on_golden_benchmark_long_items() {
    let items = FileSource::parse_file(Path::new("data/system1_golden_benchmark.jsonl"))
        .expect("Failed to load golden benchmark dataset");
    let schema_str = std::fs::read_to_string("schema/system1_benchmark_questions.json")
        .expect("Failed to read benchmark schema");
    let questions: BTreeMap<String, Question> =
        serde_json::from_str(&schema_str).expect("Failed to parse benchmark schema");

    let short_item = items.iter().find(|i| i.id == "en_01").unwrap();
    let short_text = short_item.state["text"].as_str().unwrap();
    assert!(estimate_tokens(short_text) <= 512);
    for strat in [
        ChunkStrategy::TruncatePrefix,
        ChunkStrategy::HeadTail,
        ChunkStrategy::SlidingWindow,
    ] {
        let chunks = chunk_text_for_window(short_text, 512, strat);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].as_ref(), short_text);
    }

    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let ctx_window = Checkpoint::English.context_window();
    assert_eq!(ctx_window, 512);

    let expected = [
        (
            "long_01",
            "Immediate engineering escalation required ASAP!",
            "technical",
            3usize,
            true,
        ),
        (
            "long_02",
            "Lock down tenant and escalate immediately!",
            "security",
            3usize,
            true,
        ),
        (
            "long_03",
            "unless we issue an immediate refund!",
            "billing",
            3usize,
            true,
        ),
        (
            "long_04",
            "three new hires in our account settings.",
            "account",
            1usize,
            false,
        ),
    ];

    for (id, tail_phrase, exp_dept, exp_urg, exp_esc) in expected {
        let item = items.iter().find(|i| i.id == id).unwrap();
        let text = item.state["text"].as_str().unwrap();
        assert!(
            estimate_tokens(text) > ctx_window,
            "{id} should exceed {ctx_window} tokens (got {})",
            estimate_tokens(text)
        );

        // 1. TruncatePrefix loses the tail incident conclusion beyond the 512-token window
        let prefix_chunks = chunk_text_for_window(text, ctx_window, ChunkStrategy::TruncatePrefix);
        assert_eq!(prefix_chunks.len(), 1);
        assert!(
            !prefix_chunks[0].contains(tail_phrase),
            "{id} TruncatePrefix should cut before tail conclusion"
        );

        // 2. HeadTail preserves both document header and post-preamble tail incident
        let ht_chunks = chunk_text_for_window(text, ctx_window, ChunkStrategy::HeadTail);
        assert_eq!(ht_chunks.len(), 1);
        assert!(ht_chunks[0].contains(" ... "));
        assert!(
            ht_chunks[0].starts_with('['),
            "{id} HeadTail should preserve header prefix"
        );
        assert!(
            ht_chunks[0].contains(tail_phrase),
            "{id} HeadTail should preserve post-preamble incident beyond token 560"
        );

        // Verify HeadTail recovers exact classification for 512-token model
        let ht_res = clf
            .classify(
                &serde_json::Value::String(ht_chunks[0].to_string()),
                &questions,
            )
            .unwrap();
        assert_eq!(
            ht_res.answers["department"].choice.as_deref(),
            Some(exp_dept),
            "{id} HeadTail department mismatch"
        );
        assert_eq!(
            ht_res.answers["urgency"].score.unwrap().round() as usize,
            exp_urg,
            "{id} HeadTail urgency mismatch"
        );
        assert_eq!(
            ht_res.answers["requires_escalation"].noul.unwrap() >= 0.5,
            exp_esc,
            "{id} HeadTail escalation mismatch"
        );

        // 3. SlidingWindow covers entire document across overlapping windows and pools logits accurately
        let sw_chunks = chunk_text_for_window(text, ctx_window, ChunkStrategy::SlidingWindow);
        assert!(
            sw_chunks.len() >= 2,
            "{id} SlidingWindow should produce >= 2 overlapping windows"
        );
        assert!(
            sw_chunks.last().unwrap().contains("END OF PREAMBLE"),
            "{id} SlidingWindow final chunk must contain tail incident"
        );

        // Pool logits across windows for department, urgency, and requires_escalation
        let dept_q = &questions["department"];
        let dept_opts = dept_q.render_options();
        let dept_win_logits: Vec<Vec<f64>> = sw_chunks
            .iter()
            .map(|c| clf.compute_option_logits(c, dept_q, &dept_opts).unwrap())
            .collect();
        let pooled_dept = pool_window_logits(&dept_win_logits);
        let best_dept_idx = pooled_dept
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let dept_keys: Vec<&str> = dept_opts
            .iter()
            .map(|o| o.split(':').next().unwrap().trim())
            .collect();
        assert_eq!(
            dept_keys[best_dept_idx], exp_dept,
            "{id} SlidingWindow pooled department mismatch"
        );

        let urg_q = &questions["urgency"];
        let urg_opts = urg_q.render_options();
        let urg_win_logits: Vec<Vec<f64>> = sw_chunks
            .iter()
            .map(|c| clf.compute_option_logits(c, urg_q, &urg_opts).unwrap())
            .collect();
        let pooled_urg = pool_window_logits(&urg_win_logits);
        let best_urg_idx = pooled_urg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            best_urg_idx, exp_urg,
            "{id} SlidingWindow pooled urgency mismatch"
        );

        let esc_q = &questions["requires_escalation"];
        let esc_opts = esc_q.render_options();
        let esc_win_logits: Vec<Vec<f64>> = sw_chunks
            .iter()
            .map(|c| clf.compute_option_logits(c, esc_q, &esc_opts).unwrap())
            .collect();
        let pooled_esc = pool_window_logits(&esc_win_logits);
        assert_eq!(
            pooled_esc[1] > pooled_esc[0],
            exp_esc,
            "{id} SlidingWindow pooled escalation mismatch"
        );

        // 4. Verify LayaClassifier::with_chunk_strategy E2E on both HeadTail and SlidingWindow
        for strat in [ChunkStrategy::HeadTail, ChunkStrategy::SlidingWindow] {
            let strat_clf =
                LayaClassifier::with_checkpoint(Checkpoint::English).with_chunk_strategy(strat);
            let res = strat_clf.classify(&item.state, &questions).unwrap();
            assert_eq!(
                res.answers["department"].choice.as_deref(),
                Some(exp_dept),
                "{id} {strat:?} classifier department mismatch"
            );
            assert_eq!(
                res.answers["urgency"].score.unwrap().round() as usize,
                exp_urg,
                "{id} {strat:?} classifier urgency mismatch"
            );
            assert_eq!(
                res.answers["requires_escalation"].noul.unwrap() >= 0.5,
                exp_esc,
                "{id} {strat:?} classifier escalation mismatch"
            );
        }
    }
}

#[test]
fn test_pool_window_logits_entropy_confidence_dominance() {
    // Single window returns identical logits
    let single = vec![vec![1.2, 3.4, 0.5]];
    assert_eq!(pool_window_logits(&single), vec![1.2, 3.4, 0.5]);

    // Low-entropy (high-confidence) window dominates over high-entropy (uniform) preamble window
    let routine_preamble = vec![0.25, 0.24, 0.26, 0.25];
    let incident_window = vec![0.1, 6.5, 0.1, 0.1];
    let pooled = pool_window_logits(&[routine_preamble, incident_window]);
    let argmax = pooled
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax, 1);
}
