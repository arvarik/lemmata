//! Every installed native model returns well-formed answers on hostile inputs.
//!
//! The test needs weights in `models/`, so it skips checkpoints that are not installed (all of
//! them on CI). Set `LEMMATA_DEVICE=metal` to run it on the GPU.

use std::collections::BTreeMap;

use lemmata::engine::LayaClassifier;
use lemmata::model::native::{Backend, native_available, set_backend};
use lemmata::model::{Checkpoint, ChunkStrategy, Question, QuestionType};

fn questions(with_many_labels: bool) -> BTreeMap<String, Question> {
    let mut q: BTreeMap<String, Question> = serde_json::from_value(serde_json::json!({
        "department": {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "invoices and refunds", "technical": "bugs and outages",
                         "sales": "pricing", "other": "anything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
            "criteria": ["0: not urgent", "1: normal", "2: urgent", "3: critical outage"]},
        "escalate": {"type": "noul", "instructions": "Does this need a supervisor?",
            "criteria": ["false: routine request", "true: urgent escalation"]},
        "churn": {"type": "noul", "instructions": "The customer threatens to cancel."}
    }))
    .unwrap();
    if with_many_labels {
        let many: Vec<String> = (0..100).map(|i| format!("intent_{i:03}")).collect();
        q.insert("intent".into(), Question::new_choice("Which intent?", many));
    }
    q
}

fn inputs() -> Vec<(&'static str, serde_json::Value)> {
    let long = "The checkout page crashes after login and nobody answers the ticket. ".repeat(250);
    vec![
        ("empty", serde_json::json!("   ")),
        ("one_char", serde_json::json!("x")),
        (
            "mask_tokens",
            serde_json::json!("[MASK] <mask> <|mask|> refund [CLS] [SEP]"),
        ),
        (
            "mixed_scripts",
            serde_json::json!("账单重复扣款 Doppelte Abbuchung 🚨 двойное списание فاتورة"),
        ),
        (
            "control_chars",
            serde_json::json!("refund\u{0}\t\r\n\u{7}please"),
        ),
        ("no_spaces", serde_json::json!("a".repeat(6000))),
        (
            "json_object",
            serde_json::json!({"subject": "Refund", "body": "Billed twice", "n": 3}),
        ),
        ("long", serde_json::json!(long)),
    ]
}

fn check(name: &str, res: &lemmata::model::DecisionResult, questions: &BTreeMap<String, Question>) {
    assert!(
        !res.model.starts_with("mock:"),
        "{name}: simulated result {}",
        res.model
    );
    assert!(res.input_tokens > 0, "{name}: no input tokens");
    assert!(
        res.latency_ms.is_finite() && res.latency_ms >= 0.0,
        "{name}: latency"
    );
    for (qid, q) in questions {
        let a = &res.answers[qid];
        let n_options = q.render_options().len();
        assert!(
            (0.0..=1.0).contains(&a.confidence),
            "{name}/{qid}: confidence {}",
            a.confidence
        );
        let logits = a.logits.as_ref().expect("raw logits");
        assert_eq!(logits.len(), n_options, "{name}/{qid}: logit count");
        assert!(
            logits.iter().all(|z| z.is_finite()),
            "{name}/{qid}: logits {logits:?}"
        );
        let probs: Vec<f64> = a.probabilities.values().copied().collect();
        assert!(
            probs
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "{name}/{qid}: probabilities {probs:?}"
        );
        // Probabilities are rounded to 4 decimals, so 100 of them can drift by up to 0.005.
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 6e-3,
            "{name}/{qid}: probabilities sum to {sum}"
        );
        match q.qtype {
            QuestionType::Choice => {
                let choice = a.choice.as_deref().expect("choice");
                assert!(
                    a.probabilities.contains_key(choice),
                    "{name}/{qid}: choice {choice}"
                );
            }
            QuestionType::Score => {
                let score = a.score.expect("score");
                assert!(
                    (0.0..=(n_options - 1) as f64).contains(&score),
                    "{name}/{qid}: score"
                );
            }
            QuestionType::Noul => {
                let p = a.noul.expect("noul");
                assert!((0.0..=1.0).contains(&p), "{name}/{qid}: noul {p}");
            }
        }
    }
}

#[test]
fn native_models_return_well_formed_answers_on_hostile_inputs() {
    set_backend(Backend::Native);
    let checkpoints: Vec<Checkpoint> = Checkpoint::all()
        .iter()
        .copied()
        .filter(|&c| native_available(c))
        .collect();
    if checkpoints.is_empty() {
        eprintln!("skipping: no native weights installed");
        return;
    }
    for ckpt in checkpoints {
        for (input_name, state) in inputs() {
            let strategies: &[ChunkStrategy] = if input_name == "long" {
                &[
                    ChunkStrategy::TruncatePrefix,
                    ChunkStrategy::HeadTail,
                    ChunkStrategy::SlidingWindow,
                ]
            } else {
                &[ChunkStrategy::TruncatePrefix]
            };
            // 100 labels on two short inputs only: on long inputs NLI would score 100 hypotheses
            // per window, which takes minutes on a CPU.
            let questions = questions(matches!(input_name, "one_char" | "mixed_scripts"));
            for &strategy in strategies {
                let name = format!("{}/{input_name}/{}", ckpt.short_name(), strategy.as_str());
                let clf = LayaClassifier::with_checkpoint(ckpt).with_chunk_strategy(strategy);
                let res = clf
                    .classify(&state, &questions)
                    .unwrap_or_else(|e| panic!("{name}: {e:#}"));
                check(&name, &res, &questions);
            }
        }
        lemmata::model::native::release_models();
        eprintln!("{}: ok", ckpt.short_name());
    }
}
