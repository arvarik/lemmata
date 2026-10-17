//! Native answers stay the same when the computation changes but the question does not.
//!
//! Runs on every installed native checkpoint. Needs weights in `models/`, so it skips on CI.

use std::collections::BTreeMap;

use lemmata::datasource::DataItem;
use lemmata::engine::LayaClassifier;
use lemmata::model::native::{Backend, native_available, set_backend};
use lemmata::model::{Checkpoint, ChunkStrategy, DecisionResult, Question};

const TOL: f64 = 2e-3;

fn schema() -> BTreeMap<String, Question> {
    serde_json::from_value(serde_json::json!({
        "department": {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "invoices and refunds", "technical": "bugs and outages",
                         "sales": "pricing", "other": "anything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
            "criteria": ["not urgent", "normal", "critical outage"]},
        "churn": {"type": "noul", "instructions": "The customer threatens to cancel."}
    }))
    .unwrap()
}

const TEXTS: [&str; 3] = [
    "We were billed twice for invoice 4411. Please refund the duplicate charge.",
    "The API returns 500 errors on every checkout since the deploy.",
    "Your product is too expensive, we are moving to a competitor next month.",
];

fn same(name: &str, a: &DecisionResult, b: &DecisionResult) {
    for (qid, x) in &a.answers {
        let y = &b.answers[qid];
        assert_eq!(x.choice, y.choice, "{name}/{qid}: choice");
        for (label, p) in &x.probabilities {
            let q = y.probabilities[label];
            assert!((p - q).abs() < TOL, "{name}/{qid}/{label}: {p} vs {q}");
        }
    }
}

fn classify(clf: &LayaClassifier, text: &str, q: &BTreeMap<String, Question>) -> DecisionResult {
    clf.classify(&serde_json::json!(text), q).unwrap()
}

#[test]
fn native_answers_do_not_depend_on_how_they_are_computed() {
    set_backend(Backend::Native);
    let questions = schema();
    let checkpoints: Vec<Checkpoint> = Checkpoint::all()
        .iter()
        .copied()
        .filter(|&c| native_available(c))
        .collect();
    if checkpoints.is_empty() {
        eprintln!("skipping: no native weights installed");
    }
    for ckpt in checkpoints {
        let name = ckpt.short_name();
        let clf = LayaClassifier::with_checkpoint(ckpt);

        // 1. Repeated calls give the same answers.
        let first: Vec<DecisionResult> = TEXTS
            .iter()
            .map(|t| classify(&clf, t, &questions))
            .collect();
        for (t, r) in TEXTS.iter().zip(&first) {
            same(&format!("{name}/repeat"), r, &classify(&clf, t, &questions));
        }

        // 2. A batch gives the same answers as one call per item.
        let items: Vec<DataItem> = TEXTS
            .iter()
            .enumerate()
            .map(|(i, t)| DataItem::new(format!("i{i}"), serde_json::json!(t), "test"))
            .collect();
        for ((_, r), single) in clf
            .classify_batch(&items, &questions)
            .unwrap()
            .iter()
            .zip(&first)
        {
            same(&format!("{name}/batch"), single, r);
        }

        // 3. A question alone gets the same answer as with the other questions.
        for (qid, q) in &questions {
            let alone = BTreeMap::from([(qid.clone(), q.clone())]);
            for (t, r) in TEXTS.iter().zip(&first) {
                let mut partial = r.clone();
                partial.answers.retain(|k, _| k == qid);
                same(
                    &format!("{name}/alone/{qid}"),
                    &partial,
                    &classify(&clf, t, &alone),
                );
            }
        }

        // 4. Every chunk strategy reads a short text the same way.
        for strategy in [ChunkStrategy::HeadTail, ChunkStrategy::SlidingWindow] {
            let chunked = LayaClassifier::with_checkpoint(ckpt).with_chunk_strategy(strategy);
            for (t, r) in TEXTS.iter().zip(&first) {
                same(
                    &format!("{name}/{}", strategy.as_str()),
                    r,
                    &classify(&chunked, t, &questions),
                );
            }
        }

        // 5. NLI scores each hypothesis on its own, so option order cannot matter.
        if matches!(
            ckpt,
            Checkpoint::DebertaV3ZeroShot
                | Checkpoint::ModernBertZeroShot
                | Checkpoint::ModernBertNli
        ) {
            let mut reordered = questions.clone();
            reordered.retain(|k, _| k == "department");
            reordered.insert(
                "department".into(),
                Question::new_choice(
                    "Which team should handle this?",
                    vec![
                        "other".into(),
                        "sales".into(),
                        "technical".into(),
                        "billing".into(),
                    ],
                ),
            );
            let original = BTreeMap::from([(
                "department".to_string(),
                Question::new_choice(
                    "Which team should handle this?",
                    vec![
                        "billing".into(),
                        "technical".into(),
                        "sales".into(),
                        "other".into(),
                    ],
                ),
            )]);
            for t in TEXTS {
                same(
                    &format!("{name}/order"),
                    &classify(&clf, t, &original),
                    &classify(&clf, t, &reordered),
                );
            }
        }
        lemmata::model::native::release_models();
        eprintln!("{name}: consistent");
    }
}
