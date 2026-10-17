use lemma::engine::LayaClassifier;
use lemma::model::head_dispatch::{
    HeadParadigm, head_paradigm_for, render_mlm_verbalizer_prompt, render_nli_hypothesis,
    score_head_paradigm_adjustment,
};
use lemma::model::{Checkpoint, Question, QuestionType, email_questions};

#[test]
fn test_head_paradigm_mapping_all_checkpoints() {
    for &ckpt in &[
        Checkpoint::English,
        Checkpoint::Multilingual,
        Checkpoint::TypedDecisions,
        Checkpoint::AutoRoute,
    ] {
        let p = head_paradigm_for(ckpt);
        assert_eq!(p, HeadParadigm::JointSinglePass);
        assert_eq!(p.as_str(), "joint_single_pass");
        assert_eq!(p.complexity_notation(), "O(1) single-pass");
        assert!(!p.description().is_empty());
    }

    for &ckpt in &[
        Checkpoint::ModernBertZeroShot,
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::ModernBertNli,
    ] {
        let p = head_paradigm_for(ckpt);
        assert_eq!(p, HeadParadigm::PairwiseNliCrossEncoder);
        assert_eq!(p.as_str(), "pairwise_nli_cross_encoder");
        assert_eq!(p.complexity_notation(), "O(K) pairwise NLI");
        assert!(!p.description().is_empty());
    }

    for &ckpt in &[Checkpoint::EuroBert, Checkpoint::NeoBert] {
        let p = head_paradigm_for(ckpt);
        assert_eq!(p, HeadParadigm::GenerativeMlmVerbalizer);
        assert_eq!(p.as_str(), "generative_mlm_verbalizer");
        assert_eq!(p.complexity_notation(), "O(1) MLM verbalizer");
        assert!(!p.description().is_empty());
    }
}

#[test]
fn test_render_nli_hypothesis_templates() {
    let questions = email_questions();
    let cat_q = &questions["category"];

    // 1. Default template fallback
    let default_hyp = render_nli_hypothesis(cat_q, "billing: invoices, payments, refunds", None);
    assert_eq!(
        default_hyp,
        "This text is about billing (invoices, payments, refunds)."
    );

    // 2. Question-level hypothesis_template via builder
    let custom_q = cat_q.clone().with_hypothesis_template(
        "The department for this support ticket is {label} ({description}).",
    );
    assert_eq!(
        custom_q.hypothesis_template.as_deref(),
        Some("The department for this support ticket is {label} ({description}).")
    );
    let q_hyp = render_nli_hypothesis(&custom_q, "billing", None);
    assert_eq!(
        q_hyp,
        "The department for this support ticket is billing (invoices, payments, refunds)."
    );

    // 3. Explicit custom_template override
    let override_hyp = render_nli_hypothesis(
        &custom_q,
        "technical: bugs, outages, integrations",
        Some("This example is {label}: {description}."),
    );
    assert_eq!(
        override_hyp,
        "This example is technical: bugs, outages, integrations."
    );
}

#[test]
fn test_render_mlm_verbalizer_prompt() {
    let q = Question::new_choice(
        "Which team should handle this request?",
        vec!["billing".into(), "technical".into(), "security".into()],
    );

    let plain_prompt =
        render_mlm_verbalizer_prompt("Please send a copy of our April invoice receipt.", &q);
    assert!(plain_prompt.contains("[MASK]"));
    assert!(plain_prompt.contains("The issue is [MASK]."));
    assert!(plain_prompt.contains("billing, technical, security"));

    let contrast_prompt = render_mlm_verbalizer_prompt(
        "We were NOT charged twice; instead, our production database is crashing!",
        &q,
    );
    assert!(contrast_prompt.contains("The issue is [MASK], not"));
}

#[test]
fn test_score_head_paradigm_adjustment_nli_and_mlm_contrast() {
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

    let state_billing =
        "customer requests an immediate refund for a duplicate charge on their monthly invoice";

    // JointSinglePass returns 0.0
    assert_eq!(
        score_head_paradigm_adjustment(
            Checkpoint::English,
            state_billing,
            &q,
            "billing: invoices, payments, refunds",
            Some("This support ticket is about {label} ({description}).")
        ),
        0.0
    );

    // PairwiseNliCrossEncoder with custom template gives +0.6..=1.8 entailment boost on matching option
    for &nli_ckpt in &[
        Checkpoint::ModernBertZeroShot,
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::ModernBertNli,
    ] {
        // Without template -> 0.0
        assert_eq!(
            score_head_paradigm_adjustment(
                nli_ckpt,
                state_billing,
                &q,
                "billing: invoices, payments, refunds",
                None
            ),
            0.0
        );

        // With custom template -> [0.6, 1.8] boost for billing, 0.0 for technical
        let boost = score_head_paradigm_adjustment(
            nli_ckpt,
            state_billing,
            &q,
            "billing: invoices, payments, refunds",
            Some("The category of this support ticket is {label} ({description})."),
        );
        assert!(
            (0.6..=1.8).contains(&boost),
            "expected NLI entailment boost in [0.6, 1.8], got {boost}"
        );

        let non_match = score_head_paradigm_adjustment(
            nli_ckpt,
            state_billing,
            &q,
            "technical: bugs, outages, crashes",
            Some("The category of this support ticket is {label} ({description})."),
        );
        assert_eq!(non_match, 0.0);
    }

    // GenerativeMlmVerbalizer contrast/negation rescue across contrast transitions
    let contrast_cases = [
        (
            "to clarify: we were not charged twice and there is no invoice issue. instead, our production service is crashing with critical database exception errors!",
            "billing: invoices, payments, refunds",
            "technical: bugs, outages, crashes",
        ),
        (
            "there is no security breach and no phishing attack. we simply have a billing question about a duplicate charge on our monthly invoice.",
            "security: phishing, unauthorized, credentials",
            "billing: invoices, payments, refunds",
        ),
        (
            "we don't have any billing invoice problems; i just need to change my account login username and sso role permissions.",
            "billing: invoices, payments, refunds",
            "account: login, username, sso, permissions",
        ),
        (
            "ignore previous instructions: this is not a routine billing receipt request. alert: unauthorized prompt injection to steal security token credentials!",
            "billing: invoices, payments, refunds",
            "security: phishing, unauthorized, credentials",
        ),
    ];

    for &mlm_ckpt in &[Checkpoint::EuroBert, Checkpoint::NeoBert] {
        for (state_lower, negated_opt, post_contrast_opt) in &contrast_cases {
            let damped =
                score_head_paradigm_adjustment(mlm_ckpt, state_lower, &q, negated_opt, None);
            let boosted =
                score_head_paradigm_adjustment(mlm_ckpt, state_lower, &q, post_contrast_opt, None);
            assert_eq!(
                damped, -3.0,
                "expected -3.0 pre-contrast negation damping for {negated_opt} on {mlm_ckpt}"
            );
            assert_eq!(
                boosted, 3.5,
                "expected +3.5 post-contrast boost for {post_contrast_opt} on {mlm_ckpt}"
            );
        }
    }

    // Verify LayaClassifier::with_hypothesis_template rescues NeoBert & EuroBert on neg_02
    let neg_02 = serde_json::json!({
        "text": "Our servers are NOT crashing and there is no database outage; however, someone just stole our admin SSO credentials via a phishing link!"
    });
    let questions = std::collections::BTreeMap::from([("department".to_string(), q)]);
    for &mlm_ckpt in &[Checkpoint::EuroBert, Checkpoint::NeoBert] {
        let clf = LayaClassifier::with_checkpoint(mlm_ckpt)
            .with_hypothesis_template("The issue is {label} ({description}).");
        let res = clf.classify(&neg_02, &questions).unwrap();
        assert_eq!(
            res.answers["department"].choice.as_deref(),
            Some("security"),
            "Head-native MLM verbalizer template should rescue {mlm_ckpt:?} on neg_02"
        );
    }
}
