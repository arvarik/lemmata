use std::path::Path;

use lemma::datasource::FileSource;
use lemma::engine::LayaClassifier;
use lemma::engine::shortlist::{expand_shortlisted_logits, shortlist_options};
use lemma::model::{Checkpoint, Question, QuestionType, estimate_tokens};

fn build_20_class_enterprise_question() -> Question {
    // 20 enterprise departments: each option is 6 whitespace words ("key: five word description")
    // so estimate_tokens(opt) == 6 * 4 / 3 + 1 == 9 tokens -> 20 * 9 = 180 tokens total,
    // and top-4 shortlisted options = 4 * 9 = 36 tokens total.
    let criteria = serde_json::json!({
        "billing": "Invoices, duplicate charges, refunds, receipts",
        "technical": "Software bugs, crashes, timeouts, outages",
        "security": "Phishing, unauthorized access, compromised credentials",
        "account": "Profile login, username, SSO permissions",
        "legal_compliance": "Regulatory subpoenas, GDPR privacy, litigation",
        "hr_payroll": "Employee benefits, salary leaves, onboarding",
        "facilities": "Office building maintenance, badge HVAC",
        "procurement": "Vendor sourcing, purchase orders, bidding",
        "investor_relations": "Shareholder quarterly earnings, SEC dividends",
        "public_relations": "Press releases, media inquiries, analysts",
        "customer_success": "Quarterly business reviews, adoption workshops",
        "sales_enablement": "Pitch decks, battlecards, territory quotas",
        "product_design": "Figma wireframes, usability studies, typography",
        "data_analytics": "Looker dashboards, ETL warehouse, cohorts",
        "quality_assurance": "Regression test plans, manual signoff",
        "supply_chain": "Warehouse inventory, freight logistics, customs",
        "tax_audit": "Corporate tax returns, statutory schedules",
        "partner_alliances": "Reseller channel agreements, co-marketing sponsorships",
        "corporate_travel": "Flight bookings, hotel reimbursements, visas",
        "executive_office": "Board meeting minutes, keynote governance"
    });

    Question {
        qtype: QuestionType::Choice,
        instructions:
            "Classify the primary routing department for this incoming customer or system event."
                .into(),
        criteria: Some(criteria),
        hypothesis_template: None,
    }
}

#[test]
fn test_shortlist_20_class_taxonomy_token_reduction_and_recall() {
    let q = build_20_class_enterprise_question();
    let options = q.render_options();
    assert_eq!(options.len(), 20);

    let full_option_tokens: usize = options.iter().map(|o| estimate_tokens(o)).sum();
    assert_eq!(full_option_tokens, 180);

    // Passthrough when top_k == 0 or top_k >= options.len()
    let passthrough = shortlist_options("Need a refund on my invoice", &q, &options, 0);
    assert_eq!(passthrough.original_cardinality, 20);
    assert_eq!(passthrough.shortlisted_cardinality, 20);
    assert_eq!(passthrough.kept_options.len(), 20);

    let items = FileSource::parse_file(Path::new("data/system1_golden_benchmark.jsonl"))
        .expect("Failed to parse golden benchmark dataset");
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);

    let target_ids = [
        "en_01", "en_02", "en_03", "en_04", "en_05", "en_06", "neg_01", "neg_02", "neg_03",
        "neg_04",
    ];

    let all_keys: Vec<String> = options
        .iter()
        .map(|o| o.split(':').next().unwrap().trim().to_string())
        .collect();

    let mut total_full_evals = 0usize;
    let mut total_shortlisted_evals = 0usize;

    for id in target_ids {
        let item = items
            .iter()
            .find(|it| it.id == id)
            .unwrap_or_else(|| panic!("Missing benchmark item {id}"));
        let text = item.state["text"].as_str().unwrap();
        let expected_dept = item
            .ground_truth
            .as_ref()
            .unwrap()
            .get("department")
            .and_then(|v| v.as_str())
            .unwrap();

        // 1. Stage-1 Bi-Encoder Top-4 Shortlisting
        let sl = shortlist_options(text, &q, &options, 4);
        assert_eq!(sl.original_cardinality, 20);
        assert_eq!(sl.shortlisted_cardinality, 4);
        assert_eq!(sl.kept_indices.len(), 4);
        assert_eq!(sl.kept_options.len(), 4);

        // Verify kept_indices are strictly ascending (preserving original option order)
        assert!(
            sl.kept_indices.windows(2).all(|w| w[0] < w[1]),
            "{id}: kept_indices must be sorted in ascending order: {:?}",
            sl.kept_indices
        );

        // Verify ~180 tokens -> ~36 tokens (5x token budget reduction)
        let shortlisted_tokens: usize = sl.kept_options.iter().map(|o| estimate_tokens(o)).sum();
        assert_eq!(shortlisted_tokens, 36);
        assert!(shortlisted_tokens < Checkpoint::English.head_budget());

        // 2. Verify ground-truth department is retained in kept_options across en_01..en_06 and neg_01..neg_04
        let kept_keys: Vec<&str> = sl
            .kept_options
            .iter()
            .map(|o| o.split(':').next().unwrap().trim())
            .collect();
        assert!(
            kept_keys.contains(&expected_dept),
            "{id}: expected ground-truth '{expected_dept}' in top-4 shortlist, got {kept_keys:?}"
        );

        // 3. Stage-2 Cross-Encoder NLI Reranking on the 4 shortlisted options + logit expansion
        total_full_evals += options.len();
        total_shortlisted_evals += sl.kept_options.len();

        let full_logits = clf.compute_option_logits(text, &q, &options).unwrap();
        let short_logits = clf
            .compute_option_logits(text, &q, &sl.kept_options)
            .unwrap();
        let expanded_logits = expand_shortlisted_logits(&short_logits, &sl.kept_indices, 20);
        assert_eq!(expanded_logits.len(), 20);

        // Verify non-shortlisted options receive min_logit - 6.0 and softmax mass >= 99.8% on shortlisted
        let min_short = short_logits.iter().copied().fold(f64::INFINITY, f64::min);
        for (idx, &val) in expanded_logits.iter().enumerate() {
            if !sl.kept_indices.contains(&idx) {
                assert!(
                    (val - (min_short - 6.0)).abs() < 1e-9,
                    "Non-shortlisted option {idx} should have logit min_logit - 6.0"
                );
            }
        }

        let max_z = expanded_logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = expanded_logits.iter().map(|&z| (z - max_z).exp()).sum();
        let probs: Vec<f64> = expanded_logits
            .iter()
            .map(|&z| (z - max_z).exp() / exp_sum)
            .collect();
        let shortlisted_mass: f64 = sl.kept_indices.iter().map(|&i| probs[i]).sum();
        assert!(
            shortlisted_mass >= 0.998,
            "{id}: shortlisted softmax mass should be >= 0.998, got {shortlisted_mass:.6}"
        );

        // Verify argmax accuracy is preserved and matches both full cross-encoder and ground truth
        let full_argmax = full_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let expanded_argmax = expanded_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;

        assert_eq!(
            all_keys[expanded_argmax], expected_dept,
            "{id}: expanded argmax '{}' != ground truth '{expected_dept}'",
            all_keys[expanded_argmax]
        );
        assert_eq!(
            expanded_argmax, full_argmax,
            "{id}: shortlisted + expanded argmax should match full 20-class argmax"
        );
    }

    // Confirm 5x reduction in cross-encoder candidate evaluations (200 -> 40 across 10 items)
    assert_eq!(total_full_evals, 200);
    assert_eq!(total_shortlisted_evals, 40);
    assert_eq!(total_full_evals / total_shortlisted_evals, 5);
}
