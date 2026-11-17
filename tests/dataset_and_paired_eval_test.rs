use std::collections::BTreeMap;
use std::path::PathBuf;

use lemmata::datasource::DataItem;
use lemmata::engine::{
    CalibrationMode, CheckSeverity, ClassifierConfig, DatasetImporter, DatasetSplitter,
    DatasetVerifier, DatasetVerifyConfig, DedupAction, ExperimentParticipant, ExperimentVerdict,
    ImportConfig, ImportSource, LayaClassifier, MinHashIndex, PairedExperimentConfig,
    PairedExperimentRunner, SliceBy, SplitConfig, SplitRatios, TrainedInputsFilter,
    compute_item_set_digest, compute_schema_hash, compute_state_merkle_sha256, file_sha256_hex,
    holm_bonferroni, infer_base_checkpoint_from_artifact, inv_normal_cdf, mcnemar_test, normal_cdf,
    normalize_state_for_leakage, normalize_template_for_grouping, parse_key_val_pairs,
    parse_value_map_specs, sha256_u64_prefix, write_dataset_jsonl,
};
use lemmata::model::native::{Backend, set_backend};
use lemmata::model::{
    Checkpoint, ChunkStrategy, DecisionAnswer, DecisionResult, Question, QuestionType,
};

struct TempTestDir {
    path: PathBuf,
}

impl TempTestDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("lemmata_pr2_{tag}_{}_{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp test dir");
        Self { path }
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn multi_question_schema() -> BTreeMap<String, Question> {
    BTreeMap::from([
        (
            "department".to_string(),
            Question::new_choice(
                "Which team owns this ticket?",
                vec![
                    "billing".into(),
                    "technical".into(),
                    "security".into(),
                    "account".into(),
                ],
            ),
        ),
        (
            "urgency".to_string(),
            Question::new_score(
                "Rate operational urgency from 0 to 3",
                vec![
                    "low".into(),
                    "normal".into(),
                    "high".into(),
                    "critical".into(),
                ],
            ),
        ),
        (
            "churnRisk".to_string(),
            Question::new_noul("Is the customer threatening to cancel?"),
        ),
    ])
}

fn make_synthetic_answer(
    qtype: QuestionType,
    keys: &[&str],
    raw_logits: Vec<f64>,
) -> DecisionAnswer {
    let max_z = raw_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = raw_logits.iter().map(|&z| (z - max_z).exp()).sum();
    let probs: Vec<f64> = raw_logits
        .iter()
        .map(|&z| (z - max_z).exp() / sum_exp)
        .collect();

    let mut prob_map = BTreeMap::new();
    let (choice, score, noul) = match qtype {
        QuestionType::Choice => {
            let mut best_i = 0;
            for (i, (&k, &p)) in keys.iter().zip(probs.iter()).enumerate() {
                prob_map.insert(k.to_string(), (p * 10000.0).round() / 10000.0);
                if p > probs[best_i] {
                    best_i = i;
                }
            }
            (Some(keys[best_i].to_string()), None, None)
        }
        QuestionType::Score => {
            let mut s = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                s += i as f64 * p;
                prob_map.insert(i.to_string(), (p * 10000.0).round() / 10000.0);
            }
            (None, Some(s), None)
        }
        QuestionType::Noul => {
            let pt = probs.get(1).copied().unwrap_or(0.5);
            prob_map.insert("false".into(), 1.0 - pt);
            prob_map.insert("true".into(), pt);
            (None, None, Some(pt))
        }
    };
    let confidence = probs.iter().copied().fold(0.0, f64::max);
    DecisionAnswer {
        qtype,
        choice,
        selected_labels: None,
        score,
        noul,
        confidence,
        probabilities: prob_map,
        act_probability: Some(1.0 - confidence),
        logits: Some(raw_logits),
    }
}

#[test]
fn normalization_digests_and_trained_filter_round_trip() {
    let raw = "\u{FEFF}  Hello\tWORLD!\nInvoice #INV-9921  ";
    assert_eq!(
        normalize_state_for_leakage(raw),
        "hello world! invoice #inv-9921"
    );

    let t1 = normalize_template_for_grouping(
        "Alert from alice@corp.io: Invoice #INV-9921 ($1,250.00) failed at https://pay.corp.io/tx/a8f90123bcdef012",
    );
    let t2 = normalize_template_for_grouping(
        "ALERT from bob@enterprise.org: Invoice #INV-4402 ($98.50) failed at https://pay.corp.io/tx/deadbeef12345678",
    );
    assert_eq!(
        t1, t2,
        "template masking collapses emails, URLs, hex IDs, and numbers"
    );

    let item_a =
        DataItem::new("a", serde_json::json!("Refund duplicate charge"), "t").with_ground_truth(
            BTreeMap::from([("department".into(), serde_json::json!("billing"))]),
        );
    let item_b =
        DataItem::new("b", serde_json::json!("API 504 crash on checkout"), "t").with_ground_truth(
            BTreeMap::from([("department".into(), serde_json::json!("technical"))]),
        );

    assert_eq!(
        compute_item_set_digest(&[item_a.clone(), item_b.clone()]),
        compute_item_set_digest(&[item_b.clone(), item_a.clone()]),
        "item_set_digest is order-invariant"
    );
    assert_eq!(
        compute_state_merkle_sha256(&[item_a.clone(), item_b.clone()]),
        compute_state_merkle_sha256(&[item_b.clone(), item_a.clone()]),
        "state_merkle_sha256 is order-invariant"
    );

    // Headered LMFT1 binary filter round-trip
    let tmp = TempTestDir::new("filter");
    let filter_path = tmp.path.join("trained_inputs.sha256.bin");
    let filter = TrainedInputsFilter::from_items(&[item_a.clone(), item_b.clone()]);
    filter.write_to_file(&filter_path).unwrap();
    let loaded = TrainedInputsFilter::load_from_file(&filter_path).unwrap();

    let norm_leak = DataItem::new(
        "t1",
        serde_json::json!("  refund   DUPLICATE charge\n"),
        "test",
    );
    let clean_item = DataItem::new(
        "t2",
        serde_json::json!("Update SSO member seat role"),
        "test",
    );
    let ov = loaded.check_overlap(&[item_a.clone(), norm_leak, clean_item]);
    assert!(ov.has_overlap());
    assert_eq!(ov.exact_overlap_ids, vec!["a"]);
    assert_eq!(ov.normalized_overlap_ids, vec!["t1"]);
    assert_eq!(ov.total_overlapping_items(), 2);

    // Flat raw little-endian u64 array compatibility (matches Python `struct.pack("<Q", ...)`)
    let raw_u64 = sha256_u64_prefix(b"custom python state");
    let flat_filter = TrainedInputsFilter::from_bytes(&raw_u64.to_le_bytes()).unwrap();
    let hit = flat_filter.check_overlap(&[DataItem::new(
        "py1",
        serde_json::json!("custom python state"),
        "py",
    )]);
    assert_eq!(hit.exact_overlap_ids, vec!["py1"]);
}

#[test]
fn import_csv_bom_mapping_score_offset_multilabel_and_metadata_isolation() {
    let tmp = TempTestDir::new("import");
    let csv_path = tmp.path.join("raw_tickets.csv");
    let out_jsonl = tmp.path.join("canonical.jsonl");

    // Includes UTF-8 BOM, padded headers, camelCase prefixed column `gt_churnRisk`,
    // 1-indexed stars `1..4`, custom value `fin` -> `billing`, and secret `tenant_id` metadata
    let csv_content = "\u{FEFF} ticket_no , body_text , queue_code , stars_1to4 , gt_churnRisk , tenant_id \n\
        101,\"Billed twice on April invoice\r\nBest regards,\nJohn\",fin,3,yes,cust_alpha\n\
        102,\"Production server 504 crash\",tech,4,0,cust_beta\n\
        103,\"   \",fin,2,no,cust_gamma\n";
    std::fs::write(&csv_path, csv_content).unwrap();

    let schema = multi_question_schema();
    let cfg = ImportConfig {
        schema: Some(schema),
        id_col: Some("ticket_no".into()),
        text_col: Some("body_text".into()),
        state_cols: vec![],
        meta_cols: vec!["tenant_id".into()],
        label_map: parse_key_val_pairs(&[
            "department=queue_code".into(),
            "urgency=stars_1to4".into(),
        ])
        .unwrap(),
        value_map: parse_value_map_specs(&[
            "department:fin=billing".into(),
            "department:tech=technical".into(),
        ])
        .unwrap(),
        int_labels_from_schema: false,
        score_offset: -1,
        multi_label_sep: None,
        normalize_nfc: true,
        clean_email: true,
        drop_empty: true,
        strict: true,
    };

    let (items, report) =
        DatasetImporter::import(&cfg, &ImportSource::File(csv_path), Some(&out_jsonl)).unwrap();
    assert_eq!(report.total_rows_read, 3);
    assert_eq!(report.dropped_empty, 1);
    assert_eq!(items.len(), 2);
    assert!(out_jsonl.exists());

    // Item 0 checks: numeric ID preserved as string, signature stripped, metadata isolated from state
    assert_eq!(items[0].id, "101");
    assert_eq!(
        items[0].state,
        serde_json::json!("Billed twice on April invoice")
    );
    assert!(
        !items[0].state_text().contains("cust_alpha"),
        "metadata column must never leak into model-visible state"
    );
    assert_eq!(
        items[0].metadata.get("tenant_id"),
        Some(&serde_json::json!("cust_alpha"))
    );
    let gt0 = items[0].ground_truth.as_ref().unwrap();
    assert_eq!(gt0.get("department"), Some(&serde_json::json!("billing")));
    assert_eq!(gt0.get("urgency"), Some(&serde_json::json!(2))); // 3 - 1 = 2
    assert_eq!(gt0.get("churnRisk"), Some(&serde_json::json!(true))); // camelCase preserved!

    // Multi-label + integer ClassLabel JSON import + NFC composition + HF dry-run
    let json_path = tmp.path.join("int_labels.json");
    std::fs::write(
        &json_path,
        serde_json::to_string(&serde_json::json!([
            {"id": 1, "text": "Cafe\u{0301} billing refund", "label": 0},
            {"id": 2, "text": "Server crash", "label": 1}
        ]))
        .unwrap(),
    )
    .unwrap();
    let single_q_schema = BTreeMap::from([(
        "dept".to_string(),
        Question::new_choice("Pick dept", vec!["billing".into(), "technical".into()]),
    )]);
    let (ml_items, _) = DatasetImporter::import(
        &ImportConfig {
            schema: Some(single_q_schema.clone()),
            int_labels_from_schema: true,
            normalize_nfc: true,
            strict: true,
            ..Default::default()
        },
        &ImportSource::File(json_path),
        None,
    )
    .unwrap();
    assert_eq!(ml_items[0].id, "1");
    assert_eq!(ml_items[0].state_text(), "Café billing refund");
    assert_eq!(
        ml_items[0].ground_truth.as_ref().unwrap().get("dept"),
        Some(&serde_json::json!("billing"))
    );

    // Multi-label semicolon CSV
    let ml_csv = tmp.path.join("ml.csv");
    std::fs::write(
        &ml_csv,
        "id,text,target\nm1,Refund and bug,billing;technical\n",
    )
    .unwrap();
    let (parsed_ml, _) = DatasetImporter::import(
        &ImportConfig {
            schema: Some(single_q_schema),
            multi_label_sep: Some(';'),
            strict: true,
            ..Default::default()
        },
        &ImportSource::File(ml_csv),
        None,
    )
    .unwrap();
    assert_eq!(
        parsed_ml[0].ground_truth.as_ref().unwrap().get("dept"),
        Some(&serde_json::json!(["billing", "technical"]))
    );

    // HF Hub dry-run check (zero network)
    let (_, hf_dry) = DatasetImporter::import(
        &ImportConfig::default(),
        &ImportSource::HuggingFace {
            repo_id: "PolyAI/banking77".into(),
            subset: Some("default".into()),
            split: Some("train".into()),
            revision: None,
            cache_dir: Some(tmp.path.clone()),
            dry_run: true,
        },
        None,
    )
    .unwrap();
    let dry_cmd = hf_dry.dry_run_command.expect("dry run command");
    assert!(dry_cmd.contains("refs/convert/parquet"));
    assert!(dry_cmd.contains("default/train/*.parquet"));
}

#[test]
fn split_groups_templates_near_duplicates_and_enforces_rare_class_train_floor() {
    let tmp = TempTestDir::new("split");
    let schema = multi_question_schema();

    // Build 60 items including:
    // - Explicit group_col pairs (`customer_id = "vip_group"`)
    // - Template duplicates differing only in invoice numbers
    // - MinHash near-duplicates (>4 words, minor paraphrase)
    // - Short <4-word items ("Thanks team", "Help asap") that MUST NOT collapse into one LSH hub
    // - Rare class `"security"` with only 2 examples (must put >=1 in train and >=1 in test!)
    let mut items = Vec::new();
    for i in 0..24 {
        let dept = if i % 2 == 0 { "billing" } else { "technical" };
        let urg = (i % 4) as i64;
        let churn = i % 3 == 0;
        let mut gt = BTreeMap::new();
        gt.insert("department".into(), serde_json::json!(dept));
        gt.insert("urgency".into(), serde_json::json!(urg));
        gt.insert("churnRisk".into(), serde_json::json!(churn));
        items.push(
            DataItem::new(
                format!("base_{i:02}"),
                serde_json::json!(format!(
                    "Unique ticket topic_{i} alpha_{} beta_{} gamma_{} delta_{} regarding {dept} workflow",
                    i * 7,
                    i * 13 + 3,
                    i * 29 + 5,
                    i * 41 + 9
                )),
                "synthetic",
            )
            .with_ground_truth(gt),
        );
    }

    // Add 3 template variants (differing only in invoice numbers & emails)
    for k in 0..3 {
        let mut gt = BTreeMap::new();
        gt.insert("department".into(), serde_json::json!("billing"));
        gt.insert("urgency".into(), serde_json::json!(2));
        gt.insert("churnRisk".into(), serde_json::json!(true));
        items.push(
            DataItem::new(
                format!("tmpl_{k}"),
                serde_json::json!(format!(
                    "Customer user{k}@acme.com reports duplicate charge on invoice #INV-{} for ${}",
                    9000 + k,
                    100 + k * 50
                )),
                "synthetic",
            )
            .with_ground_truth(gt),
        );
    }

    // Add 2 MinHash near-duplicates (high Jaccard on 5-grams, different trailing word)
    let near_a = "Our enterprise production checkout microservice cluster is throwing unhandled gateway timeout exceptions across every worker node today";
    let near_b = "Our enterprise production checkout microservice cluster is throwing unhandled gateway timeout exceptions across every worker node right now";
    let mh = MinHashIndex::new(42);
    let sim = MinHashIndex::jaccard_estimate(
        &mh.signature(&normalize_state_for_leakage(near_a)).unwrap(),
        &mh.signature(&normalize_state_for_leakage(near_b)).unwrap(),
    );
    assert!(sim >= 0.85, "expected Jaccard >= 0.85, got {sim}");
    assert!(
        mh.signature("thanks team").is_none(),
        "<4 word strings skip MinHash LSH"
    );

    for (k, txt) in [(0, near_a), (1, near_b)] {
        let mut gt = BTreeMap::new();
        gt.insert("department".into(), serde_json::json!("technical"));
        gt.insert("urgency".into(), serde_json::json!(3));
        gt.insert("churnRisk".into(), serde_json::json!(true));
        items.push(
            DataItem::new(format!("near_{k}"), serde_json::json!(txt), "synthetic")
                .with_ground_truth(gt),
        );
    }

    // Add 2 completely different texts sharing `customer_id = "cust_shared"`
    for (k, txt) in [
        (0, "First distinct question from shared account alpha zebra"),
        (
            1,
            "Totally unrelated second message about dashboard SSO login seats",
        ),
    ] {
        let mut gt = BTreeMap::new();
        gt.insert("department".into(), serde_json::json!("account"));
        gt.insert("urgency".into(), serde_json::json!(1));
        gt.insert("churnRisk".into(), serde_json::json!(false));
        let meta = BTreeMap::from([("customer_id".into(), serde_json::json!("cust_shared"))]);
        items.push(
            DataItem::new(format!("grp_{k}"), serde_json::json!(txt), "synthetic")
                .with_ground_truth(gt)
                .with_metadata(meta),
        );
    }

    // Add ultra-rare `"security"` class (only 2 distinct items across entire dataset!)
    for (k, text) in [
        (
            0,
            "Phishing credential compromise MFA bypass on executive mailbox",
        ),
        (
            1,
            "Unauthorized SSH key exfiltration detected in staging vault",
        ),
    ] {
        let mut gt = BTreeMap::new();
        gt.insert("department".into(), serde_json::json!("security"));
        gt.insert("urgency".into(), serde_json::json!(3));
        gt.insert("churnRisk".into(), serde_json::json!(true));
        items.push(
            DataItem::new(
                format!("sec_rare_{k}"),
                serde_json::json!(text),
                "synthetic",
            )
            .with_ground_truth(gt),
        );
    }

    let src_file = tmp.path.join("all.jsonl");
    write_dataset_jsonl(&src_file, &items).unwrap();

    let split_dir = tmp.path.join("splits_4way");
    let split_cfg = SplitConfig {
        ratios: SplitRatios::default(),
        seed: 42,
        group_col: Some("customer_id".into()),
        dedup_sim: 0.80,
        dedup_action: DedupAction::Group,
    };
    let manifest1 =
        DatasetSplitter::split_to_dir(&split_cfg, &src_file, &schema, &split_dir).unwrap();
    let out2 = DatasetSplitter::split_items(&split_cfg, &items, &schema, None).unwrap();
    assert_eq!(
        manifest1.splits["train"].item_set_digest, out2.manifest.splits["train"].item_set_digest,
        "deterministic seed produces identical splits"
    );

    // Verify all 4 splits exist
    for s in ["train", "val", "cal", "test"] {
        assert!(split_dir.join(format!("{s}.jsonl")).exists());
    }

    // Locate which split each item landed in
    let mut item_split = BTreeMap::new();
    for (s_name, s_items) in &out2.splits {
        for it in s_items {
            item_split.insert(it.id.clone(), s_name.clone());
        }
    }

    // Template cluster stayed together
    assert_eq!(item_split["tmpl_0"], item_split["tmpl_1"]);
    assert_eq!(item_split["tmpl_1"], item_split["tmpl_2"]);
    // MinHash near-duplicate cluster stayed together
    assert_eq!(item_split["near_0"], item_split["near_1"]);
    // Explicit customer_id cluster stayed together
    assert_eq!(item_split["grp_0"], item_split["grp_1"]);
    // Rare class `security` (2 items) put 1 in train and 1 in test (never 0 in train!)
    let sec_sup = &manifest1.per_question_support["department"]["security"];
    assert_eq!(sec_sup["train"], 1);
    assert_eq!(sec_sup["test"], 1);

    // Verify 3-way `--no-cal` split omits `cal`
    let no_cal_out = DatasetSplitter::split_items(
        &SplitConfig {
            ratios: SplitRatios::three_way_no_cal(),
            ..split_cfg
        },
        &items,
        &schema,
        None,
    )
    .unwrap();
    assert!(!no_cal_out.splits.contains_key("cal"));
    assert_eq!(no_cal_out.splits.len(), 3);

    // Verify `DatasetVerifier` reports ZERO cross-split leakage on `split_dir`!
    let verify_rep =
        DatasetVerifier::verify_dir(&DatasetVerifyConfig::default(), &split_dir, &schema).unwrap();
    assert!(
        !verify_rep
            .findings
            .iter()
            .any(|f| f.stage == "cross_split_leakage"),
        "expected 0 cross-split leakage findings, got {:?}",
        verify_rep.findings
    );
}

#[test]
fn verify_catches_all_five_preflight_failure_modes_and_truncation_collisions() {
    let tmp = TempTestDir::new("verify");
    let schema = BTreeMap::from([
        (
            "dept".to_string(),
            Question::new_choice(
                "Pick department",
                vec![
                    "billing".into(),
                    "technical".into(),
                    "dead_unused_option".into(),
                ],
            ),
        ),
        (
            "severity".to_string(),
            Question::new_score(
                "Score 0..2",
                vec!["low".into(), "med".into(), "high".into()],
            ),
        ),
    ]);

    // Build shared 450-word preamble exceeding English checkpoint's state_room (~370 tokens)
    // where tail diverges with opposing labels!
    let long_preamble = (0..450)
        .map(|i| format!("telemetry_word_{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let long_billing = format!("{long_preamble} --- ACTUAL TAIL: duplicate invoice charge refund");
    let long_tech = format!("{long_preamble} --- ACTUAL TAIL: fatal kernel panic crash");

    let train_items =
        vec![
            // Structured state containing target-named key `dept` -> TARGET_KEY_IN_STATE
            DataItem::new(
                "tr_1",
                serde_json::json!({"summary": "invoice issue", "dept": "billing"}),
                "t",
            )
            .with_ground_truth(BTreeMap::from([
                ("dept".into(), serde_json::json!("billing")),
                ("severity".into(), serde_json::json!(1)),
            ])),
            // Conflicting duplicate pair on identical state text -> Bayes ceiling < 1.0
            DataItem::new("tr_2a", serde_json::json!("ambiguous ticket text"), "t")
                .with_ground_truth(BTreeMap::from([
                    ("dept".into(), serde_json::json!("billing")),
                    ("severity".into(), serde_json::json!(2)),
                ])),
            DataItem::new("tr_2b", serde_json::json!("ambiguous ticket text"), "t")
                .with_ground_truth(BTreeMap::from([
                    ("dept".into(), serde_json::json!("technical")),
                    ("severity".into(), serde_json::json!(3)), // 1..=3 off-by-one 1-indexed score!
                ])),
            // Truncation collision pair
            DataItem::new("tr_long_a", serde_json::json!(long_billing), "t").with_ground_truth(
                BTreeMap::from([
                    ("dept".into(), serde_json::json!("billing")),
                    ("severity".into(), serde_json::json!(2)),
                ]),
            ),
            DataItem::new("tr_long_b", serde_json::json!(long_tech), "t").with_ground_truth(
                BTreeMap::from([
                    ("dept".into(), serde_json::json!("technical")),
                    ("severity".into(), serde_json::json!(2)),
                ]),
            ),
            // Template leak source
            DataItem::new(
                "tr_leak",
                serde_json::json!("Customer alice@foo.com billed twice on invoice #INV-100"),
                "t",
            )
            .with_ground_truth(BTreeMap::from([
                ("dept".into(), serde_json::json!("billing")),
                ("severity".into(), serde_json::json!(2)),
            ])),
        ];

    let cal_items = vec![
        // Tiny cal split (< 19 items -> conformal trivial floor!)
        DataItem::new(
            "cal_1",
            serde_json::json!("Unique calibration text one"),
            "c",
        )
        .with_ground_truth(BTreeMap::from([
            ("dept".into(), serde_json::json!("OOV_LABEL")),
            ("severity".into(), serde_json::json!(2)),
        ])),
    ];

    let test_items = vec![
        // Normalized template leak from `tr_leak`!
        DataItem::new(
            "te_leak",
            serde_json::json!("Customer bob@bar.org billed twice on invoice #INV-999"),
            "te",
        )
        .with_ground_truth(BTreeMap::from([
            ("dept".into(), serde_json::json!("billing")),
            ("severity".into(), serde_json::json!(2)),
        ])),
    ];

    let mut splits = BTreeMap::new();
    splits.insert("train".into(), train_items);
    splits.insert("cal".into(), cal_items);
    splits.insert("test".into(), test_items);

    let rep = DatasetVerifier::verify_splits(
        &DatasetVerifyConfig {
            checkpoint: Checkpoint::English,
            chunk_strategy: ChunkStrategy::TruncatePrefix,
            ..Default::default()
        },
        &splits,
        &schema,
        None,
        Some(&tmp.path),
    )
    .unwrap();

    assert!(!rep.ready);
    let codes: Vec<&str> = rep.findings.iter().map(|f| f.code.as_str()).collect();
    assert!(codes.contains(&"OOV_CHOICE_LABEL"), "codes={codes:?}");
    assert!(codes.contains(&"SCORE_ONE_INDEXED"), "codes={codes:?}");
    assert!(codes.contains(&"DEAD_SCHEMA_OPTION"), "codes={codes:?}");
    assert!(
        codes.contains(&"CONFLICTING_DUPLICATE_LABELS"),
        "codes={codes:?}"
    );
    assert!(codes.contains(&"TARGET_KEY_IN_STATE"), "codes={codes:?}");
    assert!(
        codes.contains(&"CAL_CONFORMAL_TRIVIAL_FLOOR"),
        "codes={codes:?}"
    );
    assert!(
        codes.contains(&"TRUNCATION_LABEL_COLLISION"),
        "codes={codes:?}"
    );
    assert!(
        codes.contains(&"CROSS_SPLIT_NORMALIZED_LEAKAGE"),
        "codes={codes:?}"
    );
    assert!(
        rep.questions["dept"].empirical_bayes_ceiling < 1.0,
        "conflicting duplicate lowers Bayes ceiling below 100%"
    );

    // Switching chunk_strategy from TruncatePrefix to HeadTail preserves the tail and eliminates the collision!
    let rep_head_tail = DatasetVerifier::verify_splits(
        &DatasetVerifyConfig {
            checkpoint: Checkpoint::English,
            chunk_strategy: ChunkStrategy::HeadTail,
            ..Default::default()
        },
        &splits,
        &schema,
        None,
        None,
    )
    .unwrap();
    assert_eq!(rep_head_tail.tokenizer.truncation_label_collisions, 0);

    let rendered = DatasetVerifier::format_table(&rep);
    assert!(rendered.contains("FAILED PREFLIGHT"));
}

#[test]
fn statistical_primitives_mcnemar_mid_p_normal_cdf_and_holm_match_exact_values() {
    // Zero discordant pairs
    let m0 = mcnemar_test(20, 0, 0, 5);
    assert_eq!(m0.p_exact, 1.0);
    assert_eq!(m0.p_mid, 1.0);
    assert!((m0.haldane_anscombe_or - 1.0).abs() < 1e-6);

    // b = 0 broken, c = 5 fixed: P(X=0) = 2^-5 = 0.03125
    // Exact two-sided = 2 * 0.03125 = 0.0625; Lancaster mid-p = 2*0 + 0.03125 = 0.03125 (< 0.05!)
    let m5 = mcnemar_test(15, 0, 5, 2);
    assert!((m5.p_exact - 0.0625).abs() < 1e-5);
    assert!((m5.p_mid - 0.03125).abs() < 1e-5);
    assert!((m5.haldane_anscombe_or - 11.0).abs() < 1e-5);

    // Symmetry: mcnemar(b, c) p-values == mcnemar(c, b)
    let m_rev = mcnemar_test(15, 5, 0, 2);
    assert_eq!(m5.p_exact, m_rev.p_exact);
    assert_eq!(m5.p_mid, m_rev.p_mid);

    // Normal CDF & Acklam Inverse Normal CDF round-trip
    for p in [0.001, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.999] {
        let z = inv_normal_cdf(p);
        let p_back = normal_cdf(z);
        assert!(
            (p - p_back).abs() < 1e-6,
            "normal_cdf(inv_normal_cdf({p})) = {p_back}"
        );
    }

    // Holm-Bonferroni step-down monotonicity: p = [0.01, 0.04, 0.03] (m = 3)
    // Sorted: 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> max(0.06, 0.04) = 0.06!
    let adj = holm_bonferroni(&[0.01, 0.04, 0.03]);
    assert!((adj[0] - 0.03).abs() < 1e-6);
    assert!((adj[2] - 0.06).abs() < 1e-6);
    assert!((adj[1] - 0.06).abs() < 1e-6);
}

#[test]
fn paired_eval_enforces_five_layer_leakage_guard_and_tamper_detection() {
    let tmp = TempTestDir::new("leak_guard");
    let schema = BTreeMap::from([(
        "dept".to_string(),
        Question::new_choice("Pick dept", vec!["billing".into(), "technical".into()]),
    )]);

    let train_item = DataItem::new(
        "tr_01",
        serde_json::json!("Duplicate billing invoice charge #99"),
        "train",
    )
    .with_ground_truth(BTreeMap::from([(
        "dept".into(),
        serde_json::json!("billing"),
    )]));
    let clean_test_item = DataItem::new(
        "te_01",
        serde_json::json!("Production API server crash timeout"),
        "test",
    )
    .with_ground_truth(BTreeMap::from([(
        "dept".into(),
        serde_json::json!("technical"),
    )]));

    let train_path = tmp.path.join("train.jsonl");
    let test_path = tmp.path.join("test.jsonl");
    write_dataset_jsonl(&train_path, std::slice::from_ref(&train_item)).unwrap();
    write_dataset_jsonl(&test_path, std::slice::from_ref(&clean_test_item)).unwrap();

    let art_dir = tmp.path.join("artifacts");
    std::fs::create_dir_all(&art_dir).unwrap();
    let adapter_path = art_dir.join("adapter_model.safetensors");
    std::fs::write(&adapter_path, b"fake_safetensors_payload").unwrap();
    let adapter_sha = file_sha256_hex(&adapter_path).unwrap();

    let filter = TrainedInputsFilter::from_items(std::slice::from_ref(&train_item));
    filter
        .write_to_file(&art_dir.join("trained_inputs.sha256.bin"))
        .unwrap();

    let train_sha = file_sha256_hex(&train_path).unwrap();
    let test_sha = file_sha256_hex(&test_path).unwrap();
    let tm_json = serde_json::json!({
        "manifest_version": "lemmata-ft-2026.1",
        "run_id": "ft_test_01",
        "base_model": {
            "checkpoint_slug": "english",
            "repo_id": "convaiinnovations/laya"
        },
        "schema": {
            "schema_sha256": compute_schema_hash(&schema)
        },
        "dataset_provenance": {
            "split_id": "sp_demo",
            "train": {
                "file_sha256": train_sha,
                "state_merkle_sha256": compute_state_merkle_sha256(std::slice::from_ref(&train_item))
            },
            "locked_test_guard": {
                "expected_test_file_sha256": test_sha,
                "trained_inputs_filter_file": "trained_inputs.sha256.bin"
            }
        },
        "artifacts": {
            "adapter_safetensors_sha256": adapter_sha
        }
    });
    std::fs::write(
        art_dir.join("training_manifest.json"),
        serde_json::to_string_pretty(&tm_json).unwrap(),
    )
    .unwrap();

    assert_eq!(
        infer_base_checkpoint_from_artifact(&art_dir).as_deref(),
        Some("english")
    );

    set_backend(Backend::Mock);
    let base =
        ExperimentParticipant::new("base", LayaClassifier::with_checkpoint(Checkpoint::English));
    let cand = ExperimentParticipant::new(
        "ft",
        LayaClassifier::with_checkpoint(Checkpoint::TypedDecisions),
    )
    .with_artifact_dir(&art_dir);

    // 1. Clean test set passes provenance guard!
    let ok_rep = PairedExperimentRunner::run_paired(
        &PairedExperimentConfig {
            bootstrap_samples: 100,
            permutations: 200,
            ..Default::default()
        },
        &base,
        &cand,
        std::slice::from_ref(&clean_test_item),
        None,
        &schema,
        Some(&test_path),
        None,
    )
    .unwrap();
    assert!(ok_rep.provenance.verified);
    assert!(!ok_rep.provenance.leakage_detected);

    // 2. Passing `train.jsonl` as test file is hard-rejected!
    let err_split = PairedExperimentRunner::run_paired(
        &PairedExperimentConfig::default(),
        &base,
        &cand,
        std::slice::from_ref(&train_item),
        None,
        &schema,
        Some(&train_path),
        None,
    )
    .unwrap_err();
    assert!(
        err_split.to_string().contains("leakage"),
        "unexpected error: {err_split}"
    );

    // 3. Passing a new file containing a whitespace/case-tweaked training item is caught by `trained_inputs.sha256.bin`!
    let sneaky_item = DataItem::new(
        "sneaky_99",
        serde_json::json!("  DUPLICATE billing   invoice charge #99\n"),
        "custom",
    )
    .with_ground_truth(BTreeMap::from([(
        "dept".into(),
        serde_json::json!("billing"),
    )]));
    let err_row = PairedExperimentRunner::run_paired(
        &PairedExperimentConfig::default(),
        &base,
        &cand,
        &[clean_test_item.clone(), sneaky_item.clone()],
        None,
        &schema,
        None,
        None,
    )
    .unwrap_err();
    assert!(err_row.to_string().contains("sneaky_99"));

    // 4. `allow_leakage: true` records override warning instead of failing
    let overridden = PairedExperimentRunner::run_paired(
        &PairedExperimentConfig {
            allow_leakage: true,
            bootstrap_samples: 50,
            permutations: 50,
            ..Default::default()
        },
        &base,
        &cand,
        &[clean_test_item.clone(), sneaky_item],
        None,
        &schema,
        None,
        None,
    )
    .unwrap();
    assert!(overridden.provenance.leakage_detected);
    assert!(overridden.provenance.leakage_overridden);

    // 5. Tampering with `adapter_model.safetensors` triggers hard integrity error
    std::fs::write(&adapter_path, b"corrupted_bytes").unwrap();
    let err_tamper = PairedExperimentRunner::run_paired(
        &PairedExperimentConfig::default(),
        &base,
        &cand,
        std::slice::from_ref(&clean_test_item),
        None,
        &schema,
        Some(&test_path),
        None,
    )
    .unwrap_err();
    assert!(err_tamper.to_string().contains("integrity check failed"));
}

#[test]
fn paired_eval_calibrates_overconfident_finetune_and_emits_promote_with_calibration() {
    let schema = multi_question_schema();
    let mut base_cfg = ClassifierConfig::with_checkpoint(Checkpoint::English);
    base_cfg.temperature = 1.0;
    base_cfg.temperature_by_options.clear();
    let cand_cfg = base_cfg.clone();

    let dept_keys = ["billing", "technical", "security", "account"];

    // Build 40 calibration items and 40 test items:
    // - Base model: ~60% accurate on dept, well-tempered logits
    // - Fine-tuned Candidate: 85% accurate on dept, BUT 6x sharpened overconfident logits (`[-6, +6]`),
    //   causing high raw ECE & NLL penalty on its 15% errors until post-hoc `cal` temperature scaling!
    let build_split = |prefix: &str, n: usize| {
        let mut items = Vec::with_capacity(n);
        let mut base_res = Vec::with_capacity(n);
        let mut cand_res = Vec::with_capacity(n);

        for i in 0..n {
            let true_dept_idx = i % 4;
            let true_dept = dept_keys[true_dept_idx];
            let true_urg = i % 4;
            let true_churn = i % 2 == 0;

            let gt = BTreeMap::from([
                ("department".into(), serde_json::json!(true_dept)),
                ("urgency".into(), serde_json::json!(true_urg)),
                ("churnRisk".into(), serde_json::json!(true_churn)),
            ]);
            let meta = BTreeMap::from([(
                "region".into(),
                serde_json::json!(if i % 2 == 0 { "us" } else { "eu" }),
            )]);
            let item = DataItem::new(
                format!("{prefix}_{i:02}"),
                serde_json::json!(format!("Support event {prefix} #{i} for {true_dept}")),
                prefix,
            )
            .with_ground_truth(gt)
            .with_metadata(meta);

            // Base: right on 60% (`i % 5 < 3`), wrong on 40%, mild logit margin 1.1
            let base_right = i % 5 < 3;
            let mut b_dept_logits = vec![0.0; 4];
            let b_pick = if base_right {
                true_dept_idx
            } else {
                (true_dept_idx + 1) % 4
            };
            b_dept_logits[b_pick] = 1.1;

            // Candidate: right on 85% (`i % 20 < 17`), wrong on 15%, extreme raw logit margin 8.5!
            let cand_right = (i % 20) < 17;
            let mut c_dept_logits = vec![0.0; 4];
            let c_pick = if cand_right {
                true_dept_idx
            } else {
                (true_dept_idx + 1) % 4
            };
            c_dept_logits[c_pick] = 8.5;

            let mut b_urg_logits = vec![0.0; 4];
            b_urg_logits[if base_right {
                true_urg
            } else {
                (true_urg + 2) % 4
            }] = 1.2;
            let mut c_urg_logits = vec![0.0; 4];
            c_urg_logits[if cand_right {
                true_urg
            } else {
                (true_urg + 1) % 4
            }] = 6.0;

            let b_churn_logits = if base_right == true_churn {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            };
            let c_churn_logits = if cand_right == true_churn {
                vec![0.0, 7.5]
            } else {
                vec![7.5, 0.0]
            };

            let b_dec = DecisionResult {
                model: "base".into(),
                answers: BTreeMap::from([
                    (
                        "department".into(),
                        make_synthetic_answer(QuestionType::Choice, &dept_keys, b_dept_logits),
                    ),
                    (
                        "urgency".into(),
                        make_synthetic_answer(
                            QuestionType::Score,
                            &["0", "1", "2", "3"],
                            b_urg_logits,
                        ),
                    ),
                    (
                        "churnRisk".into(),
                        make_synthetic_answer(
                            QuestionType::Noul,
                            &["false", "true"],
                            b_churn_logits,
                        ),
                    ),
                ]),
                input_tokens: 20,
                latency_ms: 10.0,
            };
            let c_dec = DecisionResult {
                model: "cand".into(),
                answers: BTreeMap::from([
                    (
                        "department".into(),
                        make_synthetic_answer(QuestionType::Choice, &dept_keys, c_dept_logits),
                    ),
                    (
                        "urgency".into(),
                        make_synthetic_answer(
                            QuestionType::Score,
                            &["0", "1", "2", "3"],
                            c_urg_logits,
                        ),
                    ),
                    (
                        "churnRisk".into(),
                        make_synthetic_answer(
                            QuestionType::Noul,
                            &["false", "true"],
                            c_churn_logits,
                        ),
                    ),
                ]),
                input_tokens: 20,
                latency_ms: 10.0,
            };

            base_res.push((item.id.clone(), b_dec));
            cand_res.push((item.id.clone(), c_dec));
            items.push(item);
        }
        (items, base_res, cand_res)
    };

    let (cal_items, base_cal_res, cand_cal_res) = build_split("cal", 40);
    let (test_items, base_test_res, cand_test_res) = build_split("test", 40);

    let report = PairedExperimentRunner::evaluate_precomputed_pair(
        &PairedExperimentConfig {
            calibration_mode: CalibrationMode::Both,
            bootstrap_samples: 400,
            permutations: 500,
            confidence: 0.95,
            alpha: 0.10,
            target_accuracy: 0.80,
            seed: 42,
            slice_by: Some("field:region".parse::<SliceBy>().unwrap()),
            ..Default::default()
        },
        "base-laya",
        "ft-laya-lora",
        &base_cfg,
        &cand_cfg,
        &base_test_res,
        &cand_test_res,
        &test_items,
        Some((&cal_items, &base_cal_res, &cand_cal_res)),
        &schema,
    )
    .unwrap();

    let raw = report.raw_comparison.as_ref().expect("raw regime present");
    let cal = report
        .calibrated_comparison
        .as_ref()
        .expect("calibrated regime present");

    // Raw candidate is much more accurate (+25%), but overconfident (raw ECE > base raw ECE, high raw NLL)
    assert!(raw.accuracy.delta > 0.20);
    assert!(raw.ece.candidate > raw.ece.base);
    // Post-hoc `cal` temperature scaling cools Candidate's logits (`T* > 2.0`), slashing both ECE and NLL!
    assert!(
        cal.candidate_fitted_temperatures.values().any(|&t| t > 2.0),
        "expected fitted T* > 2.0 on sharpened candidate logits, got {:?}",
        cal.candidate_fitted_temperatures
    );
    assert!(cal.nll.candidate < raw.nll.candidate);
    assert!(cal.ece.candidate < raw.ece.candidate);
    assert!(
        cal.nll.delta < 0.0,
        "calibrated candidate beats base on NLL"
    );
    assert!(cal.overall_mcnemar.p_mid < 0.01);
    assert!(cal.e_aurc.base >= 0.0 && cal.e_aurc.candidate >= 0.0);
    assert!(cal.score_mae.as_ref().unwrap().delta < 0.0);
    assert!(cal.score_rps.as_ref().unwrap().delta < 0.0);

    // Conformal prediction sets on calibrated candidate are sharper (smaller set size, higher singleton rate)
    let sr = cal
        .selective_risk
        .as_ref()
        .expect("selective risk computed");
    assert!(sr.conformal_set_size.candidate <= sr.conformal_set_size.base);
    assert!(sr.conformal_singleton_rate.candidate >= sr.conformal_singleton_rate.base);

    // Holm-Bonferroni populated across all 3 questions
    assert_eq!(cal.questions.len(), 3);
    for qr in cal.questions.values() {
        assert!(qr.mcnemar.p_mid_holm.is_some());
        assert!(qr.p_perm_nll_holm.is_some());
    }
    assert_eq!(cal.questions["department"].labels.len(), 4);
    assert_eq!(cal.slices.len(), 2); // `eu` and `us` from `item.metadata["region"]`!

    assert_eq!(report.verdict, ExperimentVerdict::PromoteWithCalibration);

    // Verify all 3 report formatters (`table`, `md`, `json`) and file export render cleanly
    let tmp = TempTestDir::new("export");
    let md_out = tmp.path.join("report.md");
    PairedExperimentRunner::export_report(&report, &md_out).unwrap();
    assert!(md_out.exists());
    let tbl = PairedExperimentRunner::format_table(&report);
    let md = PairedExperimentRunner::format_markdown(&report);
    let js = PairedExperimentRunner::format_json(&report).unwrap();
    assert!(tbl.contains("PROMOTE (WITH CALIBRATION)"));
    assert!(md.contains("## Aggregate Paired Metrics"));
    assert!(js.contains("\"PROMOTE_WITH_CALIBRATION\""));
}

#[test]
fn multi_label_paired_eval_drop_exact_and_regression_verdict() {
    let schema = BTreeMap::from([(
        "tags".to_string(),
        Question::new_choice(
            "Select applicable tags",
            vec!["billing".into(), "refund".into(), "technical".into()],
        ),
    )]);
    let mut cfg = ClassifierConfig::with_checkpoint(Checkpoint::English);
    cfg.multi_label = true;
    cfg.multi_label_threshold = 0.50;
    cfg.temperature = 1.0;
    cfg.temperature_by_options.clear();

    // Test DedupAction::DropExact on duplicate multi-label items
    let dup1 =
        DataItem::new("d1", serde_json::json!("Refund duplicate invoice"), "s").with_ground_truth(
            BTreeMap::from([("tags".into(), serde_json::json!(["billing", "refund"]))]),
        );
    let dup2 = DataItem::new("d2", serde_json::json!("  refund DUPLICATE invoice "), "s")
        .with_ground_truth(BTreeMap::from([(
            "tags".into(),
            serde_json::json!(["billing", "refund"]),
        )]));
    let other =
        DataItem::new("d3", serde_json::json!("API server crash 500"), "s").with_ground_truth(
            BTreeMap::from([("tags".into(), serde_json::json!(["technical"]))]),
        );
    let split_out = DatasetSplitter::split_items(
        &SplitConfig {
            ratios: SplitRatios::three_way_no_cal(),
            dedup_action: DedupAction::DropExact,
            ..Default::default()
        },
        &[dup1, dup2, other],
        &schema,
        None,
    )
    .unwrap();
    assert_eq!(split_out.manifest.dedup.dropped_exact_duplicates, 1);

    // Verify strict mode elevates warnings to failure via `CheckSeverity::Warn`
    let v_rep = DatasetVerifier::verify_splits(
        &DatasetVerifyConfig {
            multi_label: true,
            strict: true,
            ..Default::default()
        },
        &split_out.splits,
        &schema,
        None,
        None,
    )
    .unwrap();
    assert!(
        v_rep
            .findings
            .iter()
            .any(|f| f.severity == CheckSeverity::Warn || f.severity == CheckSeverity::Fail)
    );
    assert!(!v_rep.ready);

    // Build 20 multi-label test items where Base gets 100% right and Candidate regresses on 12/20!
    let mut test_items = Vec::new();
    let mut base_res = Vec::new();
    let mut cand_res = Vec::new();
    let tag_keys = ["billing", "refund", "technical"];

    for i in 0..20 {
        let it = DataItem::new(
            format!("ml_{i}"),
            serde_json::json!(format!("Item {i}")),
            "t",
        )
        .with_ground_truth(BTreeMap::from([(
            "tags".into(),
            serde_json::json!(["billing", "refund"]),
        )]));
        let b_ans = make_synthetic_answer(QuestionType::Choice, &tag_keys, vec![3.0, 3.0, -3.0]);
        let c_logits = if i < 12 {
            vec![-3.0, -3.0, 4.0] // high-confidence regression!
        } else {
            vec![3.0, 3.0, -3.0]
        };
        let c_ans = make_synthetic_answer(QuestionType::Choice, &tag_keys, c_logits);
        base_res.push((
            it.id.clone(),
            DecisionResult {
                model: "base".into(),
                answers: BTreeMap::from([("tags".into(), b_ans)]),
                input_tokens: 5,
                latency_ms: 1.0,
            },
        ));
        cand_res.push((
            it.id.clone(),
            DecisionResult {
                model: "cand".into(),
                answers: BTreeMap::from([("tags".into(), c_ans)]),
                input_tokens: 5,
                latency_ms: 1.0,
            },
        ));
        test_items.push(it);
    }

    let reg_report = PairedExperimentRunner::evaluate_precomputed_pair(
        &PairedExperimentConfig {
            bootstrap_samples: 200,
            permutations: 300,
            ..Default::default()
        },
        "base-ml",
        "broken-cand-ml",
        &cfg,
        &cfg,
        &base_res,
        &cand_res,
        &test_items,
        None,
        &schema,
    )
    .unwrap();

    assert_eq!(reg_report.verdict, ExperimentVerdict::Regressed);
    assert!(reg_report.primary.multi_label_f1.as_ref().unwrap().delta < -0.40);
    assert_eq!(reg_report.primary.overall_mcnemar.n10_broken, 12);
    assert_eq!(reg_report.primary.top_regressions.len(), 5);
}

#[test]
fn regression_paired_eval_custom_temperature_score_pwin_and_layer4_base_header() {
    use serde_json::json;
    // 1. Bug #1a & #1b: Adapted/custom `English` candidate (`is_custom_or_adapted() == true`,
    // empty `temperature_by_options`) must keep `T = 1.0` across `score:` and `noul:` (both raw
    // and `< min_req` cal fallback), whereas un-adapted `English` base uses `[1.6369, 1.25143, 1.9834]`.
    // Also Bug #1c: `Score` `p_win` must use rounded-level probability `probs[pred_score.round()]`.
    let mut schema = BTreeMap::new();
    schema.insert(
        "sev".to_string(),
        Question::new_score(
            "Severity",
            vec!["low".into(), "medium".into(), "high".into()],
        ),
    );
    schema.insert("urgent".to_string(), Question::new_noul("Is urgent?"));

    let mut base_cfg = ClassifierConfig::with_checkpoint(Checkpoint::English);
    base_cfg.temperature_by_options.clear(); // force fallback to `default_temperatures()` on base

    let mut cand_cfg = base_cfg.clone();
    cand_cfg.is_custom_model = true; // adapted/custom model -> must fall back to `T = 1.0`, NOT `[1.6369, 1.2514, 1.9834]`!

    // Bimodal score logits `[6.0, 0.0, 6.0]` -> `probs ≈ [0.499, 0.002, 0.499]`, `pred_score = 1.0` (correct for gt=1!),
    // so `probs[pred_idx=1] ≈ 0.002` (ECE ≈ `|1.0 - 0.002| = 0.998`), NOT `argmax ≈ 0.499` (ECE ≈ `0.501`)!
    // Noul logits `[0.0, 2.0]` with gt=true:
    // - At `T = 1.0` (adapted candidate): `nll = -ln(sigmoid(2.0)) = 0.126928`
    // - At `T = 1.9834` (zero-shot English noul default): `nll = -ln(sigmoid(2.0 / 1.9834)) = 0.3106`
    let mut items = Vec::new();
    let mut base_res = Vec::new();
    let mut cand_res = Vec::new();
    for i in 0..5 {
        let mut gt = BTreeMap::new();
        gt.insert("sev".into(), json!(1));
        gt.insert("urgent".into(), json!(true));
        let it = DataItem::new(format!("s{i}"), json!(format!("item {i}")), "test")
            .with_ground_truth(gt);
        let mut answers = BTreeMap::new();
        answers.insert(
            "sev".into(),
            DecisionAnswer {
                qtype: QuestionType::Score,
                choice: None,
                selected_labels: None,
                score: Some(1.0),
                noul: None,
                confidence: 0.5,
                probabilities: BTreeMap::new(),
                act_probability: None,
                logits: Some(vec![6.0, 0.0, 6.0]),
            },
        );
        answers.insert(
            "urgent".into(),
            DecisionAnswer {
                qtype: QuestionType::Noul,
                choice: None,
                selected_labels: None,
                score: None,
                noul: Some(0.8808),
                confidence: 0.8,
                probabilities: BTreeMap::new(),
                act_probability: None,
                logits: Some(vec![0.0, 2.0]),
            },
        );
        let dec = DecisionResult {
            model: "test".into(),
            answers,
            input_tokens: 4,
            latency_ms: 1.0,
        };
        base_res.push((it.id.clone(), dec.clone()));
        cand_res.push((it.id.clone(), dec));
        items.push(it);
    }

    // Pass 5 `cal` items (< 10 `min_req`!) so `fit_posthoc_on_cal` exercises `prior_t` fallback for `score:3-5` and `noul:2`!
    let rep = PairedExperimentRunner::evaluate_precomputed_pair(
        &PairedExperimentConfig {
            calibration_mode: CalibrationMode::Both,
            bootstrap_samples: 50,
            permutations: 50,
            ..Default::default()
        },
        "base-en",
        "cand-adapted",
        &base_cfg,
        &cand_cfg,
        &base_res,
        &cand_res,
        &items,
        Some((&items, &base_res, &cand_res)),
        &schema,
    )
    .unwrap();

    let cal_comp = rep.calibrated_comparison.as_ref().unwrap();
    // Base (< min_req) must keep Laya English score (`1.25143`) and noul (`1.9834`) priors, NOT Choice (`1.6369`)!
    assert!((cal_comp.base_fitted_temperatures["score:3-5"] - 1.25143).abs() < 1e-3);
    assert!((cal_comp.base_fitted_temperatures["noul:2"] - 1.9834).abs() < 1e-3);
    // Adapted candidate (< min_req) must keep `T = 1.0` on BOTH buckets!
    assert!((cal_comp.candidate_fitted_temperatures["score:3-5"] - 1.0).abs() < 1e-6);
    assert!((cal_comp.candidate_fitted_temperatures["noul:2"] - 1.0).abs() < 1e-6);

    // Candidate NLL on `urgent` at T=1.0 is `-ln(sigmoid(2.0)) ≈ 0.1269` vs base at T=1.9834 `≈ 0.3106`
    let raw_urgent = &rep.raw_comparison.as_ref().unwrap().questions["urgent"];
    assert!((raw_urgent.nll.candidate - 0.1269).abs() < 1e-3);
    assert!((raw_urgent.nll.base - 0.3106).abs() < 1e-3);

    // On a Score-only schema, rounded-level `p_win` on `[6.0, 0.0, 6.0]` (`pred_idx = 1`) gives
    // `p_win ≈ 0.0025` (100% correct), so ECE is `|1.0 - 0.0025| ≈ 0.9975` (would be `0.501` if argmax `0.499`!).
    let score_only_schema = BTreeMap::from([("sev".to_string(), schema["sev"].clone())]);
    let score_rep = PairedExperimentRunner::evaluate_precomputed_pair(
        &PairedExperimentConfig {
            calibration_mode: CalibrationMode::Raw,
            bootstrap_samples: 20,
            permutations: 20,
            ..Default::default()
        },
        "base-en",
        "cand-adapted",
        &base_cfg,
        &cand_cfg,
        &base_res,
        &cand_res,
        &items,
        None,
        &score_only_schema,
    )
    .unwrap();
    assert!(
        score_rep.primary.ece.candidate > 0.95,
        "Score p_win must use rounded-level probability (~0.0025), got ECE={}",
        score_rep.primary.ece.candidate
    );

    // 2. Bug #2: Layer 4 `remote_base_header_sha256`, `status != completed`, missing adapter weights, and tampered calibration!
    let tmp = std::env::temp_dir().join(format!("lemmata_peval_l4_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let base_dir = tmp.join("base");
    let art_dir = tmp.join("art");
    std::fs::create_dir_all(&base_dir).unwrap();
    std::fs::create_dir_all(&art_dir).unwrap();

    // Write valid minimal safetensors file: 8-byte header_len + `{}`
    let write_min_st = |p: &std::path::Path, hdr: &[u8]| {
        let mut bytes = (hdr.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(hdr);
        std::fs::write(p, bytes).unwrap();
    };
    write_min_st(&base_dir.join("model.safetensors"), b"{}");
    let real_base_hdr =
        lemmata::model::safetensors_header_sha256(&base_dir.join("model.safetensors")).unwrap();

    write_min_st(&art_dir.join("adapter_model.safetensors"), b"{}");
    let ad_sha =
        lemmata::engine::file_sha256_hex(&art_dir.join("adapter_model.safetensors")).unwrap();
    std::fs::write(
        art_dir.join("lemmata_calibration.json"),
        b"{\"temperature\":1.0}",
    )
    .unwrap();
    let cal_sha =
        lemmata::engine::file_sha256_hex(&art_dir.join("lemmata_calibration.json")).unwrap();

    let write_tm = |status: &str, remote_hdr: &str, cal_digest: &str| {
        std::fs::write(
            art_dir.join("training_manifest.json"),
            serde_json::to_string_pretty(&json!({
                "manifest_version": "lemmata-ft-2026.1",
                "run_id": "r1",
                "status": status,
                "base_model": {
                    "checkpoint_slug": "english",
                    "repo_id": "convaiinnovations/laya",
                    "remote_base_header_sha256": remote_hdr
                },
                "artifacts": {
                    "adapter_safetensors_sha256": ad_sha,
                    "calibration_sha256": cal_digest
                }
            }))
            .unwrap(),
        )
        .unwrap();
    };

    // Case A: `status != "completed"` -> error
    write_tm("spot_interrupted", &real_base_hdr, &cal_sha);
    assert!(
        lemmata::engine::verify_experiment_provenance_with_base_dir(
            &items,
            None,
            &schema,
            Some("convaiinnovations/laya"),
            None,
            Some(&art_dir),
            Some(&base_dir),
            false,
        )
        .is_err()
    );

    // Case B: Layer 4 `remote_base_header_sha256` mismatch -> error
    write_tm("completed", &"0".repeat(64), &cal_sha);
    let l4_err = lemmata::engine::verify_experiment_provenance_with_base_dir(
        &items,
        None,
        &schema,
        Some("convaiinnovations/laya"),
        None,
        Some(&art_dir),
        Some(&base_dir),
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(
        l4_err.contains("Base checkpoint header mismatch"),
        "{l4_err}"
    );

    // Case C: Tampered `calibration_sha256` -> error
    write_tm("completed", &real_base_hdr, &"f".repeat(64));
    assert!(
        lemmata::engine::verify_experiment_provenance_with_base_dir(
            &items,
            None,
            &schema,
            Some("convaiinnovations/laya"),
            None,
            Some(&art_dir),
            Some(&base_dir),
            false,
        )
        .is_err()
    );

    // Case D: Deleted `adapter_model.safetensors` -> error (never silently skipped!)
    write_tm("completed", &real_base_hdr, &cal_sha);
    std::fs::remove_file(art_dir.join("adapter_model.safetensors")).unwrap();
    assert!(
        lemmata::engine::verify_experiment_provenance_with_base_dir(
            &items,
            None,
            &schema,
            Some("convaiinnovations/laya"),
            None,
            Some(&art_dir),
            Some(&base_dir),
            false,
        )
        .is_err()
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
