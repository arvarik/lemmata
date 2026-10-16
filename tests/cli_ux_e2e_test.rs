use clap::Parser;
use lemma::cli::{Cli, Commands, GuardArgs};
use std::path::Path;
use std::process::Command;

#[test]
fn test_cli_flag_aliases_and_policy_default() {
    // 1. Verify `lemma route` alias maps to `Commands::Hierarchical`
    let cli_route = Cli::try_parse_from([
        "lemma",
        "route",
        "run",
        "--schema",
        "schema/banking_hierarchy.json",
    ])
    .expect("`lemma route run` alias must parse");
    assert!(matches!(cli_route.command, Commands::Hierarchical(_)));

    // 2. Verify `lemma guard` defaults to existing `schema/policy.json`
    let cli_guard = Cli::try_parse_from(["lemma", "guard"]).expect("`lemma guard` must parse");
    if let Commands::Guard(GuardArgs { policy, .. }) = cli_guard.command {
        assert_eq!(policy, Path::new("schema/policy.json"));
        assert!(policy.exists(), "Default policy path must exist on disk");
    } else {
        panic!("Expected Commands::Guard");
    }

    // 3. Verify documented README aliases (`--in-data`, `--test-data`, `--eval-data`, `--target-acc`, `--receipts`, `--attest-key`)
    assert!(
        Cli::try_parse_from([
            "lemma",
            "ood",
            "fit",
            "--in-data",
            "data/customer_support.jsonl",
            "--fpr-rate",
            "0.05",
        ])
        .is_ok()
    );
    assert!(
        Cli::try_parse_from([
            "lemma",
            "ood",
            "eval",
            "--test-data",
            "data/banking_tickets.jsonl",
        ])
        .is_ok()
    );
    assert!(
        Cli::try_parse_from([
            "lemma",
            "cascade",
            "tune",
            "--eval-data",
            "data/customer_support.jsonl",
            "--target-acc",
            "0.95",
        ])
        .is_ok()
    );
    assert!(
        Cli::try_parse_from([
            "lemma",
            "cascade",
            "run",
            "--data",
            "data/customer_support.jsonl",
            "--threshold",
            "0.45",
        ])
        .is_ok()
    );
    assert!(
        Cli::try_parse_from([
            "lemma",
            "attest",
            "verify",
            "--receipts",
            "logs/receipts.jsonl",
            "--attest-key",
            "secret_123",
        ])
        .is_ok()
    );
}

#[test]
fn test_cli_json_stdout_free_of_banner_pollution() {
    let bin = env!("CARGO_BIN_EXE_lemma");

    // 1. `lemma test --json` must emit pure JSON on stdout (banners go to stderr)
    let out_test = Command::new(bin)
        .args([
            "test",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma test --json");
    assert!(out_test.status.success());
    let parsed_test: serde_json::Value = serde_json::from_slice(&out_test.stdout)
        .expect("lemma test --json stdout must be valid JSON without leading banners");
    assert!(parsed_test.get("accuracy").is_some());

    // 2. `lemma guard --format json` must emit pure JSON array on stdout
    let out_guard = Command::new(bin)
        .args([
            "guard",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute lemma guard --format json");
    assert!(out_guard.status.success());
    let parsed_guard: serde_json::Value = serde_json::from_slice(&out_guard.stdout)
        .expect("lemma guard --format json stdout must be valid JSON");
    assert!(parsed_guard.is_array());
}

#[test]
fn test_arena_sweep_baseline_adversarial_and_cost_calibration_e2e() {
    let bin = env!("CARGO_BIN_EXE_lemma");
    let tmp_dir = std::env::temp_dir().join(format!("lemma_e2e_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp_dir);
    let baseline_path = tmp_dir.join("baseline.json");
    let config_path = tmp_dir.join("calibrated_cost.json");

    // #1 Head-to-Head Model Arena (`lemma compare --models ... --json`)
    let out_arena = Command::new(bin)
        .args([
            "compare",
            "--models",
            "english,english",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma compare --json");
    assert!(out_arena.status.success());
    let parsed_arena: serde_json::Value =
        serde_json::from_slice(&out_arena.stdout).expect("Arena output must be valid JSON");
    assert_eq!(parsed_arena["models"].as_array().unwrap().len(), 2);
    assert!(
        (parsed_arena["models"][1]["agreement_with_ref"]
            .as_f64()
            .unwrap()
            - 1.0)
            .abs()
            < 1e-6
    );

    // #2 Pareto Frontier Sweep (`lemma cascade sweep --json`)
    let out_sweep = Command::new(bin)
        .args([
            "cascade",
            "sweep",
            "--file",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--target-accuracy",
            "0.90",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma cascade sweep --json");
    assert!(out_sweep.status.success());
    let parsed_sweep: serde_json::Value =
        serde_json::from_slice(&out_sweep.stdout).expect("Cascade sweep output must be valid JSON");
    let sweep_arr = parsed_sweep.as_array().expect("Sweep JSON must be array");
    assert_eq!(sweep_arr.len(), 19);
    assert!(
        sweep_arr
            .iter()
            .any(|p| p["is_optimal"].as_bool() == Some(true))
    );

    // #3 CI Regression Gate & Baseline Diff (`lemma test --save-baseline` + `--baseline`)
    let out_save = Command::new(bin)
        .args([
            "test",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--save-baseline",
            baseline_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma test --save-baseline");
    assert!(out_save.status.success());
    assert!(baseline_path.exists());

    let out_diff = Command::new(bin)
        .args([
            "test",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--baseline",
            baseline_path.to_str().unwrap(),
            "--max-regression",
            "0.01",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma test --baseline");
    assert!(out_diff.status.success());
    let parsed_diff: serde_json::Value =
        serde_json::from_slice(&out_diff.stdout).expect("Baseline diff output must be valid JSON");
    assert_eq!(
        parsed_diff["baseline_diff"]["regressed"].as_bool(),
        Some(false)
    );

    // #6 Adversarial Perturbation & Boundary Stress Test (`lemma experiment --adversarial --json`)
    let out_adv = Command::new(bin)
        .args([
            "experiment",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--adversarial",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma experiment --adversarial --json");
    assert!(out_adv.status.success());
    let parsed_adv: serde_json::Value =
        serde_json::from_slice(&out_adv.stdout).expect("Adversarial output must be valid JSON");
    assert_eq!(parsed_adv["perturbations"].as_array().unwrap().len(), 5);
    assert!(parsed_adv["robustness_score"].as_f64().is_some());

    // #8 Cost-Sensitive / Asymmetric Loss Threshold Optimizer (`lemma calibrate --cost-weights ...`)
    let out_cal = Command::new(bin)
        .args([
            "calibrate",
            "--file",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--cost-weights",
            "billing=2.5,technical=1.0",
            "--export",
            config_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma calibrate --cost-weights");
    assert!(out_cal.status.success());
    let parsed_cal: serde_json::Value =
        serde_json::from_slice(&out_cal.stdout).expect("Calibrate output must be valid JSON");
    assert!(parsed_cal["raw_expected_cost"].as_f64().is_some());
    assert!(parsed_cal["calibrated_expected_cost"].as_f64().is_some());
    assert_eq!(parsed_cal["cost_weights"]["billing"].as_f64(), Some(2.5));

    // Verify exported calibrated config applies cost_weights seamlessly during `lemma test --config`
    let out_test_cfg = Command::new(bin)
        .args([
            "test",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--config",
            config_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma test --config");
    assert!(out_test_cfg.status.success());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_command_aliases_model_flags_and_positional_ergonomics() {
    // 1. Intuitive command aliases: run, predict, infer, eval, pull, models, tune, why
    let cli_run = Cli::try_parse_from([
        "lemma",
        "run",
        "My invoice was charged twice",
        "--model",
        "deberta-v3",
    ])
    .expect("`lemma run <TEXT> --model` must parse");
    match cli_run.command {
        Commands::Classify(args) => {
            assert_eq!(
                args.positional_text.as_deref(),
                Some("My invoice was charged twice")
            );
            assert_eq!(args.checkpoint, "deberta-v3");
        }
        _ => panic!("Expected Commands::Classify"),
    }

    assert!(matches!(
        Cli::try_parse_from(["lemma", "predict", "-m", "modernbert-zeroshot", "test"])
            .expect("`lemma predict -m` must parse")
            .command,
        Commands::Classify(_)
    ));
    assert!(matches!(
        Cli::try_parse_from(["lemma", "infer", "-m", "auto-best", "test"])
            .expect("`lemma infer -m` must parse")
            .command,
        Commands::Classify(_)
    ));

    let cli_eval = Cli::try_parse_from([
        "lemma",
        "eval",
        "--model",
        "eurobert",
        "-f",
        "data/customer_support.jsonl",
    ])
    .expect("`lemma eval --model` must parse");
    match cli_eval.command {
        Commands::Test(args) => assert_eq!(args.checkpoint, "eurobert"),
        _ => panic!("Expected Commands::Test"),
    }

    let cli_pull = Cli::try_parse_from(["lemma", "pull", "--all", "--verify-only"])
        .expect("`lemma pull --all --verify-only` must parse");
    match cli_pull.command {
        Commands::Download(args) => {
            assert!(args.all);
            assert!(args.verify_only);
        }
        _ => panic!("Expected Commands::Download"),
    }

    let cli_models = Cli::try_parse_from(["lemma", "models", "--model", "neobert"])
        .expect("`lemma models --model` must parse");
    match cli_models.command {
        Commands::Info(args) => assert_eq!(args.checkpoint, "neobert"),
        _ => panic!("Expected Commands::Info"),
    }

    let cli_tune = Cli::try_parse_from([
        "lemma",
        "tune",
        "-m",
        "typed-decisions",
        "--file",
        "data/customer_support.jsonl",
    ])
    .expect("`lemma tune -m` must parse");
    match cli_tune.command {
        Commands::Calibrate(args) => assert_eq!(args.checkpoint, "typed-decisions"),
        _ => panic!("Expected Commands::Calibrate"),
    }

    let cli_why = Cli::try_parse_from([
        "lemma",
        "why",
        "Urgent production database crash and stack trace",
        "--model",
        "deberta-v3",
    ])
    .expect("`lemma why <TEXT> --model` must parse");
    match cli_why.command {
        Commands::Explain(args) => {
            assert_eq!(
                args.input_text.as_deref(),
                Some("Urgent production database crash and stack trace")
            );
            assert!(args.target.is_none());
            assert!(args.contrast.is_none());
            assert_eq!(args.checkpoint, "deberta-v3");
        }
        _ => panic!("Expected Commands::Explain"),
    }

    // 2. CompareArgs adapter flags (--chunk-strategy, --shortlist-k, --hypothesis-template)
    let cli_cmp = Cli::try_parse_from([
        "lemma",
        "compare",
        "-m",
        "english,deberta-v3",
        "--chunk-strategy",
        "head-tail",
        "--shortlist-k",
        "4",
        "--hypothesis-template",
        "This ticket is about {label}.",
    ])
    .expect("`lemma compare` with adapter flags must parse");
    match cli_cmp.command {
        Commands::Compare(args) => {
            assert_eq!(args.checkpoints, "english,deberta-v3");
            assert_eq!(args.chunk_strategy, "head-tail");
            assert_eq!(args.shortlist_k, 4);
            assert_eq!(
                args.hypothesis_template.as_deref(),
                Some("This ticket is about {label}.")
            );
        }
        _ => panic!("Expected Commands::Compare"),
    }
}

#[test]
fn test_positional_classify_auto_explain_and_download_all_e2e() {
    let bin = env!("CARGO_BIN_EXE_lemma");

    // 1. Positional `lemma classify "<TEXT>"` (or `lemma run "<TEXT>"`)
    let out_cls = Command::new(bin)
        .args([
            "run",
            "My credit card invoice was charged twice, please refund immediately",
            "--model",
            "english",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute lemma run positional text");
    assert!(out_cls.status.success());
    let parsed_cls: serde_json::Value =
        serde_json::from_slice(&out_cls.stdout).expect("Positional classify must emit valid JSON");
    let arr = parsed_cls.as_array().expect("Expected JSON array");
    assert_eq!(arr.len(), 1);
    assert_eq!(
        arr[0][1]["answers"]["classification"]["choice"].as_str(),
        Some("billing")
    );

    // 2. Ergonomic `lemma explain "<TEXT>"` auto-detects winning & runner-up labels
    let out_exp = Command::new(bin)
        .args([
            "why",
            "System error 500 stack trace crash in Kubernetes pod",
            "--model",
            "english",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemma why positional text");
    assert!(out_exp.status.success());
    let parsed_exp: serde_json::Value =
        serde_json::from_slice(&out_exp.stdout).expect("Explain output must be valid JSON");
    assert_eq!(parsed_exp["target_choice"].as_str(), Some("technical"));
    assert_ne!(parsed_exp["contrast_choice"].as_str(), Some("technical"));
    assert!(!parsed_exp["attributions"].as_array().unwrap().is_empty());

    // 3. `lemma download --all --verify-only` checks all 7 local System 1 checkpoints
    let all_present = [
        lemma::model::Checkpoint::English,
        lemma::model::Checkpoint::Multilingual,
        lemma::model::Checkpoint::TypedDecisions,
        lemma::model::Checkpoint::ModernBertZeroShot,
        lemma::model::Checkpoint::DebertaV3ZeroShot,
        lemma::model::Checkpoint::EuroBert,
        lemma::model::Checkpoint::NeoBert,
    ]
    .iter()
    .all(|c| c.local_dir().join("model.safetensors").exists());

    let out_dl = Command::new(bin)
        .args(["download", "--all", "--verify-only"])
        .output()
        .expect("Failed to execute lemma download --all --verify-only");
    let combined_out = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out_dl.stdout),
        String::from_utf8_lossy(&out_dl.stderr)
    );
    assert!(combined_out.contains("convaiinnovations/laya"));
    assert!(combined_out.contains("MoritzLaurer/ModernBERT-large-zeroshot-v2.0"));
    assert!(combined_out.contains("MoritzLaurer/deberta-v3-large-zeroshot-v2.0"));
    assert!(combined_out.contains("EuroBERT/EuroBERT-210m"));
    assert!(combined_out.contains("chandar-lab/NeoBERT"));
    if all_present {
        assert!(out_dl.status.success());
    }
}
