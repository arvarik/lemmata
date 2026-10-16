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
