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
