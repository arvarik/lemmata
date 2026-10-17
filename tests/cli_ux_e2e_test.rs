use clap::Parser;
use lemmata::cli::{Cli, Commands};
use std::process::Command;

#[test]
fn test_cli_flag_aliases() {
    // 1. `lemmata taxonomy run` parses, and the old `route` alias is gone
    let cli = Cli::try_parse_from([
        "lemmata",
        "taxonomy",
        "run",
        "--schema",
        "schema/banking_hierarchy.json",
    ])
    .expect("`lemmata taxonomy run` must parse");
    assert!(matches!(cli.command, Commands::Taxonomy(_)));
    assert!(
        Cli::try_parse_from([
            "lemmata",
            "route",
            "run",
            "--schema",
            "schema/banking_hierarchy.json"
        ])
        .is_err()
    );

    // 2. Documented README aliases (`--eval-data`, `--target-acc`, `--data`, `--threshold`, `--receipts`, `--attest-key`)
    assert!(
        Cli::try_parse_from([
            "lemmata",
            "defer",
            "fit",
            "--eval-data",
            "data/customer_support.jsonl",
            "--schema",
            "schema/questions.json",
            "--target-acc",
            "0.95",
        ])
        .is_ok()
    );
    assert!(
        Cli::try_parse_from([
            "lemmata",
            "defer",
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
            "lemmata",
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
    let bin = env!("CARGO_BIN_EXE_lemmata");

    // 1. `lemmata test --json` must emit pure JSON on stdout (banners go to stderr)
    let out_test = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
        .args([
            "test",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemmata test --json");
    assert!(out_test.status.success());
    let parsed_test: serde_json::Value = serde_json::from_slice(&out_test.stdout)
        .expect("lemmata test --json stdout must be valid JSON without leading banners");
    assert!(parsed_test.get("accuracy").is_some());

    // 2. `lemmata ensemble --json` must emit a pure JSON array on stdout
    let out_ensemble = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
        .args([
            "ensemble",
            "-f",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemmata ensemble --json");
    assert!(out_ensemble.status.success());
    let parsed_ensemble: serde_json::Value = serde_json::from_slice(&out_ensemble.stdout)
        .expect("lemmata ensemble --json stdout must be valid JSON");
    assert!(parsed_ensemble.is_array());
}

#[test]
fn test_arena_sweep_baseline_and_cost_calibration_e2e() {
    let bin = env!("CARGO_BIN_EXE_lemmata");
    let tmp_dir = std::env::temp_dir().join(format!("lemmata_e2e_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp_dir);
    let baseline_path = tmp_dir.join("baseline.json");
    let config_path = tmp_dir.join("calibrated_cost.json");

    // #1 Head-to-Head Model Arena (`lemmata compare --models ... --json`)
    let out_arena = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
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
        .expect("Failed to execute lemmata compare --json");
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

    // #2 Deferral fit (`lemmata defer fit --json`): 6 labelled items cannot certify 90% accuracy
    let out_fit = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
        .args([
            "defer",
            "fit",
            "--val-data",
            "data/customer_support.jsonl",
            "-s",
            "schema/questions.json",
            "--target-accuracy",
            "0.90",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemmata defer fit --json");
    assert!(!out_fit.status.success(), "no threshold can be certified");
    assert!(String::from_utf8_lossy(&out_fit.stderr).contains("needs at least"));
    let parsed_fit: serde_json::Value =
        serde_json::from_slice(&out_fit.stdout).expect("Defer fit output must be valid JSON");
    assert_eq!(parsed_fit["points"].as_array().map(Vec::len), Some(20));
    assert!(parsed_fit["policy"].is_null());

    // #3 CI Regression Gate & Baseline Diff (`lemmata test --save-baseline` + `--baseline`)
    let out_save = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
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
        .expect("Failed to execute lemmata test --save-baseline");
    assert!(out_save.status.success());
    assert!(baseline_path.exists());

    let out_diff = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
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
        .expect("Failed to execute lemmata test --baseline");
    assert!(out_diff.status.success());
    let parsed_diff: serde_json::Value =
        serde_json::from_slice(&out_diff.stdout).expect("Baseline diff output must be valid JSON");
    assert_eq!(
        parsed_diff["baseline_diff"]["regressed"].as_bool(),
        Some(false)
    );

    // #8 Cost-Sensitive / Asymmetric Loss Threshold Optimizer (`lemmata calibrate --cost-weights ...`)
    let out_cal = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
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
        .expect("Failed to execute lemmata calibrate --cost-weights");
    assert!(out_cal.status.success());
    let parsed_cal: serde_json::Value =
        serde_json::from_slice(&out_cal.stdout).expect("Calibrate output must be valid JSON");
    assert!(parsed_cal["raw_expected_cost"].as_f64().is_some());
    assert!(parsed_cal["calibrated_expected_cost"].as_f64().is_some());
    assert_eq!(parsed_cal["cost_weights"]["billing"].as_f64(), Some(2.5));

    // Verify exported calibrated config applies cost_weights seamlessly during `lemmata test --config`
    let out_test_cfg = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
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
        .expect("Failed to execute lemmata test --config");
    assert!(out_test_cfg.status.success());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_command_aliases_model_flags_and_positional_ergonomics() {
    // 1. Intuitive command aliases: run, predict, infer, eval, pull, models, tune, why
    let cli_run = Cli::try_parse_from([
        "lemmata",
        "run",
        "My invoice was charged twice",
        "--model",
        "deberta-v3",
    ])
    .expect("`lemmata run <TEXT> --model` must parse");
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
        Cli::try_parse_from(["lemmata", "predict", "-m", "modernbert-zeroshot", "test"])
            .expect("`lemmata predict -m` must parse")
            .command,
        Commands::Classify(_)
    ));
    assert!(matches!(
        Cli::try_parse_from(["lemmata", "infer", "-m", "auto-best", "test"])
            .expect("`lemmata infer -m` must parse")
            .command,
        Commands::Classify(_)
    ));

    let cli_eval = Cli::try_parse_from([
        "lemmata",
        "eval",
        "--model",
        "eurobert",
        "-f",
        "data/customer_support.jsonl",
    ])
    .expect("`lemmata eval --model` must parse");
    match cli_eval.command {
        Commands::Test(args) => assert_eq!(args.checkpoint, "eurobert"),
        _ => panic!("Expected Commands::Test"),
    }

    let cli_pull = Cli::try_parse_from(["lemmata", "pull", "--all", "--verify-only"])
        .expect("`lemmata pull --all --verify-only` must parse");
    match cli_pull.command {
        Commands::Download(args) => {
            assert!(args.all);
            assert!(args.verify_only);
        }
        _ => panic!("Expected Commands::Download"),
    }

    let cli_models = Cli::try_parse_from(["lemmata", "models", "--model", "neobert"])
        .expect("`lemmata models --model` must parse");
    match cli_models.command {
        Commands::Info(args) => assert_eq!(args.checkpoint, "neobert"),
        _ => panic!("Expected Commands::Info"),
    }

    let cli_tune = Cli::try_parse_from([
        "lemmata",
        "tune",
        "-m",
        "typed-decisions",
        "--file",
        "data/customer_support.jsonl",
    ])
    .expect("`lemmata tune -m` must parse");
    match cli_tune.command {
        Commands::Calibrate(args) => assert_eq!(args.checkpoint, "typed-decisions"),
        _ => panic!("Expected Commands::Calibrate"),
    }

    let cli_why = Cli::try_parse_from([
        "lemmata",
        "why",
        "Urgent production database crash and stack trace",
        "--model",
        "deberta-v3",
    ])
    .expect("`lemmata why <TEXT> --model` must parse");
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
        "lemmata",
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
    .expect("`lemmata compare` with adapter flags must parse");
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
    let bin = env!("CARGO_BIN_EXE_lemmata");

    // 1. Positional `lemmata classify "<TEXT>"` (or `lemmata run "<TEXT>"`)
    let out_cls = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
        .args([
            "run",
            "My credit card invoice was charged twice, please refund immediately",
            "--model",
            "english",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to execute lemmata run positional text");
    assert!(out_cls.status.success());
    let parsed_cls: serde_json::Value =
        serde_json::from_slice(&out_cls.stdout).expect("Positional classify must emit valid JSON");
    let arr = parsed_cls.as_array().expect("Expected JSON array");
    assert_eq!(arr.len(), 1);
    assert_eq!(
        arr[0][1]["answers"]["classification"]["choice"].as_str(),
        Some("billing")
    );

    // 2. Ergonomic `lemmata explain "<TEXT>"` auto-detects winning & runner-up labels
    let out_exp = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
        .args([
            "why",
            "System error 500 stack trace crash in Kubernetes pod",
            "--model",
            "english",
            "--json",
        ])
        .output()
        .expect("Failed to execute lemmata why positional text");
    assert!(out_exp.status.success());
    let parsed_exp: serde_json::Value =
        serde_json::from_slice(&out_exp.stdout).expect("Explain output must be valid JSON");
    assert_eq!(parsed_exp["target_choice"].as_str(), Some("technical"));
    assert_ne!(parsed_exp["contrast_choice"].as_str(), Some("technical"));
    assert!(!parsed_exp["attributions"].as_array().unwrap().is_empty());

    // 3. `lemmata download --all --verify-only` checks all 7 local System 1 checkpoints
    let all_present = [
        lemmata::model::Checkpoint::English,
        lemmata::model::Checkpoint::Multilingual,
        lemmata::model::Checkpoint::TypedDecisions,
        lemmata::model::Checkpoint::ModernBertZeroShot,
        lemmata::model::Checkpoint::DebertaV3ZeroShot,
        lemmata::model::Checkpoint::EuroBert,
        lemmata::model::Checkpoint::NeoBert,
    ]
    .iter()
    .all(|c| c.local_dir().join("model.safetensors").exists());

    let out_dl = Command::new(bin)
        .env("LEMMATA_BACKEND", "mock")
        .args(["download", "--all", "--verify-only"])
        .output()
        .expect("Failed to execute lemmata download --all --verify-only");
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

#[test]
fn test_learn_never_panics_under_real_backends() {
    let bin = env!("CARGO_BIN_EXE_lemmata");
    // `auto` runs real weights where installed and the labelled mock elsewhere.
    for phase in ["2", "3"] {
        let out = Command::new(bin)
            .env("LEMMATA_BACKEND", "auto")
            .args(["learn", phase])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "learn {phase} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // `native` either runs every lesson model or stops with an error (exit 1), never a panic (101).
    let out = Command::new(bin)
        .env("LEMMATA_BACKEND", "native")
        .args(["learn", "3"])
        .output()
        .unwrap();
    match out.status.code() {
        Some(0) => {}
        Some(1) => assert!(String::from_utf8_lossy(&out.stderr).contains("--backend auto")),
        other => panic!(
            "learn under native exited with {other:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        ),
    }
}

/// Split a documented command into words, honoring single and double quotes. Stops at a pipe,
/// a redirect, or a shell comment, because those belong to the shell, not to lemmata.
fn shell_words(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quote = None;
    let mut started = false;
    for c in command.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => word.push(c),
            (None, '"' | '\'') => {
                quote = Some(c);
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            (None, c) => {
                word.push(c);
                started = true;
            }
        }
    }
    if started {
        words.push(word);
    }
    let end = words
        .iter()
        .position(|w| {
            w == "|" || w == "&&" || w == "#" || w.starts_with('>') || w.starts_with("2>")
        })
        .unwrap_or(words.len());
    words.truncate(end);
    words
}

/// Every `lemmata ...` command in the lessons, recipes, glossary, quiz, reading list, and README
/// must parse. Prose may name a command without its required values, so those errors pass.
#[test]
fn test_documented_commands_parse() {
    use clap::error::ErrorKind;
    use lemmata::engine::{
        CurriculumPhase, build_all_quiz_questions, build_glossary, build_phase_lesson,
        build_reading_list, build_workflow_recipes,
    };

    fn strings(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::String(s) => out.push(s.clone()),
            serde_json::Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
            serde_json::Value::Object(o) => o.values().for_each(|x| strings(x, out)),
            _ => {}
        }
    }
    let mut texts = Vec::new();
    for phase in CurriculumPhase::all() {
        strings(
            &serde_json::to_value(build_phase_lesson(phase)).unwrap(),
            &mut texts,
        );
    }
    strings(
        &serde_json::to_value(build_workflow_recipes()).unwrap(),
        &mut texts,
    );
    strings(&serde_json::to_value(build_glossary()).unwrap(), &mut texts);
    strings(
        &serde_json::to_value(build_all_quiz_questions()).unwrap(),
        &mut texts,
    );
    strings(
        &serde_json::to_value(build_reading_list(None)).unwrap(),
        &mut texts,
    );

    let readme = std::fs::read_to_string("README.md").unwrap();
    let mut in_block = false;
    let mut pending = String::new();
    for line in readme.lines() {
        if line.trim_start().starts_with("```") {
            in_block = !in_block;
            continue;
        }
        if in_block {
            pending.push_str(line.trim_end().trim_end_matches('\\'));
            pending.push(' ');
            if !line.trim_end().ends_with('\\') {
                texts.push(std::mem::take(&mut pending).trim().to_string());
            }
        } else {
            // In prose, only inline code is a command.
            texts.extend(
                line.split('`')
                    .skip(1)
                    .step_by(2)
                    .map(|span| format!("`{span}`")),
            );
        }
    }

    // The examples in each subcommand's --help.
    use clap::CommandFactory;
    for sub in Cli::command().get_subcommands() {
        if let Some(help) = sub.get_after_help() {
            texts.extend(help.to_string().lines().map(|l| l.trim().to_string()));
        }
    }

    let mut commands = Vec::new();
    for text in &texts {
        if text.starts_with("lemmata ") {
            commands.extend(
                text.split(" / ")
                    .filter(|c| c.starts_with("lemmata "))
                    .map(str::to_string),
            );
        }
        for span in text.split('`').skip(1).step_by(2) {
            if span.starts_with("lemmata ") {
                commands.push(span.to_string());
            }
        }
    }
    assert!(
        commands.len() > 50,
        "found only {} documented commands",
        commands.len()
    );

    let failures: Vec<String> = commands
        .iter()
        .filter_map(|command| {
            let err = Cli::try_parse_from(shell_words(command)).err()?;
            match err.kind() {
                ErrorKind::DisplayHelp
                | ErrorKind::DisplayVersion
                | ErrorKind::MissingRequiredArgument
                | ErrorKind::MissingSubcommand
                | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => None,
                _ => Some(format!(
                    "{command}\n    {}",
                    err.to_string().lines().next().unwrap_or("")
                )),
            }
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} documented commands do not parse:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Two `classify --receipts` runs append to one log and continue its hash chain.
#[test]
fn test_classify_receipts_append_and_verify() {
    let bin = env!("CARGO_BIN_EXE_lemmata");
    let log = std::env::temp_dir().join(format!("lemmata_receipts_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let run = |args: &[&str]| {
        Command::new(bin)
            .env("LEMMATA_BACKEND", "mock")
            .env("LEMMATA_ATTEST_KEY", "test-key")
            .args(args)
            .output()
            .unwrap()
    };
    let log_arg = log.to_str().unwrap();
    assert!(
        run(&["classify", "--text", "refund me", "--receipts", log_arg])
            .status
            .success()
    );
    let file_run = [
        "classify",
        "-f",
        "data/customer_support.jsonl",
        "-s",
        "schema/questions.json",
        "--receipts",
        log_arg,
    ];
    assert!(run(&file_run).status.success());

    let out = run(&["attest", "verify", "--receipts", log_arg, "--json"]);
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let _ = std::fs::remove_file(&log);
    assert_eq!(report["total_receipts"], 7);
    assert_eq!(report["valid_receipts"], 7);
    assert_eq!(report["chain_intact"], true);
}
