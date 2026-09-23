use std::str::FromStr;
use anyhow::Result;
use clap::Parser;
use comfy_table::{Cell, Color, Table};

use lemma::benchmark::BenchmarkRunner;
use lemma::cli::{Cli, Commands, DownloadArgs, ExperimentMode, InfoArgs, OutputFormat};
use lemma::engine::{ClassifierConfig, Evaluator, ExperimentRunner, LayaClassifier};
use lemma::model::{Checkpoint, Downloader};
use lemma::{create_inline_question, load_questions_from_file, resolve_data_source};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download(args) => handle_download(args),
        Commands::Classify(args) => handle_classify(args),
        Commands::Experiment(args) => handle_experiment(args),
        Commands::Test(args) => handle_test(args),
        Commands::Bench(args) => handle_bench(args),
        Commands::Info(args) => handle_info(args),
    }
}

fn handle_download(args: DownloadArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint)
        .map_err(|e| anyhow::anyhow!(e))?;

    let downloader = Downloader::detect()?;

    if args.verify_only {
        println!("🔍 Verifying checkpoint files in {}", args.target_dir.display());
        let report = downloader.verify(&args.target_dir)?;
        if report.is_valid {
            println!("✅ All required checkpoint files are valid and present!");
            for (f, sz) in report.found {
                println!("   - {}: {:.2} MB", f, sz as f64 / 1_048_576.0);
            }
        } else {
            println!("❌ Missing required files in {}:", args.target_dir.display());
            for m in report.missing {
                println!("   - {}", m);
            }
        }
        return Ok(());
    }

    downloader.download(checkpoint, &args.target_dir, args.dry_run)?;
    Ok(())
}

fn handle_classify(args: lemma::cli::ClassifyArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint)
        .map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(&args.qtype, &args.instruction, &args.labels)?
    };

    let source = resolve_data_source(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
    )?;

    let mut items = source.read_items()?;
    if let Some(limit) = args.limit {
        items.truncate(limit);
    }

    if items.is_empty() {
        eprintln!("⚠️  No data items found to classify. Check input path or data/ folder.");
        return Ok(());
    }

    let mut config = ClassifierConfig::default();
    config.checkpoint = checkpoint;
    config.temperature = args.temperature;
    let classifier = LayaClassifier::new(config);

    let results = classifier.classify_batch(&items, &questions)?;

    match args.format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string(&results)?);
        }
        OutputFormat::Pretty => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        OutputFormat::Compact => {
            for (id, res) in results {
                print!("{id} => ");
                for (qid, ans) in res.answers {
                    if let Some(c) = ans.choice {
                        print!("{qid}:{c} ({:.2}%) ", ans.confidence * 100.0);
                    } else if let Some(s) = ans.score {
                        print!("{qid}:{s:.2} ");
                    } else if let Some(n) = ans.noul {
                        print!("{qid}:{n:.4} ");
                    }
                }
                println!("[{:.1}ms]", res.latency_ms);
            }
        }
        OutputFormat::Table => {
            let mut table = Table::new();
            table.set_header(vec![
                Cell::new("Item ID").fg(Color::Cyan),
                Cell::new("Input Snippet").fg(Color::White),
                Cell::new("Question").fg(Color::Blue),
                Cell::new("Prediction").fg(Color::Green),
                Cell::new("Confidence").fg(Color::Yellow),
                Cell::new("Escalate?").fg(Color::DarkGrey),
                Cell::new("Latency").fg(Color::DarkGrey),
            ]);

            for (idx, (id, res)) in results.iter().enumerate() {
                let snippet = items.get(idx).map(|it| it.preview_text(40)).unwrap_or_default();

                for (qid, ans) in &res.answers {
                    let pred_str = if let Some(c) = &ans.choice {
                        c.clone()
                    } else if let Some(s) = ans.score {
                        format!("{:.2}", s)
                    } else if let Some(n) = ans.noul {
                        format!("true: {:.2}%", n * 100.0)
                    } else {
                        "N/A".into()
                    };

                    let esc_str = ans
                        .act_probability
                        .map(|p| if p > 0.5 { "YES (⚠️)" } else { "no" })
                        .unwrap_or("no");

                    table.add_row(vec![
                        Cell::new(id),
                        Cell::new(&snippet),
                        Cell::new(qid),
                        Cell::new(pred_str),
                        Cell::new(format!("{:.1}%", ans.confidence * 100.0)),
                        Cell::new(esc_str),
                        Cell::new(format!("{:.1}ms", res.latency_ms)),
                    ]);
                }
            }

            println!("\n{table}\n");
            println!(
                "Classified {} items across {} questions using {} checkpoint.\n",
                results.len(),
                questions.len(),
                checkpoint.repo_id()
            );
        }
    }

    Ok(())
}

fn handle_experiment(args: lemma::cli::ExperimentArgs) -> Result<()> {
    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(
            "choice",
            "Classify the topic or sentiment",
            "billing,technical,sales,other",
        )?
    };

    let source = resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?;
    let items = source.read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No data items found for experiment.");
        return Ok(());
    }

    let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);

    match args.mode {
        ExperimentMode::Order => {
            let (qid, question) = questions
                .iter()
                .find(|(_, q)| q.qtype == lemma::model::QuestionType::Choice)
                .unwrap_or_else(|| questions.iter().next().unwrap());
            ExperimentRunner::test_option_order_invariance(
                &classifier,
                &items,
                qid,
                question,
                args.permutations,
            )?;
        }
        ExperimentMode::Temp => {
            let temps = [0.5, 0.8, 1.0, 1.25, 1.5, 1.76, 2.0, 2.5];
            ExperimentRunner::sweep_temperature_calibration(&items, &questions, &temps)?;
        }
        ExperimentMode::Checkpoints => {
            ExperimentRunner::compare_checkpoints(&items, &questions)?;
        }
    }

    Ok(())
}

fn handle_test(args: lemma::cli::TestArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint)
        .map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(
            "choice",
            "Classify topic",
            "billing,technical,sales,other",
        )?
    };

    let source = resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?;
    let items = source.read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No labeled items found in data source.");
        return Ok(());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let evaluator = Evaluator::new(classifier);

    println!(
        "🔬 Evaluating {} items against ground truth with checkpoint {}...",
        items.len(),
        checkpoint.repo_id()
    );

    let metrics = evaluator.evaluate(&items, &questions)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&metrics)?);
    } else {
        Evaluator::print_report(&metrics);
    }

    Ok(())
}

fn handle_bench(args: lemma::cli::BenchArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint)
        .map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(
            "choice",
            "Classify department and priority",
            "billing,technical,sales,other",
        )?
    };

    let source = resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?;
    let items = source.read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No items found to benchmark. Please supply items in data/ or via --file.");
        return Ok(());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let runner = BenchmarkRunner::new(classifier).with_warmup(args.warmup);

    runner.run(&items, &questions, args.iterations)?;
    Ok(())
}

fn handle_info(args: InfoArgs) -> Result<()> {
    println!("\n🧠 Laya System 1 Decision Model Family");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Checkpoint").fg(Color::Cyan),
        Cell::new("Backbone").fg(Color::Blue),
        Cell::new("Parameters").fg(Color::Green),
        Cell::new("Context").fg(Color::Yellow),
        Cell::new("Head Budget").fg(Color::Magenta),
        Cell::new("Key Strengths / Domain").fg(Color::White),
    ]);

    table.add_row(vec![
        Cell::new("convaiinnovations/laya"),
        Cell::new("ModernBERT-large"),
        Cell::new("421M"),
        Cell::new("512 tokens"),
        Cell::new("192 tokens"),
        Cell::new("English text classification, email triage, toxicity guardrails (~39.5ms)"),
    ]);

    table.add_row(vec![
        Cell::new("convaiinnovations/laya-multilingual"),
        Cell::new("mmBERT-base"),
        Cell::new("322M"),
        Cell::new("1024 (up to 8k)"),
        Cell::new("256 tokens"),
        Cell::new("100+ languages, 2.2x faster forward pass (~32.8ms)"),
    ]);

    table.add_row(vec![
        Cell::new("convaiinnovations/laya-typed-decisions"),
        Cell::new("ModernBERT-large"),
        Cell::new("421M"),
        Cell::new("1024 tokens"),
        Cell::new("256 tokens"),
        Cell::new("Specialized enterprise workflows (0.766 acc, beats Jev teacher)"),
    ]);

    println!("{table}\n");

    println!("Local Cache Inspection ({}):", args.local_dir.display());
    if args.local_dir.exists() {
        if let Ok(downloader) = Downloader::detect() {
            let rep = downloader.verify(&args.local_dir)?;
            if rep.is_valid {
                println!("   Status: Local checkpoint ready and verified ✅");
                for (f, sz) in rep.found {
                    println!("     • {}: {:.2} MB", f, sz as f64 / 1_048_576.0);
                }
            } else {
                println!("   Status: Incomplete or missing files:");
                for m in rep.missing {
                    println!("     • missing: {}", m);
                }
            }
        }
    } else {
        println!("   Status: Directory does not exist yet. Run `lemma download` to fetch weights.");
    }
    println!();

    Ok(())
}
