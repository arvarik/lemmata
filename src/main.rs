use std::io::Write;
use std::str::FromStr;
use anyhow::{Result, bail};
use clap::Parser;
use comfy_table::{Cell, Color, Table};

use lemma::benchmark::BenchmarkRunner;
use lemma::cli::{
    BenchArgs, Cli, Commands, ConformalArgs, DownloadArgs, ExplainArgs, ExperimentArgs,
    ExperimentMode, InfoArgs, InspectArgs, OutputFormat, SchemaArgs, SchemaCommands, TestArgs,
};
use lemma::engine::{
    AttributionEngine, ClassifierConfig, ConformalEngine, Evaluator, ExperimentRunner,
    LayaClassifier,
};
use lemma::model::{Checkpoint, Downloader, inspect_safetensors};
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
        Commands::Inspect(args) => handle_inspect(args),
        Commands::Explain(args) => handle_explain(args),
        Commands::Conformal(args) => handle_conformal(args),
        Commands::Schema(args) => handle_schema(args),
    }
}

fn handle_download(args: DownloadArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
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
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

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

    let raw_results = classifier.classify_batch(&items, &questions)?;

    // Optional filter for supervisor escalation
    let results: Vec<_> = if args.escalate_only {
        raw_results
            .into_iter()
            .filter(|(_, res)| {
                res.answers
                    .values()
                    .any(|ans| ans.act_probability.unwrap_or(0.0) > 0.5)
            })
            .collect()
    } else {
        raw_results
    };

    let mut out_writer: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(std::io::BufWriter::new(std::fs::File::create(path)?)),
        None => Box::new(std::io::stdout()),
    };

    match args.format {
        OutputFormat::Json => {
            writeln!(out_writer, "{}", serde_json::to_string(&results)?)?;
        }
        OutputFormat::Pretty => {
            writeln!(out_writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
        OutputFormat::Jsonl => {
            for (id, res) in &results {
                let row = serde_json::json!({
                    "id": id,
                    "answers": res.answers,
                    "latency_ms": res.latency_ms,
                });
                writeln!(out_writer, "{}", serde_json::to_string(&row)?)?;
            }
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(out_writer);
            wtr.write_record(&[
                "item_id",
                "question_id",
                "prediction",
                "confidence",
                "act_probability",
                "latency_ms",
            ])?;

            for (id, res) in &results {
                for (qid, ans) in &res.answers {
                    let pred_str = if let Some(c) = &ans.choice {
                        c.clone()
                    } else if let Some(s) = ans.score {
                        format!("{:.2}", s)
                    } else if let Some(n) = ans.noul {
                        format!("{:.4}", n)
                    } else {
                        "N/A".into()
                    };

                    wtr.write_record(&[
                        id.as_str(),
                        qid.as_str(),
                        &pred_str,
                        &format!("{:.4}", ans.confidence),
                        &ans.act_probability
                            .map(|p| format!("{:.4}", p))
                            .unwrap_or_default(),
                        &format!("{:.2}", res.latency_ms),
                    ])?;
                }
            }
            wtr.flush()?;
        }
        OutputFormat::Compact => {
            for (id, res) in &results {
                write!(out_writer, "{id} => ")?;
                for (qid, ans) in &res.answers {
                    if let Some(c) = &ans.choice {
                        write!(out_writer, "{qid}:{c} ({:.2}%) ", ans.confidence * 100.0)?;
                    } else if let Some(s) = ans.score {
                        write!(out_writer, "{qid}:{s:.2} ")?;
                    } else if let Some(n) = ans.noul {
                        write!(out_writer, "{qid}:{n:.4} ")?;
                    }
                }
                writeln!(out_writer, "[{:.1}ms]", res.latency_ms)?;
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

            writeln!(out_writer, "\n{table}\n")?;
            writeln!(
                out_writer,
                "Classified {} items across {} questions using {} checkpoint.\n",
                results.len(),
                questions.len(),
                checkpoint.repo_id()
            )?;
        }
    }

    Ok(())
}

fn handle_experiment(args: ExperimentArgs) -> Result<()> {
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

fn handle_test(args: TestArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

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

    let mut cfg = ClassifierConfig::default();
    cfg.checkpoint = checkpoint;
    if let Some(t) = args.temperature {
        cfg.temperature = t;
    }
    let classifier = LayaClassifier::new(cfg);
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

    if let Some(min_acc) = args.min_accuracy {
        if metrics.accuracy < min_acc {
            bail!(
                "Accuracy test failed! Observed: {:.2}%, Required minimum: {:.2}%",
                metrics.accuracy * 100.0,
                min_acc * 100.0
            );
        } else {
            println!(
                "✅ Accuracy test passed: {:.2}% >= {:.2}%",
                metrics.accuracy * 100.0,
                min_acc * 100.0
            );
        }
    }

    Ok(())
}

fn handle_bench(args: BenchArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

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

    let report = runner.run(&items, &questions, args.iterations)?;

    if args.json {
        let json_report = serde_json::json!({
            "items_per_sec": report.throughput_items_per_sec,
            "decisions_per_sec": report.throughput_questions_per_sec,
            "latency_p50_ms": report.latency_p50_ms,
            "latency_p90_ms": report.latency_p90_ms,
            "latency_p95_ms": report.latency_p95_ms,
            "latency_p99_ms": report.latency_p99_ms,
            "latency_avg_ms": report.latency_avg_ms,
            "total_items": report.total_items,
        });
        println!("{}", serde_json::to_string_pretty(&json_report)?);
    }

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

    let show_all = args.checkpoint == "all";
    let target_ckpt = Checkpoint::from_str(&args.checkpoint).ok();

    if show_all || target_ckpt == Some(Checkpoint::English) {
        table.add_row(vec![
            Cell::new("convaiinnovations/laya"),
            Cell::new("ModernBERT-large"),
            Cell::new("421M"),
            Cell::new("512 tokens"),
            Cell::new("192 tokens"),
            Cell::new("English text classification, email triage, toxicity guardrails (~39.5ms)"),
        ]);
    }

    if show_all || target_ckpt == Some(Checkpoint::Multilingual) {
        table.add_row(vec![
            Cell::new("convaiinnovations/laya-multilingual"),
            Cell::new("mmBERT-base"),
            Cell::new("322M"),
            Cell::new("1024 (up to 8k)"),
            Cell::new("256 tokens"),
            Cell::new("100+ languages, 2.2x faster forward pass (~32.8ms)"),
        ]);
    }

    if show_all || target_ckpt == Some(Checkpoint::TypedDecisions) {
        table.add_row(vec![
            Cell::new("convaiinnovations/laya-typed-decisions"),
            Cell::new("ModernBERT-large"),
            Cell::new("421M"),
            Cell::new("1024 tokens"),
            Cell::new("256 tokens"),
            Cell::new("Specialized enterprise workflows (0.766 acc, beats Jev teacher)"),
        ]);
    }

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

fn handle_inspect(args: InspectArgs) -> Result<()> {
    println!("🔍 Inspecting Safetensors weights: {}", args.file.display());
    let rep = inspect_safetensors(&args.file)?;

    if args.json {
        let json_val = serde_json::json!({
            "total_tensors": rep.total_tensors,
            "total_parameters": rep.total_parameters,
            "dtypes": rep.dtypes,
            "is_laya_compatible": rep.is_laya_compatible,
            "has_act_head": rep.has_act_head,
            "has_decision_scorer": rep.has_decision_scorer,
            "file_size_bytes": rep.file_size_bytes,
            "header_size_bytes": rep.header_size_bytes,
        });
        println!("{}", serde_json::to_string_pretty(&json_val)?);
        return Ok(());
    }

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Inspection Property").fg(Color::Cyan),
        Cell::new("Observed Value").fg(Color::Green),
        Cell::new("Verification Details").fg(Color::DarkGrey),
    ]);

    table.add_row(vec![
        Cell::new("Total Tensors"),
        Cell::new(format!("{}", rep.total_tensors)),
        Cell::new("Number of distinct weight arrays"),
    ]);
    table.add_row(vec![
        Cell::new("Total Parameters"),
        Cell::new(format!("{:.2}M ({} elements)", rep.total_parameters as f64 / 1_000_000.0, rep.total_parameters)),
        Cell::new("ModernBERT-large + decision head expected ~421M"),
    ]);
    table.add_row(vec![
        Cell::new("File Size"),
        Cell::new(format!("{:.2} MB", rep.file_size_bytes as f64 / 1_048_576.0)),
        Cell::new(format!("Header JSON size: {} bytes", rep.header_size_bytes)),
    ]);
    table.add_row(vec![
        Cell::new("Dtype Distribution"),
        Cell::new(format!("{:?}", rep.dtypes)),
        Cell::new("Precision format (F16 / BF16 / F32)"),
    ]);
    table.add_row(vec![
        Cell::new("Laya Decision Head"),
        Cell::new(if rep.is_laya_compatible { "VERIFIED (✅)" } else { "NOT FOUND (⚠️)" }),
        Cell::new(format!("Act head: {}, Mask Scorer: {}", rep.has_act_head, rep.has_decision_scorer)),
    ]);

    println!("\n{table}\n");

    println!("Sample Tensor Shapes:");
    for (name, shape) in rep.tensor_shapes.iter().take(8) {
        println!("  • {:<45} {:?}", name, shape);
    }
    println!("  ... and {} more tensors.\n", rep.total_tensors.saturating_sub(8));

    Ok(())
}

fn handle_explain(args: ExplainArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(
            "choice",
            "Classify department",
            "billing,technical,sales,other",
        )?
    };

    let target_lower = args.target.to_lowercase();
    let qid = args
        .qid
        .as_deref()
        .or_else(|| {
            questions
                .iter()
                .find(|(_, q)| {
                    q.render_options()
                        .iter()
                        .any(|o| o.to_lowercase().contains(&target_lower))
                })
                .map(|(k, _)| k.as_str())
        })
        .or_else(|| questions.keys().next().map(|s| s.as_str()))
        .unwrap_or("department");

    let question = questions
        .get(qid)
        .ok_or_else(|| anyhow::anyhow!("Question '{}' not found in schema", qid))?;

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let engine = AttributionEngine::new(&classifier);

    let report = engine.explain(&args.text, qid, question, &args.target, &args.contrast)?;
    AttributionEngine::print_report(&report);

    Ok(())
}

fn handle_conformal(args: ConformalArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(
            "choice",
            "Classify department",
            "billing,technical,sales,other",
        )?
    };

    let source = resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?;
    let items = source.read_items()?;

    let (qid, question) = questions
        .iter()
        .find(|(_, q)| q.qtype == lemma::model::QuestionType::Choice)
        .unwrap_or_else(|| questions.iter().next().unwrap());

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let (report, prediction_items) =
        ConformalEngine::evaluate(&classifier, &items, qid, question, args.alpha)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        ConformalEngine::print_report(&report, &prediction_items);
    }

    Ok(())
}

fn handle_schema(args: SchemaArgs) -> Result<()> {
    match args.action {
        SchemaCommands::Validate(val_args) => {
            let checkpoint = Checkpoint::from_str(&val_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let questions = load_questions_from_file(&val_args.schema)?;
            let head_budget = checkpoint.head_budget();

            println!(
                "\n📋 Validating Schema '{}' against Checkpoint '{}' (Head Budget: {} tokens)",
                val_args.schema.display(),
                checkpoint.repo_id(),
                head_budget
            );

            let mut all_valid = true;
            let mut total_warnings = 0;

            for (qid, q) in &questions {
                match q.validate(qid, head_budget) {
                    Ok(warnings) => {
                        let opts = q.render_options();
                        println!("  ✅ Question '{}' (type: {:?}, {} options): valid", qid, q.qtype, opts.len());
                        for w in warnings {
                            println!("     ⚠️  {}", w);
                            total_warnings += 1;
                        }
                    }
                    Err(err) => {
                        println!("  ❌ Question '{}': {}", qid, err);
                        all_valid = false;
                    }
                }
            }

            println!();
            if all_valid {
                println!("✨ Schema validation passed successfully ({} warnings).\n", total_warnings);
            } else {
                bail!("Schema validation failed! Resolve errors above.");
            }
        }
    }
    Ok(())
}
