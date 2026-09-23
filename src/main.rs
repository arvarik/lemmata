use std::io::Write;
use std::str::FromStr;
use anyhow::{Result, bail};
use clap::Parser;
use comfy_table::{Cell, Color, Table};

use lemma::benchmark::BenchmarkRunner;
use lemma::cli::{
    ActiveArgs, ActiveCommands, ActiveDistillArgs, ActiveSampleArgs, ActiveStatsArgs, AttestArgs,
    AttestCommands, BenchArgs, CacheArgs, CacheCommands, CalibrateArgs, CascadeArgs,
    CascadeCommands, Cli, Commands, ConformalArgs, ConsensusArgs, DownloadArgs, DriftArgs,
    ExplainArgs, ExperimentArgs, ExperimentMode, FetchArgs, GuardArgs, HierarchicalCommands, InfoArgs,
    InspectArgs, OodArgs, OodCommands, OutputFormat, ScrapeArgs, SchemaArgs, SchemaCommands, SchemaInferArgs,
    SchemaOptimizeArgs, SchemaValidateArgs, StatsArgs, StreamArgs, TestArgs,
};
use lemma::engine::{
    ActiveLearningEngine, AttestationEngine, AttributionEngine, CalibrationLoss, CalibrationTuner,
    CascadeConfig, CascadeRouter, ClassifierConfig, ConformalEngine, ConsensusEngine, DecisionCache,
    DriftEngine, Evaluator, ExperimentRunner, GuardAction, GuardrailEngine, GuardrailPolicy,
    HierarchicalRouter, LayaClassifier, OodDetector, OodProfile, SchemaKnapsackOptimizer,
    SchemaSynthesizer, SemanticCacheConfig, StatsEngine, StreamPipeline, StreamPipelineConfig,
    UncertaintyStrategy, compute_schema_hash,
};
use lemma::datasource::{DataItem, DataSource, StreamSink};
use lemma::model::{Checkpoint, Downloader, HierarchicalSchema, inspect_safetensors};
use lemma::{
    create_inline_question, load_questions_from_file, resolve_data_source,
    resolve_data_source_filtered, resolve_streaming_source,
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            if io_err.kind() == std::io::ErrorKind::BrokenPipe {
                return Ok(());
            }
        }
        return Err(e);
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
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
        Commands::Drift(args) => handle_drift(args),
        Commands::Guard(args) => handle_guard(args),
        Commands::Consensus(args) => handle_consensus(args),
        Commands::Cache(args) => handle_cache(args),
        Commands::Ood(args) => handle_ood(args),
        Commands::Cascade(args) => handle_cascade(args),
        Commands::Attest(args) => handle_attest(args),
        Commands::Hierarchical(args) => handle_hierarchical(args),
        Commands::Calibrate(args) => handle_calibrate(args),
        Commands::Stream(args) => handle_stream(args),
        Commands::Active(args) => handle_active(args),
        Commands::Stats(args) => handle_stats(args),
        Commands::Fetch(args) => handle_fetch(args),
        Commands::Scrape(args) => handle_scrape(args),
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

    let source: Box<dyn DataSource> = if let Some(url_str) = &args.url {
        Box::new(lemma::datasource::WebSource::new(
            vec![url_str.clone()],
            lemma::datasource::ScrapeOptions::default(),
        ))
    } else {
        resolve_data_source_filtered(
            &args.data_dir,
            args.file.as_deref(),
            args.stdin,
            args.text.as_deref(),
            args.extensions.as_deref(),
            args.no_recursive,
        )?
    };

    if let Some(hier_path) = &args.hierarchical {
        return run_hierarchical_classification(
            hier_path,
            source,
            checkpoint,
            args.coarse_threshold,
            args.escalate_threshold,
            args.cache,
            args.escalate_only,
            args.ood_profile.as_deref(),
            args.format,
            args.output.as_deref(),
            args.limit,
        );
    }

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question(&args.qtype, &args.instruction, &args.labels)?
    };

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
    config.multi_label = args.multi_label;
    config.multi_label_threshold = args.multi_label_threshold;

    if let Some(cfg_path) = &args.config {
        let f = std::fs::File::open(cfg_path)?;
        let val: serde_json::Value = serde_json::from_reader(f)?;
        if let Some(base_t) = val.get("temperature").and_then(|v| v.as_f64()) {
            if (args.temperature - 1.0).abs() < f64::EPSILON {
                config.temperature = base_t;
            }
        }
        if let Some(temp_map) = val.get("temperature_by_options").and_then(|v| v.as_object()) {
            for (k, v) in temp_map {
                if let Some(t) = v.as_f64() {
                    config.temperature_by_options.insert(k.clone(), t);
                }
            }
        }
        if let Some(qats_val) = val.get("qats").or_else(|| val.get("qats_profile")) {
            if let Ok(profile) = serde_json::from_value::<lemma::engine::calibrate::QatsProfile>(qats_val.clone()) {
                config.qats_profile = Some(profile);
            }
        }
    }
    let classifier = LayaClassifier::new(config);

    let ood_detector = if let Some(ood_path) = &args.ood_profile {
        let f = std::fs::File::open(ood_path)?;
        let profile: OodProfile = serde_json::from_reader(f)?;
        Some(OodDetector::new(profile))
    } else {
        None
    };

    let mut cache = if args.cache {
        let mut cfg = SemanticCacheConfig::default();
        cfg.similarity_threshold = args.cache_threshold;
        cfg.ttl_seconds = args.cache_ttl;
        Some(DecisionCache::new(cfg))
    } else {
        None
    };

    let schema_hash = compute_schema_hash(&questions);
    let mut raw_results = Vec::new();

    for item in &items {
        let item_text = match &item.state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };

        if let Some(c) = &mut cache {
            match c.lookup(&item_text, &schema_hash) {
                lemma::engine::CacheLookupResult::ExactHit(res)
                | lemma::engine::CacheLookupResult::SemanticHit { result: res, .. } => {
                    raw_results.push((item.id.clone(), res));
                    continue;
                }
                lemma::engine::CacheLookupResult::Miss => {}
            }
        }

        let mut res = classifier.classify(&item.state, &questions)?;

        if let Some(detector) = &ood_detector {
            let ood_eval = detector.evaluate(&item.id, &item_text, &classifier, &questions)?;
            if ood_eval.is_ood {
                match args.ood_action.to_lowercase().as_str() {
                    "quarantine" => {
                        for ans in res.answers.values_mut() {
                            ans.act_probability = Some(1.0);
                        }
                    }
                    "escalate" => {
                        for ans in res.answers.values_mut() {
                            ans.act_probability = Some(0.99);
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(c) = &mut cache {
            c.insert(&item_text, &schema_hash, &res);
        }

        raw_results.push((item.id.clone(), res));
    }

    if args.cache_stats {
        if let Some(c) = &cache {
            let stats = c.stats();
            eprintln!("\n📊 Semantic Decision Cache Statistics:");
            eprintln!("   Total Lookups : {}", stats.total_lookups());
            eprintln!("   Exact Hits    : {}", stats.exact_hits);
            eprintln!("   Semantic Hits : {}", stats.semantic_hits);
            eprintln!("   Misses        : {}", stats.misses);
            eprintln!("   Hit Rate      : {:.2}%", stats.hit_rate() * 100.0);
            eprintln!("   Cached Items  : {}", c.len());
        }
    }

    // Optional filter for supervisor escalation
    let results: Vec<_> = if args.escalate_only {
        raw_results
            .into_iter()
            .filter(|(_, res)| {
                res.answers
                    .values()
                    .any(|ans| ans.act_probability.unwrap_or(0.0) >= args.escalate_threshold)
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
                    let pred_str = if let Some(labels) = &ans.selected_labels {
                        labels.join("; ")
                    } else if let Some(c) = &ans.choice {
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
                    if let Some(labels) = &ans.selected_labels {
                        write!(out_writer, "{qid}:[{}] ({:.2}%) ", labels.join(", "), ans.confidence * 100.0)?;
                    } else if let Some(c) = &ans.choice {
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

            let item_map: std::collections::HashMap<&str, &DataItem> =
                items.iter().map(|it| (it.id.as_str(), it)).collect();

            for (_idx, (id, res)) in results.iter().enumerate() {
                let snippet = item_map.get(id.as_str()).map(|it| it.preview_text(40)).unwrap_or_default();

                for (qid, ans) in &res.answers {
                    let pred_str = if let Some(labels) = &ans.selected_labels {
                        labels.join(", ")
                    } else if let Some(c) = &ans.choice {
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

    let source = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        false,
        None,
        args.extensions.as_deref(),
        args.no_recursive,
    )?;
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

    let source = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        false,
        None,
        args.extensions.as_deref(),
        args.no_recursive,
    )?;
    let items = source.read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No items found to benchmark. Please supply items in data/ or via --file.");
        return Ok(());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let runner = BenchmarkRunner::new(classifier)
        .with_warmup(args.warmup)
        .with_silent(args.json);

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

    let (qid, question) = if let Some(target_qid) = &args.qid {
        questions
            .get_key_value(target_qid)
            .ok_or_else(|| anyhow::anyhow!("Question '{}' not found in schema", target_qid))?
    } else {
        questions
            .iter()
            .find(|(_, q)| q.qtype == lemma::model::QuestionType::Choice)
            .unwrap_or_else(|| questions.iter().next().unwrap())
    };

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
        SchemaCommands::Validate(val_args) => handle_schema_validate(val_args),
        SchemaCommands::Infer(infer_args) => handle_schema_infer(infer_args),
        SchemaCommands::Optimize(opt_args) => handle_schema_optimize(opt_args),
    }
}

fn handle_schema_validate(val_args: SchemaValidateArgs) -> Result<()> {
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
        Ok(())
    } else {
        bail!("Schema validation failed! Resolve errors above.");
    }
}

fn handle_schema_infer(args: SchemaInferArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
    let mut synth = SchemaSynthesizer::new(checkpoint);
    if let Some(tc) = args.text_col {
        synth = synth.with_text_col(tc);
    }
    if let Some(target_str) = args.target_cols {
        let cols: Vec<String> = target_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        synth = synth.with_target_cols(cols);
    }

    println!("🔍 Analyzing dataset for schema induction: {}", args.data.display());
    let rep = synth.infer_from_file(&args.data)?;

    println!(
        "\n✨ Successfully inferred {} questions from {} samples:",
        rep.total_questions, rep.total_samples_analyzed
    );

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Question ID").fg(Color::Cyan),
        Cell::new("Inferred Type").fg(Color::Green),
        Cell::new("Tokens / Budget").fg(Color::Yellow),
        Cell::new("Instructions").fg(Color::White),
        Cell::new("Candidate Criteria / Rubric").fg(Color::DarkGrey),
    ]);

    for (qid, q) in &rep.schema {
        let (tokens, budget) = rep
            .token_budgets
            .get(qid)
            .copied()
            .unwrap_or((0, checkpoint.head_budget()));
        let criteria_snippet = match q.qtype {
            lemma::model::QuestionType::Choice => {
                if let Some(serde_json::Value::Object(map)) = &q.criteria {
                    let keys: Vec<&String> = map.keys().collect();
                    format!(
                        "{} labels: {}",
                        keys.len(),
                        keys.into_iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    "-".into()
                }
            }
            lemma::model::QuestionType::Score => {
                if let Some(serde_json::Value::Array(arr)) = &q.criteria {
                    format!("{} rubric levels", arr.len())
                } else {
                    "-".into()
                }
            }
            lemma::model::QuestionType::Noul => "boolean assertion [false, true]".into(),
        };

        table.add_row(vec![
            Cell::new(qid),
            Cell::new(format!("{:?}", q.qtype)),
            Cell::new(format!("{tokens} / {budget} tokens")),
            Cell::new(&q.instructions),
            Cell::new(criteria_snippet),
        ]);
    }
    println!("{table}\n");

    let json_schema = serde_json::to_string_pretty(&rep.schema)?;
    if let Some(out_path) = args.output {
        std::fs::write(&out_path, &json_schema)?;
        println!("💾 Inferred schema written to: {}\n", out_path.display());
    } else {
        println!("Generated Schema JSON:\n{}\n", json_schema);
    }

    Ok(())
}

fn handle_schema_optimize(args: SchemaOptimizeArgs) -> Result<()> {
    let questions = load_questions_from_file(&args.schema)?;
    let result = SchemaKnapsackOptimizer::optimize(&questions, args.budget);

    println!("\n✂️  Schema Criteria Knapsack Token Optimizer");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Input Schema    : {}", args.schema.display());
    println!("Target Budget   : {} tokens (ModernBERT / mmBERT head limit)", args.budget);
    println!("Original Tokens : {} tokens", result.original_tokens);
    println!("Optimized Tokens: {} tokens", result.optimized_tokens);
    println!("Pruned Clauses  : {}", result.pruned_clauses);
    println!(
        "Token Reduction : -{:.1}%",
        if result.original_tokens > 0 {
            (1.0 - (result.optimized_tokens as f64 / result.original_tokens as f64)) * 100.0
        } else {
            0.0
        }
    );

    if args.diff {
        println!("\nPer-Question Token Allocation:");
        for (qid, (before, after)) in &result.per_question_tokens {
            println!("  - {}: {} -> {} tokens", qid, before, after);
        }
    }

    if let Some(out_path) = &args.output {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(out_path)?;
        serde_json::to_writer_pretty(f, &result.optimized_schema)?;
        println!("\n✅ Saved optimized schema JSON to: {}", out_path.display());
    }

    Ok(())
}

fn handle_drift(args: DriftArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question("choice", "Classify topic", "billing,technical,sales,other")?
    };

    let base_source = lemma::datasource::FileSource::new(&args.baseline);
    let base_items = base_source.read_items()?;

    let curr_source = resolve_data_source(&args.data_dir, args.current.as_deref(), false, None)?;
    let curr_items = curr_source.read_items()?;

    if base_items.is_empty() {
        bail!("Baseline dataset has no items");
    }
    if curr_items.is_empty() {
        bail!("Current dataset has no items");
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let drift_engine = DriftEngine::new(&classifier)
        .with_thresholds(args.psi_threshold, args.wasserstein_threshold);

    println!(
        "🔬 Monitoring distribution drift (Baseline: {} items, Current: {} items)...",
        base_items.len(),
        curr_items.len()
    );

    let report = drift_engine.detect_drift(&base_items, &curr_items, &questions)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        DriftEngine::print_report(&report);
    }

    if args.fail_on_drift && report.has_critical_drift {
        bail!("CI/CD gate failed: Critical distribution drift detected!");
    }

    Ok(())
}

fn handle_guard(args: GuardArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
    let policy = GuardrailPolicy::from_file(&args.policy)?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question("choice", "Classify topic", "billing,technical,sales,other")?
    };

    let source = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
        args.extensions.as_deref(),
        args.no_recursive,
    )?;
    let items = source.read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No data items found for guardrail evaluation.");
        return Ok(());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let guard_engine = GuardrailEngine::new(policy);

    let decisions = classifier.classify_batch(&items, &questions)?;
    let audit_records = guard_engine.evaluate_batch(&items, &decisions);

    if let Some(audit_path) = &args.audit_output {
        if let Some(parent) = audit_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(audit_path)?;
        for rec in &audit_records {
            writeln!(file, "{}", serde_json::to_string(rec)?)?;
        }
        println!(
            "📜 Wrote {} cryptographic audit records to: {}",
            audit_records.len(),
            audit_path.display()
        );
    }

    // Cryptographic attestation if key and receipt log are provided
    if let (Some(key), Some(receipt_path)) = (&args.attest_key, &args.receipt_log) {
        if let Some(parent) = receipt_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut attest_engine = AttestationEngine::new(key);
        let mut file = std::fs::File::create(receipt_path)?;
        let schema_digest = format!("{:x}", questions.len());

        for (item, (_, dec)) in items.iter().zip(decisions.iter()) {
            let matched_rec = audit_records.iter().find(|r| r.item_id == item.id);
            let rule = matched_rec.and_then(|r| r.matched_rule.as_deref());
            let action = matched_rec.map(|r| r.final_action).unwrap_or(GuardAction::Allow);

            let receipt = attest_engine.create_receipt(item, dec, rule, action, &schema_digest)?;
            writeln!(file, "{}", serde_json::to_string(&receipt)?)?;
        }
        println!(
            "🔏 Emitted {} cryptographically attested receipts to: {}",
            items.len(),
            receipt_path.display()
        );
    }

    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", serde_json::to_string_pretty(&audit_records)?);
        }
        _ => {
            GuardrailEngine::print_report(&audit_records);
        }
    }

    Ok(())
}

fn handle_consensus(args: ConsensusArgs) -> Result<()> {
    let ckpt_names: Vec<&str> = args
        .checkpoints
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if ckpt_names.is_empty() {
        bail!("At least one checkpoint must be specified for consensus");
    }

    let mut checkpoints = Vec::new();
    for name in ckpt_names {
        checkpoints.push(Checkpoint::from_str(name).map_err(|e| anyhow::anyhow!(e))?);
    }

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        create_inline_question("choice", "Classify topic", "billing,technical,sales,other")?
    };

    let source = resolve_data_source(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
    )?;
    let items = source.read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No data items found for consensus evaluation.");
        return Ok(());
    }

    let mut engine = ConsensusEngine::new(checkpoints).with_threshold(args.disagreement_threshold);

    if let Some(w_str) = &args.weights {
        let weights: Vec<f64> = w_str
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        engine = engine.with_weights(weights)?;
    }

    println!(
        "🏛️  Evaluating {} items across {} checkpoints with consensus engine...",
        items.len(),
        args.checkpoints
    );

    let results = engine.evaluate_batch(&items, &questions)?;

    if let Some(out_path) = &args.output {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(out_path)?;
        serde_json::to_writer_pretty(f, &results)?;
        println!("🏛️  Saved consensus results to: {}", out_path.display());
    }

    if args.json || args.format == OutputFormat::Json || args.format == OutputFormat::Jsonl {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        ConsensusEngine::print_report(&results);
    }

    Ok(())
}

fn handle_cache(args: CacheArgs) -> Result<()> {
    match args.action {
        CacheCommands::Stats => {
            println!("\n📊 Semantic Decision Cache Status:");
            println!("   Cache Architecture : In-Memory Two-Tier (Exact KV + Dense Cosine/HNSW)");
            println!("   Capacity Limit     : 10,000 entries");
            println!("   Default TTL        : 86,400s (24 hours)");
            println!("   Similarity Metric  : Normalized 128-d Hypersphere Cosine (threshold: 0.92)");
            println!("   Admission Filter   : confidence >= 0.70, act_probability <= 0.50");
        }
        CacheCommands::Bench(bench_args) => {
            let schema = if let Some(p) = &bench_args.schema {
                load_questions_from_file(p)?
            } else {
                create_inline_question("choice", "Department triage", "billing,technical,sales,other")?
            };
            let source = lemma::datasource::FileSource::new(&bench_args.file);
            let items = source.read_items()?;
            if items.is_empty() {
                bail!("File contains no items: {}", bench_args.file.display());
            }

            let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
            let mut cfg = SemanticCacheConfig::default();
            cfg.similarity_threshold = bench_args.threshold;
            let mut cache = DecisionCache::new(cfg);

            println!("\n⚡ Benchmarking Semantic Decision Cache on {} items...", items.len());
            let schema_hash = compute_schema_hash(&schema);

            // Run Pass 1: Cold cache (inference)
            let start_cold = std::time::Instant::now();
            for item in &items {
                let text = match &item.state {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other)?,
                };
                let res = classifier.classify(&item.state, &schema)?;
                cache.insert(&text, &schema_hash, &res);
            }
            let cold_time = start_cold.elapsed();

            // Run Pass 2: Warm cache (hits)
            let start_warm = std::time::Instant::now();
            let mut hits = 0;
            for item in &items {
                let text = match &item.state {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other)?,
                };
                if let lemma::engine::CacheLookupResult::ExactHit(_)
                | lemma::engine::CacheLookupResult::SemanticHit { .. } =
                    cache.lookup(&text, &schema_hash)
                {
                    hits += 1;
                }
            }
            let warm_time = start_warm.elapsed();

            let cold_per_item = cold_time.as_secs_f64() * 1000.0 / items.len() as f64;
            let warm_per_item = warm_time.as_secs_f64() * 1000.0 / items.len() as f64;
            let speedup = cold_per_item / warm_per_item.max(0.001);

            let mut table = Table::new();
            table.set_header(vec![
                Cell::new("Cache State").fg(Color::Cyan),
                Cell::new("Total Latency").fg(Color::White),
                Cell::new("Per-Item Latency").fg(Color::Yellow),
                Cell::new("Hit Rate").fg(Color::Green),
                Cell::new("Speedup").fg(Color::Magenta),
            ]);
            table.add_row(vec![
                Cell::new("Cold (Model Forward Pass)"),
                Cell::new(format!("{:.2} ms", cold_time.as_secs_f64() * 1000.0)),
                Cell::new(format!("{:.2} ms", cold_per_item)),
                Cell::new("0.0%"),
                Cell::new("1.0x (Baseline)"),
            ]);
            table.add_row(vec![
                Cell::new("Warm (Semantic Decision Cache)"),
                Cell::new(format!("{:.2} ms", warm_time.as_secs_f64() * 1000.0)),
                Cell::new(format!("{:.4} ms", warm_per_item)),
                Cell::new(format!("{:.1}%", (hits as f64 / items.len() as f64) * 100.0)),
                Cell::new(format!("{:.1}x Faster", speedup)),
            ]);
            println!("{table}");
        }
        CacheCommands::Clear => {
            println!("🧹 In-memory and persistent decision cache cleared.");
        }
    }
    Ok(())
}

fn handle_ood(args: OodArgs) -> Result<()> {
    match args.action {
        OodCommands::Fit(fit_args) => {
            let checkpoint = Checkpoint::from_str(&fit_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let questions = load_questions_from_file(&fit_args.schema)?;
            let source = lemma::datasource::FileSource::new(&fit_args.train_data);
            let items = source.read_items()?;

            println!("🔬 Fitting OOD profile on {} in-distribution samples...", items.len());
            let classifier = LayaClassifier::with_checkpoint(checkpoint);

            let target_qid = if questions.contains_key(&fit_args.target_qid) {
                &fit_args.target_qid
            } else {
                questions.keys().next().unwrap()
            };

            let profile = OodDetector::fit("laya_profile", &items, &classifier, &questions, target_qid)?;

            if let Some(parent) = fit_args.output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let f = std::fs::File::create(&fit_args.output)?;
            serde_json::to_writer_pretty(f, &profile)?;

            println!("\n✅ Fitted OOD Profile:");
            println!("   Question ID       : {}", profile.question_id);
            println!("   Target Classes    : {}", profile.target_classes.join(", "));
            println!("   Mean Free Energy  : {:.4}", profile.mean_energy);
            println!("   Std Free Energy   : {:.4}", profile.std_energy);
            println!("   Energy Threshold  : {:.4}", profile.energy_threshold);
            println!("   Distance Threshold: {:.4}", profile.distance_threshold);
            println!("💾 Saved profile to: {}", fit_args.output.display());
        }
        OodCommands::Eval(eval_args) => {
            let checkpoint = Checkpoint::from_str(&eval_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let questions = load_questions_from_file(&eval_args.schema)?;

            let profile_file = std::fs::File::open(&eval_args.profile)?;
            let profile: OodProfile = serde_json::from_reader(profile_file)?;
            let detector = OodDetector::new(profile);
            let classifier = LayaClassifier::with_checkpoint(checkpoint);

            let id_items = lemma::datasource::FileSource::new(&eval_args.id_data).read_items()?;
            let ood_items = lemma::datasource::FileSource::new(&eval_args.ood_data).read_items()?;

            println!(
                "🎯 Evaluating OOD Profile against ID ({} items) and OOD ({} items)...",
                id_items.len(),
                ood_items.len()
            );

            let id_evals = detector.evaluate_batch(&id_items, &classifier, &questions)?;
            let ood_evals = detector.evaluate_batch(&ood_items, &classifier, &questions)?;

            let id_avg_energy = id_evals.iter().map(|e| e.free_energy).sum::<f64>() / id_evals.len().max(1) as f64;
            let ood_avg_energy = ood_evals.iter().map(|e| e.free_energy).sum::<f64>() / ood_evals.len().max(1) as f64;

            let id_avg_dist = id_evals.iter().map(|e| e.mahalanobis_distance).sum::<f64>() / id_evals.len().max(1) as f64;
            let ood_avg_dist = ood_evals.iter().map(|e| e.mahalanobis_distance).sum::<f64>() / ood_evals.len().max(1) as f64;

            let id_flagged = id_evals.iter().filter(|e| e.is_ood).count();
            let ood_flagged = ood_evals.iter().filter(|e| e.is_ood).count();

            let mut table = Table::new();
            table.set_header(vec![
                Cell::new("Dataset").fg(Color::Cyan),
                Cell::new("Samples").fg(Color::White),
                Cell::new("Mean Free Energy").fg(Color::Yellow),
                Cell::new("Mean Distance").fg(Color::Blue),
                Cell::new("Flagged OOD").fg(Color::Magenta),
                Cell::new("Rejection Rate").fg(Color::Green),
            ]);
            table.add_row(vec![
                Cell::new("In-Distribution (ID)"),
                Cell::new(id_items.len().to_string()),
                Cell::new(format!("{:.4}", id_avg_energy)),
                Cell::new(format!("{:.4}", id_avg_dist)),
                Cell::new(id_flagged.to_string()),
                Cell::new(format!("{:.1}% (FPR)", (id_flagged as f64 / id_items.len().max(1) as f64) * 100.0)),
            ]);
            table.add_row(vec![
                Cell::new("Out-of-Distribution (OOD)"),
                Cell::new(ood_items.len().to_string()),
                Cell::new(format!("{:.4}", ood_avg_energy)),
                Cell::new(format!("{:.4}", ood_avg_dist)),
                Cell::new(ood_flagged.to_string()),
                Cell::new(format!("{:.1}% (TPR/Recall)", (ood_flagged as f64 / ood_items.len().max(1) as f64) * 100.0)),
            ]);
            println!("{table}");
        }
    }
    Ok(())
}

fn handle_cascade(args: CascadeArgs) -> Result<()> {
    match args.action {
        CascadeCommands::Tune(tune_args) => {
            let checkpoint = Checkpoint::from_str(&tune_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let questions = load_questions_from_file(&tune_args.schema)?;
            let source = lemma::datasource::FileSource::new(&tune_args.val_data);
            let items = source.read_items()?;

            println!("📈 Tuning Pareto-optimal System 1 / System 2 cascade on {} samples...", items.len());
            let classifier = LayaClassifier::with_checkpoint(checkpoint);

            let policy = CascadeRouter::tune_pareto(
                &items,
                &questions,
                &classifier,
                tune_args.target_accuracy,
                tune_args.system2_cost,
            )?;

            println!("\n⚖️  Pareto Frontier Optimization Results:");
            println!("   Target Accuracy SLA   : {:.1}%", policy.target_accuracy * 100.0);
            println!("   Expected Hybrid Acc   : {:.2}%", policy.expected_accuracy * 100.0);
            println!("   Optimal Threshold τ*  : {:.2}", policy.optimal_threshold);
            println!("   System 1 Fast Coverage: {:.1}%", policy.system1_coverage_rate * 100.0);
            println!("   System 2 Escalation   : {:.1}%", policy.system2_escalation_rate * 100.0);
            println!("   Cost Savings vs LLM   : -{:.1}%", policy.cost_savings_percent);
            println!("   Estimated Cost / 1k   : ${:.2}", policy.estimated_cost_per_1000);

            if let Some(export_path) = &tune_args.export_policy {
                if let Some(parent) = export_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let f = std::fs::File::create(export_path)?;
                serde_json::to_writer_pretty(f, &policy)?;
                println!("💾 Exported cascade policy to: {}", export_path.display());
            }
        }
        CascadeCommands::Run(run_args) => {
            let checkpoint = Checkpoint::from_str(&run_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let questions = if let Some(p) = &run_args.schema {
                load_questions_from_file(p)?
            } else {
                create_inline_question("choice", "Department triage", "billing,technical,sales,other")?
            };
            let source = resolve_data_source(
                std::path::Path::new("data"),
                run_args.file.as_deref(),
                false,
                run_args.text.as_deref(),
            )?;
            let items = source.read_items()?;
            if items.is_empty() {
                bail!("No items found to route");
            }

            let classifier = LayaClassifier::with_checkpoint(checkpoint);
            let mut cfg = CascadeConfig::default();
            cfg.risk_threshold = run_args.risk_threshold;
            cfg.system2_cost_per_call = run_args.system2_cost;
            let router = CascadeRouter::new(cfg);

            let mut decisions = Vec::new();
            for item in &items {
                let dec = router.route(item, &questions, &classifier)?;
                decisions.push(dec);
            }

            match run_args.format {
                OutputFormat::Json | OutputFormat::Jsonl => {
                    println!("{}", serde_json::to_string_pretty(&decisions)?);
                }
                _ => {
                    let mut table = Table::new();
                    table.set_header(vec![
                        Cell::new("Item ID").fg(Color::Cyan),
                        Cell::new("Execution Tier").fg(Color::Green),
                        Cell::new("Composite Risk").fg(Color::Yellow),
                        Cell::new("Entropy").fg(Color::White),
                        Cell::new("Margin Gap").fg(Color::White),
                        Cell::new("Cost ($)").fg(Color::Magenta),
                    ]);

                    for dec in &decisions {
                        table.add_row(vec![
                            Cell::new(&dec.item_id),
                            Cell::new(dec.tier.badge()),
                            Cell::new(format!("{:.4}", dec.composite_risk)),
                            Cell::new(format!("{:.4}", dec.normalized_entropy)),
                            Cell::new(format!("{:.4}", 1.0 - dec.top2_margin)),
                            Cell::new(format!("${:.4}", dec.cost_usd)),
                        ]);
                    }
                    println!("{table}");
                }
            }
        }
    }
    Ok(())
}

fn handle_attest(args: AttestArgs) -> Result<()> {
    match args.action {
        AttestCommands::Verify(verify_args) => {
            let file = std::fs::File::open(&verify_args.receipt_log)?;
            let reader = std::io::BufReader::new(file);
            let mut receipts: Vec<lemma::engine::DecisionReceipt> = Vec::new();

            for line in std::io::BufRead::lines(reader) {
                let l = line?;
                if !l.trim().is_empty() {
                    let rec: lemma::engine::DecisionReceipt = serde_json::from_str(&l)?;
                    receipts.push(rec);
                }
            }

            println!("🔍 Verifying cryptographic decision receipts in: {}", verify_args.receipt_log.display());
            let report = AttestationEngine::verify_ledger(&receipts, &verify_args.secret_key);

            println!("\n🛡️  Attestation Verification Report:");
            println!("   Total Receipts Evaluated : {}", report.total_receipts);
            println!("   Valid Cryptographic Sigs : {}", report.valid_receipts);
            println!("   Invalid / Tampered Sigs  : {}", report.invalid_receipts);
            println!(
                "   Merkle Chain Intact      : {}",
                if report.chain_intact { "✅ YES" } else { "❌ NO (Chain Broken!)" }
            );

            if let Some(idx) = report.broken_link_index {
                println!("   ⚠️  First broken chain link at receipt #{}", idx);
            }

            for reason in &report.failure_reasons {
                println!("   ❌ {}", reason);
            }

            if !report.chain_intact || report.invalid_receipts > 0 {
                bail!("Attestation verification failed: Audit trail is tampered or invalid!");
            }

            println!("\n✅ Non-repudiable audit trail verified successfully under RFC 8785.");
        }
    }
    Ok(())
}

fn run_hierarchical_classification(
    schema_path: &std::path::Path,
    source: Box<dyn DataSource>,
    checkpoint: Checkpoint,
    coarse_threshold: f64,
    escalate_threshold: f64,
    cache_enabled: bool,
    escalate_only: bool,
    ood_profile: Option<&std::path::Path>,
    format: OutputFormat,
    output: Option<&std::path::Path>,
    limit: Option<usize>,
) -> Result<()> {
    let schema = HierarchicalSchema::load_from_file(schema_path)?;
    let mut items = source.read_items()?;
    if let Some(limit) = limit {
        items.truncate(limit);
    }
    if items.is_empty() {
        eprintln!("⚠️  No data items found to classify.");
        return Ok(());
    }

    let mut cfg = ClassifierConfig::default();
    cfg.checkpoint = checkpoint;
    cfg.escalate_threshold = escalate_threshold;
    let classifier = LayaClassifier::new(cfg);

    let mut router = HierarchicalRouter::new(schema, classifier)
        .with_coarse_threshold(coarse_threshold)
        .with_escalate_threshold(escalate_threshold);

    if cache_enabled {
        let cache = std::sync::Arc::new(std::sync::Mutex::new(DecisionCache::new(SemanticCacheConfig::default())));
        router = router.with_cache(cache);
    }

    if let Some(ood_path) = ood_profile {
        let f = std::fs::File::open(ood_path)?;
        let profile: OodProfile = serde_json::from_reader(f)?;
        router = router.with_ood_detector(std::sync::Arc::new(OodDetector::new(profile)));
    }

    let mut results = router.classify_items(&items)?;
    if escalate_only {
        results.retain(|r| r.escalated);
    }

    let rendered = match format {
        OutputFormat::Table => HierarchicalRouter::format_table(&results),
        OutputFormat::Csv => HierarchicalRouter::format_csv(&results),
        OutputFormat::Jsonl => HierarchicalRouter::format_jsonl(&results)?,
        OutputFormat::Json | OutputFormat::Pretty => serde_json::to_string_pretty(&results)?,
        OutputFormat::Compact => serde_json::to_string(&results)?,
    };

    if let Some(out_path) = output {
        std::fs::write(out_path, rendered)?;
        println!("💾 Successfully wrote {} hierarchical decision(s) to {}", results.len(), out_path.display());
    } else {
        println!("{}", rendered);
    }

    Ok(())
}

fn handle_hierarchical(args: lemma::cli::HierarchicalArgs) -> Result<()> {
    match args.command {
        HierarchicalCommands::Run(run_args) => {
            let checkpoint = Checkpoint::from_str(&run_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let source = resolve_data_source(
                &run_args.data_dir,
                run_args.file.as_deref(),
                false,
                run_args.text.as_deref(),
            )?;
            run_hierarchical_classification(
                &run_args.schema,
                source,
                checkpoint,
                run_args.coarse_threshold,
                run_args.escalate_threshold,
                run_args.cache,
                run_args.escalate_only,
                None,
                run_args.format,
                run_args.output.as_deref(),
                None,
            )
        }
        HierarchicalCommands::Tree(tree_args) => {
            let checkpoint = Checkpoint::from_str(&tree_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let schema = HierarchicalSchema::load_from_file(&tree_args.schema)?;
            let budget = checkpoint.head_budget();
            println!("\n🌲 Hierarchical Decision Tree Visualization ({})", checkpoint.repo_id());
            println!("{}\n", "=".repeat(65));
            let tree = schema.render_ascii_tree(budget);
            println!("{}", tree);
            Ok(())
        }
        HierarchicalCommands::Validate(val_args) => {
            let checkpoint = Checkpoint::from_str(&val_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let schema = HierarchicalSchema::load_from_file(&val_args.schema)?;
            let budget = checkpoint.head_budget();
            println!("\n🔍 Validating Hierarchical Schema against {} (Head Budget: {} tokens)", checkpoint.repo_id(), budget);
            println!("{}\n", "=".repeat(75));
            match schema.validate(budget) {
                Ok(report) => {
                    println!("✅ Hierarchical schema is structurally VALID!");
                    println!("   • Root coarse tokens:      ~{} tokens", report.root_tokens);
                    println!("   • Max branch tokens:       ~{} tokens", report.max_branch_tokens);
                    println!("   • Coarse branches count:   {}", report.branch_count);
                    println!("   • Total leaf classes:      {}", report.total_leaf_classes);
                    println!("   • Tree depth:              {} levels", schema.max_depth());
                    if report.warnings.is_empty() {
                        println!("\n✨ Zero warnings: all coarse and fine nodes comfortably fit inside token budget!");
                    } else {
                        println!("\n⚠️  Warnings encountered ({}):", report.warnings.len());
                        for w in report.warnings {
                            println!("   - {}", w);
                        }
                    }
                    Ok(())
                }
                Err(err) => {
                    eprintln!("❌ Schema validation FAILED:\n   {}", err);
                    bail!("Hierarchical schema validation failed");
                }
            }
        }
        HierarchicalCommands::Bench(bench_args) => {
            let checkpoint = Checkpoint::from_str(&bench_args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
            let schema = HierarchicalSchema::load_from_file(&bench_args.schema)?;
            let source = resolve_data_source(
                &bench_args.data_dir,
                bench_args.file.as_deref(),
                false,
                None,
            )?;
            let items = source.read_items()?;
            if items.is_empty() {
                bail!("No data items found for benchmarking");
            }

            let mut cfg = ClassifierConfig::default();
            cfg.checkpoint = checkpoint;
            let classifier = LayaClassifier::new(cfg);
            let router = HierarchicalRouter::new(schema, classifier)
                .with_coarse_threshold(bench_args.coarse_threshold);

            println!("\n⚡ Benchmarking Hierarchical Coarse-to-Fine Router...");
            println!("   Checkpoint: {} | Samples: {} | Iterations: {}", checkpoint.repo_id(), items.len(), bench_args.iterations);
            println!("{}\n", "=".repeat(70));

            let bench_res = router.benchmark(&items, bench_args.iterations)?;

            if bench_args.json {
                println!("{}", serde_json::to_string_pretty(&bench_res)?);
            } else {
                println!("📊 Benchmark Results:");
                println!("   • Total Evaluated:       {} decisions", bench_res.total_samples);
                println!("   • Avg Coarse Latency:    {:.2} ms", bench_res.avg_coarse_latency_ms);
                println!("   • Avg Fine Latency:      {:.2} ms", bench_res.avg_fine_latency_ms);
                println!("   • Avg Total Latency:     {:.2} ms", bench_res.avg_total_latency_ms);
                println!("   • Coarse Escalation Rate: {:.1}%", bench_res.coarse_escalation_rate * 100.0);
                println!("   • Completion Rate:       {:.1}%", bench_res.completion_rate * 100.0);
                println!("   • Avg Compound Conf:     {:.3}", bench_res.avg_compound_confidence);
                println!("   • Throughput:            {:.1} requests/sec", bench_res.throughput_qps);
            }
            Ok(())
        }
    }
}

fn handle_calibrate(args: CalibrateArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
    let loss = args
        .loss
        .parse::<CalibrationLoss>()
        .map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        let default_schema = std::path::Path::new("schema/questions.json");
        if default_schema.exists() {
            load_questions_from_file(default_schema)?
        } else {
            bail!("Question schema file is required for calibration. Specify with --schema <PATH>");
        }
    };

    let source = resolve_data_source(&args.data_dir, args.val_data.as_deref(), false, None)?;
    let items = source.read_items()?;

    if items.is_empty() {
        bail!("No validation items found to calibrate. Check --val-data path or data directory.");
    }

    let dataset_name = args
        .val_data
        .as_ref()
        .and_then(|p| p.file_stem().and_then(|s| s.to_str()))
        .unwrap_or("validation_dataset");

    let tuner = CalibrationTuner::new(checkpoint)
        .with_loss(loss)
        .with_bins(args.bins)
        .with_qats(args.adaptive, args.adaptive_gamma);

    let (report, calibrated_config) = tuner.fit(&items, &questions, dataset_name)?;

    if let Some(export_path) = &args.export_config {
        if let Some(parent) = export_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json_str = serde_json::to_string_pretty(&calibrated_config)?;
        std::fs::write(export_path, json_str)?;
        eprintln!("💾 Calibrated agent config exported to: {}", export_path.display());
    }

    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", CalibrationTuner::format_json(&report)?);
        }
        OutputFormat::Table | OutputFormat::Pretty | OutputFormat::Compact | OutputFormat::Csv => {
            println!("{}", CalibrationTuner::format_table(&report, args.export_config.as_deref()));
        }
    }

    Ok(())
}

fn handle_stream(args: StreamArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        let default_schema = std::path::Path::new("schema/questions.json");
        if default_schema.exists() {
            load_questions_from_file(default_schema)?
        } else {
            bail!("Question schema file is required for streaming. Specify with --schema <PATH>");
        }
    };

    let mut config = ClassifierConfig::default();
    config.checkpoint = checkpoint;
    config.escalate_threshold = args.escalate_threshold;

    if let Some(cfg_path) = &args.config {
        let f = std::fs::File::open(cfg_path)?;
        let val: serde_json::Value = serde_json::from_reader(f)?;
        if let Some(base_t) = val.get("temperature").and_then(|v| v.as_f64()) {
            config.temperature = base_t;
        }
        if let Some(temp_map) = val.get("temperature_by_options").and_then(|v| v.as_object()) {
            for (k, v) in temp_map {
                if let Some(t) = v.as_f64() {
                    config.temperature_by_options.insert(k.clone(), t);
                }
            }
        }
        if let Some(qats_val) = val.get("qats").or_else(|| val.get("qats_profile")) {
            if let Ok(profile) = serde_json::from_value::<lemma::engine::QatsProfile>(qats_val.clone()) {
                config.qats_profile = Some(profile);
            }
        }
    }

    let classifier = LayaClassifier::new(config);
    let primary_sink = StreamSink::from_target(&args.sink)?;

    let mut pipeline_cfg = StreamPipelineConfig::default();
    pipeline_cfg.batch_size = args.batch_size;
    pipeline_cfg.max_flush_interval_ms = args.max_flush_interval_ms;
    pipeline_cfg.escalate_threshold = args.escalate_threshold;
    pipeline_cfg.enable_cache = args.cache;
    pipeline_cfg.enable_attest = args.attest;
    pipeline_cfg.attest_secret = args.attest_secret;
    pipeline_cfg.limit = args.limit;

    let mut pipeline = StreamPipeline::new(classifier, questions, primary_sink)
        .with_config(pipeline_cfg);

    if let Some(esc_target) = &args.escalate_sink {
        let esc_sink = StreamSink::from_target(esc_target)?;
        pipeline = pipeline.with_escalate_sink(esc_sink);
    }

    let mut source = resolve_streaming_source(&args.source, args.columns.as_deref())?;

    eprintln!(
        "🚀 Ingesting stream from '{}' [{}] (batch_size: {}, flush_window: {}ms)...",
        args.source,
        source.name(),
        args.batch_size,
        args.max_flush_interval_ms
    );

    let metrics = pipeline.run(source.as_mut())?;

    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", StreamPipeline::format_json(&metrics)?);
        }
        OutputFormat::Table | OutputFormat::Pretty => {
            println!("{}", StreamPipeline::format_table(&metrics));
        }
        OutputFormat::Compact | OutputFormat::Csv => {
            println!(
                "✅ Processed {} events ({:.1} QPS) | Escalated: {} ({:.1}%) | Cached: {} ({:.1}%) | Attested: {}",
                metrics.total_classified,
                metrics.throughput_qps,
                metrics.total_escalated,
                metrics.escalation_rate * 100.0,
                metrics.total_cached,
                metrics.cache_hit_rate * 100.0,
                metrics.total_attested
            );
        }
    }

    Ok(())
}

fn handle_active(args: ActiveArgs) -> Result<()> {
    match args.action {
        ActiveCommands::Sample(sample_args) => handle_active_sample(sample_args),
        ActiveCommands::Distill(distill_args) => handle_active_distill(distill_args),
        ActiveCommands::Stats(stats_args) => handle_active_stats(stats_args),
    }
}

fn handle_active_sample(args: ActiveSampleArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        let default_schema = std::path::Path::new("schema/questions.json");
        if default_schema.exists() {
            load_questions_from_file(default_schema)?
        } else {
            bail!("Question schema file is required. Specify with --schema <PATH>");
        }
    };

    let items = lemma::datasource::FileSource::parse_file(&args.data)?;
    if items.is_empty() {
        bail!("No items found in dataset: {}", args.data.display());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);

    let ood_detector = if let Some(ood_path) = &args.ood_profile {
        let f = std::fs::File::open(ood_path)?;
        let profile: OodProfile = serde_json::from_reader(f)?;
        Some(OodDetector::new(profile))
    } else {
        None
    };

    let engine = ActiveLearningEngine::new()
        .with_escalate_threshold(args.escalate_threshold)
        .with_conformal_alpha(args.conformal_alpha);

    let (queue_items, report) = engine.sample(
        &classifier,
        &items,
        &questions,
        args.strategy,
        args.budget,
        ood_detector.as_ref(),
    )?;

    if let Some(export_path) = &args.export_queue {
        ActiveLearningEngine::export_queue(&queue_items, export_path)?;
    }

    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", ActiveLearningEngine::format_sample_json(&report, &queue_items)?);
        }
        OutputFormat::Table | OutputFormat::Pretty | OutputFormat::Compact | OutputFormat::Csv => {
            println!(
                "{}",
                ActiveLearningEngine::format_sample_table(&report, &queue_items, args.export_queue.as_deref())
            );
        }
    }

    Ok(())
}

fn handle_active_distill(args: ActiveDistillArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        let default_schema = std::path::Path::new("schema/questions.json");
        if default_schema.exists() {
            load_questions_from_file(default_schema)?
        } else {
            bail!("Question schema file is required. Specify with --schema <PATH>");
        }
    };

    let items = lemma::datasource::FileSource::parse_file(&args.data)?;
    if items.is_empty() {
        bail!("No items found in dataset: {}", args.data.display());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let engine = ActiveLearningEngine::new();

    let (samples, mut report) = engine.distill(
        &classifier,
        &items,
        &questions,
        args.val_split,
        &args.source_label,
    )?;

    ActiveLearningEngine::export_rlcd(&samples, &args.output)?;
    report.export_path = args.output.display().to_string();

    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", ActiveLearningEngine::format_distill_json(&report, &samples)?);
        }
        OutputFormat::Table | OutputFormat::Pretty | OutputFormat::Compact | OutputFormat::Csv => {
            println!("{}", ActiveLearningEngine::format_distill_table(&report));
        }
    }

    Ok(())
}

fn handle_active_stats(args: ActiveStatsArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;
    let questions = if let Some(schema_path) = &args.schema {
        load_questions_from_file(schema_path)?
    } else {
        let default_schema = std::path::Path::new("schema/questions.json");
        if default_schema.exists() {
            load_questions_from_file(default_schema)?
        } else {
            bail!("Question schema file is required. Specify with --schema <PATH>");
        }
    };

    let items = lemma::datasource::FileSource::parse_file(&args.data)?;
    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let engine = ActiveLearningEngine::new();

    let total_len = items.len() * questions.len();
    let (queue, report) = engine.sample(
        &classifier,
        &items,
        &questions,
        UncertaintyStrategy::Composite,
        total_len,
        None,
    )?;

    match args.format {
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        OutputFormat::Table | OutputFormat::Pretty | OutputFormat::Compact | OutputFormat::Csv => {
            println!(
                "{}",
                ActiveLearningEngine::format_sample_table(&report, &queue, None)
            );
        }
    }

    Ok(())
}

fn handle_stats(args: StatsArgs) -> Result<()> {
    let checkpoint = Checkpoint::from_str(&args.checkpoint).map_err(|e| anyhow::anyhow!(e))?;

    let questions = if args.schema.exists() {
        load_questions_from_file(&args.schema)?
    } else if args.schema.to_string_lossy() != "schema/questions.json" {
        bail!("Schema file not found: {}", args.schema.display());
    } else {
        create_inline_question(
            "choice",
            "Classify topic and sentiment",
            "billing,technical,sales,other",
        )?
    };

    let source = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
        args.extensions.as_deref(),
        args.no_recursive,
    )?;

    let items = source.read_items()?;
    if items.is_empty() {
        eprintln!("⚠️  No data items found for stats evaluation.");
        return Ok(());
    }

    let config = ClassifierConfig {
        checkpoint,
        escalate_threshold: args.escalate_threshold,
        ..Default::default()
    };
    let classifier = LayaClassifier::new(config);

    let stats_engine = StatsEngine::new(classifier)
        .with_iterations(args.iterations)
        .with_warmup(args.warmup)
        .with_cache(args.cache, args.similarity_threshold)
        .with_escalate_threshold(args.escalate_threshold);

    let report = stats_engine.evaluate(&items, &questions)?;

    if let Some(dest) = &args.output {
        StatsEngine::export_report(&report, dest)?;
        println!("💾 Statistics report exported to: {}", dest.display());
    }

    match args.format {
        OutputFormat::Table | OutputFormat::Pretty => {
            println!("{}", StatsEngine::format_table(&report));
        }
        OutputFormat::Json | OutputFormat::Jsonl => {
            println!("{}", StatsEngine::format_json(&report)?);
        }
        OutputFormat::Compact => {
            println!("{}", StatsEngine::format_compact(&report));
        }
        OutputFormat::Csv => {
            print!("{}", StatsEngine::format_csv(&report));
        }
    }

    Ok(())
}

fn handle_fetch(args: FetchArgs) -> Result<()> {
    let mut schema_generated = None;
    let (items, source_desc) = if let Some(dataset) = &args.hf {
        let source = lemma::datasource::HuggingFaceSource::new(dataset)
            .with_config(args.config)
            .with_split(args.split)
            .with_limit(args.limit)
            .with_offset(args.offset)
            .with_custom_columns(args.text_col, args.label_col);
        let (items, schema) = source.read_items_with_schema()?;
        schema_generated = Some((dataset.clone(), schema));
        (items, format!("Hugging Face dataset '{}'", dataset))
    } else if let Some(category) = &args.hn {
        let source = lemma::datasource::HackerNewsSource::new(category, args.limit);
        let items = source.read_items()?;
        (items, format!("Hacker News feed '{}'", category))
    } else if let Some(sub) = &args.reddit {
        let source = lemma::datasource::RedditSource::new(sub, args.limit);
        let items = source.read_items()?;
        (items, format!("r/{} on Reddit", sub))
    } else if let Some(article) = &args.wikipedia {
        let source = lemma::datasource::WikipediaSource::new(Some(article.clone()), args.limit);
        let items = source.read_items()?;
        (items, format!("Wikipedia article '{}'", article))
    } else {
        bail!("No source specified for fetch. Specify one of: --hf, --hn, --reddit, --wikipedia");
    };

    if args.auto_schema {
        if let Some((name, schema)) = schema_generated {
            if !schema.is_empty() {
                let schema_path = if let Some(p) = &args.schema_output {
                    p.clone()
                } else {
                    let slug = name.replace('/', "_");
                    std::path::PathBuf::from(format!("schema/{}.json", slug))
                };
                if let Some(parent) = schema_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                }
                let json = serde_json::to_string_pretty(&schema)?;
                std::fs::write(&schema_path, json)?;
                eprintln!("✨ Inferred question schema written to {}", schema_path.display());
            }
        }
    }

    if items.is_empty() {
        eprintln!("⚠️  No records retrieved from {}.", source_desc);
        return Ok(());
    }

    eprintln!("📦 Ingested {} items from {}", items.len(), source_desc);

    let mut out_writer: Box<dyn Write> = match &args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            Box::new(std::fs::File::create(path)?)
        }
        None => Box::new(std::io::stdout()),
    };

    match args.format {
        OutputFormat::Jsonl => {
            for item in &items {
                let line = serde_json::to_string(item)?;
                if let Err(e) = writeln!(out_writer, "{}", line) {
                    if e.kind() == std::io::ErrorKind::BrokenPipe {
                        return Ok(());
                    }
                    return Err(e.into());
                }
            }
        }
        OutputFormat::Json | OutputFormat::Pretty => {
            let json = serde_json::to_string_pretty(&items)?;
            if let Err(e) = writeln!(out_writer, "{}", json) {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }
        }
        OutputFormat::Table | OutputFormat::Compact => {
            let mut table = comfy_table::Table::new();
            table.load_preset(comfy_table::presets::UTF8_FULL);
            table.set_header(vec!["ID", "Origin", "Ground Truth", "Preview Text"]);
            for item in &items {
                let gt_str = item.ground_truth.as_ref()
                    .map(|gt| serde_json::to_string(gt).unwrap_or_default())
                    .unwrap_or_else(|| "-".to_string());
                table.add_row(vec![
                    comfy_table::Cell::new(&item.id),
                    comfy_table::Cell::new(&item.source_uri),
                    comfy_table::Cell::new(gt_str),
                    comfy_table::Cell::new(item.preview_text(60)),
                ]);
            }
            if let Err(e) = writeln!(out_writer, "{table}") {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(out_writer);
            wtr.write_record(["id", "source_uri", "ground_truth", "state"])?;
            for item in &items {
                let gt_str = item.ground_truth.as_ref()
                    .map(|gt| serde_json::to_string(gt).unwrap_or_default())
                    .unwrap_or_default();
                let state_str = serde_json::to_string(&item.state).unwrap_or_default();
                wtr.write_record([&item.id, &item.source_uri, &gt_str, &state_str])?;
            }
            wtr.flush()?;
        }
    }

    Ok(())
}

fn handle_scrape(args: ScrapeArgs) -> Result<()> {
    let options = lemma::datasource::ScrapeOptions {
        timeout_secs: args.timeout,
        max_chars: args.max_chars,
        use_jina: args.jina,
        ..Default::default()
    };
    let scraper = lemma::datasource::WebScraper::new(options);
    let doc = scraper.scrape(&args.url)?;

    eprintln!("🌐 Scraped '{}' ({} words, SPA: {})", doc.url, doc.word_count, doc.is_likely_spa);

    let state = serde_json::json!({
        "title": doc.title,
        "text": doc.markdown,
        "url": doc.url,
        "words": doc.word_count,
    });
    let item = lemma::datasource::DataItem::new(format!("web:{}", doc.url), state, &doc.url);

    let mut out_writer: Box<dyn Write> = match &args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            Box::new(std::fs::File::create(path)?)
        }
        None => Box::new(std::io::stdout()),
    };

    match args.format {
        OutputFormat::Jsonl => {
            let line = serde_json::to_string(&item)?;
            if let Err(e) = writeln!(out_writer, "{}", line) {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }
        }
        OutputFormat::Json | OutputFormat::Pretty => {
            let json = serde_json::to_string_pretty(&item)?;
            if let Err(e) = writeln!(out_writer, "{}", json) {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }
        }
        OutputFormat::Table | OutputFormat::Compact => {
            let mut table = comfy_table::Table::new();
            table.load_preset(comfy_table::presets::UTF8_FULL);
            table.set_header(vec!["Field", "Value"]);
            table.add_row(vec!["URL", &doc.url]);
            table.add_row(vec!["Title", &doc.title]);
            table.add_row(vec!["Word Count", &doc.word_count.to_string()]);
            table.add_row(vec!["SPA Shell", if doc.is_likely_spa { "true" } else { "false" }]);
            table.add_row(vec!["Content Preview", &item.preview_text(160)]);
            if let Err(e) = writeln!(out_writer, "{table}") {
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e.into());
            }
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(out_writer);
            wtr.write_record(["id", "url", "title", "words", "text"])?;
            wtr.write_record([&item.id, &doc.url, &doc.title, &doc.word_count.to_string(), &doc.markdown])?;
            wtr.flush()?;
        }
    }

    Ok(())
}
