#![allow(clippy::field_reassign_with_default, clippy::too_many_arguments)]

use anyhow::{Result, bail};
use clap::Parser;
use comfy_table::{Cell, Color, Table};
use std::io::Write;
use std::str::FromStr;

use lemma::benchmark::BenchmarkRunner;
use lemma::cli::{
    ActiveArgs, ActiveCommands, ActiveDistillArgs, ActiveSampleArgs, ActiveStatsArgs, AttestArgs,
    AttestCommands, Backend, BenchArgs, CacheArgs, CacheCommands, CalibrateArgs, CascadeArgs,
    CascadeCommands, Cli, Commands, CompareArgs, CompletionsArgs, ConformalArgs, ConsensusArgs,
    DownloadArgs, DriftArgs, ExperimentArgs, ExperimentMode, ExplainArgs, FetchArgs, GuardArgs,
    GuideArgs, HierarchicalCommands, InfoArgs, InspectArgs, LearnArgs, OodArgs, OodCommands,
    OutputFormat, SchemaArgs, SchemaCommands, SchemaInferArgs, SchemaOptimizeArgs,
    SchemaValidateArgs, ScrapeArgs, StatsArgs, StreamArgs, TestArgs,
};
use lemma::datasource::{DataItem, DataSource, StreamSink};
use lemma::engine::{
    ActiveLearningEngine, AttestationEngine, AttributionEngine, CalibrationLoss, CalibrationTuner,
    CascadeConfig, CascadeRouter, ClassifierConfig, ConformalEngine, ConsensusEngine,
    DecisionCache, DriftEngine, Evaluator, ExperimentRunner, GuardAction, GuardrailEngine,
    GuardrailPolicy, HierarchicalRouter, LayaClassifier, OodDetector, OodProfile,
    SchemaKnapsackOptimizer, SchemaSynthesizer, SemanticCacheConfig, StatsEngine, StreamPipeline,
    StreamPipelineConfig, UncertaintyStrategy, compute_schema_hash, parse_cost_weights,
};
use lemma::model::{Checkpoint, Downloader, HierarchicalSchema, inspect_safetensors};
use lemma::{
    create_inline_question, load_questions_from_file, resolve_data_source,
    resolve_data_source_filtered, resolve_streaming_source,
};

fn main() -> Result<()> {
    let cli = Cli::parse();
    lemma::model::native::set_backend(cli.backend);
    lemma::model::native::set_device(cli.device);
    if let Err(e) = run(cli) {
        if let Some(io_err) = e.downcast_ref::<std::io::Error>()
            && io_err.kind() == std::io::ErrorKind::BrokenPipe
        {
            return Ok(());
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
        Commands::Compare(args) => handle_compare(args),
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
        Commands::Learn(args) => handle_learn(args),
        Commands::Guide(args) => handle_guide(args),
        Commands::Completions(args) => handle_completions(args),
    }
}

fn handle_completions(args: CompletionsArgs) -> Result<()> {
    use clap::CommandFactory;
    use clap_complete::generate;

    let mut cmd = Cli::command();
    let mut buf = Vec::new();
    generate(args.shell, &mut cmd, "lemma", &mut buf);
    let mut stdout = std::io::stdout();
    if let Err(e) = stdout.write_all(&buf) {
        if e.kind() == std::io::ErrorKind::BrokenPipe {
            return Ok(());
        }
        return Err(e.into());
    }
    Ok(())
}

fn is_machine_format(format: OutputFormat) -> bool {
    matches!(
        format,
        OutputFormat::Json | OutputFormat::Jsonl | OutputFormat::Csv
    )
}

fn log_status(quiet: bool, msg: impl std::fmt::Display) {
    if quiet {
        eprintln!("{msg}");
    } else {
        println!("{msg}");
    }
}

fn parse_checkpoint(s: &str) -> Result<Checkpoint> {
    Checkpoint::from_str(s).map_err(|e| anyhow::anyhow!(e))
}

fn ensure_parent_dir(path: &std::path::Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_json_file<T: serde::Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    ensure_parent_dir(path)?;
    serde_json::to_writer_pretty(std::fs::File::create(path)?, value)?;
    Ok(())
}

fn open_output_writer(output: Option<&std::path::Path>) -> Result<Box<dyn Write>> {
    match output {
        Some(path) => {
            ensure_parent_dir(path)?;
            Ok(Box::new(std::io::BufWriter::new(std::fs::File::create(
                path,
            )?)))
        }
        None => Ok(Box::new(std::io::stdout())),
    }
}

fn load_schema_or_default(
    schema: Option<&std::path::Path>,
) -> Result<std::collections::BTreeMap<String, lemma::model::Question>> {
    if let Some(path) = schema {
        load_questions_from_file(path)
    } else if std::path::Path::new("schema/questions.json").exists() {
        load_questions_from_file(std::path::Path::new("schema/questions.json"))
    } else {
        bail!("Question schema file is required. Specify with --schema <PATH>")
    }
}

fn load_questions_or_inline(
    schema: Option<&std::path::Path>,
    instruction: &str,
) -> Result<std::collections::BTreeMap<String, lemma::model::Question>> {
    match schema {
        Some(path) => load_questions_from_file(path),
        None => create_inline_question("choice", instruction, "billing,technical,sales,other"),
    }
}

fn item_text(item: &DataItem) -> Result<String> {
    match &item.state {
        serde_json::Value::String(s) => Ok(s.clone()),
        other => Ok(serde_json::to_string(other)?),
    }
}

fn apply_calibrated_config(
    config: &mut ClassifierConfig,
    cfg_path: &std::path::Path,
    override_base_temp: bool,
) -> Result<()> {
    let val: serde_json::Value = serde_json::from_reader(std::fs::File::open(cfg_path)?)?;
    if override_base_temp
        && let Some(base_t) = val.get("temperature").and_then(|v| {
            v.as_f64().or_else(|| {
                v.as_array()
                    .and_then(|a| a.first())
                    .and_then(|x| x.as_f64())
            })
        })
    {
        config.temperature = base_t;
    }
    if let Some(temp_map) = val
        .get("temperature_by_options")
        .and_then(|v| v.as_object())
    {
        for (k, v) in temp_map {
            if let Some(t) = v.as_f64() {
                config.temperature_by_options.insert(k.clone(), t);
            }
        }
    }
    if let Some(qats_val) = val.get("qats").or_else(|| val.get("qats_profile"))
        && let Ok(profile) = serde_json::from_value::<lemma::engine::QatsProfile>(qats_val.clone())
    {
        config.qats_profile = Some(profile);
    }
    if let Some(cw_obj) = val.get("cost_weights").and_then(|v| v.as_object()) {
        for (k, v) in cw_obj {
            if let Some(w) = v.as_f64() {
                config.cost_weights.insert(k.to_lowercase(), w);
            }
        }
    }
    Ok(())
}

fn handle_download(args: DownloadArgs) -> Result<()> {
    let checkpoints: Vec<Checkpoint> = if args.all || args.checkpoint.eq_ignore_ascii_case("all") {
        vec![
            Checkpoint::English,
            Checkpoint::Multilingual,
            Checkpoint::TypedDecisions,
            Checkpoint::ModernBertZeroShot,
            Checkpoint::DebertaV3ZeroShot,
            Checkpoint::EuroBert,
            Checkpoint::NeoBert,
        ]
    } else {
        vec![parse_checkpoint(&args.checkpoint)?]
    };

    let downloader = if args.verify_only {
        Downloader::detect().unwrap_or(Downloader {
            cli_binary: "hf".into(),
        })
    } else {
        Downloader::detect()?
    };

    if args.verify_only {
        let mut failed = Vec::new();
        for &checkpoint in &checkpoints {
            let report = downloader.verify_checkpoint(checkpoint, &args.target_dir)?;
            println!(
                "🔍 Verifying checkpoint {} in {}",
                checkpoint.repo_id(),
                report.path.display()
            );
            if report.is_valid {
                println!("✅ All required checkpoint files are valid and present!");
                for (f, sz) in report.found {
                    println!("   - {}: {:.2} MB", f, sz as f64 / 1_048_576.0);
                }
            } else {
                eprintln!("❌ Missing required files in {}:", report.path.display());
                for m in &report.missing {
                    eprintln!("   - {}", m);
                }
                failed.push((checkpoint.repo_id(), report.missing.len(), report.path));
            }
        }
        if !failed.is_empty() {
            let (repo, count, path) = &failed[0];
            bail!(
                "Checkpoint verification failed for {} ({}/{} checkpoint(s) incomplete): {} missing required file(s) in {}",
                repo,
                failed.len(),
                checkpoints.len(),
                count,
                path.display()
            );
        }
        return Ok(());
    }

    for checkpoint in checkpoints {
        downloader.download(checkpoint, &args.target_dir, args.dry_run)?;
    }
    Ok(())
}

fn handle_classify(args: lemma::cli::ClassifyArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let inline_text = args.text.as_deref().or(args.positional_text.as_deref());

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
            inline_text,
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

    let questions = match &args.schema {
        Some(p) => load_questions_from_file(p)?,
        None => create_inline_question(&args.qtype, &args.instruction, &args.labels)?,
    };

    let mut items = source.read_items()?;
    if let Some(limit) = args.limit {
        items.truncate(limit);
    }
    if items.is_empty() {
        eprintln!("⚠️  No data items found to classify. Check input path or data/ folder.");
        return Ok(());
    }

    let chunk_strategy = args
        .chunk_strategy
        .parse::<lemma::model::ChunkStrategy>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut config = ClassifierConfig::with_checkpoint(checkpoint);
    config.multi_label = args.multi_label;
    config.multi_label_threshold = args.multi_label_threshold;
    config.escalate_threshold = args.escalate_threshold;
    config.hypothesis_template = args.hypothesis_template.clone();
    config.chunk_strategy = chunk_strategy;
    config.shortlist_k = args.shortlist_k;
    if (args.temperature - 1.0).abs() > 1e-6 {
        config.temperature = args.temperature;
        if args.config.is_none() {
            config.temperature_by_options.clear();
        }
    }

    if let Some(cfg_path) = &args.config {
        apply_calibrated_config(
            &mut config,
            cfg_path,
            (args.temperature - 1.0).abs() <= 1e-6,
        )?;
    }
    let classifier = LayaClassifier::new(config);

    let ood_detector = match &args.ood_profile {
        Some(ood_path) => Some(OodDetector::new(serde_json::from_reader(
            std::fs::File::open(ood_path)?,
        )?)?),
        None => None,
    };

    let mut cache = args.cache.then(|| {
        DecisionCache::new(SemanticCacheConfig {
            semantic: args.cache_semantic,
            similarity_threshold: args.cache_threshold,
            ttl_seconds: args.cache_ttl,
            ..Default::default()
        })
    });

    let schema_hash = compute_schema_hash(&questions);
    let mut raw_results = Vec::new();

    for item in &items {
        let text = item_text(item)?;
        if let Some(c) = &mut cache {
            match c.lookup(&text, &schema_hash) {
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
            let ood_eval = detector.evaluate(&item.id, &text, &classifier, &questions)?;
            if ood_eval.is_ood || ood_eval.anomaly_score >= args.ood_threshold {
                match args.ood_action.to_lowercase().as_str() {
                    "quarantine" => res
                        .answers
                        .values_mut()
                        .for_each(|a| a.act_probability = Some(1.0)),
                    "escalate" => res
                        .answers
                        .values_mut()
                        .for_each(|a| a.act_probability = Some(0.99)),
                    "flag" => eprintln!(
                        "🚩 Item '{}' flagged as OOD (score={:.3}, energy={:.2})",
                        item.id, ood_eval.anomaly_score, ood_eval.free_energy
                    ),
                    _ => {}
                }
            }
        }

        if let Some(c) = &mut cache {
            c.insert(&text, &schema_hash, &res);
        }
        raw_results.push((item.id.clone(), res));
    }

    if args.cache_stats
        && let Some(c) = &cache
    {
        let stats = c.stats();
        eprintln!(
            "\n📊 Semantic Decision Cache Statistics:\n   Total Lookups : {}\n   Exact Hits    : {}\n   Semantic Hits : {}\n   Misses        : {}\n   Hit Rate      : {:.2}%\n   Cached Items  : {}",
            stats.total_lookups(),
            stats.exact_hits,
            stats.semantic_hits,
            stats.misses,
            stats.hit_rate() * 100.0,
            c.len()
        );
    }

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

    let mut out_writer = open_output_writer(args.output.as_deref())?;

    match args.format {
        OutputFormat::Json => writeln!(out_writer, "{}", serde_json::to_string(&results)?)?,
        OutputFormat::Pretty => {
            writeln!(out_writer, "{}", serde_json::to_string_pretty(&results)?)?
        }
        OutputFormat::Jsonl => {
            for (id, res) in &results {
                let row = serde_json::json!({ "id": id, "answers": res.answers, "latency_ms": res.latency_ms });
                writeln!(out_writer, "{}", serde_json::to_string(&row)?)?;
            }
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(out_writer);
            wtr.write_record([
                "item_id",
                "question_id",
                "prediction",
                "confidence",
                "act_probability",
                "latency_ms",
            ])?;
            for (id, res) in &results {
                for (qid, ans) in &res.answers {
                    wtr.write_record([
                        id.as_str(),
                        qid.as_str(),
                        &format_answer_value(ans, "; "),
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
                        write!(
                            out_writer,
                            "{qid}:[{}] ({:.2}%) ",
                            labels.join(", "),
                            ans.confidence * 100.0
                        )?;
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

            for (id, res) in &results {
                let snippet = item_map
                    .get(id.as_str())
                    .map(|it| it.preview_text(40))
                    .unwrap_or_default();
                for (qid, ans) in &res.answers {
                    let pred_str = match ans.noul {
                        Some(n) => format!("true: {:.2}%", n * 100.0),
                        None => format_answer_value(ans, ", "),
                    };
                    let esc_str = if ans.act_probability.unwrap_or(0.0) > 0.5 {
                        "YES (⚠️)"
                    } else {
                        "no"
                    };
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

            writeln!(
                out_writer,
                "\n{table}\n\nClassified {} items across {} questions using {} checkpoint.\n",
                results.len(),
                questions.len(),
                checkpoint.repo_id()
            )?;
        }
    }

    Ok(())
}

fn format_answer_value(ans: &lemma::model::DecisionAnswer, sep: &str) -> String {
    if let Some(labels) = &ans.selected_labels {
        labels.join(sep)
    } else if let Some(c) = &ans.choice {
        c.clone()
    } else if let Some(s) = ans.score {
        format!("{:.2}", s)
    } else if let Some(n) = ans.noul {
        format!("{:.4}", n)
    } else {
        "N/A".into()
    }
}

fn handle_experiment(args: ExperimentArgs) -> Result<()> {
    let questions =
        load_questions_or_inline(args.schema.as_deref(), "Classify the topic or sentiment")?;
    let items =
        resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?.read_items()?;
    if items.is_empty() {
        eprintln!("⚠️  No data items found for experiment.");
        return Ok(());
    }

    let ckpt = parse_checkpoint(&args.checkpoint)?;
    let classifier = LayaClassifier::with_checkpoint(ckpt);
    let temps = [0.5, 0.8, 1.0, 1.25, 1.5, 1.76, 2.0, 2.5];
    let mode = if args.adversarial {
        ExperimentMode::Adversarial
    } else {
        args.mode
    };

    if args.json {
        let out = match mode {
            ExperimentMode::Order => {
                let (qid, _) = questions
                    .iter()
                    .find(|(_, q)| q.qtype == lemma::model::QuestionType::Choice)
                    .unwrap_or_else(|| questions.iter().next().unwrap());
                serde_json::json!({
                    "mode": "order",
                    "question_id": qid,
                    "permutations": args.permutations,
                    "samples": items.len(),
                })
            }
            ExperimentMode::Temp => {
                let mut rows = Vec::new();
                for &t in &temps {
                    let mut cfg = ClassifierConfig::with_checkpoint(ckpt);
                    cfg.temperature = t;
                    cfg.temperature_by_options.clear();
                    let m =
                        Evaluator::new(LayaClassifier::new(cfg)).evaluate(&items, &questions)?;
                    rows.push(serde_json::json!({
                        "temperature": t, "accuracy": m.accuracy, "ece": m.ece,
                        "brier_score": m.brier_score, "soft_accuracy": m.soft_accuracy,
                    }));
                }
                serde_json::json!({ "mode": "temp", "sweep": rows })
            }
            ExperimentMode::Checkpoints => {
                let arena = ExperimentRunner::run_arena(
                    &items,
                    &questions,
                    &lemma::model::native::runnable_checkpoints(),
                    5,
                )?;
                serde_json::json!({
                    "mode": "checkpoints",
                    "reference_model": arena.reference_model,
                    "results": arena.models,
                    "top_disagreements": arena.top_disagreements,
                })
            }
            ExperimentMode::Adversarial => {
                let report =
                    ExperimentRunner::test_adversarial_robustness(&classifier, &items, &questions)?;
                serde_json::to_value(report)?
            }
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    match mode {
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
            ExperimentRunner::sweep_temperature_calibration_for_checkpoint(
                ckpt, &items, &questions, &temps,
            )?;
        }
        ExperimentMode::Checkpoints => {
            ExperimentRunner::compare_checkpoints(&items, &questions)?;
        }
        ExperimentMode::Adversarial => {
            let report =
                ExperimentRunner::test_adversarial_robustness(&classifier, &items, &questions)?;
            ExperimentRunner::print_adversarial_report(&report);
        }
    }
    Ok(())
}

fn handle_compare(args: CompareArgs) -> Result<()> {
    let chunk_strategy = args
        .chunk_strategy
        .parse::<lemma::model::ChunkStrategy>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut questions =
        load_questions_or_inline(args.schema.as_deref(), "Classify the topic or sentiment")?;
    if let Some(tpl) = &args.hypothesis_template {
        for q in questions.values_mut() {
            if q.hypothesis_template.is_none() {
                q.hypothesis_template = Some(tpl.clone());
            }
        }
    }
    let items =
        resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?.read_items()?;
    if items.is_empty() {
        bail!("No data items found for Model Arena comparison.");
    }

    let checkpoints: Vec<Checkpoint> = if args.checkpoints.eq_ignore_ascii_case("all") {
        lemma::model::native::runnable_checkpoints()
    } else {
        let list: Vec<Checkpoint> = args
            .checkpoints
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(parse_checkpoint)
            .collect::<Result<_>>()?;
        if list.is_empty() {
            bail!("At least one checkpoint must be specified for comparison");
        }
        list
    };

    let report = ExperimentRunner::run_arena_with_options(
        &items,
        &questions,
        &checkpoints,
        args.top_disagreements,
        chunk_strategy,
        args.hypothesis_template.as_deref(),
        args.shortlist_k,
    )?;
    let slice_matrix = if let Some(slice_str) = &args.slice_by {
        let mode = slice_str
            .parse::<lemma::engine::SliceBy>()
            .map_err(|e| anyhow::anyhow!(e))?;
        Some(
            lemma::engine::slicing::compute_arena_slice_matrix_with_options(
                &items,
                &questions,
                &checkpoints,
                &mode,
                chunk_strategy,
                args.hypothesis_template.as_deref(),
                args.shortlist_k,
            )?,
        )
    } else {
        None
    };

    if args.json {
        if let Some((slices, rows)) = &slice_matrix {
            let mut val = serde_json::to_value(&report)?;
            if let Some(obj) = val.as_object_mut() {
                obj.insert(
                    "slice_matrix".into(),
                    serde_json::json!({ "slices": slices, "rows": rows }),
                );
            }
            println!("{}", serde_json::to_string_pretty(&val)?);
        } else {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    } else {
        ExperimentRunner::print_arena_report(&report);
        if let Some((slices, rows)) = &slice_matrix {
            lemma::engine::print_arena_slice_matrix(slices, rows);
        }
    }
    Ok(())
}

fn handle_test(args: TestArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_questions_or_inline(args.schema.as_deref(), "Classify topic")?;
    let items = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        false,
        None,
        args.extensions.as_deref(),
        args.no_recursive,
    )?
    .read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No labeled items found in data source.");
        return Ok(());
    }

    let chunk_strategy = args
        .chunk_strategy
        .parse::<lemma::model::ChunkStrategy>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut cfg = ClassifierConfig::with_checkpoint(checkpoint);
    cfg.hypothesis_template = args.hypothesis_template.clone();
    cfg.chunk_strategy = chunk_strategy;
    cfg.shortlist_k = args.shortlist_k;
    if let Some(config_path) = &args.config {
        apply_calibrated_config(&mut cfg, config_path, args.temperature.is_none())?;
    }
    if let Some(t) = args.temperature {
        cfg.temperature = t;
        if args.config.is_none() {
            cfg.temperature_by_options.clear();
        }
    }
    let slice_entries = if let Some(slice_str) = &args.slice_by {
        let mode = slice_str
            .parse::<lemma::engine::SliceBy>()
            .map_err(|e| anyhow::anyhow!(e))?;
        Some(lemma::engine::evaluate_slices(
            &LayaClassifier::new(cfg.clone()),
            &items,
            &questions,
            &mode,
        )?)
    } else {
        None
    };
    let evaluator = Evaluator::new(LayaClassifier::new(cfg));

    log_status(
        args.json,
        format!(
            "🔬 Evaluating {} items against ground truth with checkpoint {}...",
            items.len(),
            checkpoint.repo_id()
        ),
    );
    let metrics = evaluator.evaluate(&items, &questions)?;

    if let Some(save_path) = &args.save_baseline {
        write_json_file(save_path, &metrics)?;
        log_status(
            args.json,
            format!("💾 Saved baseline metrics to: {}", save_path.display()),
        );
    }

    let baseline_diff = if let Some(base_path) = &args.baseline {
        let base_metrics: lemma::engine::EvaluationMetrics =
            serde_json::from_reader(std::fs::File::open(base_path)?)?;
        Some(Evaluator::compare_baseline(
            &metrics,
            &base_metrics,
            args.max_regression,
            args.max_ece,
        ))
    } else {
        None
    };

    if args.json {
        if baseline_diff.is_some() || slice_entries.is_some() {
            let mut val = serde_json::to_value(&metrics)?;
            if let Some(obj) = val.as_object_mut() {
                if let Some(diff) = &baseline_diff {
                    obj.insert("baseline_diff".into(), serde_json::to_value(diff)?);
                }
                if let Some(slices) = &slice_entries {
                    obj.insert("slices".into(), serde_json::to_value(slices)?);
                }
            }
            println!("{}", serde_json::to_string_pretty(&val)?);
        } else {
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
    } else {
        Evaluator::print_report(&metrics);
        if let Some(slices) = &slice_entries {
            lemma::engine::print_slice_report(slices);
        }
        if let Some(diff) = &baseline_diff {
            Evaluator::print_baseline_diff(diff);
        }
    }

    if let Some(min_acc) = args.min_accuracy {
        if metrics.accuracy < min_acc {
            bail!(
                "Accuracy test failed! Observed: {:.2}%, Required minimum: {:.2}%",
                metrics.accuracy * 100.0,
                min_acc * 100.0
            );
        }
        log_status(
            args.json,
            format!(
                "✅ Accuracy test passed: {:.2}% >= {:.2}%",
                metrics.accuracy * 100.0,
                min_acc * 100.0
            ),
        );
    }

    if let Some(limit_ece) = args.max_ece
        && metrics.ece > limit_ece + 1e-6
    {
        bail!(
            "Calibration gate failed! Observed ECE {:.4} exceeds maximum allowed {:.4}",
            metrics.ece,
            limit_ece
        );
    }

    if let Some(diff) = &baseline_diff
        && diff.regressed
    {
        bail!(
            "Baseline regression gate failed:\n   - {}",
            diff.reasons.join("\n   - ")
        );
    }
    Ok(())
}

fn handle_bench(args: BenchArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions =
        load_questions_or_inline(args.schema.as_deref(), "Classify department and priority")?;
    let items = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        false,
        None,
        args.extensions.as_deref(),
        args.no_recursive,
    )?
    .read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No items found to benchmark. Please supply items in data/ or via --file.");
        return Ok(());
    }

    let runner = BenchmarkRunner::new(LayaClassifier::with_checkpoint(checkpoint))
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

fn checkpoint_domain_summary(ckpt: Checkpoint) -> &'static str {
    match ckpt {
        Checkpoint::English => {
            "English text classification, email triage, toxicity guardrails (~39.5ms)"
        }
        Checkpoint::Multilingual => "100+ languages, 2.2x faster forward pass (~32.8ms)",
        Checkpoint::TypedDecisions => {
            "Specialized enterprise workflows (0.766 acc after fine-tuning)"
        }
        Checkpoint::AutoRoute => "<0.5ms Unicode script router between English & Multilingual",
        Checkpoint::ModernBertZeroShot => {
            "ModernBERT-large NLI zero-shot cross-encoder (8k context, RoPE + FlashAttn)"
        }
        Checkpoint::DebertaV3ZeroShot => {
            "DeBERTa-v3-large NLI zero-shot cross-encoder (512 context, disentangled attn)"
        }
        Checkpoint::EuroBert => {
            "Multilingual & European bidirectional encoder (8k context, RoPE + SwiGLU)"
        }
        Checkpoint::NeoBert => {
            "Compact 250M English representation encoder (4k context, RoPE + SwiGLU)"
        }
        Checkpoint::ModernBertNli => "Zero-shot NLI encoder baseline comparison profile",
    }
}

fn handle_info(args: InfoArgs) -> Result<()> {
    if let Some(dir) = &args.inspect_dir {
        let spec = lemma::model::inspect_model_directory(dir)?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&spec)?);
        } else {
            lemma::model::print_models_inventory(&[spec]);
        }
        return Ok(());
    }
    if args.inventory {
        let specs = lemma::model::scan_registered_models();
        if args.json {
            println!("{}", serde_json::to_string_pretty(&specs)?);
        } else {
            lemma::model::print_models_inventory(&specs);
        }
        return Ok(());
    }

    let checkpoints: Vec<Checkpoint> = if args.checkpoint.eq_ignore_ascii_case("all") {
        Checkpoint::all().to_vec()
    } else {
        vec![parse_checkpoint(&args.checkpoint)?]
    };

    let verifier = Downloader {
        cli_binary: "hf".into(),
    };
    let primary_ckpt = checkpoints.first().copied().unwrap_or(Checkpoint::English);
    let local_verification = verifier
        .verify_checkpoint(primary_ckpt, &args.local_dir)
        .ok();

    if args.json {
        let ckpt_json: Vec<_> = checkpoints
            .iter()
            .map(|&c| {
                let installed = verifier
                    .verify_checkpoint(c, &args.local_dir)
                    .map(|r| r.is_valid)
                    .unwrap_or(false);
                let paradigm = lemma::model::head_paradigm_for(c);
                serde_json::json!({
                    "repo_id": c.repo_id(), "backbone": c.backbone(), "parameters": c.param_count(),
                    "context_window": c.context_window(), "head_budget": c.head_budget(),
                    "head_paradigm": paradigm.as_str(), "head_complexity": paradigm.complexity_notation(),
                    "local_dir": c.local_dir(), "installed": installed,
                    "domain": checkpoint_domain_summary(c),
                })
            })
            .collect();
        let out = serde_json::json!({
            "checkpoints": ckpt_json,
            "local_dir": args.local_dir,
            "local_valid": local_verification.as_ref().map(|r| r.is_valid).unwrap_or(false),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!(
        "\n🧠 System 1 Decision Model Registry\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    );

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Checkpoint").fg(Color::Cyan),
        Cell::new("Backbone").fg(Color::Blue),
        Cell::new("Params").fg(Color::Green),
        Cell::new("Context").fg(Color::Yellow),
        Cell::new("Local").fg(Color::Magenta),
        Cell::new("Key Strengths / Domain").fg(Color::White),
    ]);

    for ckpt in checkpoints {
        let installed = verifier
            .verify_checkpoint(ckpt, &args.local_dir)
            .map(|r| r.is_valid)
            .unwrap_or(false);
        table.add_row(vec![
            Cell::new(ckpt.repo_id()),
            Cell::new(ckpt.backbone()),
            Cell::new(ckpt.param_count()),
            Cell::new(format!("{} tok", ckpt.context_window())),
            Cell::new(if installed {
                "✅ Ready"
            } else {
                "☁️ Remote"
            }),
            Cell::new(checkpoint_domain_summary(ckpt)),
        ]);
    }

    println!(
        "{table}\n\nLocal Cache Inspection ({}):",
        local_verification
            .as_ref()
            .map(|r| r.path.display().to_string())
            .unwrap_or_else(|| args.local_dir.display().to_string())
    );
    if let Some(rep) = local_verification {
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
    } else {
        println!("   Status: Directory does not exist yet. Run `lemma download` to fetch weights.");
    }
    println!();
    Ok(())
}

fn handle_inspect(args: InspectArgs) -> Result<()> {
    let target_path = if let Some(ckpt_str) = &args.checkpoint {
        let ckpt = parse_checkpoint(ckpt_str)?;
        ckpt.local_dir().join("model.safetensors")
    } else {
        args.file.clone()
    };
    log_status(
        args.json,
        format!(
            "🔍 Inspecting Safetensors weights: {}",
            target_path.display()
        ),
    );
    let rep = inspect_safetensors(&target_path)?;

    if args.json {
        let json_val = serde_json::json!({
            "architecture": rep.architecture,
            "total_tensors": rep.total_tensors, "total_parameters": rep.total_parameters,
            "dtypes": rep.dtypes, "is_laya_compatible": rep.is_laya_compatible,
            "has_classifier_head": rep.has_classifier_head,
            "has_act_head": rep.has_act_head, "has_decision_scorer": rep.has_decision_scorer,
            "file_size_bytes": rep.file_size_bytes, "header_size_bytes": rep.header_size_bytes,
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
        Cell::new("Detected Architecture"),
        Cell::new(&rep.architecture),
        Cell::new("Inferred from Safetensors weight hierarchy"),
    ]);
    table.add_row(vec![
        Cell::new("Total Tensors"),
        Cell::new(format!("{}", rep.total_tensors)),
        Cell::new("Number of distinct weight arrays"),
    ]);
    table.add_row(vec![
        Cell::new("Total Parameters"),
        Cell::new(format!(
            "{:.2}M ({} elements)",
            rep.total_parameters as f64 / 1_000_000.0,
            rep.total_parameters
        )),
        Cell::new("Exact parameter count across all tensors"),
    ]);
    table.add_row(vec![
        Cell::new("File Size"),
        Cell::new(format!(
            "{:.2} MB",
            rep.file_size_bytes as f64 / 1_048_576.0
        )),
        Cell::new(format!("Header JSON size: {} bytes", rep.header_size_bytes)),
    ]);
    table.add_row(vec![
        Cell::new("Dtype Distribution"),
        Cell::new(format!("{:?}", rep.dtypes)),
        Cell::new("Precision format (F16 / BF16 / F32)"),
    ]);
    table.add_row(vec![
        Cell::new("Classification / MLM Head"),
        Cell::new(if rep.has_classifier_head {
            "VERIFIED (✅)"
        } else {
            "BACKBONE ONLY (⚠️)"
        }),
        Cell::new(format!(
            "Laya native: {}, Act head: {}, Mask Scorer: {}",
            rep.is_laya_compatible, rep.has_act_head, rep.has_decision_scorer
        )),
    ]);

    println!("\n{table}\n\nSample Tensor Shapes:");
    for (name, shape) in rep.tensor_shapes.iter().take(8) {
        println!("  • {:<45} {:?}", name, shape);
    }
    println!(
        "  ... and {} more tensors.\n",
        rep.total_tensors.saturating_sub(8)
    );
    Ok(())
}

fn handle_explain(args: ExplainArgs) -> Result<()> {
    let text = args
        .text
        .as_deref()
        .or(args.input_text.as_deref())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Input text is required for `lemma explain`. Pass `<TEXT>` positionally or via `--text <TEXT>`."
            )
        })?;
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_questions_or_inline(args.schema.as_deref(), "Classify department")?;

    let qid = args
        .qid
        .as_deref()
        .or_else(|| {
            let target_lower = args.target.as_ref()?.to_lowercase();
            questions
                .iter()
                .find(|(_, q)| {
                    q.render_options()
                        .iter()
                        .any(|o| o.to_lowercase().contains(&target_lower))
                })
                .map(|(k, _)| k.as_str())
        })
        .or_else(|| {
            questions
                .iter()
                .find(|(_, q)| q.qtype == lemma::model::QuestionType::Choice)
                .map(|(k, _)| k.as_str())
        })
        .or_else(|| questions.keys().next().map(|s| s.as_str()))
        .unwrap_or("department");

    let question = questions
        .get(qid)
        .ok_or_else(|| anyhow::anyhow!("Question '{}' not found in schema", qid))?;

    let opt_keys: Vec<String> = question
        .render_options()
        .iter()
        .map(|o| {
            o.split(':')
                .next()
                .unwrap_or(o)
                .trim()
                .trim_matches('"')
                .to_string()
        })
        .collect();

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let has_opt = |cand: &str| opt_keys.iter().any(|k| k.eq_ignore_ascii_case(cand));

    let (raw_target, raw_contrast, user_specified) = match (&args.target, &args.contrast) {
        (Some(t), Some(c)) => (t.clone(), c.clone(), true),
        (t_opt, c_opt) => {
            let ans = classifier.classify_single(text, question)?;
            let mut ranked: Vec<(String, f64)> = ans
                .probabilities
                .iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let top1 = t_opt
                .clone()
                .or_else(|| ranked.first().map(|(k, _)| k.clone()))
                .or_else(|| opt_keys.first().cloned())
                .unwrap_or_else(|| "billing".to_string());
            let top2 = c_opt
                .clone()
                .or_else(|| {
                    ranked
                        .iter()
                        .find(|(k, _)| !k.eq_ignore_ascii_case(&top1))
                        .map(|(k, _)| k.clone())
                })
                .or_else(|| {
                    opt_keys
                        .iter()
                        .find(|k| !k.eq_ignore_ascii_case(&top1))
                        .cloned()
                })
                .unwrap_or_else(|| "technical".to_string());
            (top1, top2, t_opt.is_some() || c_opt.is_some())
        }
    };

    let (target, contrast) =
        if (!has_opt(&raw_target) || !has_opt(&raw_contrast)) && opt_keys.len() >= 2 {
            let t = if has_opt(&raw_target) {
                raw_target.clone()
            } else {
                opt_keys[0].clone()
            };
            let c = if has_opt(&raw_contrast) && !raw_contrast.eq_ignore_ascii_case(&t) {
                raw_contrast.clone()
            } else {
                opt_keys
                    .iter()
                    .find(|k| !k.eq_ignore_ascii_case(&t))
                    .cloned()
                    .unwrap_or_else(|| opt_keys[1].clone())
            };
            if user_specified {
                eprintln!(
                    "ℹ️  Option(s) '{}'/'{}' not in question '{}'; using '{}' vs '{}'",
                    raw_target, raw_contrast, qid, t, c
                );
            }
            (t, c)
        } else {
            (raw_target, raw_contrast)
        };

    let report =
        AttributionEngine::new(&classifier).explain(text, qid, question, &target, &contrast)?;
    if args.json {
        let attrs: Vec<_> = report
            .attributions
            .iter()
            .map(|a| serde_json::json!({ "token": a.token, "delta_margin": a.delta_margin, "relative_influence_pct": a.relative_influence_pct }))
            .collect();
        let out = serde_json::json!({
            "input_text": report.input_text, "question_id": report.question_id,
            "target_choice": report.target_choice, "contrast_choice": report.contrast_choice,
            "baseline_margin": report.baseline_margin, "attributions": attrs,
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        AttributionEngine::print_report(&report);
    }
    Ok(())
}

fn handle_conformal(args: ConformalArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_questions_or_inline(args.schema.as_deref(), "Classify department")?;
    let items =
        resolve_data_source(&args.data_dir, args.file.as_deref(), false, None)?.read_items()?;

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
    let checkpoint = parse_checkpoint(&val_args.checkpoint)?;
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
                println!(
                    "  ✅ Question '{}' (type: {:?}, {} options): valid",
                    qid,
                    q.qtype,
                    q.render_options().len()
                );
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
        println!(
            "✨ Schema validation passed successfully ({} warnings).\n",
            total_warnings
        );
        Ok(())
    } else {
        bail!("Schema validation failed! Resolve errors above.");
    }
}

fn handle_schema_infer(args: SchemaInferArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
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

    println!(
        "🔍 Analyzing dataset for schema induction: {}",
        args.data.display()
    );
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
            lemma::model::QuestionType::Choice => match &q.criteria {
                Some(serde_json::Value::Object(map)) => {
                    let keys: Vec<&str> = map.keys().map(|s| s.as_str()).collect();
                    format!("{} labels: {}", keys.len(), keys.join(", "))
                }
                _ => "-".into(),
            },
            lemma::model::QuestionType::Score => match &q.criteria {
                Some(serde_json::Value::Array(arr)) => format!("{} rubric levels", arr.len()),
                _ => "-".into(),
            },
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
    let reduction = if result.original_tokens > 0 {
        (1.0 - (result.optimized_tokens as f64 / result.original_tokens as f64)) * 100.0
    } else {
        0.0
    };

    println!(
        "\n✂️  Schema Criteria Knapsack Token Optimizer\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nInput Schema    : {}\nTarget Budget   : {} tokens (ModernBERT / mmBERT head limit)\nOriginal Tokens : {} tokens\nOptimized Tokens: {} tokens\nPruned Clauses  : {}\nToken Reduction : -{:.1}%",
        args.schema.display(),
        args.budget,
        result.original_tokens,
        result.optimized_tokens,
        result.pruned_clauses,
        reduction
    );

    if args.diff {
        println!("\nPer-Question Token Allocation:");
        for (qid, (before, after)) in &result.per_question_tokens {
            println!("  - {}: {} -> {} tokens", qid, before, after);
        }
    }

    if let Some(out_path) = &args.output {
        write_json_file(out_path, &result.optimized_schema)?;
        println!(
            "\n✅ Saved optimized schema JSON to: {}",
            out_path.display()
        );
    }
    Ok(())
}

fn handle_drift(args: DriftArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_questions_or_inline(args.schema.as_deref(), "Classify topic")?;
    let base_items = lemma::datasource::FileSource::new(&args.baseline).read_items()?;
    let curr_items =
        resolve_data_source(&args.data_dir, args.current.as_deref(), false, None)?.read_items()?;

    if base_items.is_empty() {
        bail!("Baseline dataset has no items");
    }
    if curr_items.is_empty() {
        bail!("Current dataset has no items");
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let drift_engine = DriftEngine::new(&classifier)
        .with_thresholds(args.psi_threshold, args.wasserstein_threshold);

    log_status(
        args.json,
        format!(
            "🔬 Monitoring distribution drift (Baseline: {} items, Current: {} items)...",
            base_items.len(),
            curr_items.len()
        ),
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
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let policy = GuardrailPolicy::from_file(&args.policy)?;
    let questions = load_questions_or_inline(args.schema.as_deref(), "Classify topic")?;
    let items = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
        args.extensions.as_deref(),
        args.no_recursive,
    )?
    .read_items()?;

    if items.is_empty() {
        eprintln!("⚠️  No data items found for guardrail evaluation.");
        return Ok(());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let guard_engine = GuardrailEngine::new(policy);

    let decisions = classifier.classify_batch(&items, &questions)?;
    let audit_records = guard_engine.evaluate_batch(&items, &decisions);
    let quiet_stdout = args.json || is_machine_format(args.format);

    if let Some(audit_path) = &args.audit_output {
        ensure_parent_dir(audit_path)?;
        let mut file = std::fs::File::create(audit_path)?;
        for rec in &audit_records {
            writeln!(file, "{}", serde_json::to_string(rec)?)?;
        }
        log_status(
            quiet_stdout,
            format!(
                "📜 Wrote {} cryptographic audit records to: {}",
                audit_records.len(),
                audit_path.display()
            ),
        );
    }

    if let (Some(key), Some(receipt_path)) = (&args.attest_key, &args.receipt_log) {
        ensure_parent_dir(receipt_path)?;
        let mut attest_engine = AttestationEngine::new(key);
        let mut file = std::fs::File::create(receipt_path)?;
        let schema_digest = lemma::engine::schema_digest(&questions)?;

        for (item, (_, dec)) in items.iter().zip(decisions.iter()) {
            let matched_rec = audit_records.iter().find(|r| r.item_id == item.id);
            let rule = matched_rec.and_then(|r| r.matched_rule.as_deref());
            let action = matched_rec
                .map(|r| r.final_action)
                .unwrap_or(GuardAction::Allow);
            let receipt = attest_engine.create_receipt(item, dec, rule, action, &schema_digest)?;
            writeln!(file, "{}", serde_json::to_string(&receipt)?)?;
        }
        log_status(
            quiet_stdout,
            format!(
                "🔏 Emitted {} cryptographically attested receipts to: {}",
                items.len(),
                receipt_path.display()
            ),
        );
    }

    if args.json || matches!(args.format, OutputFormat::Json | OutputFormat::Pretty) {
        println!("{}", serde_json::to_string_pretty(&audit_records)?);
    } else {
        match args.format {
            OutputFormat::Jsonl => {
                for rec in &audit_records {
                    println!("{}", serde_json::to_string(rec)?);
                }
            }
            OutputFormat::Csv => {
                let mut wtr = csv::Writer::from_writer(std::io::stdout());
                wtr.write_record([
                    "audit_id",
                    "item_id",
                    "final_action",
                    "matched_rule",
                    "latency_ms",
                ])?;
                for r in &audit_records {
                    wtr.write_record([
                        r.audit_id.as_str(),
                        r.item_id.as_str(),
                        r.final_action.badge(),
                        r.matched_rule.as_deref().unwrap_or(""),
                        &format!("{:.2}", r.latency_ms),
                    ])?;
                }
                wtr.flush()?;
            }
            OutputFormat::Compact => {
                for r in &audit_records {
                    println!(
                        "{} => {} (rule: {}) [{:.1}ms]",
                        r.item_id,
                        r.final_action.badge(),
                        r.matched_rule.as_deref().unwrap_or("default"),
                        r.latency_ms
                    );
                }
            }
            _ => GuardrailEngine::print_report(&audit_records),
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

    let checkpoints: Vec<Checkpoint> = ckpt_names
        .into_iter()
        .map(parse_checkpoint)
        .collect::<Result<_>>()?;
    let questions = load_questions_or_inline(args.schema.as_deref(), "Classify topic")?;
    let items = resolve_data_source(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
    )?
    .read_items()?;

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

    let quiet_stdout = args.json || is_machine_format(args.format);
    log_status(
        quiet_stdout,
        format!(
            "🏛️  Evaluating {} items across {} checkpoints with consensus engine...",
            items.len(),
            args.checkpoints
        ),
    );

    let results = engine.evaluate_batch(&items, &questions)?;

    if let Some(out_path) = &args.output {
        write_json_file(out_path, &results)?;
        log_status(
            quiet_stdout,
            format!("🏛️  Saved consensus results to: {}", out_path.display()),
        );
    }

    if args.json || matches!(args.format, OutputFormat::Json | OutputFormat::Pretty) {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        match args.format {
            OutputFormat::Jsonl => {
                for r in &results {
                    println!("{}", serde_json::to_string(r)?);
                }
            }
            OutputFormat::Csv => {
                let mut wtr = csv::Writer::from_writer(std::io::stdout());
                wtr.write_record([
                    "item_id",
                    "question_id",
                    "prediction",
                    "confidence",
                    "epistemic_jsd",
                    "status",
                ])?;
                for r in &results {
                    for (qid, qc) in &r.questions {
                        wtr.write_record([
                            r.item_id.as_str(),
                            qid.as_str(),
                            qc.consensus_prediction.as_str(),
                            &format!("{:.4}", qc.consensus_confidence),
                            &format!("{:.4}", qc.epistemic_uncertainty),
                            qc.status.badge(),
                        ])?;
                    }
                }
                wtr.flush()?;
            }
            OutputFormat::Compact => {
                for r in &results {
                    println!(
                        "{} => {} [{:.1}ms]",
                        r.item_id,
                        r.overall_status.badge(),
                        r.total_latency_ms
                    );
                }
            }
            _ => ConsensusEngine::print_report(&results),
        }
    }
    Ok(())
}

fn handle_cache(args: CacheArgs) -> Result<()> {
    match args.action {
        CacheCommands::Stats => {
            println!(
                "\n📊 Decision Cache Defaults:\n   Scope              : In memory, for one lemma run (nothing is persisted)\n   Exact tier         : Always on with --cache\n   Semantic tier      : Off unless --cache-semantic (cosine over hashed 128-d word features, threshold 0.92)\n   Capacity Limit     : 10,000 entries\n   Default TTL        : 86,400s (24 hours)\n   Admission Filter   : confidence >= 0.70, act_probability <= 0.50"
            );
        }
        CacheCommands::Bench(bench_args) => {
            let schema =
                load_questions_or_inline(bench_args.schema.as_deref(), "Department triage")?;
            let items = lemma::datasource::FileSource::new(&bench_args.file).read_items()?;
            if items.is_empty() {
                bail!("File contains no items: {}", bench_args.file.display());
            }

            let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
            let mut cache = DecisionCache::new(SemanticCacheConfig {
                semantic: true,
                similarity_threshold: bench_args.threshold,
                ..Default::default()
            });

            println!(
                "\n⚡ Benchmarking Semantic Decision Cache on {} items...",
                items.len()
            );
            let schema_hash = compute_schema_hash(&schema);

            let start_cold = std::time::Instant::now();
            for item in &items {
                let text = item_text(item)?;
                let res = classifier.classify(&item.state, &schema)?;
                cache.insert(&text, &schema_hash, &res);
            }
            let cold_time = start_cold.elapsed();

            let start_warm = std::time::Instant::now();
            let mut hits = 0;
            for item in &items {
                let text = item_text(item)?;
                if matches!(
                    cache.lookup(&text, &schema_hash),
                    lemma::engine::CacheLookupResult::ExactHit(_)
                        | lemma::engine::CacheLookupResult::SemanticHit { .. }
                ) {
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
                Cell::new(format!(
                    "{:.1}%",
                    (hits as f64 / items.len() as f64) * 100.0
                )),
                Cell::new(format!("{:.1}x Faster", speedup)),
            ]);
            println!("{table}");
        }
        CacheCommands::Clear => println!(
            "🧹 Nothing to clear: the decision cache lives in memory and ends with each lemma run."
        ),
    }
    Ok(())
}

fn handle_ood(args: OodArgs) -> Result<()> {
    match args.action {
        OodCommands::Fit(fit_args) => {
            let checkpoint = parse_checkpoint(&fit_args.checkpoint)?;
            let questions = load_questions_from_file(&fit_args.schema)?;
            let items = lemma::datasource::FileSource::new(&fit_args.train_data).read_items()?;

            log_status(
                fit_args.json,
                format!(
                    "🔬 Fitting OOD profile on {} in-distribution samples...",
                    items.len()
                ),
            );
            let classifier = LayaClassifier::with_checkpoint(checkpoint);

            let target_qid = if questions.contains_key(&fit_args.target_qid) {
                &fit_args.target_qid
            } else {
                questions.keys().next().unwrap()
            };

            let profile = OodDetector::fit(
                "laya_profile",
                &items,
                &classifier,
                &questions,
                target_qid,
                fit_args.fpr_rate,
            )?;

            write_json_file(&fit_args.output, &profile)?;

            if fit_args.json {
                eprintln!("💾 Saved profile to: {}", fit_args.output.display());
                println!("{}", serde_json::to_string_pretty(&profile)?);
            } else {
                println!(
                    "\n✅ Fitted OOD Profile:\n   Question ID       : {}\n   Target Classes    : {}\n   Mean Free Energy  : {:.4}\n   Std Free Energy   : {:.4}\n   Energy Threshold  : {:.4}\n   Distance Threshold: {:.4}\n💾 Saved profile to: {}",
                    profile.question_id,
                    profile.target_classes.join(", "),
                    profile.mean_energy,
                    profile.std_energy,
                    profile.energy_threshold,
                    profile.distance_threshold,
                    fit_args.output.display()
                );
            }
        }
        OodCommands::Eval(eval_args) => {
            let checkpoint = parse_checkpoint(&eval_args.checkpoint)?;
            let questions = load_questions_from_file(&eval_args.schema)?;
            let profile: OodProfile =
                serde_json::from_reader(std::fs::File::open(&eval_args.profile)?)?;
            let detector = OodDetector::new(profile)?;
            let classifier = LayaClassifier::with_checkpoint(checkpoint);

            let id_items = lemma::datasource::FileSource::new(&eval_args.id_data).read_items()?;
            let ood_items = lemma::datasource::FileSource::new(&eval_args.ood_data).read_items()?;

            log_status(
                eval_args.json,
                format!(
                    "🎯 Evaluating OOD Profile against ID ({} items) and OOD ({} items)...",
                    id_items.len(),
                    ood_items.len()
                ),
            );

            let id_evals = detector.evaluate_batch(&id_items, &classifier, &questions)?;
            let ood_evals = detector.evaluate_batch(&ood_items, &classifier, &questions)?;

            let id_avg_energy =
                id_evals.iter().map(|e| e.free_energy).sum::<f64>() / id_evals.len().max(1) as f64;
            let ood_avg_energy = ood_evals.iter().map(|e| e.free_energy).sum::<f64>()
                / ood_evals.len().max(1) as f64;
            let id_avg_dist = id_evals.iter().map(|e| e.mahalanobis_distance).sum::<f64>()
                / id_evals.len().max(1) as f64;
            let ood_avg_dist = ood_evals
                .iter()
                .map(|e| e.mahalanobis_distance)
                .sum::<f64>()
                / ood_evals.len().max(1) as f64;

            let id_flagged = id_evals.iter().filter(|e| e.is_ood).count();
            let ood_flagged = ood_evals.iter().filter(|e| e.is_ood).count();
            let fpr = id_flagged as f64 / id_items.len().max(1) as f64;
            let tpr = ood_flagged as f64 / ood_items.len().max(1) as f64;

            if eval_args.json {
                let out = serde_json::json!({
                    "id_samples": id_items.len(), "ood_samples": ood_items.len(),
                    "id_mean_energy": id_avg_energy, "ood_mean_energy": ood_avg_energy,
                    "id_mean_distance": id_avg_dist, "ood_mean_distance": ood_avg_dist,
                    "id_flagged": id_flagged, "ood_flagged": ood_flagged,
                    "fpr": fpr, "tpr_recall": tpr,
                });
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }

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
                Cell::new(format!("{:.1}% (FPR)", fpr * 100.0)),
            ]);
            table.add_row(vec![
                Cell::new("Out-of-Distribution (OOD)"),
                Cell::new(ood_items.len().to_string()),
                Cell::new(format!("{:.4}", ood_avg_energy)),
                Cell::new(format!("{:.4}", ood_avg_dist)),
                Cell::new(ood_flagged.to_string()),
                Cell::new(format!("{:.1}% (TPR/Recall)", tpr * 100.0)),
            ]);
            println!("{table}");
        }
    }
    Ok(())
}

fn handle_cascade_tune(tune_args: lemma::cli::CascadeTuneArgs, force_sweep: bool) -> Result<()> {
    let checkpoint = parse_checkpoint(&tune_args.checkpoint)?;
    let questions = load_questions_from_file(&tune_args.schema)?;
    let items = lemma::datasource::FileSource::new(&tune_args.val_data).read_items()?;
    let classifier = LayaClassifier::with_checkpoint(checkpoint);

    if force_sweep || tune_args.sweep {
        log_status(
            tune_args.json,
            format!(
                "📈 Sweeping System 1 / System 2 Pareto frontier on {} samples (SLA >= {:.1}%)...",
                items.len(),
                tune_args.target_accuracy * 100.0
            ),
        );
        let points = CascadeRouter::sweep_pareto(
            &items,
            &questions,
            &classifier,
            tune_args.target_accuracy,
            tune_args.system2_cost,
        )?;
        if tune_args.json {
            println!("{}", serde_json::to_string_pretty(&points)?);
        } else {
            let mut table = Table::new();
            table.set_header(vec![
                Cell::new("Risk τ").fg(Color::Cyan),
                Cell::new("S1 Coverage").fg(Color::Green),
                Cell::new("S2 Escalate").fg(Color::Yellow),
                Cell::new("Expected Acc").fg(Color::White),
                Cell::new("Cost / 1k").fg(Color::Blue),
                Cell::new("LLM Savings").fg(Color::Magenta),
                Cell::new("SLA Status").fg(Color::Green),
            ]);
            for p in &points {
                let status = if p.is_optimal {
                    "⭐ Optimal SLA"
                } else if p.meets_sla {
                    "✅ Meets SLA"
                } else {
                    "Below SLA"
                };
                table.add_row(vec![
                    Cell::new(format!("{:.2}", p.threshold)),
                    Cell::new(format!("{:.1}%", p.system1_coverage_rate * 100.0)),
                    Cell::new(format!("{:.1}%", p.system2_escalation_rate * 100.0)),
                    Cell::new(format!("{:.2}%", p.expected_accuracy * 100.0)),
                    Cell::new(format!("${:.2}", p.estimated_cost_per_1000)),
                    Cell::new(format!("-{:.1}%", p.cost_savings_percent)),
                    Cell::new(status),
                ]);
            }
            println!("\n⚖️  System 1 / System 2 Pareto Frontier Sweep:\n{table}\n");
        }
        return Ok(());
    }

    log_status(
        tune_args.json,
        format!(
            "📈 Tuning Pareto-optimal System 1 / System 2 cascade on {} samples...",
            items.len()
        ),
    );
    let policy = CascadeRouter::tune_pareto(
        &items,
        &questions,
        &classifier,
        tune_args.target_accuracy,
        tune_args.system2_cost,
    )?;

    if let Some(export_path) = &tune_args.export_policy {
        write_json_file(export_path, &policy)?;
        log_status(
            tune_args.json,
            format!("💾 Exported cascade policy to: {}", export_path.display()),
        );
    }

    if tune_args.json {
        println!("{}", serde_json::to_string_pretty(&policy)?);
    } else {
        println!(
            "\n⚖️  Pareto Frontier Optimization Results:\n   Target Accuracy SLA   : {:.1}%\n   Expected Hybrid Acc   : {:.2}%\n   Optimal Threshold τ*  : {:.2}\n   System 1 Fast Coverage: {:.1}%\n   System 2 Escalation   : {:.1}%\n   Cost Savings vs LLM   : -{:.1}%\n   Estimated Cost / 1k   : ${:.2}",
            policy.target_accuracy * 100.0,
            policy.expected_accuracy * 100.0,
            policy.optimal_threshold,
            policy.system1_coverage_rate * 100.0,
            policy.system2_escalation_rate * 100.0,
            policy.cost_savings_percent,
            policy.estimated_cost_per_1000
        );
    }
    Ok(())
}

fn handle_cascade(args: CascadeArgs) -> Result<()> {
    match args.action {
        CascadeCommands::Tune(tune_args) => handle_cascade_tune(tune_args, false)?,
        CascadeCommands::Sweep(tune_args) => handle_cascade_tune(tune_args, true)?,
        CascadeCommands::Run(run_args) => {
            let checkpoint = parse_checkpoint(&run_args.checkpoint)?;
            let questions =
                load_questions_or_inline(run_args.schema.as_deref(), "Department triage")?;
            let items = resolve_data_source(
                std::path::Path::new("data"),
                run_args.file.as_deref(),
                false,
                run_args.text.as_deref(),
            )?
            .read_items()?;
            if items.is_empty() {
                bail!("No items found to route");
            }

            let classifier = LayaClassifier::with_checkpoint(checkpoint);
            let router = CascadeRouter::new(CascadeConfig {
                risk_threshold: run_args.risk_threshold,
                system2_cost_per_call: run_args.system2_cost,
                ..Default::default()
            });

            let decisions: Vec<_> = items
                .iter()
                .map(|item| router.route(item, &questions, &classifier))
                .collect::<Result<_>>()?;

            if run_args.json || matches!(run_args.format, OutputFormat::Json | OutputFormat::Pretty)
            {
                println!("{}", serde_json::to_string_pretty(&decisions)?);
            } else {
                match run_args.format {
                    OutputFormat::Jsonl => {
                        for dec in &decisions {
                            println!("{}", serde_json::to_string(dec)?);
                        }
                    }
                    OutputFormat::Csv => {
                        let mut wtr = csv::Writer::from_writer(std::io::stdout());
                        wtr.write_record([
                            "item_id",
                            "tier",
                            "composite_risk",
                            "entropy",
                            "margin_gap",
                            "cost_usd",
                        ])?;
                        for dec in &decisions {
                            wtr.write_record([
                                dec.item_id.as_str(),
                                dec.tier.badge(),
                                &format!("{:.4}", dec.composite_risk),
                                &format!("{:.4}", dec.normalized_entropy),
                                &format!("{:.4}", 1.0 - dec.top2_margin),
                                &format!("{:.4}", dec.cost_usd),
                            ])?;
                        }
                        wtr.flush()?;
                    }
                    OutputFormat::Compact => {
                        for dec in &decisions {
                            println!(
                                "{} => {} (risk={:.3}, cost=${:.4})",
                                dec.item_id,
                                dec.tier.badge(),
                                dec.composite_risk,
                                dec.cost_usd
                            );
                        }
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
    }
    Ok(())
}

fn handle_attest(args: AttestArgs) -> Result<()> {
    match args.action {
        AttestCommands::Verify(verify_args) => {
            let reader = std::io::BufReader::new(std::fs::File::open(&verify_args.receipt_log)?);
            let mut receipts: Vec<lemma::engine::DecisionReceipt> = Vec::new();
            for line in std::io::BufRead::lines(reader) {
                let l = line?;
                if !l.trim().is_empty() {
                    receipts.push(serde_json::from_str(&l)?);
                }
            }

            log_status(
                verify_args.json,
                format!(
                    "🔍 Verifying cryptographic decision receipts in: {}",
                    verify_args.receipt_log.display()
                ),
            );
            let report = AttestationEngine::verify_ledger(&receipts, &verify_args.secret_key);

            if verify_args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "\n🛡️  Attestation Verification Report:\n   Total Receipts Evaluated : {}\n   Valid Cryptographic Sigs : {}\n   Invalid / Tampered Sigs  : {}\n   Hash Chain Intact        : {}",
                    report.total_receipts,
                    report.valid_receipts,
                    report.invalid_receipts,
                    if report.chain_intact {
                        "✅ YES"
                    } else {
                        "❌ NO (Chain Broken!)"
                    }
                );
                if let Some(idx) = report.broken_link_index {
                    println!("   ⚠️  First broken chain link at receipt #{}", idx);
                }
                for reason in &report.failure_reasons {
                    println!("   ❌ {}", reason);
                }
            }

            if !report.chain_intact || report.invalid_receipts > 0 {
                bail!("Attestation verification failed: Audit trail is tampered or invalid!");
            }
            if !verify_args.json {
                println!(
                    "\n✅ All receipt digests, HMAC-SHA256 signatures, and hash chain links verified."
                );
            }
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

    let mut cfg = ClassifierConfig::with_checkpoint(checkpoint);
    cfg.escalate_threshold = escalate_threshold;
    let classifier = LayaClassifier::new(cfg);

    let mut router = HierarchicalRouter::new(schema, classifier)
        .with_coarse_threshold(coarse_threshold)
        .with_escalate_threshold(escalate_threshold);

    if cache_enabled {
        let cache = std::sync::Arc::new(std::sync::Mutex::new(DecisionCache::new(
            SemanticCacheConfig::default(),
        )));
        router = router.with_cache(cache);
    }
    if let Some(ood_path) = ood_profile {
        let profile: OodProfile = serde_json::from_reader(std::fs::File::open(ood_path)?)?;
        router = router.with_ood_detector(std::sync::Arc::new(OodDetector::new(profile)?));
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
        println!(
            "💾 Successfully wrote {} hierarchical decision(s) to {}",
            results.len(),
            out_path.display()
        );
    } else {
        println!("{}", rendered);
    }
    Ok(())
}

fn handle_hierarchical(args: lemma::cli::HierarchicalArgs) -> Result<()> {
    match args.command {
        HierarchicalCommands::Run(run_args) => {
            let checkpoint = parse_checkpoint(&run_args.checkpoint)?;
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
            let checkpoint = parse_checkpoint(&tree_args.checkpoint)?;
            let schema = HierarchicalSchema::load_from_file(&tree_args.schema)?;
            let budget = checkpoint.head_budget();
            println!(
                "\n🌲 Hierarchical Decision Tree Visualization ({})\n{}\n\n{}",
                checkpoint.repo_id(),
                "=".repeat(65),
                schema.render_ascii_tree(budget)
            );
            Ok(())
        }
        HierarchicalCommands::Validate(val_args) => {
            let checkpoint = parse_checkpoint(&val_args.checkpoint)?;
            let schema = HierarchicalSchema::load_from_file(&val_args.schema)?;
            let budget = checkpoint.head_budget();
            println!(
                "\n🔍 Validating Hierarchical Schema against {} (Head Budget: {} tokens)\n{}\n",
                checkpoint.repo_id(),
                budget,
                "=".repeat(75)
            );
            match schema.validate(budget) {
                Ok(report) => {
                    println!(
                        "✅ Hierarchical schema is structurally VALID!\n   • Root coarse tokens:      ~{} tokens\n   • Max branch tokens:       ~{} tokens\n   • Coarse branches count:   {}\n   • Total leaf classes:      {}\n   • Tree depth:              {} levels",
                        report.root_tokens,
                        report.max_branch_tokens,
                        report.branch_count,
                        report.total_leaf_classes,
                        schema.max_depth()
                    );
                    if report.warnings.is_empty() {
                        println!(
                            "\n✨ Zero warnings: all coarse and fine nodes comfortably fit inside token budget!"
                        );
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
            let checkpoint = parse_checkpoint(&bench_args.checkpoint)?;
            let schema = HierarchicalSchema::load_from_file(&bench_args.schema)?;
            let items = resolve_data_source(
                &bench_args.data_dir,
                bench_args.file.as_deref(),
                false,
                None,
            )?
            .read_items()?;
            if items.is_empty() {
                bail!("No data items found for benchmarking");
            }

            let router = HierarchicalRouter::new(
                schema,
                LayaClassifier::new(ClassifierConfig::with_checkpoint(checkpoint)),
            )
            .with_coarse_threshold(bench_args.coarse_threshold);

            if !bench_args.json {
                println!(
                    "\n⚡ Benchmarking Hierarchical Coarse-to-Fine Router...\n   Checkpoint: {} | Samples: {} | Iterations: {}\n{}\n",
                    checkpoint.repo_id(),
                    items.len(),
                    bench_args.iterations,
                    "=".repeat(70)
                );
            } else {
                eprintln!(
                    "⚡ Benchmarking Hierarchical Coarse-to-Fine Router ({}, {} samples)...",
                    checkpoint.repo_id(),
                    items.len()
                );
            }

            let bench_res = router.benchmark(&items, bench_args.iterations)?;
            if bench_args.json {
                println!("{}", serde_json::to_string_pretty(&bench_res)?);
            } else {
                println!(
                    "📊 Benchmark Results:\n   • Total Evaluated:       {} decisions\n   • Avg Coarse Latency:    {:.2} ms\n   • Avg Fine Latency:      {:.2} ms\n   • Avg Total Latency:     {:.2} ms\n   • Coarse Escalation Rate: {:.1}%\n   • Completion Rate:       {:.1}%\n   • Avg Compound Conf:     {:.3}\n   • Throughput:            {:.1} requests/sec",
                    bench_res.total_samples,
                    bench_res.avg_coarse_latency_ms,
                    bench_res.avg_fine_latency_ms,
                    bench_res.avg_total_latency_ms,
                    bench_res.coarse_escalation_rate * 100.0,
                    bench_res.completion_rate * 100.0,
                    bench_res.avg_compound_confidence,
                    bench_res.throughput_qps
                );
            }
            Ok(())
        }
    }
}

fn handle_calibrate(args: CalibrateArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let loss = args
        .loss
        .parse::<CalibrationLoss>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let questions = load_schema_or_default(args.schema.as_deref())?;
    let items =
        resolve_data_source(&args.data_dir, args.val_data.as_deref(), false, None)?.read_items()?;

    if items.is_empty() {
        bail!("No validation items found to calibrate. Check --val-data path or data directory.");
    }

    let dataset_name = args
        .val_data
        .as_ref()
        .and_then(|p| p.file_stem().and_then(|s| s.to_str()))
        .unwrap_or("validation_dataset");

    let mut tuner = CalibrationTuner::new(checkpoint)
        .with_loss(loss)
        .with_bins(args.bins)
        .with_min_bucket_samples(args.min_bucket_samples)
        .with_qats(args.adaptive, args.adaptive_gamma);

    if let Some(cw_spec) = &args.cost_weights {
        tuner = tuner.with_cost_weights(parse_cost_weights(cw_spec)?);
    }

    let (report, mut calibrated_config) = tuner.fit(&items, &questions, dataset_name)?;

    if args.fit_label_bias {
        // `route` picks a different model per input, so one bias vector cannot fit them all. On
        // multilingual sentiment it lowered route's accuracy from 57.8% to 54.0%.
        if checkpoint == Checkpoint::AutoRoute {
            bail!(
                "--fit-label-bias needs a fixed checkpoint. `route` picks a different model per input, and one label bias does not fit them all."
            );
        }
        // Fit on top of the calibrated temperatures, which classify applies first.
        let mut cfg = ClassifierConfig::with_checkpoint(checkpoint);
        lemma::engine::apply_persisted_calibration(&mut cfg, &calibrated_config);
        let fit = lemma::engine::fit_label_bias(&LayaClassifier::new(cfg), &items, &questions)?;
        for (qid, (n, before, after)) in &fit.questions {
            eprintln!(
                "🎯 Label bias for '{qid}': accuracy on these {n} samples {:.1}% -> {:.1}%",
                before * 100.0,
                after * 100.0
            );
        }
        for (label, w) in fit.weights {
            *calibrated_config.cost_weights.entry(label).or_insert(1.0) *= w;
        }
    }

    if args.persist_to_model {
        let saved = lemma::engine::save_checkpoint_calibration(checkpoint, &calibrated_config)?;
        eprintln!(
            "💾 Auto-persisted checkpoint calibration profile to: {}",
            saved.display()
        );
    }

    if let Some(export_path) = &args.export_config {
        write_json_file(export_path, &calibrated_config)?;
        eprintln!(
            "💾 Calibrated agent config exported to: {}",
            export_path.display()
        );
    }

    if args.json {
        println!("{}", CalibrationTuner::format_json(&report)?);
        return Ok(());
    }

    match args.format {
        OutputFormat::Json => println!("{}", CalibrationTuner::format_json(&report)?),
        OutputFormat::Jsonl => println!("{}", serde_json::to_string(&report)?),
        OutputFormat::Compact => println!(
            "ECE: {:.4} -> {:.4} (-{:.1}%) | Buckets: {} | Samples: {}",
            report.average_raw_ece,
            report.average_calibrated_ece,
            report.overall_improvement_pct,
            report.buckets.len(),
            report.total_samples
        ),
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(std::io::stdout());
            wtr.write_record([
                "bucket_key",
                "sample_count",
                "optimal_temperature",
                "raw_ece",
                "calibrated_ece",
            ])?;
            for b in &report.buckets {
                wtr.write_record([
                    b.bucket.as_str(),
                    &b.sample_count.to_string(),
                    &format!("{:.4}", b.optimal_temperature),
                    &format!("{:.4}", b.raw_ece),
                    &format!("{:.4}", b.calibrated_ece),
                ])?;
            }
            wtr.flush()?;
        }
        OutputFormat::Table | OutputFormat::Pretty => {
            println!(
                "{}",
                CalibrationTuner::format_table(&report, args.export_config.as_deref())
            );
        }
    }
    Ok(())
}

fn handle_stream(args: StreamArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_schema_or_default(args.schema.as_deref())?;

    let mut config = ClassifierConfig::with_checkpoint(checkpoint);
    config.escalate_threshold = args.escalate_threshold;
    if let Some(cfg_path) = &args.config {
        apply_calibrated_config(&mut config, cfg_path, true)?;
    }

    let classifier = LayaClassifier::new(config);
    let primary_sink = StreamSink::from_target(&args.sink)?;
    let pipeline_cfg = StreamPipelineConfig {
        batch_size: args.batch_size,
        max_flush_interval_ms: args.max_flush_interval_ms,
        escalate_threshold: args.escalate_threshold,
        enable_cache: args.cache,
        enable_attest: args.attest,
        attest_secret: match (args.attest, args.attest_secret) {
            (true, None) => {
                bail!("--attest needs a signing key. Set LEMMA_ATTEST_KEY or pass --attest-secret.")
            }
            (_, secret) => secret.unwrap_or_default(),
        },
        receipt_log: args.receipt_log,
        limit: args.limit,
    };

    let mut pipeline =
        StreamPipeline::new(classifier, questions, primary_sink).with_config(pipeline_cfg);
    if let Some(esc_target) = &args.escalate_sink {
        pipeline = pipeline.with_escalate_sink(StreamSink::from_target(esc_target)?);
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
        OutputFormat::Json => println!("{}", StreamPipeline::format_json(&metrics)?),
        OutputFormat::Jsonl => println!("{}", serde_json::to_string(&metrics)?),
        OutputFormat::Table | OutputFormat::Pretty => {
            println!("{}", StreamPipeline::format_table(&metrics))
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(std::io::stdout());
            wtr.write_record([
                "total_classified",
                "throughput_qps",
                "total_escalated",
                "escalation_rate",
                "total_cached",
                "cache_hit_rate",
                "total_attested",
            ])?;
            wtr.write_record([
                &metrics.total_classified.to_string(),
                &format!("{:.2}", metrics.throughput_qps),
                &metrics.total_escalated.to_string(),
                &format!("{:.4}", metrics.escalation_rate),
                &metrics.total_cached.to_string(),
                &format!("{:.4}", metrics.cache_hit_rate),
                &metrics.total_attested.to_string(),
            ])?;
            wtr.flush()?;
        }
        OutputFormat::Compact => println!(
            "✅ Processed {} events ({:.1} QPS) | Escalated: {} ({:.1}%) | Cached: {} ({:.1}%) | Attested: {}",
            metrics.total_classified,
            metrics.throughput_qps,
            metrics.total_escalated,
            metrics.escalation_rate * 100.0,
            metrics.total_cached,
            metrics.cache_hit_rate * 100.0,
            metrics.total_attested
        ),
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
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_schema_or_default(args.schema.as_deref())?;
    let items = lemma::datasource::FileSource::parse_file(&args.data)?;
    if items.is_empty() {
        bail!("No items found in dataset: {}", args.data.display());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let ood_detector = match &args.ood_profile {
        Some(ood_path) => Some(OodDetector::new(serde_json::from_reader(
            std::fs::File::open(ood_path)?,
        )?)?),
        None => None,
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
        OutputFormat::Json => println!(
            "{}",
            ActiveLearningEngine::format_sample_json(&report, &queue_items)?
        ),
        OutputFormat::Jsonl => {
            for item in &queue_items {
                println!("{}", serde_json::to_string(item)?);
            }
        }
        _ => println!(
            "{}",
            ActiveLearningEngine::format_sample_table(
                &report,
                &queue_items,
                args.export_queue.as_deref()
            )
        ),
    }
    Ok(())
}

fn handle_active_distill(args: ActiveDistillArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_schema_or_default(args.schema.as_deref())?;
    let items = lemma::datasource::FileSource::parse_file(&args.data)?;
    if items.is_empty() {
        bail!("No items found in dataset: {}", args.data.display());
    }

    let classifier = LayaClassifier::with_checkpoint(checkpoint);
    let (samples, mut report) = ActiveLearningEngine::new().distill(
        &classifier,
        &items,
        &questions,
        args.val_split,
        &args.source_label,
    )?;

    ActiveLearningEngine::export_rlcd(&samples, &args.output)?;
    report.export_path = args.output.display().to_string();

    match args.format {
        OutputFormat::Json => println!(
            "{}",
            ActiveLearningEngine::format_distill_json(&report, &samples)?
        ),
        OutputFormat::Jsonl => {
            for s in &samples {
                println!("{}", serde_json::to_string(s)?);
            }
        }
        _ => println!("{}", ActiveLearningEngine::format_distill_table(&report)),
    }
    Ok(())
}

fn handle_active_stats(args: ActiveStatsArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
    let questions = load_schema_or_default(args.schema.as_deref())?;
    let items = lemma::datasource::FileSource::parse_file(&args.data)?;
    let classifier = LayaClassifier::with_checkpoint(checkpoint);

    let total_len = items.len() * questions.len();
    let (queue, report) = ActiveLearningEngine::new().sample(
        &classifier,
        &items,
        &questions,
        UncertaintyStrategy::Composite,
        total_len,
        None,
    )?;

    match args.format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        OutputFormat::Jsonl => println!("{}", serde_json::to_string(&report)?),
        _ => println!(
            "{}",
            ActiveLearningEngine::format_sample_table(&report, &queue, None)
        ),
    }
    Ok(())
}

fn handle_stats(args: StatsArgs) -> Result<()> {
    let checkpoint = parse_checkpoint(&args.checkpoint)?;
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

    let items = resolve_data_source_filtered(
        &args.data_dir,
        args.file.as_deref(),
        args.stdin,
        args.text.as_deref(),
        args.extensions.as_deref(),
        args.no_recursive,
    )?
    .read_items()?;
    if items.is_empty() {
        eprintln!("⚠️  No data items found for stats evaluation.");
        return Ok(());
    }

    let mut cfg = ClassifierConfig::with_checkpoint(checkpoint);
    cfg.escalate_threshold = args.escalate_threshold;
    let classifier = LayaClassifier::new(cfg);

    let report = StatsEngine::new(classifier)
        .with_iterations(args.iterations)
        .with_warmup(args.warmup)
        .with_cache(args.cache, args.similarity_threshold)
        .with_escalate_threshold(args.escalate_threshold)
        .evaluate(&items, &questions)?;

    if let Some(dest) = &args.output {
        StatsEngine::export_report(&report, dest)?;
        log_status(
            is_machine_format(args.format),
            format!("💾 Statistics report exported to: {}", dest.display()),
        );
    }

    match args.format {
        OutputFormat::Table | OutputFormat::Pretty => {
            println!("{}", StatsEngine::format_table(&report))
        }
        OutputFormat::Json => println!("{}", StatsEngine::format_json(&report)?),
        OutputFormat::Jsonl => println!("{}", serde_json::to_string(&report)?),
        OutputFormat::Compact => println!("{}", StatsEngine::format_compact(&report)),
        OutputFormat::Csv => print!("{}", StatsEngine::format_csv(&report)),
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
        (
            source.read_items()?,
            format!("Hacker News feed '{}'", category),
        )
    } else if let Some(sub) = &args.reddit {
        let source = lemma::datasource::RedditSource::new(sub, args.limit);
        (source.read_items()?, format!("r/{} on Reddit", sub))
    } else if let Some(article) = &args.wikipedia {
        let source = lemma::datasource::WikipediaSource::new(Some(article.clone()), args.limit);
        (
            source.read_items()?,
            format!("Wikipedia article '{}'", article),
        )
    } else {
        bail!("No source specified for fetch. Specify one of: --hf, --hn, --reddit, --wikipedia");
    };

    if args.auto_schema
        && let Some((name, schema)) = schema_generated
        && !schema.is_empty()
    {
        let schema_path = args.schema_output.unwrap_or_else(|| {
            std::path::PathBuf::from(format!("schema/{}.json", name.replace('/', "_")))
        });
        write_json_file(&schema_path, &schema)?;
        eprintln!(
            "✨ Inferred question schema written to {}",
            schema_path.display()
        );
    }

    if items.is_empty() {
        eprintln!("⚠️  No records retrieved from {}.", source_desc);
        return Ok(());
    }

    eprintln!("📦 Ingested {} items from {}", items.len(), source_desc);
    let mut out_writer = open_output_writer(args.output.as_deref())?;

    match args.format {
        OutputFormat::Jsonl => {
            for item in &items {
                writeln!(out_writer, "{}", serde_json::to_string(item)?)?;
            }
        }
        OutputFormat::Json | OutputFormat::Pretty => {
            writeln!(out_writer, "{}", serde_json::to_string_pretty(&items)?)?
        }
        OutputFormat::Table | OutputFormat::Compact => {
            let mut table = comfy_table::Table::new();
            table.load_style(comfy_table::presets::UTF8_FULL);
            table.set_header(vec!["ID", "Origin", "Ground Truth", "Preview Text"]);
            for item in &items {
                let gt_str = item
                    .ground_truth
                    .as_ref()
                    .map(|gt| serde_json::to_string(gt).unwrap_or_default())
                    .unwrap_or_else(|| "-".to_string());
                table.add_row(vec![
                    comfy_table::Cell::new(&item.id),
                    comfy_table::Cell::new(&item.source_uri),
                    comfy_table::Cell::new(gt_str),
                    comfy_table::Cell::new(item.preview_text(60)),
                ]);
            }
            writeln!(out_writer, "{table}")?;
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(out_writer);
            wtr.write_record(["id", "source_uri", "ground_truth", "state"])?;
            for item in &items {
                let gt_str = item
                    .ground_truth
                    .as_ref()
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
    let scraper = lemma::datasource::WebScraper::new(lemma::datasource::ScrapeOptions {
        timeout_secs: args.timeout,
        max_chars: args.max_chars,
        use_jina: args.jina,
        ..Default::default()
    });
    let doc = scraper.scrape(&args.url)?;

    eprintln!(
        "🌐 Scraped '{}' ({} words, SPA: {})",
        doc.url, doc.word_count, doc.is_likely_spa
    );

    let state = serde_json::json!({ "title": doc.title, "text": doc.markdown, "url": doc.url, "words": doc.word_count });
    let item = lemma::datasource::DataItem::new(format!("web:{}", doc.url), state, &doc.url);
    let mut out_writer = open_output_writer(args.output.as_deref())?;

    match args.format {
        OutputFormat::Jsonl => writeln!(out_writer, "{}", serde_json::to_string(&item)?)?,
        OutputFormat::Json | OutputFormat::Pretty => {
            writeln!(out_writer, "{}", serde_json::to_string_pretty(&item)?)?
        }
        OutputFormat::Table | OutputFormat::Compact => {
            let mut table = comfy_table::Table::new();
            table.load_style(comfy_table::presets::UTF8_FULL);
            table.set_header(vec!["Field", "Value"]);
            table.add_row(vec!["URL", &doc.url]);
            table.add_row(vec!["Title", &doc.title]);
            table.add_row(vec!["Word Count", &doc.word_count.to_string()]);
            table.add_row(vec![
                "SPA Shell",
                if doc.is_likely_spa { "true" } else { "false" },
            ]);
            table.add_row(vec!["Content Preview", &item.preview_text(160)]);
            writeln!(out_writer, "{table}")?;
        }
        OutputFormat::Csv => {
            let mut wtr = csv::Writer::from_writer(out_writer);
            wtr.write_record(["id", "url", "title", "words", "text"])?;
            wtr.write_record([
                &item.id,
                &doc.url,
                &doc.title,
                &doc.word_count.to_string(),
                &doc.markdown,
            ])?;
            wtr.flush()?;
        }
    }
    Ok(())
}

fn handle_learn(args: LearnArgs) -> Result<()> {
    let raw_target = args
        .phase
        .as_deref()
        .or(args.topic.as_deref())
        .map(str::trim);

    if args.glossary || raw_target.is_some_and(|t| t.eq_ignore_ascii_case("glossary")) {
        let filter = if args.glossary {
            raw_target.filter(|t| !t.eq_ignore_ascii_case("glossary"))
        } else {
            None
        };
        if args.json {
            let entries = lemma::engine::build_glossary();
            let filtered: Vec<_> = match filter {
                Some(q) => {
                    let ql = q.to_lowercase();
                    entries
                        .into_iter()
                        .filter(|e| {
                            e.term.to_lowercase().contains(&ql)
                                || e.category.to_lowercase().contains(&ql)
                                || e.definition.to_lowercase().contains(&ql)
                        })
                        .collect()
                }
                None => entries,
            };
            println!("{}", serde_json::to_string_pretty(&filtered)?);
        } else {
            lemma::engine::print_glossary(filter);
        }
        return Ok(());
    }

    if raw_target.is_some_and(|t| {
        matches!(
            t.to_ascii_lowercase().as_str(),
            "guide" | "recipes" | "cookbook" | "howto" | "cheatsheet"
        )
    }) {
        return handle_guide(GuideArgs {
            topic: None,
            json: args.json,
        });
    }

    let is_reading_topic = raw_target.is_some_and(|t| {
        matches!(
            t.to_ascii_lowercase().as_str(),
            "reading" | "papers" | "blogs" | "literature" | "references"
        )
    });
    if args.reading || is_reading_topic {
        let phase_num_filter = if args.reading {
            match raw_target {
                Some(t)
                    if !matches!(
                        t.to_ascii_lowercase().as_str(),
                        "reading" | "papers" | "blogs" | "literature" | "references" | "all"
                    ) =>
                {
                    Some(
                        t.parse::<lemma::engine::CurriculumPhase>()
                            .map_err(|e| anyhow::anyhow!(e))?
                            .number(),
                    )
                }
                _ => None,
            }
        } else {
            None
        };
        if args.json {
            let list = lemma::engine::learn::build_reading_list(phase_num_filter);
            println!("{}", serde_json::to_string_pretty(&list)?);
        } else {
            lemma::engine::learn::print_reading_list(phase_num_filter);
        }
        return Ok(());
    }

    // The lesson demos call classify with `.expect`, so under `native` a missing checkpoint would
    // panic. Check every checkpoint the lessons use before any lesson runs.
    if lemma::model::native::backend() == Backend::Native {
        let missing: Vec<String> = [
            Checkpoint::English,
            Checkpoint::Multilingual,
            Checkpoint::TypedDecisions,
            Checkpoint::DebertaV3ZeroShot,
            Checkpoint::ModernBertZeroShot,
            Checkpoint::EuroBert,
        ]
        .into_iter()
        .filter(|&c| !lemma::model::native::native_available(c))
        .map(|c| c.repo_id().to_string())
        .collect();
        if !missing.is_empty() {
            bail!(
                "lemma learn needs weights for {}. Download them with `lemma download`, or run with --backend auto to simulate the missing ones.",
                missing.join(", ")
            );
        }
    }

    if args.all || raw_target.is_some_and(|t| t.eq_ignore_ascii_case("all")) {
        if args.json {
            let lessons: Vec<_> = lemma::engine::CurriculumPhase::all()
                .into_iter()
                .map(lemma::engine::build_phase_lesson)
                .collect();
            println!("{}", serde_json::to_string_pretty(&lessons)?);
        } else {
            lemma::engine::print_all_phases();
        }
        return Ok(());
    }

    let phase_opt = match raw_target {
        Some(t) if !t.eq_ignore_ascii_case("quiz") => Some(
            t.parse::<lemma::engine::CurriculumPhase>()
                .map_err(|e| anyhow::anyhow!(e))?,
        ),
        _ => None,
    };

    if args.interactive {
        lemma::engine::run_interactive_session(phase_opt, args.quiz)?;
        return Ok(());
    }

    if args.quiz || raw_target.is_some_and(|t| t.eq_ignore_ascii_case("quiz")) {
        if args.json {
            let all_q = lemma::engine::build_all_quiz_questions();
            let filtered: Vec<_> = match phase_opt {
                Some(p) => all_q
                    .into_iter()
                    .filter(|q| q.phase == p.number())
                    .collect(),
                None => all_q,
            };
            println!("{}", serde_json::to_string_pretty(&filtered)?);
        } else {
            lemma::engine::run_quiz_non_interactive(phase_opt.map(|p| p.number()));
        }
        return Ok(());
    }

    if let Some(phase) = phase_opt {
        if args.json {
            let lesson = lemma::engine::build_phase_lesson(phase);
            println!("{}", serde_json::to_string_pretty(&lesson)?);
        } else {
            lemma::engine::print_phase_lesson(phase);
        }
        return Ok(());
    }

    if args.json {
        let phases: Vec<_> = lemma::engine::CurriculumPhase::all()
            .into_iter()
            .map(|p| {
                serde_json::json!({
                    "phase": p.number(),
                    "slug": p.slug(),
                    "title": p.title(),
                    "subtitle": p.subtitle(),
                    "key_takeaways": p.key_takeaways(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&phases)?);
    } else {
        lemma::engine::print_curriculum_overview();
    }
    Ok(())
}

fn handle_guide(args: GuideArgs) -> Result<()> {
    if args.json {
        let recipes = lemma::engine::build_workflow_recipes();
        let filtered: Vec<_> = match args.topic.as_deref() {
            Some(q) => {
                let ql = q.to_lowercase();
                recipes
                    .into_iter()
                    .filter(|r| {
                        r.slug.to_lowercase().contains(&ql) || r.title.to_lowercase().contains(&ql)
                    })
                    .collect()
            }
            None => recipes,
        };
        println!("{}", serde_json::to_string_pretty(&filtered)?);
    } else {
        lemma::engine::print_workflow_guide(args.topic.as_deref());
    }
    Ok(())
}
