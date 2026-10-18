use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "lemmata",
    about = "High-performance System 1 decision platform for bidirectional encoders (Laya, ModernBERT, DeBERTa-v3, EuroBERT, NeoBERT)",
    version,
    author
)]
pub struct Cli {
    /// Inference backend. `native` runs real weights and fails when a checkpoint has none
    #[arg(
        long,
        global = true,
        value_enum,
        env = "LEMMATA_BACKEND",
        default_value_t = Backend::Native
    )]
    pub backend: Backend,

    /// Device for native models. `auto` uses the Metal GPU when one is present
    #[arg(
        long,
        global = true,
        value_enum,
        env = "LEMMATA_DEVICE",
        default_value_t = DeviceChoice::Auto
    )]
    pub device: DeviceChoice,

    #[command(subcommand)]
    pub command: Commands,
}

pub use crate::model::native::{Backend, DeviceChoice};

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Download or verify System 1 model checkpoints from Hugging Face Hub
    #[command(visible_alias = "pull")]
    Download(DownloadArgs),

    /// Classify data ingested from data/ folder, files, stdin, or inline text
    #[command(visible_alias = "run", alias = "predict", alias = "infer")]
    Classify(ClassifyArgs),

    /// Evaluate classification accuracy and calibration against ground-truth labels
    #[command(visible_alias = "eval")]
    Test(TestArgs),

    /// Benchmark classification throughput and latency percentiles
    Bench(BenchArgs),

    /// Display System 1 model architecture specifications and local environment status
    #[command(visible_alias = "models")]
    Info(InfoArgs),

    /// Inspect local Safetensors weights, tensor shapes, parameter counts, and integrity
    Inspect(InspectArgs),

    /// Explain decisions via contrastive token leave-one-out margin ablation
    #[command(visible_alias = "why")]
    Explain(ExplainArgs),

    /// Calibrate and evaluate conformal prediction sets with guaranteed (1 - α) coverage
    Conformal(ConformalArgs),

    /// Validate and lint question schemas against Laya head token budgets
    Schema(SchemaArgs),

    /// Pool several checkpoints into one ensemble and show where they disagree
    Ensemble(EnsembleArgs),

    /// Manage sub-millisecond semantic and exact decision cache
    Cache(CacheArgs),

    /// Decide which items to defer to a larger model or a person (a System 1 / System 2 cascade)
    Defer(DeferArgs),

    /// Verify signed, hash-chained decision receipts
    Attest(AttestArgs),

    /// Classify against a label taxonomy: a coarse label first, then a fine label
    Taxonomy(TaxonomyArgs),

    /// Automatically tune and calibrate temperature parameters per question bucket
    #[command(visible_alias = "tune")]
    Calibrate(CalibrateArgs),

    /// Fit supervised linear probes on labelled data
    Probe(ProbeArgs),

    /// Full evaluation report for one checkpoint: accuracy, calibration, speed, and escalation
    Report(ReportArgs),

    /// Head-to-head System 1 model arena comparing accuracy, agreement, Cohen's kappa, McNemar's chi2, and speedup
    #[command(visible_alias = "arena")]
    Compare(CompareArgs),

    /// Interactive 5-phase System 1 mastery curriculum, live concept labs, glossary, and Socratic quizzes
    #[command(visible_alias = "educate", alias = "tutorial", alias = "course")]
    Learn(LearnArgs),

    /// Practical end-to-end CLI workflow recipes and quickstart cookbook
    #[command(
        visible_alias = "cookbook",
        visible_alias = "howto",
        alias = "cheatsheet"
    )]
    Guide(GuideArgs),

    /// Generate shell auto-completions for bash, elvish, fish, powershell, or zsh
    Completions(CompletionsArgs),
}

#[derive(Args, Debug)]
pub struct CompletionsArgs {
    /// Target shell to generate completions for
    #[arg(value_enum)]
    pub shell: clap_complete::Shell,
}

#[derive(Args, Debug)]
pub struct DownloadArgs {
    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Download or verify all 7 local System 1 checkpoints
    #[arg(long)]
    pub all: bool,

    /// Destination directory for model weights
    #[arg(short, long, default_value = "models/laya")]
    pub target_dir: PathBuf,

    /// Show download command without fetching weights
    #[arg(long)]
    pub dry_run: bool,

    /// Only verify existing local model files
    #[arg(long)]
    pub verify_only: bool,
}

#[derive(Args, Debug)]
pub struct ClassifyArgs {
    /// Inline text snippet to classify (positional shortcut for --text)
    #[arg(value_name = "TEXT")]
    pub positional_text: Option<String>,

    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to ingest (JSON, JSONL, CSV, TXT)
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Read data streamed from standard input
    #[arg(long)]
    pub stdin: bool,

    /// Classify inline text snippet directly from command line
    #[arg(short, long)]
    pub text: Option<String>,

    /// Target web URL to scrape, extract, and classify directly
    #[arg(short = 'u', long)]
    pub url: Option<String>,

    /// Path to a JSON schema file defining questions (Laya format)
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Inline question type: choice, score, or noul
    #[arg(long, default_value = "choice")]
    pub qtype: String,

    /// Inline question instructions
    #[arg(
        short,
        long,
        default_value = "Classify the topic or sentiment of this text"
    )]
    pub instruction: String,

    /// Comma-separated labels for choice question (e.g. "billing,technical,sales,other")
    #[arg(short, long, default_value = "billing,technical,sales,other")]
    pub labels: String,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Temperature scaling factor
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,

    /// Output presentation format (table, json, pretty, compact, csv, jsonl)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Write output to specified file instead of stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Filter output to only include decisions flagged for supervisor escalation
    #[arg(long)]
    pub escalate_only: bool,

    /// Limit maximum number of ingested items
    #[arg(short = 'n', long)]
    pub limit: Option<usize>,

    /// Enable multi-label classification using independent sigmoid scoring
    #[arg(long)]
    pub multi_label: bool,

    /// Multi-label decision threshold for sigmoid probabilities
    #[arg(long, default_value_t = 0.40)]
    pub multi_label_threshold: f64,

    /// Path to custom JSON model/agent config (e.g. rl_agent_config.json with temperature buckets)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Supervisor escalation probability threshold (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub escalate_threshold: f64,

    /// Enable sub-millisecond semantic decision cache
    #[arg(long)]
    pub cache: bool,

    /// Also reuse cached decisions for similar text (hashed word features, not the model). Negations can match
    #[arg(long, requires = "cache")]
    pub cache_semantic: bool,

    /// Similarity threshold for semantic cache match (default: 0.92)
    #[arg(long, default_value_t = 0.92)]
    pub cache_threshold: f64,

    /// Cache TTL in seconds (default: 86400)
    #[arg(long, default_value_t = 86400)]
    pub cache_ttl: u64,

    /// Print semantic cache hit/miss statistics summary
    #[arg(long)]
    pub cache_stats: bool,

    /// Append one signed, hash-chained receipt per decision to this JSONL file. Check it with `lemmata attest verify`
    #[arg(long, conflicts_with = "taxonomy")]
    pub receipts: Option<PathBuf>,

    /// HMAC-SHA256 key that signs the receipts. Prefer the LEMMATA_ATTEST_KEY environment variable: a key on the command line stays in shell history
    #[arg(long, env = "LEMMATA_ATTEST_KEY", hide_env_values = true)]
    pub attest_key: Option<String>,

    /// Comma-separated file extensions to ingest from directory (e.g. "json,jsonl,csv")
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,

    /// Path to a taxonomy schema (coarse labels, then fine labels), as for `lemmata taxonomy run`
    #[arg(long)]
    pub taxonomy: Option<PathBuf>,

    /// Coarse router confidence threshold to proceed to fine sub-classifier (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub coarse_threshold: f64,

    /// Custom NLI hypothesis or MLM verbalizer template (e.g. "This support ticket is about {label}.")
    #[arg(long)]
    pub hypothesis_template: Option<String>,

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate, head-tail, sliding)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Keep only the top-k options by lexical match (hashed word features and a synonym table) before model scoring (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: usize,

    /// Supervised probe from `lemmata probe fit`. It replaces zero-shot scoring for its question
    #[arg(long)]
    pub probe: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct CompareArgs {
    /// Comma-separated list of System 1 checkpoints to compare (default: all)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "models",
        visible_alias = "model",
        default_value = "all"
    )]
    pub checkpoints: String,

    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to evaluate
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Number of top model disagreement examples to highlight
    #[arg(long, default_value_t = 5)]
    pub top_disagreements: usize,

    /// Disaggregate accuracy by diagnostic slice (auto, prefix, script, length, or `field:<name>`)
    #[arg(long)]
    pub slice_by: Option<String>,

    /// Custom NLI hypothesis or MLM verbalizer template (e.g. "This support ticket is about {label}.")
    #[arg(long)]
    pub hypothesis_template: Option<String>,

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate, head-tail, sliding)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Keep only the top-k options by lexical match (hashed word features and a synonym table) before model scoring (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: usize,

    /// Output arena report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct TestArgs {
    /// Directory to ingest labeled data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific labeled file to evaluate (must contain ground_truth or label field)
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Temperature scaling factor during evaluation
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Path to custom JSON model/agent config (e.g. calibrated config with temperature & cost_weights)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Minimum required accuracy (exits with error if accuracy falls below this threshold, e.g. 0.85)
    #[arg(long, visible_alias = "fail-under")]
    pub min_accuracy: Option<f64>,

    /// Path to baseline evaluation metrics JSON to compare against for CI regression gating
    #[arg(long)]
    pub baseline: Option<PathBuf>,

    /// Save current evaluation metrics JSON to a baseline file
    #[arg(long)]
    pub save_baseline: Option<PathBuf>,

    /// Maximum allowed accuracy drop relative to --baseline before failing CI (default: 0.0)
    #[arg(long, default_value_t = 0.0)]
    pub max_regression: f64,

    /// Maximum allowed Expected Calibration Error (ECE) before failing CI
    #[arg(long)]
    pub max_ece: Option<f64>,

    /// Print evaluation output as JSON instead of table
    #[arg(long)]
    pub json: bool,

    /// Comma-separated file extensions to ingest from directory
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,

    /// Custom NLI hypothesis or MLM verbalizer template (e.g. "This support ticket is about {label}.")
    #[arg(long)]
    pub hypothesis_template: Option<String>,

    /// Disaggregate evaluation metrics by diagnostic slice (auto, prefix, script, length, or `field:<name>`)
    #[arg(long)]
    pub slice_by: Option<String>,

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate, head-tail, sliding)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Keep only the top-k options by lexical match (hashed word features and a synonym table) before model scoring (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: usize,

    /// Supervised probe from `lemmata probe fit`. It replaces zero-shot scoring for its question
    #[arg(long)]
    pub probe: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to benchmark
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Number of benchmark passes over the dataset
    #[arg(short, long, default_value_t = 10)]
    pub iterations: usize,

    /// Number of warmup passes before recording metrics
    #[arg(short, long, default_value_t = 3)]
    pub warmup: usize,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Output benchmark metrics as JSON
    #[arg(long)]
    pub json: bool,

    /// Comma-separated file extensions to ingest from directory
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Model checkpoint (all, english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "all"
    )]
    pub checkpoint: String,

    /// Check local cache directory for downloaded checkpoints
    #[arg(long, default_value = "models/laya")]
    pub local_dir: PathBuf,

    /// Scan and display live disk & architecture inventory across all local checkpoints (and custom HF directories)
    #[arg(long)]
    pub inventory: bool,

    /// Inspect a specific local Hugging Face checkpoint directory (config.json + model.safetensors)
    #[arg(long)]
    pub inspect_dir: Option<PathBuf>,

    /// Output model specifications as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Path to model.safetensors file or model directory to inspect
    #[arg(short, long, default_value = "models/laya/model.safetensors")]
    pub file: PathBuf,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(short, long, short_alias = 'm', visible_alias = "model")]
    pub checkpoint: Option<String>,

    /// Output report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ExplainArgs {
    /// Text to explain (can also be passed as a positional argument)
    #[arg(short, long)]
    pub text: Option<String>,

    /// Inline text to explain (positional)
    #[arg(value_name = "TEXT")]
    pub input_text: Option<String>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Comma-separated labels for an inline question, used when no --schema is given
    #[arg(short, long, default_value = "billing,technical,sales,other")]
    pub labels: String,

    /// Specific question ID to explain
    #[arg(long)]
    pub qid: Option<String>,

    /// Target winning choice to explain (auto-detected top-1 prediction if omitted)
    #[arg(long)]
    pub target: Option<String>,

    /// Contrast runner-up choice (auto-detected top-2 prediction if omitted)
    #[arg(long)]
    pub contrast: Option<String>,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Output explanation report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ConformalArgs {
    /// Directory to ingest calibration data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific labeled file for calibration
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Significance error rate alpha (e.g. 0.05 for 95% coverage guarantee)
    #[arg(short, long, default_value_t = 0.05)]
    pub alpha: f64,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Specific question ID to calibrate (default: first choice question)
    #[arg(long)]
    pub qid: Option<String>,

    /// Output report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct SchemaArgs {
    #[command(subcommand)]
    pub action: SchemaCommands,
}

#[derive(Subcommand, Debug)]
pub enum SchemaCommands {
    /// Validate question schema criteria and verify token budget limits
    Validate(SchemaValidateArgs),

    /// Automatically infer questions and synthesize criteria from raw data
    Infer(SchemaInferArgs),

    /// Dynamic 0/1 knapsack criteria optimizer to fit ModernBERT/mmBERT head token budgets
    Optimize(SchemaOptimizeArgs),
}

#[derive(Args, Debug)]
pub struct SchemaValidateArgs {
    /// Path to schema JSON file to validate
    pub schema: PathBuf,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,
}

#[derive(Args, Debug)]
pub struct SchemaInferArgs {
    /// Path to CSV or JSONL data file to infer schema from
    #[arg(short, long)]
    pub data: PathBuf,

    /// Name of column containing input text (auto-detected if omitted)
    #[arg(long)]
    pub text_col: Option<String>,

    /// Comma-separated target column names to infer questions for (auto-detected if omitted)
    #[arg(long)]
    pub target_cols: Option<String>,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Output file path to save inferred schema JSON
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct EnsembleArgs {
    /// Comma-separated list of checkpoints (default: "english,multilingual,typed-decisions")
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "models",
        visible_alias = "model",
        default_value = "english,multilingual,typed-decisions"
    )]
    pub checkpoints: String,

    /// Optional comma-separated reliability weights (e.g. "0.4,0.3,0.3")
    #[arg(short, long)]
    pub weights: Option<String>,

    /// Epistemic uncertainty disagreement threshold (triggers escalation when exceeded)
    #[arg(long, default_value_t = 0.15)]
    pub disagreement_threshold: f64,

    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to ingest
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Read data streamed from standard input
    #[arg(long)]
    pub stdin: bool,

    /// Classify inline text snippet directly from command line
    #[arg(short, long)]
    pub text: Option<String>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Optional path to save consensus output to file
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Output presentation format (table, json, pretty, compact)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Output report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub action: CacheCommands,
}

#[derive(Subcommand, Debug)]
pub enum CacheCommands {
    /// Benchmark latency with and without semantic cache
    Bench(CacheBenchArgs),

    /// Inspect cache performance metrics
    Stats,

    /// Clear in-memory / persisted cache
    Clear,
}

#[derive(Args, Debug)]
pub struct CacheBenchArgs {
    /// Specific file to benchmark
    #[arg(short, long)]
    pub file: PathBuf,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Semantic similarity threshold
    #[arg(long, default_value_t = 0.92)]
    pub threshold: f64,
}

#[derive(Args, Debug)]
pub struct DeferArgs {
    #[command(subcommand)]
    pub action: DeferCommands,
}

#[derive(Subcommand, Debug)]
pub enum DeferCommands {
    /// Tune Pareto-optimal threshold given target accuracy SLA
    Tune(DeferTuneArgs),

    /// Sweep thresholds across [0.05, 0.95] and display full Cost / Coverage / Accuracy Pareto table
    Sweep(DeferTuneArgs),

    /// Run cost-aware cascading router on inputs
    Run(DeferRunArgs),
}

#[derive(Args, Debug)]
pub struct DeferTuneArgs {
    /// Labeled validation dataset
    #[arg(
        short,
        long,
        visible_alias = "eval-data",
        visible_alias = "file",
        default_value = "data/customer_support.jsonl"
    )]
    pub val_data: PathBuf,

    /// Path to questions schema JSON file
    #[arg(short, long, default_value = "schema/questions.json")]
    pub schema: PathBuf,

    /// Target accuracy SLA (e.g. 0.95 for 95%)
    #[arg(long, visible_alias = "target-acc", default_value_t = 0.95)]
    pub target_accuracy: f64,

    /// Estimated System 2 cost per invocation in USD
    #[arg(long, default_value_t = 0.005)]
    pub system2_cost: f64,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Sweep all candidate risk thresholds in [0.05, 0.95] and print the Pareto frontier table
    #[arg(long)]
    pub sweep: bool,

    /// Optional path to export tuned cascade policy JSON
    #[arg(long)]
    pub export_policy: Option<PathBuf>,

    /// Output tuned policy as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct DeferRunArgs {
    /// Specific file to route
    #[arg(short, long, visible_alias = "data")]
    pub file: Option<PathBuf>,

    /// Inline text snippet
    #[arg(short, long)]
    pub text: Option<String>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Risk threshold tau (items with risk > tau cascade to System 2)
    #[arg(short, long, visible_alias = "threshold", default_value_t = 0.45)]
    pub risk_threshold: f64,

    /// Estimated System 2 cost per invocation in USD
    #[arg(long, default_value_t = 0.005)]
    pub system2_cost: f64,

    /// Output presentation format
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Output decisions as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct AttestArgs {
    #[command(subcommand)]
    pub action: AttestCommands,
}

#[derive(Subcommand, Debug)]
pub enum AttestCommands {
    /// Verify receipt digests, HMAC-SHA256 signatures, and hash chain links in a receipts log
    Verify(AttestVerifyArgs),
}

#[derive(Args, Debug)]
pub struct AttestVerifyArgs {
    /// Path to decision receipts JSONL log
    #[arg(
        short,
        long,
        visible_alias = "receipts",
        default_value = "logs/receipts.jsonl"
    )]
    pub receipt_log: PathBuf,

    /// HMAC-SHA256 key that signed the receipts. Prefer the LEMMATA_ATTEST_KEY environment variable: a key on the command line stays in shell history
    #[arg(
        short,
        long,
        visible_alias = "attest-key",
        env = "LEMMATA_ATTEST_KEY",
        hide_env_values = true
    )]
    pub secret_key: String,

    /// Output verification report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct SchemaOptimizeArgs {
    /// Path to schema JSON file to optimize
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Target head token budget (default: 192 for English, 256 for Multilingual)
    #[arg(short, long, default_value_t = 192)]
    pub budget: usize,

    /// Output path to save optimized schema JSON
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Show detailed clause diff
    #[arg(long)]
    pub diff: bool,
}

#[derive(Args, Debug)]
pub struct TaxonomyArgs {
    #[command(subcommand)]
    pub command: TaxonomyCommands,
}

#[derive(Subcommand, Debug)]
pub enum TaxonomyCommands {
    /// Classify inputs through a hierarchical coarse-to-fine decision tree
    Run(TaxonomyRunArgs),
    /// Print a taxonomy schema as a tree
    Show(TaxonomyShowArgs),
    /// Validate token budgets across all coarse and fine hierarchical nodes
    Validate(TaxonomyValidateArgs),
    /// Benchmark hierarchical coarse vs fine latency and throughput
    Bench(TaxonomyBenchArgs),
}

#[derive(Args, Debug)]
pub struct TaxonomyRunArgs {
    /// Path to hierarchical question schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Specific file to ingest (JSON, JSONL, CSV, TXT)
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Classify inline text snippet directly from command line
    #[arg(short, long)]
    pub text: Option<String>,

    /// Coarse router confidence threshold to proceed to fine sub-classifier (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub coarse_threshold: f64,

    /// Supervisor escalation probability threshold (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub escalate_threshold: f64,

    /// Output presentation format (table, json, csv, jsonl)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Write output to specified file instead of stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Enable sub-millisecond semantic decision cache
    #[arg(long)]
    pub cache: bool,

    /// Filter output to only include decisions flagged for escalation
    #[arg(long)]
    pub escalate_only: bool,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,
}

#[derive(Args, Debug)]
pub struct TaxonomyShowArgs {
    /// Path to hierarchical question schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,
}

#[derive(Args, Debug)]
pub struct TaxonomyValidateArgs {
    /// Path to hierarchical question schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,
}

#[derive(Args, Debug)]
pub struct TaxonomyBenchArgs {
    /// Path to hierarchical question schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Specific file to ingest
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Number of benchmark iterations
    #[arg(short, long, default_value_t = 20)]
    pub iterations: usize,

    /// Coarse router confidence threshold (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub coarse_threshold: f64,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,
}

#[derive(Args, Debug)]
pub struct CalibrateArgs {
    /// Path to labeled validation dataset (JSON, JSONL, CSV with ground truth)
    #[arg(short = 'v', long, alias = "file")]
    pub val_data: Option<PathBuf>,

    /// Directory to ingest validation data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Path to question schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Loss function to minimize during calibration: nll, brier, or ece (default: nll)
    #[arg(long, default_value = "nll")]
    pub loss: String,

    /// Number of confidence bins for Expected Calibration Error (ECE) calculation
    #[arg(long, default_value_t = 10)]
    pub bins: usize,

    /// Minimum labeled samples per bucket before lemmata fits its temperature
    #[arg(long, default_value_t = crate::engine::calibrate::DEFAULT_MIN_BUCKET_SAMPLES)]
    pub min_bucket_samples: usize,

    /// Also fit a bias per choice label on the labelled data and export it as cost weights.
    /// It removes a model's preference for some labels. Fit on data you will not test on
    #[arg(long)]
    pub fit_label_bias: bool,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Export calibrated configuration to specified JSON file (e.g. config/laya_calibrated.json)
    #[arg(short = 'e', long, alias = "output", visible_alias = "export")]
    pub export_config: Option<PathBuf>,

    /// Persist calibrated profile directly into `<checkpoint_local_dir>/lemmata_calibration.json` for auto-loading
    #[arg(long)]
    pub persist_to_model: bool,

    /// Enable 2026 Quantile-Adaptive Temperature Scaling (Q-ATS) under covariate shift
    #[arg(long)]
    pub adaptive: bool,

    /// Q-ATS adaptive scaling factor gamma (default: 0.30)
    #[arg(long, default_value_t = 0.30)]
    pub adaptive_gamma: f64,

    /// Asymmetric class cost weights (e.g. "billing=5.0,security=10.0" or path to JSON file)
    #[arg(long, visible_alias = "cost-matrix")]
    pub cost_weights: Option<String>,

    /// Output report format (table, json, pretty)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Output calibration report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ReportArgs {
    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to evaluate (JSON, JSONL, CSV)
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Read inputs from standard input (stdin)
    #[arg(long)]
    pub stdin: bool,

    /// Inline text string to classify and evaluate
    #[arg(short, long)]
    pub text: Option<String>,

    /// Path to question schema JSON file
    #[arg(short, long, default_value = "schema/questions.json")]
    pub schema: PathBuf,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Number of benchmark passes for timing stability (default: 3)
    #[arg(short, long, default_value_t = 3)]
    pub iterations: usize,

    /// Number of warmup passes before recording timing metrics (default: 2)
    #[arg(short, long, default_value_t = 2)]
    pub warmup: usize,

    /// Enable semantic decision cache to evaluate caching speedup and hit rates
    #[arg(long)]
    pub cache: bool,

    /// Similarity threshold for semantic cache match (default: 0.92)
    #[arg(long, default_value_t = 0.92)]
    pub similarity_threshold: f32,

    /// Supervisor escalation probability threshold (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub escalate_threshold: f64,

    /// Comma-separated file extensions to ingest from directory
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,

    /// Output report format: table, json, compact, or csv
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Export statistics report to output file (JSON or CSV)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Table,
    Json,
    Pretty,
    Compact,
    Csv,
    Jsonl,
}

#[derive(Args, Debug)]
pub struct LearnArgs {
    /// Optional phase number (`1`..`5`), topic slug (`foundations`, `paradigms`, `scaling`, `calibration`, `production`), `all`, `glossary`, or `reading`. For recipes, run `lemmata guide`
    #[arg(value_name = "PHASE_OR_TOPIC")]
    pub topic: Option<String>,

    /// Specific curriculum phase (`1`..`5`, `foundations`, `paradigms`, `scaling`, `calibration`, `production`, or `all`)
    #[arg(short, long)]
    pub phase: Option<String>,

    /// Launch step-by-step interactive terminal tutor with live labs and graded Socratic checkpoints
    #[arg(short, long)]
    pub interactive: bool,

    /// Run Socratic diagnostic multiple-choice quiz to test conceptual intuition
    #[arg(short, long)]
    pub quiz: bool,

    /// Display A-to-Z System 1 concept and mathematical glossary (optionally filtered by topic)
    #[arg(short, long)]
    pub glossary: bool,

    /// Display curated reading trail of foundational papers, blogs, and practitioner guides
    #[arg(short = 'r', long)]
    pub reading: bool,

    /// Render all 5 curriculum phases sequentially
    #[arg(short, long)]
    pub all: bool,

    /// Output curriculum, lesson, glossary, reading list, or quiz as structured JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct GuideArgs {
    /// Optional workflow recipe slug (`quickstart`, `arena`, `calibration`, `scaling`, `production`)
    #[arg(value_name = "RECIPE")]
    pub topic: Option<String>,

    /// Output workflow recipes as structured JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ProbeArgs {
    #[command(subcommand)]
    pub action: ProbeCommands,
}

#[derive(Subcommand, Debug)]
pub enum ProbeCommands {
    /// Fit a probe for one choice question on labelled items and save it as JSON
    Fit(ProbeFitArgs),
}

#[derive(Args, Debug)]
pub struct ProbeFitArgs {
    /// Labelled training data (JSON, JSONL, or CSV with ground truth). Do not test on it
    #[arg(short, long)]
    pub data: PathBuf,

    /// Question schema
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Choice question to probe (default: the schema's only choice question)
    #[arg(short, long)]
    pub question: Option<String>,

    /// Checkpoint whose embeddings the probe reads
    #[arg(short, long, visible_alias = "model", default_value = "english")]
    pub checkpoint: String,

    /// Long-text strategy for the embeddings (truncate, head-tail, sliding-window)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// File to write the probe to
    #[arg(short, long)]
    pub output: PathBuf,
}
