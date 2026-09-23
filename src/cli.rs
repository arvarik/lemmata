use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "lemma",
    about = "High-performance CLI for Laya System 1 decision and classification models",
    version,
    author
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Download or verify Laya model checkpoints from Hugging Face Hub
    Download(DownloadArgs),

    /// Classify data ingested from data/ folder, files, stdin, or inline text
    Classify(ClassifyArgs),

    /// Run robustness and calibration experiments (order invariance, temperature sweeps)
    Experiment(ExperimentArgs),

    /// Evaluate classification accuracy and calibration against ground-truth labels
    Test(TestArgs),

    /// Benchmark classification throughput and latency percentiles
    Bench(BenchArgs),

    /// Display Laya model architecture specifications and local environment status
    Info(InfoArgs),

    /// Inspect local Safetensors weights, tensor shapes, parameter counts, and integrity
    Inspect(InspectArgs),

    /// Explain decisions via contrastive token leave-one-out margin ablation
    Explain(ExplainArgs),

    /// Calibrate and evaluate conformal prediction sets with guaranteed (1 - α) coverage
    Conformal(ConformalArgs),

    /// Validate and lint question schemas against Laya head token budgets
    Schema(SchemaArgs),
}

#[derive(Args, Debug)]
pub struct DownloadArgs {
    /// Checkpoint to download: english, multilingual, or typed-decisions
    #[arg(short, long, default_value = "english")]
    pub checkpoint: String,

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

    /// Path to a JSON schema file defining questions (Jev/Laya format)
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Inline question type: choice, score, or noul
    #[arg(long, default_value = "choice")]
    pub qtype: String,

    /// Inline question instructions
    #[arg(short, long, default_value = "Classify the topic or sentiment of this text")]
    pub instruction: String,

    /// Comma-separated labels for choice question (e.g. "billing,technical,sales,other")
    #[arg(short, long, default_value = "billing,technical,sales,other")]
    pub labels: String,

    /// Checkpoint to use: english, multilingual, or typed-decisions
    #[arg(short, long, default_value = "english")]
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
}

#[derive(Args, Debug)]
pub struct ExperimentArgs {
    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to ingest
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Experiment mode: order (position invariance), temp (calibration sweep), or checkpoints (compare all 3)
    #[arg(short, long, default_value = "order")]
    pub mode: ExperimentMode,

    /// Number of permutations for option ordering experiment
    #[arg(long, default_value_t = 5)]
    pub permutations: usize,
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

    /// Checkpoint to evaluate
    #[arg(short, long, default_value = "english")]
    pub checkpoint: String,

    /// Temperature scaling factor during evaluation
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Minimum required accuracy (exits with error if accuracy falls below this threshold, e.g. 0.85)
    #[arg(long)]
    pub min_accuracy: Option<f64>,

    /// Print evaluation output as JSON instead of table
    #[arg(long)]
    pub json: bool,
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

    /// Checkpoint to benchmark
    #[arg(short, long, default_value = "english")]
    pub checkpoint: String,

    /// Output benchmark metrics as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Checkpoint to inspect: all, english, multilingual, or typed-decisions
    #[arg(short, long, default_value = "all")]
    pub checkpoint: String,

    /// Check local cache directory for downloaded checkpoints
    #[arg(long, default_value = "models/laya")]
    pub local_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Path to model.safetensors file to inspect
    #[arg(short, long, default_value = "models/laya/model.safetensors")]
    pub file: PathBuf,

    /// Output report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ExplainArgs {
    /// Text to explain
    #[arg(short, long)]
    pub text: String,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Specific question ID to explain
    #[arg(long)]
    pub qid: Option<String>,

    /// Target winning choice to explain (e.g. "billing")
    #[arg(long, default_value = "billing")]
    pub target: String,

    /// Contrast runner-up choice (e.g. "technical")
    #[arg(long, default_value = "technical")]
    pub contrast: String,

    /// Checkpoint to use
    #[arg(short, long, default_value = "english")]
    pub checkpoint: String,
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

    /// Checkpoint to use
    #[arg(short, long, default_value = "english")]
    pub checkpoint: String,

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
}

#[derive(Args, Debug)]
pub struct SchemaValidateArgs {
    /// Path to schema JSON file to validate
    pub schema: PathBuf,

    /// Checkpoint to validate against (determines head token budget: 192 vs 256)
    #[arg(short, long, default_value = "english")]
    pub checkpoint: String,
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

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentMode {
    Order,
    Temp,
    Checkpoints,
}
