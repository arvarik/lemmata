use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "lemma",
    about = "High-performance System 1 decision platform for bidirectional encoders (Laya, ModernBERT, DeBERTa-v3, EuroBERT, NeoBERT)",
    version,
    author
)]
pub struct Cli {
    /// Inference backend. `auto` runs real weights for each checkpoint that has them in models/
    #[arg(
        long,
        global = true,
        value_enum,
        env = "LEMMA_BACKEND",
        default_value_t = Backend::Auto
    )]
    pub backend: Backend,

    /// Device for native models. `auto` uses the Metal GPU when one is present
    #[arg(
        long,
        global = true,
        value_enum,
        env = "LEMMA_DEVICE",
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

    /// Fetch public datasets or live discussions from Hugging Face, Hacker News, Reddit, or Wikipedia
    Fetch(FetchArgs),

    /// Scrape web pages or articles into clean LLM-ready markdown or text
    Scrape(ScrapeArgs),

    /// Run robustness and calibration experiments (order invariance, temperature sweeps)
    Experiment(ExperimentArgs),

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

    /// Detect concept drift and distribution shift between baseline and live inputs
    Drift(DriftArgs),

    /// Enforce declarative policy guardrails and emit immutable audit records
    Guard(GuardArgs),

    /// Multi-checkpoint mixture-of-decisions consensus with epistemic uncertainty
    Consensus(ConsensusArgs),

    /// Manage sub-millisecond semantic and exact decision cache
    Cache(CacheArgs),

    /// Free-energy and Mahalanobis++ out-of-distribution (OOD) detection
    Ood(OodArgs),

    /// Cost-aware System 1 / System 2 cascading hybrid router
    Cascade(CascadeArgs),

    /// Verify signed, hash-chained decision receipts
    Attest(AttestArgs),

    /// Hierarchical coarse-to-fine decision trees for high-cardinality routing
    #[command(visible_alias = "route")]
    Hierarchical(HierarchicalArgs),

    /// Automatically tune and calibrate temperature parameters per question bucket
    #[command(visible_alias = "tune")]
    Calibrate(CalibrateArgs),

    /// Process live streaming events with micro-batching, lakehouse columnar projection, and dual sinks
    Stream(StreamArgs),

    /// Active learning uncertainty sampling, supervisor review queue, and RLCD contrastive distillation
    Active(ActiveArgs),

    /// Comprehensive system speed, throughput performance, and classification accuracy statistics
    Stats(StatsArgs),

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

    /// Path to fitted OOD profile JSON to check for anomalies
    #[arg(long)]
    pub ood_profile: Option<PathBuf>,

    /// Critical OOD threshold (anomaly score >= threshold triggers action, default: 0.60)
    #[arg(long, default_value_t = 0.60)]
    pub ood_threshold: f64,

    /// Action when OOD detected (allow, flag, escalate, quarantine)
    #[arg(long, default_value = "escalate")]
    pub ood_action: String,

    /// Comma-separated file extensions to ingest from directory (e.g. "json,jsonl,csv")
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,

    /// Path to hierarchical question schema (coarse-to-fine decision tree)
    #[arg(long)]
    pub hierarchical: Option<PathBuf>,

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

    /// Experiment mode: order (position invariance), temp (calibration sweep), checkpoints (compare models), or adversarial
    #[arg(short, long, default_value = "order")]
    pub mode: ExperimentMode,

    /// Run adversarial perturbation and boundary stress test (shortcut for --mode adversarial)
    #[arg(long)]
    pub adversarial: bool,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(short, long, visible_alias = "model", default_value = "english")]
    pub checkpoint: String,

    /// Number of permutations for option ordering experiment
    #[arg(long, default_value_t = 5)]
    pub permutations: usize,

    /// Output experiment results as JSON
    #[arg(long)]
    pub json: bool,
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
pub struct DriftArgs {
    /// Path to golden baseline reference dataset (JSONL or CSV)
    #[arg(short, long)]
    pub baseline: PathBuf,

    /// Specific current production file to test for drift
    #[arg(long)]
    pub current: Option<PathBuf>,

    /// Directory to ingest current production data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

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

    /// Critical threshold for Population Stability Index (PSI)
    #[arg(long, default_value_t = 0.20)]
    pub psi_threshold: f64,

    /// Critical threshold for continuous Wasserstein-1 Earth Mover's Distance
    #[arg(long, default_value_t = 0.15)]
    pub wasserstein_threshold: f64,

    /// Exit with error code if critical drift is detected (for CI/CD gating)
    #[arg(long)]
    pub fail_on_drift: bool,

    /// Output report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct GuardArgs {
    /// Path to declarative policy JSON file defining guardrail rules
    #[arg(short, long, default_value = "schema/policy.json")]
    pub policy: PathBuf,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

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

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Optional path to write JSONL audit log records
    #[arg(long)]
    pub audit_output: Option<PathBuf>,

    /// HMAC-SHA256 key that signs decision receipts. Prefer the LEMMA_ATTEST_KEY environment variable: a key on the command line stays in shell history
    #[arg(long, env = "LEMMA_ATTEST_KEY", hide_env_values = true)]
    pub attest_key: Option<String>,

    /// Path to output cryptographically signed receipts JSONL
    #[arg(long)]
    pub receipt_log: Option<PathBuf>,

    /// Comma-separated file extensions to ingest from directory
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,

    /// Output presentation format (table, json, pretty, compact, csv, jsonl)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Output report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct ConsensusArgs {
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
pub struct OodArgs {
    #[command(subcommand)]
    pub action: OodCommands,
}

#[derive(Subcommand, Debug)]
pub enum OodCommands {
    /// Fit OOD detection profile on baseline in-distribution data
    Fit(OodFitArgs),

    /// Evaluate OOD detection on in-distribution vs anomaly datasets
    Eval(OodEvalArgs),
}

#[derive(Args, Debug)]
pub struct OodFitArgs {
    /// In-distribution training dataset (JSONL or CSV)
    #[arg(
        short,
        long,
        visible_alias = "in-data",
        visible_alias = "file",
        default_value = "data/customer_support.jsonl"
    )]
    pub train_data: PathBuf,

    /// Path to questions schema JSON file
    #[arg(short, long, default_value = "schema/questions.json")]
    pub schema: PathBuf,

    /// Target question ID to fit centroids for
    #[arg(long, default_value = "classification")]
    pub target_qid: String,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Target false-positive rate (FPR) for threshold calibration
    #[arg(long, default_value_t = 0.05)]
    pub fpr_rate: f64,

    /// Output file to save fitted OOD profile JSON
    #[arg(short, long, default_value = "models/laya/ood_profile.json")]
    pub output: PathBuf,

    /// Output fitted profile as JSON to stdout
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct OodEvalArgs {
    /// In-distribution evaluation dataset
    #[arg(long, default_value = "data/customer_support.jsonl")]
    pub id_data: PathBuf,

    /// Out-of-distribution anomaly dataset
    #[arg(
        long,
        visible_alias = "test-data",
        default_value = "data/banking_tickets.jsonl"
    )]
    pub ood_data: PathBuf,

    /// Path to questions schema JSON file
    #[arg(short, long, default_value = "schema/questions.json")]
    pub schema: PathBuf,

    /// Path to fitted OOD profile JSON
    #[arg(short, long, default_value = "models/laya/ood_profile.json")]
    pub profile: PathBuf,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Output evaluation metrics as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct CascadeArgs {
    #[command(subcommand)]
    pub action: CascadeCommands,
}

#[derive(Subcommand, Debug)]
pub enum CascadeCommands {
    /// Tune Pareto-optimal threshold given target accuracy SLA
    Tune(CascadeTuneArgs),

    /// Sweep thresholds across [0.05, 0.95] and display full Cost / Coverage / Accuracy Pareto table
    Sweep(CascadeTuneArgs),

    /// Run cost-aware cascading router on inputs
    Run(CascadeRunArgs),
}

#[derive(Args, Debug)]
pub struct CascadeTuneArgs {
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
pub struct CascadeRunArgs {
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

    /// HMAC-SHA256 key that signed the receipts. Prefer the LEMMA_ATTEST_KEY environment variable: a key on the command line stays in shell history
    #[arg(
        short,
        long,
        visible_alias = "attest-key",
        env = "LEMMA_ATTEST_KEY",
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
pub struct HierarchicalArgs {
    #[command(subcommand)]
    pub command: HierarchicalCommands,
}

#[derive(Subcommand, Debug)]
pub enum HierarchicalCommands {
    /// Classify inputs through a hierarchical coarse-to-fine decision tree
    Run(HierarchicalRunArgs),
    /// Inspect and print ASCII tree visualization of a hierarchical schema
    Tree(HierarchicalTreeArgs),
    /// Validate token budgets across all coarse and fine hierarchical nodes
    Validate(HierarchicalValidateArgs),
    /// Benchmark hierarchical coarse vs fine latency and throughput
    Bench(HierarchicalBenchArgs),
}

#[derive(Args, Debug)]
pub struct HierarchicalRunArgs {
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
pub struct HierarchicalTreeArgs {
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
pub struct HierarchicalValidateArgs {
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
pub struct HierarchicalBenchArgs {
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

    /// Minimum labeled samples per bucket before lemma fits its temperature
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

    /// Persist calibrated profile directly into `<checkpoint_local_dir>/lemma_calibration.json` for auto-loading
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
pub struct StreamArgs {
    /// Input: a local path, - for stdin, lakehouse://path?columns=..., or kafka://broker/topic?file=path (replays the local file)
    #[arg(short = 'i', long, alias = "input")]
    pub source: String,

    /// Path to question schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Output file path, - for stdout, or kafka://broker/topic?file=path (writes the local file)
    #[arg(long, default_value = "-")]
    pub sink: String,

    /// Escalation output file path, or kafka://broker/topic?file=path (writes the local file)
    #[arg(long)]
    pub escalate_sink: Option<String>,

    /// Micro-batch accumulation size limit (default: 32)
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// Maximum accumulation flush window in milliseconds (default: 50)
    #[arg(long, default_value_t = 50)]
    pub max_flush_interval_ms: u64,

    /// Confidence threshold below which events are diverted to the escalation sink (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub escalate_threshold: f64,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Optional calibrated agent config (rl_agent_config.json)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Columnar projection list for lakehouse / tabular sources (comma-separated, e.g. "id,body,target")
    #[arg(long)]
    pub columns: Option<String>,

    /// Enable semantic decision cache
    #[arg(long)]
    pub cache: bool,

    /// Emit signed, hash-chained decision receipts (needs a signing key)
    #[arg(long)]
    pub attest: bool,

    /// HMAC-SHA256 key that signs receipts. Prefer the LEMMA_ATTEST_KEY environment variable: a key on the command line stays in shell history
    #[arg(long, env = "LEMMA_ATTEST_KEY", hide_env_values = true)]
    pub attest_secret: Option<String>,

    /// Write the full signed receipts to this JSONL file, for `lemma attest verify`
    #[arg(long, requires = "attest")]
    pub receipt_log: Option<PathBuf>,

    /// Maximum number of events to process before stopping (default: unlimited)
    #[arg(long)]
    pub limit: Option<usize>,

    /// Output report format (compact, table, json)
    #[arg(long, default_value = "compact")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct ActiveArgs {
    #[command(subcommand)]
    pub action: ActiveCommands,
}

#[derive(Subcommand, Debug)]
pub enum ActiveCommands {
    /// Select high-uncertainty samples for supervisor review queue using active learning
    Sample(ActiveSampleArgs),

    /// Export supervisor resolutions or labeled data into RLCD contrastive distillation pairs
    Distill(ActiveDistillArgs),

    /// Inspect uncertainty distributions and active learning pool diagnostics
    Stats(ActiveStatsArgs),
}

#[derive(Args, Debug)]
pub struct ActiveSampleArgs {
    /// Unlabeled or candidate evaluation dataset file (JSON, JSONL, CSV)
    #[arg(short = 'd', long, alias = "file")]
    pub data: PathBuf,

    /// Path to question schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Uncertainty sampling strategy: composite, entropy, margin, conformal, energy (default: composite)
    #[arg(long, default_value = "composite")]
    pub strategy: crate::engine::active::UncertaintyStrategy,

    /// Maximum number of high-uncertainty samples to acquire for review (default: 50)
    #[arg(short = 'k', long, default_value_t = 50)]
    pub budget: usize,

    /// Conformal error rate alpha for set cardinality calculation (default: 0.10)
    #[arg(long, default_value_t = 0.10)]
    pub conformal_alpha: f64,

    /// Escalation threshold for flagging critical items (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub escalate_threshold: f64,

    /// Optional path to pre-fitted OOD profile for free-energy calculation
    #[arg(long)]
    pub ood_profile: Option<PathBuf>,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Export prioritized review queue items to specified JSONL file (e.g. data/review_queue.jsonl)
    #[arg(short = 'o', long, alias = "output")]
    pub export_queue: Option<PathBuf>,

    /// Output report format (table, json, compact)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct ActiveDistillArgs {
    /// Path to reviewed queue or labeled ground-truth dataset (JSON, JSONL, CSV)
    #[arg(short = 'd', long, alias = "file")]
    pub data: PathBuf,

    /// Path to question schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Destination path for RLCD contrastive dataset (default: data/rlcd_dataset.jsonl)
    #[arg(short = 'o', long, default_value = "data/rlcd_dataset.jsonl")]
    pub output: PathBuf,

    /// Fraction of samples allocated to validation split (default: 0.15)
    #[arg(long, default_value_t = 0.15)]
    pub val_split: f64,

    /// Data provenance source label (default: human_supervisor_resolution)
    #[arg(long, default_value = "human_supervisor_resolution")]
    pub source_label: String,

    /// Model checkpoint (english, multilingual, typed-decisions, auto-best, modernbert-zeroshot, deberta-v3, eurobert, neobert)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Output report format (table, json, compact)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct ActiveStatsArgs {
    /// Dataset file to inspect for uncertainty distributions
    #[arg(short = 'd', long, alias = "file")]
    pub data: PathBuf,

    /// Path to question schema JSON file
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

    /// Output report format (table, json, compact)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct StatsArgs {
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

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentMode {
    Order,
    Temp,
    Checkpoints,
    Adversarial,
}

#[derive(Args, Debug)]
#[command(group(
    clap::ArgGroup::new("source")
        .required(true)
        .args(["hf", "hn", "reddit", "wikipedia"])
))]
pub struct FetchArgs {
    /// Hugging Face dataset identifier (e.g. "ag_news", "fancyzhx/ag_news", "PolyAI/banking77", "imdb")
    #[arg(long, group = "source", value_name = "DATASET")]
    pub hf: Option<String>,

    /// Hacker News feed ("top", "new", "best", "ask", "show")
    #[arg(long, group = "source", value_name = "CATEGORY")]
    pub hn: Option<String>,

    /// Reddit subreddit name to fetch from (e.g. "rust", "machinelearning", "localllama")
    #[arg(long, group = "source", value_name = "SUBREDDIT")]
    pub reddit: Option<String>,

    /// Wikipedia article title or search term (e.g. "Rust_(programming_language)", "Artificial_intelligence")
    #[arg(long, group = "source", value_name = "ARTICLE")]
    pub wikipedia: Option<String>,

    /// Dataset configuration / subset (for multi-subset Hugging Face datasets, default: "default")
    #[arg(long, default_value = "default")]
    pub config: String,

    /// Dataset split to retrieve (default: "train" for Hugging Face)
    #[arg(short, long, default_value = "train")]
    pub split: String,

    /// Maximum number of records or stories to fetch (default: 50)
    #[arg(short = 'n', long, default_value_t = 50)]
    pub limit: usize,

    /// Record or pagination offset (default: 0)
    #[arg(long, default_value_t = 0)]
    pub offset: usize,

    /// Custom text column override
    #[arg(long)]
    pub text_col: Option<String>,

    /// Custom label column override
    #[arg(long)]
    pub label_col: Option<String>,

    /// Automatically infer and generate an aligned Laya question schema from dataset features
    #[arg(long)]
    pub auto_schema: bool,

    /// Optional file destination to save inferred schema JSON (when --auto-schema is set)
    #[arg(long, value_name = "PATH")]
    pub schema_output: Option<PathBuf>,

    /// Write output to file instead of standard output (stdout)
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Output presentation format (jsonl, json, csv, table; default: jsonl for piping)
    #[arg(long, default_value = "jsonl")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct ScrapeArgs {
    /// Target web page URL to scrape (e.g. `https://news.ycombinator.com`)
    #[arg(short = 'u', long, value_name = "URL")]
    pub url: String,

    /// Maximum Unicode character count for scraped content (default: 4000)
    #[arg(long, default_value_t = 4000)]
    pub max_chars: usize,

    /// Request timeout in seconds (default: 15)
    #[arg(long, default_value_t = 15)]
    pub timeout: u64,

    /// Route through Jina Reader (`https://r.jina.ai/<url>`) for headless dynamic markdown rendering
    #[arg(long)]
    pub jina: bool,

    /// Write output to file instead of standard output (stdout)
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Output presentation format (jsonl, json, table; default: jsonl)
    #[arg(long, default_value = "jsonl")]
    pub format: OutputFormat,
}

#[derive(Args, Debug)]
pub struct LearnArgs {
    /// Optional phase number (`1`..`5`), topic slug (`foundations`, `paradigms`, `scaling`, `calibration`, `production`), `all`, `glossary`, `reading`, or `guide`
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
