use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "lemmata",
    about = "High-performance System 1 decision platform for bidirectional encoders (Laya, ModernBERT, DeBERTa-v3, EuroBERT, NeoBERT)",
    version,
    author,
    after_help = "Get started: `lemmata guide quickstart`. Sample data and schemas: `lemmata guide --samples .`"
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
    #[command(
        visible_alias = "run",
        alias = "predict",
        alias = "infer",
        after_help = "Examples:\n  lemmata classify \"I was charged twice\" --labels billing,technical,sales\n  lemmata classify --file data/customer_support.jsonl --schema schema/questions.json --format jsonl\n  lemmata classify -c deberta-v3 --file data/customer_support.jsonl --schema schema/questions.json --cache"
    )]
    Classify(ClassifyArgs),

    /// Evaluate classification accuracy and calibration against ground-truth labels
    #[command(
        visible_alias = "eval",
        after_help = "Examples:\n  lemmata test --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json\n  lemmata test -c deberta-v3 --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --json"
    )]
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
    #[command(
        after_help = "Examples:\n  lemmata ensemble --file data/customer_support.jsonl --schema schema/questions.json\n  lemmata ensemble --checkpoints english,deberta-v3 --file data/customer_support.jsonl --schema schema/questions.json --fit-weights data/system1_golden_benchmark.jsonl"
    )]
    Ensemble(EnsembleArgs),

    /// Show or clear the saved decision cache that `--cache` reads and writes
    #[command(
        after_help = "Examples:\n  lemmata classify --file data/customer_support.jsonl --schema schema/questions.json --cache\n  lemmata cache stats\n  lemmata cache clear"
    )]
    Cache(CacheArgs),

    /// Decide which items to defer to a larger model or a person (a System 1 / System 2 cascade)
    #[command(
        after_help = "Examples:\n  lemmata defer fit --val-data data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --target-accuracy 0.80 --export-policy policy.json\n  lemmata defer run --file data/customer_support.jsonl --schema schema/questions.json --policy policy.json"
    )]
    Defer(DeferArgs),

    /// Verify signed, hash-chained decision receipts
    #[command(
        after_help = "Examples:\n  lemmata classify --file data/customer_support.jsonl --schema schema/questions.json --receipts logs/receipts.jsonl\n  lemmata attest verify --receipts logs/receipts.jsonl"
    )]
    Attest(AttestArgs),

    /// Classify against a label taxonomy: a coarse label first, then a fine label
    #[command(
        after_help = "Examples:\n  lemmata taxonomy show --schema schema/banking_hierarchy.json\n  lemmata taxonomy run --schema schema/banking_hierarchy.json --file data/banking_tickets.jsonl --beam 2"
    )]
    Taxonomy(TaxonomyArgs),

    /// Automatically tune and calibrate temperature parameters per question bucket
    #[command(
        visible_alias = "tune",
        after_help = "Examples:\n  lemmata calibrate -c deberta-v3 --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --persist-to-model"
    )]
    Calibrate(CalibrateArgs),

    /// Fit supervised linear probes on labelled data
    #[command(
        after_help = "Examples:\n  lemmata probe fit -c eurobert --data data/train.jsonl --schema schema/topics.json --output probes/eurobert-topics.json\n  lemmata classify -c eurobert --probe probes/eurobert-topics.json --file data/test.jsonl --schema schema/topics.json"
    )]
    Probe(ProbeArgs),

    /// Full evaluation report for one checkpoint: accuracy, calibration, speed, and escalation
    #[command(
        after_help = "Examples:\n  lemmata report --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json\n  lemmata report --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --format md --output report.md"
    )]
    Report(ReportArgs),

    /// Head-to-head System 1 model arena comparing accuracy, agreement, Cohen's kappa, McNemar's chi2, and speedup
    #[command(
        visible_alias = "arena",
        after_help = "Examples:\n  lemmata compare --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json\n  lemmata compare -c english,english+runs/ft_v1/artifacts --file splits/test.jsonl --schema schema/questions.json"
    )]
    Compare(CompareArgs),

    /// Canonicalize datasets, create leakage-safe splits, and run 5-stage preflight verification
    #[command(
        visible_alias = "data",
        after_help = "Examples:\n  lemmata dataset import --file raw/tickets.csv --schema schema/questions.json --output data/canonical.jsonl\n  lemmata dataset split --file data/canonical.jsonl --schema schema/questions.json --out-dir splits\n  lemmata dataset verify --split-dir splits --schema schema/questions.json"
    )]
    Dataset(DatasetArgs),

    /// Generate external-GPU fine-tuning bundles, verify PEFT artifacts, merge weights, and sync runs
    #[command(
        visible_alias = "ft",
        after_help = "Examples:\n  lemmata finetune prepare --split-dir splits --schema schema/questions.json -c english --method lora --out runs/ft_v1\n  lemmata finetune push --run runs/ft_v1 --remote ubuntu@gpu-box:~/ft/ft_v1 --dry-run\n  lemmata finetune verify --artifact runs/ft_v1/artifacts --split-dir splits\n  lemmata finetune merge --adapter runs/ft_v1/artifacts --output models/laya-ft"
    )]
    Finetune(FinetuneArgs),

    /// Paired base-vs-candidate statistical experimentation with McNemar mid-p, BCa bootstrap CIs, and zero-leakage guards
    #[command(
        visible_alias = "ab",
        after_help = "Examples:\n  lemmata experiment --base english --candidate runs/ft_v1/artifacts --split-dir splits --schema schema/questions.json\n  lemmata experiment --candidate runs/ft_v1/artifacts --file splits/test.jsonl --cal-file splits/cal.jsonl --schema schema/questions.json --format md"
    )]
    Experiment(ExperimentArgs),

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
    #[arg(value_name = "TEXT", conflicts_with_all = ["text", "file", "stdin", "url"])]
    pub positional_text: Option<String>,

    /// Directory to ingest data from (default: data)
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Specific file to ingest (JSON, JSONL, CSV, TXT)
    #[arg(short, long, conflicts_with_all = ["stdin", "text", "url"])]
    pub file: Option<PathBuf>,

    /// Read data streamed from standard input
    #[arg(long, conflicts_with_all = ["text", "url"])]
    pub stdin: bool,

    /// Classify inline text snippet directly from command line
    #[arg(short, long, conflicts_with = "url")]
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

    /// Model checkpoint slug or local checkpoint directory (english, multilingual, typed-decisions, route, modernbert-zeroshot, modernbert-nli, deberta-v3, eurobert, neobert, or `<DIR>`)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Path to a fine-tuned PEFT adapter directory (`adapter_config.json` + `adapter_model.safetensors`) folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,

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

    /// Reuse saved decisions for inputs seen before with the same model and settings (see `lemmata cache`)
    #[arg(long)]
    pub cache: bool,

    /// Print the cache hit and miss counts for this run
    #[arg(long, requires = "cache")]
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

    /// With --taxonomy: escalate an item when its best path probability is below this value
    #[arg(long, default_value_t = 0.50, requires = "taxonomy")]
    pub path_threshold: f64,

    /// With --taxonomy: how many top options to follow at each level (1 is greedy top-down)
    #[arg(long, default_value_t = 2, requires = "taxonomy")]
    pub beam: usize,

    /// Custom NLI hypothesis or MLM verbalizer template (e.g. "This support ticket is about {label}.")
    #[arg(long)]
    pub hypothesis_template: Option<String>,

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate-prefix, head-tail, sliding-window)
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

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate-prefix, head-tail, sliding-window)
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

    /// Model checkpoint slug or local checkpoint directory (english, multilingual, typed-decisions, route, modernbert-zeroshot, modernbert-nli, deberta-v3, eurobert, neobert, or `<DIR>`)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Path to a fine-tuned PEFT adapter directory (`adapter_config.json` + `adapter_model.safetensors`) folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,

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

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate-prefix, head-tail, sliding-window)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Keep only the top-k options by lexical match (hashed word features and a synonym table) before model scoring (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: usize,

    /// Supervised probe from `lemmata probe fit`. It replaces zero-shot scoring for its question
    #[arg(long)]
    pub probe: Option<PathBuf>,

    /// Override hard failure when evaluating a fine-tuned adapter on items that overlap with its training split
    #[arg(long)]
    pub allow_leakage: bool,
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

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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
    /// Comma-separated checkpoints. Default: the installed ones of english, deberta-v3, and modernbert-zeroshot, one per model family
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "models",
        visible_alias = "model"
    )]
    pub checkpoints: Option<String>,

    /// Comma-separated member weights (e.g. "0.4,0.3,0.3"). Default: equal weights
    #[arg(short, long, conflicts_with = "fit_weights")]
    pub weights: Option<String>,

    /// Fit the member weights on this labelled file (lowest log loss of the true label), print them, and use them
    #[arg(long, value_name = "LABELLED_FILE")]
    pub fit_weights: Option<PathBuf>,

    /// Flag a question as disputed when the members' disagreement (mutual information, in nats) exceeds this value
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
    /// Show the saved cache files: one per model fingerprint, with decision counts and sizes
    Stats {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Delete every saved cache file
    Clear,
}

#[derive(Args, Debug)]
pub struct DeferArgs {
    #[command(subcommand)]
    pub action: DeferCommands,
}

#[derive(Subcommand, Debug)]
pub enum DeferCommands {
    /// Choose the loosest risk threshold whose kept items reach a target accuracy with a stated confidence
    Fit(DeferFitArgs),

    /// Keep or defer each item, with a threshold from `defer fit`
    Run(DeferRunArgs),
}

#[derive(Args, Debug)]
pub struct DeferFitArgs {
    /// Labelled validation data (JSON, JSONL, or CSV with ground truth), not the data you test on
    #[arg(short, long, visible_alias = "eval-data", visible_alias = "file")]
    pub val_data: PathBuf,

    /// Path to questions schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Accuracy that the kept items must reach (e.g. 0.95 for 95%)
    #[arg(long, visible_alias = "target-acc", default_value_t = 0.95)]
    pub target_accuracy: f64,

    /// Probability that the kept items reach the target accuracy on new data from the same source
    #[arg(long, default_value_t = 0.95)]
    pub confidence: f64,

    /// System 2 cost per deferred item in USD, for the cost column
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

    /// Write the chosen threshold to this JSON file, for `defer run --policy`
    #[arg(long)]
    pub export_policy: Option<PathBuf>,

    /// Output the fit as JSON
    #[arg(long)]
    pub json: bool,

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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

    /// Keep an item when its risk (1 minus its least certain top probability) is at or below this value
    #[arg(short, long, visible_alias = "threshold", conflicts_with = "policy")]
    pub risk_threshold: Option<f64>,

    /// A policy file from `defer fit --export-policy`, instead of --risk-threshold
    #[arg(long)]
    pub policy: Option<PathBuf>,

    /// System 2 cost per deferred item in USD
    #[arg(long, default_value_t = 0.005)]
    pub system2_cost: f64,

    /// Output presentation format
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Output decisions as JSON
    #[arg(long)]
    pub json: bool,

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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
    /// Path to the taxonomy schema JSON file
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

    /// Escalate an item when its best path probability is below this value
    #[arg(long, default_value_t = 0.50)]
    pub path_threshold: f64,

    /// How many top options to follow at each level (1 is greedy top-down)
    #[arg(long, default_value_t = 2)]
    pub beam: usize,

    /// Output presentation format (table, json, csv, jsonl)
    #[arg(long, default_value = "table")]
    pub format: OutputFormat,

    /// Write output to specified file instead of stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Reuse saved decisions for inputs seen before with the same model and settings (see `lemmata cache`)
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

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct TaxonomyShowArgs {
    /// Path to the taxonomy schema JSON file
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
    /// Path to the taxonomy schema JSON file
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
    /// Path to the taxonomy schema JSON file
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

    /// Escalate an item when its best path probability is below this value
    #[arg(long, default_value_t = 0.50)]
    pub path_threshold: f64,

    /// How many top options to follow at each level (1 is greedy top-down)
    #[arg(long, default_value_t = 2)]
    pub beam: usize,

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

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
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
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Model checkpoint slug or local checkpoint directory (english, multilingual, typed-decisions, route, modernbert-zeroshot, modernbert-nli, deberta-v3, eurobert, neobert, or `<DIR>`)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Path to a fine-tuned PEFT adapter directory (`adapter_config.json` + `adapter_model.safetensors`) folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,

    /// Temperature scaling factor
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Path to custom JSON model/agent config (e.g. calibrated config with temperature & cost_weights)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Custom NLI hypothesis or MLM verbalizer template (e.g. "This support ticket is about {label}.")
    #[arg(long)]
    pub hypothesis_template: Option<String>,

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate-prefix, head-tail, sliding-window)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Keep only the top-k options by lexical match before model scoring (0 = disabled)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: usize,

    /// Supervised probe from `lemmata probe fit`. It replaces zero-shot scoring for its question
    #[arg(long)]
    pub probe: Option<PathBuf>,

    /// Number of benchmark passes for timing stability (default: 3)
    #[arg(short, long, default_value_t = 3)]
    pub iterations: usize,

    /// Number of warmup passes before recording timing metrics (default: 2)
    #[arg(short, long, default_value_t = 2)]
    pub warmup: usize,

    /// Supervisor escalation probability threshold (default: 0.50)
    #[arg(long, default_value_t = 0.50)]
    pub escalate_threshold: f64,

    /// Comma-separated file extensions to ingest from directory
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories
    #[arg(long)]
    pub no_recursive: bool,

    /// Output report format
    #[arg(long, value_enum, default_value_t = ReportFormat::Table)]
    pub format: ReportFormat,

    /// Also write the report to this file: .csv, .md, or JSON for any other extension
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Override hard failure when evaluating a fine-tuned adapter on items that overlap with its training split
    #[arg(long)]
    pub allow_leakage: bool,
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

/// Output formats for `lemmata report`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportFormat {
    Table,
    Json,
    Compact,
    Csv,
    /// Markdown, for a PR description or a model card
    Md,
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

    /// Write the sample data and schemas that the recipes use into DIR/data and DIR/schema. Existing files stay
    #[arg(long, value_name = "DIR")]
    pub samples: Option<PathBuf>,
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
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Long-context adaptation strategy when input exceeds checkpoint window (truncate-prefix, head-tail, sliding-window)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// File to write the probe to
    #[arg(short, long)]
    pub output: PathBuf,

    /// Path to a fine-tuned PEFT adapter directory folded at load time
    #[arg(long, value_name = "DIR")]
    pub adapter: Option<PathBuf>,
}

// ============================================================================
// Dataset Lifecycle Commands (`lemmata dataset` / `lemmata data`)
// ============================================================================

#[derive(Args, Debug)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub action: DatasetCommands,
}

#[derive(Subcommand, Debug)]
pub enum DatasetCommands {
    /// Import and canonicalize CSV, TSV, JSON/JSONL, Parquet, directory, or Hugging Face Hub datasets
    Import(Box<DatasetImportArgs>),

    /// Partition canonical JSONL into deterministic, leakage-safe train/val/cal/test splits
    Split(DatasetSplitArgs),

    /// Run 5-stage preflight verification (schema bijection, MinHash leakage, sample floors, tokenizer budgets)
    Verify(DatasetVerifyArgs),
}

#[derive(Args, Debug)]
pub struct DatasetImportArgs {
    /// Input file path (.jsonl, .json, .csv, .tsv, .parquet)
    #[arg(short, long, conflicts_with_all = ["dir", "hf"])]
    pub file: Option<PathBuf>,

    /// Input directory containing dataset files
    #[arg(long, conflicts_with_all = ["file", "hf"])]
    pub dir: Option<PathBuf>,

    /// Hugging Face Hub dataset repository ID (e.g. `PolyAI/banking77`)
    #[arg(long, value_name = "REPO_ID", conflicts_with_all = ["file", "dir"])]
    pub hf: Option<String>,

    /// Hugging Face dataset subset/configuration name
    #[arg(long)]
    pub subset: Option<String>,

    /// Hugging Face dataset split name (e.g. `train`, `test`)
    #[arg(long)]
    pub split: Option<String>,

    /// Hugging Face Git revision or commit SHA
    #[arg(long)]
    pub revision: Option<String>,

    /// Cache directory for downloaded Hugging Face Parquet shards
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Print the Hugging Face download command without fetching remote files
    #[arg(long)]
    pub dry_run: bool,

    /// Output canonical `.jsonl` file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Optional question schema JSON file for label domain & integer ClassLabel validation
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Explicit record ID column name
    #[arg(long)]
    pub id_col: Option<String>,

    /// Explicit model-visible text column name
    #[arg(long)]
    pub text_col: Option<String>,

    /// Comma-separated structured state column names
    #[arg(long)]
    pub state_cols: Option<String>,

    /// Comma-separated non-model metadata column names (excluded from encoder input text)
    #[arg(long)]
    pub meta_cols: Option<String>,

    /// Repeatable `<question_id>=<source_column>` ground-truth column mapping
    #[arg(long, value_name = "QID=COL")]
    pub label_map: Vec<String>,

    /// Repeatable `<question_id>:<raw>=<canonical>` label value remapping
    #[arg(long, value_name = "QID:RAW=CANON")]
    pub value_map: Vec<String>,

    /// Map integer ClassLabel indices (`0..K-1`) to schema choice keys in schema order
    #[arg(long)]
    pub int_labels_from_schema: bool,

    /// Additive offset applied to numeric `score` targets (e.g. `-1` for `1..5 -> 0..4`)
    #[arg(long, default_value_t = 0)]
    pub score_offset: i64,

    /// Delimiter character splitting string multi-label targets into arrays (e.g. `;` or `|`)
    #[arg(long)]
    pub multi_label_sep: Option<char>,

    /// Normalize line endings, strip UTF-8 BOM/NULs, and compose decomposed accents
    #[arg(long)]
    pub normalize_nfc: bool,

    /// Strip quoted email reply headers, `>` quote lines, and legal disclaimers
    #[arg(long)]
    pub clean_email: bool,

    /// Drop records whose model-visible state is blank
    #[arg(long)]
    pub drop_empty: bool,

    /// Fail immediately on malformed rows, duplicate IDs, or out-of-domain schema labels
    #[arg(long)]
    pub strict: bool,

    /// Comma-separated file extensions when importing from `--dir`
    #[arg(long)]
    pub extensions: Option<String>,

    /// Do not recursively traverse subdirectories when importing from `--dir`
    #[arg(long)]
    pub no_recursive: bool,

    /// Emit import summary report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct DatasetSplitArgs {
    /// Input dataset file (.jsonl, .json, .csv, .tsv, .parquet)
    #[arg(short, long, visible_alias = "data", visible_alias = "input")]
    pub file: PathBuf,

    /// Question schema JSON file used for multi-label iterative stratification
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Output split directory where `{train,val,cal,test}.jsonl` and `dataset_manifest.json` are written
    #[arg(
        short,
        long,
        visible_alias = "output",
        visible_alias = "out",
        default_value = "splits"
    )]
    pub out_dir: PathBuf,

    /// Comma-separated split ratios `train,val,cal,test` (or 3-way `train,val,test`)
    #[arg(long, default_value = "0.70,0.10,0.10,0.10")]
    pub ratios: String,

    /// Omit the `cal.jsonl` split (`0.80,0.10,0.0,0.10` when `--ratios` is at default)
    #[arg(long)]
    pub no_cal: bool,

    /// Deterministic PRNG seed for Xoshiro256** and MinHash LSH permutations
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Metadata or state column locking records (e.g. `customer_id`, `thread_id`) into the same split
    #[arg(long)]
    pub group_col: Option<String>,

    /// MinHash 5-gram Jaccard similarity threshold in `(0.0, 1.0]` for near-duplicate grouping
    #[arg(long, default_value_t = 0.85)]
    pub dedup_sim: f64,

    /// Exact duplicate handling before splitting: `group` or `drop-exact`
    #[arg(long, default_value = "group")]
    pub dedup_action: String,

    /// Emit `dataset_manifest.json` summary as JSON to stdout
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct DatasetVerifyArgs {
    /// Split directory containing `{train,val,cal,test}.jsonl` and optional `dataset_manifest.json`
    #[arg(long, visible_alias = "dir", conflicts_with = "file")]
    pub split_dir: Option<PathBuf>,

    /// Single dataset file to verify prior to splitting
    #[arg(short, long, visible_alias = "data", conflicts_with = "split_dir")]
    pub file: Option<PathBuf>,

    /// Question schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Target checkpoint for context-window and head-budget preflight checks
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Long-context chunking strategy (`truncate`, `head-tail`, `sliding-window`)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Validate ground-truth targets under multi-label semantics
    #[arg(long)]
    pub multi_label: bool,

    /// Target conformal error rate `alpha` for calibration sample-size checks
    #[arg(long, default_value_t = 0.05)]
    pub alpha: f64,

    /// Target deferral accuracy for calibration sample-size checks
    #[arg(long, default_value_t = 0.95)]
    pub target_accuracy: f64,

    /// Deferral certification confidence level
    #[arg(long, default_value_t = 0.95)]
    pub confidence: f64,

    /// Cross-split MinHash Jaccard similarity threshold
    #[arg(long, default_value_t = 0.85)]
    pub dedup_sim: f64,

    /// Allow unlabeled or partially labeled items without failing Stage 1
    #[arg(long)]
    pub allow_partial_gt: bool,

    /// Require local Hugging Face `tokenizer.json` instead of heuristic token estimation
    #[arg(long)]
    pub require_tokenizer: bool,

    /// Treat preflight warnings as hard failures
    #[arg(long)]
    pub strict: bool,

    /// Emit preflight verification report as JSON
    #[arg(long)]
    pub json: bool,
}

// ============================================================================
// External-GPU Fine-Tuning Commands (`lemmata finetune` / `lemmata ft`)
// ============================================================================

#[derive(Args, Debug)]
pub struct FinetuneArgs {
    #[command(subcommand)]
    pub action: FinetuneCommands,
}

#[derive(Subcommand, Debug)]
pub enum FinetuneCommands {
    /// Validate splits & schema locally and generate a self-bootstrapping PEP 723 `uv run` GPU bundle
    #[command(visible_alias = "bundle")]
    Prepare(Box<FinetunePrepareArgs>),

    /// Run 5-layer local integrity & zero-leakage verification on a pulled fine-tuning artifact
    Verify(FinetuneVerifyArgs),

    /// Merge a PEFT LoRA/DoRA adapter into a standalone local Safetensors checkpoint in Rust
    Merge(FinetuneMergeArgs),

    /// Push a lightweight fine-tuning bundle (`bundle/`) to an SSH host or S3 URI
    Push(FinetuneSyncArgs),

    /// Pull adapter weights, calibration, and manifests back from an SSH host or S3 URI
    Pull(FinetuneSyncArgs),
}

#[derive(Args, Debug)]
pub struct FinetunePrepareArgs {
    /// Split directory containing `train.jsonl`, `val.jsonl`, optional `cal.jsonl`, and locked `test.jsonl`
    #[arg(long, visible_alias = "splits")]
    pub split_dir: Option<PathBuf>,

    /// Explicit training split JSONL/CSV file (when not using `--split-dir`)
    #[arg(long)]
    pub train: Option<PathBuf>,

    /// Explicit validation split JSONL/CSV file (when not using `--split-dir`)
    #[arg(long)]
    pub val: Option<PathBuf>,

    /// Optional post-training temperature calibration split file
    #[arg(long)]
    pub cal: Option<PathBuf>,

    /// Locked evaluation test split file to guard against overlap (NEVER copied into `bundle/`)
    #[arg(long)]
    pub forbid_test: Option<PathBuf>,

    /// Question schema JSON file
    #[arg(short, long)]
    pub schema: PathBuf,

    /// Base checkpoint to fine-tune (`english`, `multilingual`, `typed-decisions`, `modernbert-zeroshot`, `modernbert-nli`, `deberta-v3`, `eurobert`, `neobert`)
    #[arg(
        short,
        long,
        short_alias = 'm',
        visible_alias = "model",
        default_value = "english"
    )]
    pub checkpoint: String,

    /// Fine-tuning adaptation strategy: `lora` (rsLoRA), `dora`, or `full`
    #[arg(long, default_value = "lora")]
    pub method: String,

    /// LoRA / DoRA rank `r`
    #[arg(short = 'r', long, default_value_t = 16)]
    pub rank: usize,

    /// LoRA / DoRA scaling numerator `lora_alpha`
    #[arg(long, default_value_t = 32.0)]
    pub alpha: f64,

    /// Disable rank-stabilized scaling (`alpha / sqrt(r)`) and use classic `alpha / r`
    #[arg(long)]
    pub no_rslora: bool,

    /// LoRA dropout probability
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f64,

    /// Fully unfreeze Laya's `head.layers.*` (`in_proj_weight`) via `modules_to_save` alongside `type_emb` + `scorer`
    #[arg(long)]
    pub unfreeze_laya_head: bool,

    /// Number of training epochs
    #[arg(long, default_value_t = 4)]
    pub epochs: usize,

    /// Explicit encoder learning rate override (defaults to family-tuned 2026 recipe)
    #[arg(long)]
    pub lr: Option<f64>,

    /// Per-device micro-batch size
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Gradient accumulation steps
    #[arg(long, default_value_t = 2)]
    pub grad_accum: usize,

    /// Mild cross-entropy label smoothing (`0.0` to disable; Keep small to avoid post-hoc ECE inflation)
    #[arg(long, default_value_t = 0.02)]
    pub label_smoothing: f64,

    /// NEFTune uniform embedding noise `alpha` (`0.0` to disable)
    #[arg(long, default_value_t = 5.0)]
    pub neftune_alpha: f64,

    /// Optional R-Drop symmetric KL regularization weight (`0.0` disabled; e.g. `0.5` for `<2k` rows)
    #[arg(long, default_value_t = 0.0)]
    pub rdrop_alpha: f64,

    /// Train `choice` questions with independent `BCEWithLogitsLoss` for multi-label classification
    #[arg(long)]
    pub multi_label: bool,

    /// PyTorch wheel index target written into PEP 723 metadata (`cu126`, `cu128`, or `cpu`)
    #[arg(long, default_value = "cu126")]
    pub cuda: String,

    /// Random seed for reproducible training
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Also emit a single-file `<out>/bundle.tar.gz` archive
    #[arg(long)]
    pub tar: bool,

    /// Output run directory where `bundle/` is created
    #[arg(
        short,
        long,
        visible_alias = "out-dir",
        visible_alias = "output",
        default_value = "runs/ft_latest"
    )]
    pub out: PathBuf,

    /// Emit bundle summary as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct FinetuneVerifyArgs {
    /// Pulled fine-tuning `artifacts/` directory (containing `training_manifest.json` & `adapter_model.safetensors`)
    #[arg(long, visible_alias = "adapter", visible_alias = "dir")]
    pub artifact: PathBuf,

    /// Optional local split directory to cross-verify `train`/`val`/`cal` digests and `test.jsonl` zero-leakage
    #[arg(long, visible_alias = "splits")]
    pub split_dir: Option<PathBuf>,

    /// Explicit evaluation test file to check against `trained_inputs.sha256.bin`
    #[arg(short = 'f', long, visible_alias = "file")]
    pub test_file: Option<PathBuf>,

    /// Optional question schema JSON file to verify `schema_sha256` parity
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Optional local base checkpoint directory to verify Safetensors header SHA-256 parity
    #[arg(long)]
    pub base_dir: Option<PathBuf>,

    /// Downgrade train/test overlap from hard error to warning
    #[arg(long)]
    pub allow_leakage: bool,

    /// Emit verification report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct FinetuneMergeArgs {
    /// Fine-tuned PEFT adapter directory (`adapter_config.json` + `adapter_model.safetensors`)
    #[arg(long, visible_alias = "artifact")]
    pub adapter: PathBuf,

    /// Optional base checkpoint slug or local directory (auto-inferred from adapter metadata if omitted)
    #[arg(short, long, short_alias = 'm', visible_alias = "model")]
    pub checkpoint: Option<String>,

    /// Destination directory for standalone merged `model.safetensors` + configs + tokenizer
    #[arg(short, long, visible_alias = "out-dir", visible_alias = "out")]
    pub output: PathBuf,

    /// Overwrite destination directory if `model.safetensors` already exists
    #[arg(long)]
    pub force: bool,

    /// Upcast merged Safetensors weights to F32 instead of preserving base precision
    #[arg(long)]
    pub f32: bool,

    /// Emit merge report as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct FinetuneSyncArgs {
    /// Local fine-tuning run directory (containing `bundle/` and `artifacts/`)
    #[arg(long, visible_alias = "dir")]
    pub run: PathBuf,

    /// Remote SSH destination (`user@host:~/path`) or S3 URI (`s3://bucket/prefix`)
    #[arg(long)]
    pub remote: String,

    /// Print the `rsync` or `aws s3 sync` command without executing it
    #[arg(long)]
    pub dry_run: bool,
}

// ============================================================================
// Paired Model Experimentation (`lemmata experiment` / `lemmata ab`)
// ============================================================================

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExperimentFormat {
    Table,
    Json,
    #[value(alias = "markdown")]
    Md,
}

#[derive(Args, Debug)]
pub struct ExperimentArgs {
    /// Base model checkpoint slug, local checkpoint dir, or `base+adapter` spec (auto-inferred from `--candidate` if omitted)
    #[arg(short, long)]
    pub base: Option<String>,

    /// Candidate model(s): PEFT adapter dir (`runs/ft1/artifacts`), run dir (`runs/ft1`), `base+adapter`, or local checkpoint dir
    #[arg(
        short = 'k',
        long,
        visible_alias = "adapter",
        required = true,
        num_args = 1..
    )]
    pub candidate: Vec<String>,

    /// Split directory convenience flag (auto-resolves `<DIR>/test.jsonl`, `<DIR>/cal.jsonl`, and `<DIR>/dataset_manifest.json`)
    #[arg(long, visible_alias = "splits")]
    pub split_dir: Option<PathBuf>,

    /// Locked test split file (`.jsonl`, `.json`, `.csv`, `.parquet`)
    #[arg(short, long, visible_alias = "test-data", visible_alias = "test-file")]
    pub file: Option<PathBuf>,

    /// Disjoint calibration split file for fair post-hoc temperature, conformal, and deferral fitting
    #[arg(long, visible_alias = "cal-data")]
    pub cal_file: Option<PathBuf>,

    /// Explicit `dataset_manifest.json` path for cryptographic split verification
    #[arg(long)]
    pub dataset_manifest: Option<PathBuf>,

    /// Question schema JSON file
    #[arg(short, long)]
    pub schema: Option<PathBuf>,

    /// Calibration evaluation protocol: `raw`, `calibrated`, or `both` (default: `both`)
    #[arg(long, default_value = "both")]
    pub calibration_mode: String,

    /// Number of paired cluster-bootstrap resamples for 95% BCa confidence intervals
    #[arg(long, default_value_t = 2000)]
    pub bootstrap_samples: usize,

    /// Number of paired sign-flip permutations for exact Phipson-Smyth p-values
    #[arg(long, default_value_t = 5000)]
    pub permutations: usize,

    /// Confidence level for bootstrap CIs and deferral bounds
    #[arg(long, default_value_t = 0.95)]
    pub confidence: f64,

    /// Significance threshold `alpha` for hypothesis tests and conformal sets
    #[arg(long, default_value_t = 0.05)]
    pub alpha: f64,

    /// Target accuracy for System 1 / System 2 deferral coverage comparison
    #[arg(long, default_value_t = 0.95)]
    pub target_accuracy: f64,

    /// System 2 fallback cost per deferred item in USD
    #[arg(long, default_value_t = 0.005)]
    pub system2_cost: f64,

    /// Deterministic PRNG seed for bootstrap and permutation resamples
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Disaggregate paired accuracy deltas by diagnostic slice (`auto`, `prefix`, `script`, `length`, or `field:<name>`)
    #[arg(long)]
    pub slice_by: Option<String>,

    /// Warmup passes before ABBA interleaved latency timing
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// ABBA interleaved timing passes over `test`
    #[arg(short, long, default_value_t = 1)]
    pub iterations: usize,

    /// Long-context chunking strategy (`truncate`, `head-tail`, `sliding-window`)
    #[arg(long, default_value = "truncate")]
    pub chunk_strategy: String,

    /// Custom NLI hypothesis or MLM verbalizer template
    #[arg(long)]
    pub hypothesis_template: Option<String>,

    /// Lexical option shortlist size (`0` = disabled)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: usize,

    /// Enable multi-label sigmoid scoring for both models
    #[arg(long)]
    pub multi_label: bool,

    /// Multi-label sigmoid probability threshold
    #[arg(long, default_value_t = 0.40)]
    pub multi_label_threshold: f64,

    /// Override hard failure when train/val/cal leakage is detected in `test`
    #[arg(long)]
    pub allow_leakage: bool,

    /// Exit with non-zero status if any candidate receives a `REGRESSED` verdict
    #[arg(long)]
    pub fail_on_regression: bool,

    /// Output format (`table`, `md`, or `json`)
    #[arg(long, value_enum, default_value_t = ExperimentFormat::Table)]
    pub format: ExperimentFormat,

    /// Export paired experiment report to file (`.md` for Markdown, `.json` for JSON)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Emit paired experiment report as JSON to stdout
    #[arg(long)]
    pub json: bool,
}
