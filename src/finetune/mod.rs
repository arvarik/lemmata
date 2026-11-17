//! External-GPU fine-tuning bundle generator, 5-layer artifact & leakage verifier,
//! standalone PEFT merger, and zero-crate `rsync` / `aws s3` transport helpers.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::datasource::{DataItem, DataSource, FileSource};
use crate::engine::compute_schema_hash;
use crate::engine::metrics::resolve_ground_truth_for_question;
use crate::model::{
    Checkpoint, MergeReport, ModelSpec, Question, TRAINING_MANIFEST_VERSION, TrainingManifest,
    merge_adapter_to_dir, safetensors_header_sha256, sha256_file,
};

/// Embedded PEP 723 self-bootstrapping training script (`uv run train.py`).
pub const TRAIN_PY_TEMPLATE: &str = include_str!("templates/train.py");

/// Binary magic header for `trained_inputs.sha256.bin` (shared with `engine::dataset`).
pub const FILTER_MAGIC: &[u8; 8] = crate::engine::dataset::TRAINED_FILTER_MAGIC;
const LEGACY_FILTER_MAGIC: &[u8; 8] = b"LMFTFLT1";

/// Fine-tuning adaptation strategy on the external GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FinetuneMethod {
    /// Rank-stabilized Low-Rank Adaptation (`LoRA` / `rsLoRA`).
    Lora,
    /// Weight-Decomposed Low-Rank Adaptation (`DoRA`).
    Dora,
    /// Full parameter fine-tuning of encoder + classification/decision head.
    Full,
}

impl FinetuneMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::Dora => "dora",
            Self::Full => "full",
        }
    }
}

impl FromStr for FinetuneMethod {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lora" | "rslora" => Ok(Self::Lora),
            "dora" => Ok(Self::Dora),
            "full" | "fft" => Ok(Self::Full),
            other => bail!("Unknown fine-tuning method '{other}'. Supported: lora, dora, full"),
        }
    }
}

/// PyTorch wheel index target written into the PEP 723 `uv run` header of `train.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CudaBackend {
    /// CUDA 12.6 wheels (supports AWS T4 `g4dn`, A10G `g5`, L4 `g6`, L40S `g6e`, A100 `p4d`, H100 `p5`).
    Cu126,
    /// CUDA 12.8 wheels (supports Ada/Hopper/Blackwell `B200`/`p6`).
    Cu128,
    /// CPU-only PyTorch wheels (for local smoke testing).
    Cpu,
}

impl CudaBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cu126 => "cu126",
            Self::Cu128 => "cu128",
            Self::Cpu => "cpu",
        }
    }
}

impl FromStr for CudaBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cu126" | "cuda126" | "12.6" => Ok(Self::Cu126),
            "cu128" | "cuda128" | "12.8" => Ok(Self::Cu128),
            "cpu" => Ok(Self::Cpu),
            other => bail!("Unknown CUDA wheel target '{other}'. Supported: cu126, cu128, cpu"),
        }
    }
}

fn canonical_sorted_json_value(val: &serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::Object(map) => {
            let mut sorted: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            sorted.sort_by_key(|(a, _)| *a);
            let mut out = serde_json::Map::with_capacity(sorted.len());
            for (k, v) in sorted {
                out.insert(k.clone(), canonical_sorted_json_value(v));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonical_sorted_json_value).collect())
        }
        other => other.clone(),
    }
}

/// Canonical compact JSON bytes with recursively sorted object keys (matching Python
/// `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`).
pub fn canonical_json_string<T: Serialize>(value: &T) -> Result<String> {
    let raw = serde_json::to_value(value)?;
    let sorted = canonical_sorted_json_value(&raw);
    Ok(serde_json::to_string(&sorted)?)
}

/// Extract model-visible state text from a `serde_json::Value`, unwrapping single-key
/// `{"text": "..."}` wrappers and preserving JSON object insertion order (`preserve_order`)
/// identically to `DataItem::state_text()` and `train.py::extract_state_text`.
pub fn canonical_value_state_text(state: &serde_json::Value) -> String {
    crate::datasource::traits::state_to_text(state).into_owned()
}

/// Extract canonical state text from a `DataItem`.
pub fn canonical_item_state_text(item: &DataItem) -> String {
    canonical_value_state_text(&item.state)
}

/// Normalize state text (BOM/NUL stripped, Unicode lowercased, whitespace-collapsed)
/// identically across `dataset`, `finetune`, `train.py`, and `paired_eval`.
pub fn normalize_state_text(text: &str) -> String {
    crate::engine::dataset::normalize_state_for_leakage(text)
}

/// Compute the little-endian `u64` prefix of `SHA-256(text.as_bytes())`.
pub fn sha256_u64_le(text: &str) -> u64 {
    let digest = Sha256::digest(text.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

/// Sorted binary filter (`trained_inputs.sha256.bin`) storing exact and normalized `u64`
/// SHA-256 prefixes of all `train ∪ val ∪ cal` state texts for sub-millisecond leakage checks.
#[derive(Debug, Clone, Default)]
pub struct TrainedInputsFilter {
    pub exact_hashes: Vec<u64>,
    pub norm_hashes: Vec<u64>,
}

/// Result of checking evaluation items against `TrainedInputsFilter`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterLeakageCheck {
    pub tested_items: usize,
    pub exact_leaked_ids: Vec<String>,
    pub normalized_leaked_ids: Vec<String>,
}

impl FilterLeakageCheck {
    pub fn has_leakage(&self) -> bool {
        !self.exact_leaked_ids.is_empty() || !self.normalized_leaked_ids.is_empty()
    }
}

impl TrainedInputsFilter {
    /// Build a deduplicated, sorted filter over `items`.
    pub fn from_items(items: &[DataItem]) -> Self {
        let mut exact = BTreeSet::new();
        let mut norm = BTreeSet::new();
        for item in items {
            let st = canonical_item_state_text(item);
            let nm = normalize_state_text(&st);
            exact.insert(sha256_u64_le(&st));
            if !nm.is_empty() {
                norm.insert(sha256_u64_le(&nm));
            }
        }
        Self {
            exact_hashes: exact.into_iter().collect(),
            norm_hashes: norm.into_iter().collect(),
        }
    }

    /// Merge another filter into `self`, keeping both arrays sorted and deduplicated.
    pub fn extend_from(&mut self, other: &Self) {
        let exact: BTreeSet<u64> = self
            .exact_hashes
            .iter()
            .chain(&other.exact_hashes)
            .copied()
            .collect();
        let norm: BTreeSet<u64> = self
            .norm_hashes
            .iter()
            .chain(&other.norm_hashes)
            .copied()
            .collect();
        self.exact_hashes = exact.into_iter().collect();
        self.norm_hashes = norm.into_iter().collect();
    }

    /// Serialize to `trained_inputs.sha256.bin` (`LMFTFLT1` header + sorted `u64` arrays).
    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("bin.tmp");
        {
            let mut f = std::io::BufWriter::new(File::create(&tmp)?);
            f.write_all(FILTER_MAGIC)?;
            f.write_all(&(self.exact_hashes.len() as u64).to_le_bytes())?;
            f.write_all(&(self.norm_hashes.len() as u64).to_le_bytes())?;
            for &v in &self.exact_hashes {
                f.write_all(&v.to_le_bytes())?;
            }
            for &v in &self.norm_hashes {
                f.write_all(&v.to_le_bytes())?;
            }
            f.flush()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load `trained_inputs.sha256.bin` from disk.
    pub fn read(path: &Path) -> Result<Self> {
        let mut buf = Vec::new();
        File::open(path)
            .with_context(|| format!("Failed to open leakage filter {}", path.display()))?
            .read_to_end(&mut buf)?;

        if buf.len() >= 24 && (&buf[..8] == FILTER_MAGIC || &buf[..8] == LEGACY_FILTER_MAGIC) {
            let n_exact = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
            let n_norm = u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize;
            let expected_bytes = 24usize
                .checked_add(
                    n_exact
                        .checked_add(n_norm)
                        .and_then(|n| n.checked_mul(8))
                        .ok_or_else(|| anyhow!("Overflow in filter header counts"))?,
                )
                .ok_or_else(|| anyhow!("Overflow in filter byte size"))?;
            if buf.len() != expected_bytes {
                bail!(
                    "Corrupted {}: expected {} bytes from header, got {}",
                    path.display(),
                    expected_bytes,
                    buf.len()
                );
            }
            let mut exact_hashes = Vec::with_capacity(n_exact);
            let mut cursor = 24;
            for _ in 0..n_exact {
                exact_hashes.push(u64::from_le_bytes(
                    buf[cursor..cursor + 8].try_into().unwrap(),
                ));
                cursor += 8;
            }
            let mut norm_hashes = Vec::with_capacity(n_norm);
            for _ in 0..n_norm {
                norm_hashes.push(u64::from_le_bytes(
                    buf[cursor..cursor + 8].try_into().unwrap(),
                ));
                cursor += 8;
            }
            exact_hashes.sort_unstable();
            norm_hashes.sort_unstable();
            Ok(Self {
                exact_hashes,
                norm_hashes,
            })
        } else if buf.len().is_multiple_of(8) {
            let count = buf.len() / 8;
            let mut hashes = Vec::with_capacity(count);
            for i in 0..count {
                hashes.push(u64::from_le_bytes(
                    buf[i * 8..(i + 1) * 8].try_into().unwrap(),
                ));
            }
            hashes.sort_unstable();
            Ok(Self {
                exact_hashes: hashes.clone(),
                norm_hashes: hashes,
            })
        } else {
            bail!(
                "Invalid binary leakage filter {} (size {} bytes)",
                path.display(),
                buf.len()
            )
        }
    }

    /// Binary-search every item in `items` against `exact_hashes` and `norm_hashes` in `<1 ms`.
    pub fn check_items(&self, items: &[DataItem]) -> FilterLeakageCheck {
        let mut exact_leaked_ids = Vec::new();
        let mut normalized_leaked_ids = Vec::new();
        for item in items {
            let st = canonical_item_state_text(item);
            let ex_h = sha256_u64_le(&st);
            if self.exact_hashes.binary_search(&ex_h).is_ok() {
                exact_leaked_ids.push(item.id.clone());
                continue;
            }
            let nm_h = sha256_u64_le(&normalize_state_text(&st));
            if self.norm_hashes.binary_search(&nm_h).is_ok() {
                normalized_leaked_ids.push(item.id.clone());
            }
        }
        FilterLeakageCheck {
            tested_items: items.len(),
            exact_leaked_ids,
            normalized_leaked_ids,
        }
    }
}

/// Recommended PEFT architecture defaults for a `Checkpoint`.
#[derive(Debug, Clone)]
pub struct ParadigmPeftDefaults {
    pub paradigm: &'static str,
    pub target_modules: Vec<String>,
    pub modules_to_save: Vec<String>,
    pub default_lr_lora: f64,
    pub default_lr_head: f64,
}

/// Return the native paradigm identifier and exact PEFT target modules for `ckpt`.
pub fn peft_defaults_for(
    ckpt: Checkpoint,
    unfreeze_laya_head: bool,
) -> Result<ParadigmPeftDefaults> {
    let s = |list: &[&str]| list.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
    match ckpt {
        Checkpoint::English | Checkpoint::Multilingual | Checkpoint::TypedDecisions => {
            let (targets, mts) = if unfreeze_laya_head {
                (
                    s(&["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]),
                    s(&["type_emb", "head", "scorer"]),
                )
            } else {
                (
                    s(&[
                        "attn.Wqkv",
                        "attn.Wo",
                        "mlp.Wi",
                        "mlp.Wo",
                        "self_attn.out_proj",
                        "linear1",
                        "linear2",
                    ]),
                    s(&["type_emb", "scorer"]),
                )
            };
            Ok(ParadigmPeftDefaults {
                paradigm: "laya_joint_decision",
                target_modules: targets,
                modules_to_save: mts,
                default_lr_lora: 1.5e-4,
                default_lr_head: 1.0e-4,
            })
        }
        Checkpoint::ModernBertZeroShot | Checkpoint::ModernBertNli => Ok(ParadigmPeftDefaults {
            paradigm: "nli_cross_encoder",
            target_modules: s(&["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]),
            modules_to_save: s(&["head", "classifier"]),
            default_lr_lora: 1.5e-4,
            default_lr_head: 1.0e-4,
        }),
        Checkpoint::DebertaV3ZeroShot => Ok(ParadigmPeftDefaults {
            paradigm: "nli_cross_encoder",
            target_modules: s(&[
                "query_proj",
                "key_proj",
                "value_proj",
                "pos_key_proj",
                "pos_query_proj",
                "dense",
            ]),
            modules_to_save: s(&["pooler", "classifier"]),
            default_lr_lora: 1.0e-4,
            default_lr_head: 1.0e-4,
        }),
        Checkpoint::EuroBert => Ok(ParadigmPeftDefaults {
            paradigm: "mlm_cloze",
            target_modules: s(&[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]),
            modules_to_save: Vec::new(),
            default_lr_lora: 2.0e-4,
            default_lr_head: 1.0e-4,
        }),
        Checkpoint::NeoBert => Ok(ParadigmPeftDefaults {
            paradigm: "mlm_cloze",
            target_modules: s(&["qkv", "wo", "w12", "w3"]),
            modules_to_save: Vec::new(),
            default_lr_lora: 2.0e-4,
            default_lr_head: 1.0e-4,
        }),
        Checkpoint::AutoRoute => bail!(
            "Cannot fine-tune meta-router 'route' directly. Pick a concrete checkpoint such as 'english', 'multilingual', 'typed-decisions', 'modernbert-zeroshot', 'deberta-v3', 'eurobert', or 'neobert'."
        ),
    }
}

/// Options for `lemmata finetune prepare`.
#[derive(Debug, Clone)]
pub struct PrepareOptions {
    pub split_dir: Option<PathBuf>,
    pub train_file: Option<PathBuf>,
    pub val_file: Option<PathBuf>,
    pub cal_file: Option<PathBuf>,
    pub forbid_test_file: Option<PathBuf>,
    pub schema_path: PathBuf,
    pub checkpoint: Checkpoint,
    pub method: FinetuneMethod,
    pub rank: usize,
    pub alpha: f64,
    pub use_rslora: bool,
    pub lora_dropout: f64,
    pub unfreeze_laya_head: bool,
    pub epochs: usize,
    pub lr: Option<f64>,
    pub batch_size: usize,
    pub grad_accum: usize,
    pub label_smoothing: f64,
    pub neftune_alpha: f64,
    pub rdrop_alpha: f64,
    pub multi_label: bool,
    pub cuda: CudaBackend,
    pub seed: u64,
    pub create_tar: bool,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleSplitEntry {
    pub rel_path: String,
    pub rows: usize,
    pub file_sha256: String,
    pub state_merkle_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub manifest_version: String,
    pub run_id: String,
    pub created_by_lemmata_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_id: Option<String>,
    pub base_model: serde_json::Value,
    pub schema: serde_json::Value,
    pub splits: BTreeMap<String, BundleSplitEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_test_file_sha256: Option<String>,
    pub method: serde_json::Value,
    pub hyperparameters: serde_json::Value,
}

/// Summary returned by `prepare_bundle`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareReport {
    pub run_id: String,
    pub out_dir: PathBuf,
    pub bundle_dir: PathBuf,
    pub tar_path: Option<PathBuf>,
    pub checkpoint: String,
    pub paradigm: String,
    pub method: String,
    pub train_rows: usize,
    pub val_rows: usize,
    pub cal_rows: usize,
    pub bundle_bytes: u64,
    pub warnings: Vec<String>,
}

fn compute_state_merkle(items: &[DataItem]) -> String {
    let mut digests: Vec<String> = items
        .iter()
        .map(|it| {
            let st = canonical_item_state_text(it);
            Sha256::digest(st.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        })
        .collect();
    digests.sort_unstable();
    Sha256::digest(digests.join("\n").as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn write_canonical_jsonl(items: &[DataItem], dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = std::io::BufWriter::new(File::create(dst)?);
    for item in items {
        serde_json::to_writer(&mut w, item)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

fn copy_or_write_jsonl(src: &Path, items: &[DataItem], dst: &Path) -> Result<()> {
    if src
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"))
    {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(src, dst)?;
        Ok(())
    } else {
        write_canonical_jsonl(items, dst)
    }
}

fn render_train_py(cuda: CudaBackend) -> String {
    match cuda {
        CudaBackend::Cu126 => TRAIN_PY_TEMPLATE.to_string(),
        CudaBackend::Cu128 => TRAIN_PY_TEMPLATE.replace(
            "https://download.pytorch.org/whl/cu126",
            "https://download.pytorch.org/whl/cu128",
        ),
        CudaBackend::Cpu => TRAIN_PY_TEMPLATE.replace(
            r#"{ index = "pytorch-cuda", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }"#,
            r#"{ index = "pytorch-cpu", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }"#,
        ),
    }
}

fn render_run_md(run_id: &str, ckpt: Checkpoint, method: FinetuneMethod, out_dir: &Path) -> String {
    let out_disp = out_dir.display();
    format!(
        "# Lemmata Fine-Tuning Bundle: `{run_id}`\n\n\
         - **Base Checkpoint**: `{repo}` (`{backbone}`)\n\
         - **Method**: `{method}`\n\
         - **Safety Guarantee**: `test.jsonl` is NEVER included in this bundle.\n\n\
         ## 1. Local Preflight / Smoke Test (Optional)\n\
         ```bash\n\
         python3 {out_disp}/bundle/train.py --dry-validate\n\
         uv run {out_disp}/bundle/train.py --smoke-test --cpu\n\
         ```\n\n\
         ## 2. Run on External AWS GPU (`g6.xlarge` L4 / `g5.xlarge` A10G / `g6e.xlarge` L40S)\n\
         ```bash\n\
         # Push lightweight bundle (~1-10 MB) to GPU instance:\n\
         lemmata finetune push --run {out_disp} --remote ubuntu@ec2-gpu:~/ft/{run_id}\n\n\
         # On the GPU instance (zero virtualenv/conda setup required):\n\
         curl -LsSf https://astral.sh/uv/install.sh | sh\n\
         cd ~/ft/{run_id}/bundle && uv run train.py\n\
         # (On AWS Spot interruption, resume seamlessly with: `uv run train.py --resume`)\n\
         ```\n\n\
         ## 3. Pull Adapter Artifacts Back & Run Paired Evaluation\n\
         ```bash\n\
         lemmata finetune pull --run {out_disp} --remote ubuntu@ec2-gpu:~/ft/{run_id}\n\
         lemmata finetune verify --artifact {out_disp}/artifacts\n\
         lemmata classify --adapter {out_disp}/artifacts \"Sample text\" --schema {out_disp}/bundle/schema.json\n\
         ```\n",
        repo = ckpt.repo_id(),
        backbone = ckpt.backbone(),
        method = method.as_str(),
    )
}

fn dir_files_bytes_recursive(dir: &Path) -> u64 {
    let mut sum = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                sum += dir_files_bytes_recursive(&p);
            } else if let Ok(m) = entry.metadata() {
                sum += m.len();
            }
        }
    }
    sum
}

/// Validate dataset splits, check zero train/test overlap, and generate a portable GPU bundle.
pub fn prepare_bundle(opts: &PrepareOptions) -> Result<PrepareReport> {
    let defaults = peft_defaults_for(opts.checkpoint, opts.unfreeze_laya_head)?;
    if opts.rank == 0 && opts.method != FinetuneMethod::Full {
        bail!("LoRA/DoRA rank (--rank) must be >= 1");
    }
    if opts.epochs == 0 {
        bail!("--epochs must be >= 1");
    }
    if opts.batch_size == 0 || opts.grad_accum == 0 {
        bail!("--batch-size and --grad-accum must be >= 1");
    }

    let questions = crate::load_questions_from_file(&opts.schema_path)?;
    if questions.is_empty() {
        bail!(
            "Schema {} contains no questions",
            opts.schema_path.display()
        );
    }

    let mut warnings = Vec::new();
    let head_budget = opts.checkpoint.head_budget();
    for (qid, q) in &questions {
        match q.validate(qid, head_budget) {
            Ok(w) => warnings.extend(w),
            Err(e) => bail!("Schema validation failed: {e}"),
        }
    }

    let (train_src, val_src, cal_src, test_src, split_id) = if let Some(sdir) = &opts.split_dir {
        if !sdir.exists() {
            bail!("Split directory does not exist: {}", sdir.display());
        }
        let tr = opts
            .train_file
            .clone()
            .unwrap_or_else(|| sdir.join("train.jsonl"));
        let vl = opts
            .val_file
            .clone()
            .unwrap_or_else(|| sdir.join("val.jsonl"));
        let cl = opts
            .cal_file
            .clone()
            .or_else(|| Some(sdir.join("cal.jsonl")).filter(|p| p.exists()));
        let ts = opts
            .forbid_test_file
            .clone()
            .or_else(|| Some(sdir.join("test.jsonl")).filter(|p| p.exists()));
        let ds_manifest_id = std::fs::read_to_string(sdir.join("dataset_manifest.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                v.get("split_id")
                    .and_then(|x| x.as_str())
                    .map(String::from)
                    .or_else(|| {
                        v.get("source_sha256")
                            .and_then(|x| x.as_str())
                            .map(|s| s[..s.len().min(16)].to_string())
                    })
            });
        (tr, vl, cl, ts, ds_manifest_id)
    } else {
        let tr = opts.train_file.clone().ok_or_else(|| {
            anyhow!("Specify either --split-dir <DIR> or --train <FILE> + --val <FILE>")
        })?;
        let vl = opts
            .val_file
            .clone()
            .ok_or_else(|| anyhow!("Specify --val <FILE> alongside --train <FILE>"))?;
        (
            tr,
            vl,
            opts.cal_file.clone(),
            opts.forbid_test_file.clone(),
            None,
        )
    };

    if !train_src.exists() {
        bail!("Training split file not found: {}", train_src.display());
    }
    if !val_src.exists() {
        bail!("Validation split file not found: {}", val_src.display());
    }
    if let Some(ref ts) = test_src
        && (ts == &train_src || ts == &val_src || cal_src.as_ref() == Some(ts))
    {
        bail!(
            "Locked test file ({}) cannot also be passed as train, val, or cal!",
            ts.display()
        );
    }

    let train_items = FileSource::new(&train_src).read_items()?;
    let val_items = FileSource::new(&val_src).read_items()?;
    let cal_items = match &cal_src {
        Some(cp) => Some(FileSource::new(cp).read_items()?),
        None => None,
    };

    if train_items.is_empty() {
        bail!("Training split {} is empty", train_src.display());
    }
    if val_items.is_empty() {
        bail!("Validation split {} is empty", val_src.display());
    }

    // Verify ground-truth presence across schema questions
    let count_labeled = |items: &[DataItem]| -> usize {
        items
            .iter()
            .filter(|it| {
                questions.iter().any(|(qid, q)| {
                    resolve_ground_truth_for_question(
                        it.ground_truth.as_ref(),
                        qid,
                        q,
                        questions.len(),
                    )
                    .is_some()
                })
            })
            .count()
    };
    let labeled_train = count_labeled(&train_items);
    if labeled_train == 0 {
        bail!(
            "No items in {} have ground-truth labels matching schema {}",
            train_src.display(),
            opts.schema_path.display()
        );
    }
    if count_labeled(&val_items) == 0 {
        bail!(
            "No items in {} have ground-truth labels matching schema {}",
            val_src.display(),
            opts.schema_path.display()
        );
    }

    // Build leakage filter across train ∪ val ∪ cal and check locked test split BEFORE bundling!
    let mut filter = TrainedInputsFilter::from_items(&train_items);
    filter.extend_from(&TrainedInputsFilter::from_items(&val_items));
    if let Some(ref ci) = cal_items {
        filter.extend_from(&TrainedInputsFilter::from_items(ci));
    }

    let expected_test_file_sha256 = if let Some(ref ts) = test_src {
        let test_items = FileSource::new(ts).read_items()?;
        let leak = filter.check_items(&test_items);
        if leak.has_leakage() {
            bail!(
                "Preflight leakage check FAILED before bundle creation: {} exact and {} normalized state collisions found between train/val/cal and test split {} (e.g. {:?}). Re-run `lemmata dataset split`.",
                leak.exact_leaked_ids.len(),
                leak.normalized_leaked_ids.len(),
                ts.display(),
                leak.exact_leaked_ids
                    .iter()
                    .chain(&leak.normalized_leaked_ids)
                    .take(5)
                    .collect::<Vec<_>>()
            );
        }
        Some(sha256_file(ts)?)
    } else {
        warnings.push(
            "No locked test.jsonl was supplied at bundle creation time; pass --test-file during `finetune verify` and `experiment`."
                .to_string(),
        );
        None
    };

    // Assemble `<out_dir>/bundle/` (strictly excluding test.jsonl!)
    let bundle_dir = opts.out_dir.join("bundle");
    let data_dir = bundle_dir.join("data");
    if data_dir.join("test.jsonl").exists() {
        std::fs::remove_file(data_dir.join("test.jsonl"))?;
    }
    std::fs::create_dir_all(&data_dir)?;
    std::fs::create_dir_all(opts.out_dir.join("artifacts"))?;

    let mut splits_map = BTreeMap::new();

    let train_dst = data_dir.join("train.jsonl");
    copy_or_write_jsonl(&train_src, &train_items, &train_dst)?;
    splits_map.insert(
        "train".to_string(),
        BundleSplitEntry {
            rel_path: "data/train.jsonl".to_string(),
            rows: train_items.len(),
            file_sha256: sha256_file(&train_dst)?,
            state_merkle_sha256: compute_state_merkle(&train_items),
        },
    );

    let val_dst = data_dir.join("val.jsonl");
    copy_or_write_jsonl(&val_src, &val_items, &val_dst)?;
    splits_map.insert(
        "val".to_string(),
        BundleSplitEntry {
            rel_path: "data/val.jsonl".to_string(),
            rows: val_items.len(),
            file_sha256: sha256_file(&val_dst)?,
            state_merkle_sha256: compute_state_merkle(&val_items),
        },
    );

    let cal_rows = if let (Some(cp), Some(ci)) = (&cal_src, &cal_items) {
        let cal_dst = data_dir.join("cal.jsonl");
        copy_or_write_jsonl(cp, ci, &cal_dst)?;
        splits_map.insert(
            "cal".to_string(),
            BundleSplitEntry {
                rel_path: "data/cal.jsonl".to_string(),
                rows: ci.len(),
                file_sha256: sha256_file(&cal_dst)?,
                state_merkle_sha256: compute_state_merkle(ci),
            },
        );
        ci.len()
    } else {
        0
    };

    debug_assert!(!data_dir.join("test.jsonl").exists());

    // Write canonical schema.json
    let schema_dst = bundle_dir.join("schema.json");
    std::fs::write(&schema_dst, serde_json::to_string_pretty(&questions)?)?;
    let schema_sha256 = compute_schema_hash(&questions);

    let questions_summary: BTreeMap<String, serde_json::Value> = questions
        .iter()
        .map(|(qid, q)| {
            let opts_rendered = q.render_options();
            (
                qid.clone(),
                serde_json::json!({
                    "type": q.qtype,
                    "num_options": opts_rendered.len(),
                }),
            )
        })
        .collect();

    let local_base_st = opts.checkpoint.local_dir().join("model.safetensors");
    let local_base_header_sha256 = local_base_st
        .exists()
        .then(|| safetensors_header_sha256(&local_base_st))
        .transpose()?;

    // Deterministic short hash over config + train sha for reproducible run IDs
    let id_seed = format!(
        "{}|{}|{}|{}",
        opts.checkpoint.repo_id(),
        opts.method.as_str(),
        splits_map["train"].file_sha256,
        opts.seed
    );
    let id_hex: String = Sha256::digest(id_seed.as_bytes())
        .iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect();
    let run_id = format!(
        "ft_{}_{}",
        opts.checkpoint.short_name().replace('-', "_"),
        id_hex
    );

    let effective_lr = opts.lr.unwrap_or(match opts.method {
        FinetuneMethod::Full => 2.0e-5,
        FinetuneMethod::Lora | FinetuneMethod::Dora => defaults.default_lr_lora,
    });

    let default_max_len = match defaults.paradigm {
        "nli_cross_encoder" => opts.checkpoint.context_window().min(512),
        "mlm_cloze" => opts.checkpoint.context_window().min(2048),
        _ => opts.checkpoint.context_window(),
    };

    let bundle_manifest = BundleManifest {
        manifest_version: TRAINING_MANIFEST_VERSION.to_string(),
        run_id: run_id.clone(),
        created_by_lemmata_version: env!("CARGO_PKG_VERSION").to_string(),
        split_id,
        base_model: serde_json::json!({
            "checkpoint_slug": opts.checkpoint.short_name(),
            "repo_id": opts.checkpoint.repo_id(),
            "local_subdir": opts.checkpoint.local_subdir(),
            "paradigm": defaults.paradigm,
            "local_base_header_sha256": local_base_header_sha256,
            "max_len": default_max_len,
            "head_max_len": opts.checkpoint.head_budget(),
        }),
        schema: serde_json::json!({
            "schema_sha256": schema_sha256,
            "questions": questions_summary,
        }),
        splits: splits_map,
        expected_test_file_sha256,
        method: serde_json::json!({
            "mode": opts.method.as_str(),
            "peft_type": match opts.method {
                FinetuneMethod::Lora => Some("LORA"),
                FinetuneMethod::Dora => Some("DORA"),
                FinetuneMethod::Full => None,
            },
            "r": (opts.method != FinetuneMethod::Full).then_some(opts.rank),
            "lora_alpha": (opts.method != FinetuneMethod::Full).then_some(opts.alpha),
            "use_rslora": opts.use_rslora,
            "use_dora": opts.method == FinetuneMethod::Dora,
            "lora_dropout": opts.lora_dropout,
            "target_modules": defaults.target_modules,
            "modules_to_save": defaults.modules_to_save,
        }),
        hyperparameters: serde_json::json!({
            "seed": opts.seed,
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "grad_accum": opts.grad_accum,
            "lr_lora": effective_lr,
            "lr_head": defaults.default_lr_head,
            "weight_decay": 0.01,
            "warmup_ratio": 0.08,
            "label_smoothing": opts.label_smoothing,
            "neftune_alpha": opts.neftune_alpha,
            "rdrop_alpha": opts.rdrop_alpha,
            "multi_label": opts.multi_label,
            "cuda_wheel": opts.cuda.as_str(),
        }),
    };

    std::fs::write(
        bundle_dir.join("bundle_manifest.json"),
        serde_json::to_string_pretty(&bundle_manifest)?,
    )?;

    let train_py_path = bundle_dir.join("train.py");
    std::fs::write(&train_py_path, render_train_py(opts.cuda))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&train_py_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&train_py_path, perms)?;
    }

    std::fs::write(
        bundle_dir.join("RUN.md"),
        render_run_md(&run_id, opts.checkpoint, opts.method, &opts.out_dir),
    )?;

    let tar_path = if opts.create_tar {
        let archive = opts.out_dir.join("bundle.tar.gz");
        let status = Command::new("tar")
            .args([
                "-czf",
                &archive.display().to_string(),
                "-C",
                &opts.out_dir.display().to_string(),
                "bundle",
            ])
            .status()
            .context("Failed to execute `tar` to create bundle.tar.gz")?;
        if !status.success() {
            bail!("`tar -czf` failed with exit code {:?}", status.code());
        }
        Some(archive)
    } else {
        None
    };

    let bundle_bytes = dir_files_bytes_recursive(&bundle_dir);

    Ok(PrepareReport {
        run_id,
        out_dir: opts.out_dir.clone(),
        bundle_dir,
        tar_path,
        checkpoint: opts.checkpoint.repo_id().to_string(),
        paradigm: defaults.paradigm.to_string(),
        method: opts.method.as_str().to_string(),
        train_rows: train_items.len(),
        val_rows: val_items.len(),
        cal_rows,
        bundle_bytes,
        warnings,
    })
}

/// Options for `lemmata finetune verify`.
#[derive(Debug, Clone)]
pub struct VerifyArtifactOptions {
    pub artifact_dir: PathBuf,
    pub split_dir: Option<PathBuf>,
    pub test_file: Option<PathBuf>,
    pub schema_path: Option<PathBuf>,
    pub base_dir: Option<PathBuf>,
}

/// Detailed 5-layer artifact verification & zero-leakage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactVerificationReport {
    pub valid: bool,
    pub run_id: String,
    pub artifact_dir: PathBuf,
    pub base_checkpoint: String,
    pub method: String,
    pub weights_verified: bool,
    pub base_header_verified: Option<bool>,
    pub schema_verified: Option<bool>,
    pub split_hashes_verified: Option<bool>,
    pub row_leakage_checked: bool,
    pub leaked_item_ids: Vec<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Execute the 5-layer local integrity & zero-leakage verification on an `artifacts/` directory.
pub fn verify_artifact(opts: &VerifyArtifactOptions) -> Result<ArtifactVerificationReport> {
    let adir = if !opts.artifact_dir.join("training_manifest.json").exists()
        && opts
            .artifact_dir
            .join("artifacts/training_manifest.json")
            .exists()
    {
        opts.artifact_dir.join("artifacts")
    } else {
        opts.artifact_dir.clone()
    };

    if !adir.exists() {
        bail!("Artifact directory does not exist: {}", adir.display());
    }

    let manifest = TrainingManifest::load(&adir)?;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if manifest.manifest_version != TRAINING_MANIFEST_VERSION {
        warnings.push(format!(
            "Manifest version '{}' differs from '{}'",
            manifest.manifest_version, TRAINING_MANIFEST_VERSION
        ));
    }
    if manifest.status != "completed" {
        errors.push(format!(
            "Training run status is '{}' (expected 'completed')",
            manifest.status
        ));
    }

    // Layer 1: Weights & Calibration SHA-256 Tamper Check
    let mut weights_verified = false;
    let mode = manifest.method.mode.to_ascii_lowercase();
    if mode == "lora" || mode == "dora" || adir.join("adapter_model.safetensors").exists() {
        let st_path = adir.join("adapter_model.safetensors");
        if !st_path.exists() {
            errors.push(format!("Missing {}", st_path.display()));
        } else if !adir.join("adapter_config.json").exists() {
            errors.push(format!(
                "Missing {}",
                adir.join("adapter_config.json").display()
            ));
        } else {
            let actual_sha = sha256_file(&st_path)?;
            if let Some(expected_sha) = &manifest.artifacts.adapter_safetensors_sha256 {
                if &actual_sha != expected_sha {
                    errors.push(format!(
                        "Tamper/corruption detected on adapter_model.safetensors: expected SHA-256 {expected_sha}, got {actual_sha}"
                    ));
                } else {
                    weights_verified = true;
                }
            } else {
                warnings.push(
                    "Manifest omits artifacts.adapter_safetensors_sha256; verified header format only."
                        .to_string(),
                );
                let _ = crate::model::inspect_safetensors(&st_path)?;
                weights_verified = true;
            }
        }
    } else {
        let st_path = adir.join("model.safetensors");
        if !st_path.exists() {
            errors.push(format!("Missing merged weights {}", st_path.display()));
        } else {
            let actual_hdr = safetensors_header_sha256(&st_path)?;
            if let Some(expected_hdr) = &manifest.artifacts.merged_safetensors_header_sha256 {
                if &actual_hdr != expected_hdr {
                    errors.push(format!(
                        "Merged model.safetensors header SHA-256 mismatch: expected {expected_hdr}, got {actual_hdr}"
                    ));
                } else {
                    weights_verified = true;
                }
            } else {
                weights_verified = true;
            }
        }
    }

    if let Some(expected_cal_sha) = &manifest.artifacts.calibration_sha256 {
        let cal_path = adir.join("lemmata_calibration.json");
        if !cal_path.exists() {
            errors.push(format!(
                "Manifest records calibration_sha256, but {} is missing",
                cal_path.display()
            ));
        } else {
            let actual_cal_sha = sha256_file(&cal_path)?;
            if &actual_cal_sha != expected_cal_sha {
                errors.push(format!(
                    "lemmata_calibration.json SHA-256 mismatch: expected {expected_cal_sha}, got {actual_cal_sha}"
                ));
            }
        }
    }

    // Layer 2: Base Checkpoint Header SHA-256 Compatibility Check
    let resolved_ckpt = Checkpoint::from_str(&manifest.base_model.checkpoint_slug)
        .or_else(|_| Checkpoint::from_str(&manifest.base_model.repo_id))
        .ok();
    let base_st = opts
        .base_dir
        .as_ref()
        .map(|d| d.join("model.safetensors"))
        .or_else(|| resolved_ckpt.map(|c| c.local_dir().join("model.safetensors")));

    let base_header_verified = if let Some(bst) = base_st.filter(|p| p.exists()) {
        if let Some(remote_hdr) = &manifest.base_model.remote_base_header_sha256 {
            let local_hdr = safetensors_header_sha256(&bst)?;
            if &local_hdr != remote_hdr {
                errors.push(format!(
                    "Base checkpoint incompatibility: adapter was trained against safetensors header {}, but local {} has header {}",
                    &remote_hdr[..remote_hdr.len().min(16)],
                    bst.display(),
                    &local_hdr[..local_hdr.len().min(16)]
                ));
                Some(false)
            } else {
                Some(true)
            }
        } else {
            None
        }
    } else {
        None
    };

    // Layer 3: Schema SHA-256 Check
    let schema_verified = if let Some(sp) = &opts.schema_path {
        let questions = crate::load_questions_from_file(sp)?;
        let cur_sha = compute_schema_hash(&questions);
        if !manifest.schema.schema_sha256.is_empty() && cur_sha != manifest.schema.schema_sha256 {
            warnings.push(format!(
                "Schema digest ({}) differs from training schema digest ({})",
                &cur_sha[..12],
                &manifest.schema.schema_sha256[..manifest.schema.schema_sha256.len().min(12)]
            ));
            Some(false)
        } else {
            Some(true)
        }
    } else {
        None
    };

    // Layer 4 & Layer 5: Split File Hash Guard + Row-Level Binary Search (`trained_inputs.sha256.bin`)
    let effective_test_file = opts.test_file.clone().or_else(|| {
        opts.split_dir
            .as_ref()
            .map(|d| d.join("test.jsonl"))
            .filter(|p| p.exists())
    });

    let filter_rel = manifest
        .dataset_provenance
        .locked_test_guard
        .trained_inputs_filter_file
        .as_deref()
        .unwrap_or("trained_inputs.sha256.bin");
    let filter_path = adir.join(filter_rel);

    let mut split_hashes_verified = None;
    let mut row_leakage_checked = false;
    let mut leaked_item_ids = Vec::new();

    if let Some(ref test_path) = effective_test_file {
        if !test_path.exists() {
            bail!("Test file does not exist: {}", test_path.display());
        }
        let test_sha = sha256_file(test_path)?;
        let test_items = FileSource::new(test_path).read_items()?;
        let test_merkle = compute_state_merkle(&test_items);
        let dp = &manifest.dataset_provenance;
        let mut split_collision = false;

        for (split_label, split_prov) in [
            ("train", Some(&dp.train)),
            ("val", dp.val.as_ref()),
            ("cal", dp.cal.as_ref()),
        ] {
            if let Some(sp) = split_prov {
                let file_match = !sp.file_sha256.is_empty() && sp.file_sha256 == test_sha;
                let merkle_match =
                    !sp.state_merkle_sha256.is_empty() && sp.state_merkle_sha256 == test_merkle;
                if file_match || merkle_match {
                    errors.push(format!(
                        "DATA LEAKAGE DETECTED: test file {} matches training '{split_label}' split digest!",
                        test_path.display()
                    ));
                    split_collision = true;
                }
            }
        }

        if let Some(expected_test_sha) = &dp.locked_test_guard.expected_test_file_sha256
            && expected_test_sha != &test_sha
        {
            warnings.push(format!(
                "Test file {} SHA-256 ({}) differs from locked test split recorded at prepare time ({}). Running row-level overlap guard.",
                test_path.display(),
                &test_sha[..12],
                &expected_test_sha[..expected_test_sha.len().min(12)]
            ));
        }
        split_hashes_verified = Some(!split_collision);

        if filter_path.exists() {
            let filter = TrainedInputsFilter::read(&filter_path)?;
            let check = filter.check_items(&test_items);
            row_leakage_checked = true;
            if check.has_leakage() {
                for id in &check.exact_leaked_ids {
                    leaked_item_ids.push(format!("{id} (exact)"));
                }
                for id in &check.normalized_leaked_ids {
                    leaked_item_ids.push(format!("{id} (normalized)"));
                }
                errors.push(format!(
                    "ROW-LEVEL DATA LEAKAGE DETECTED: {} exact and {} normalized test items in {} were present in train/val/cal during fine-tuning!",
                    check.exact_leaked_ids.len(),
                    check.normalized_leaked_ids.len(),
                    test_path.display()
                ));
            }
        } else {
            warnings.push(format!(
                "Missing {} — row-level state leakage filter could not be checked.",
                filter_path.display()
            ));
        }
    } else if filter_path.exists() {
        // Still validate binary integrity of `trained_inputs.sha256.bin` even when no test file is given.
        let _ = TrainedInputsFilter::read(&filter_path)?;
    }

    let valid = errors.is_empty();
    Ok(ArtifactVerificationReport {
        valid,
        run_id: manifest.run_id,
        artifact_dir: adir,
        base_checkpoint: manifest.base_model.repo_id,
        method: manifest.method.mode,
        weights_verified,
        base_header_verified,
        schema_verified,
        split_hashes_verified,
        row_leakage_checked,
        leaked_item_ids,
        errors,
        warnings,
    })
}

/// Guard helper invoked by `experiment` and `test` before evaluating a fine-tuned model/adapter
/// on `test_items`. Returns `Ok(Some(report))` when `candidate_dir` contains `training_manifest.json`.
pub fn guard_eval_against_training_leakage(
    spec: &ModelSpec,
    test_file: Option<&Path>,
    test_items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    allow_leakage: bool,
) -> Result<Option<ArtifactVerificationReport>> {
    let active_dir = spec.active_config_dir();
    if !active_dir.join("training_manifest.json").exists() {
        return Ok(None);
    }

    let mut report = verify_artifact(&VerifyArtifactOptions {
        artifact_dir: active_dir.clone(),
        split_dir: None,
        test_file: test_file.map(Path::to_path_buf),
        schema_path: None,
        base_dir: Some(spec.base_weights_dir()),
    })?;

    let cur_schema_sha = compute_schema_hash(questions);
    if let Some(manifest) = TrainingManifest::load_opt(&active_dir)
        && !manifest.schema.schema_sha256.is_empty()
    {
        if manifest.schema.schema_sha256 != cur_schema_sha {
            report.schema_verified = Some(false);
            report.warnings.push(format!(
                "Evaluation schema digest ({}) differs from fine-tuning schema digest ({})",
                &cur_schema_sha[..12],
                &manifest.schema.schema_sha256[..manifest.schema.schema_sha256.len().min(12)]
            ));
        } else {
            report.schema_verified = Some(true);
        }
    }

    // Even if `test_file` was `None` (e.g. directory or in-memory items), check `test_items` against `trained_inputs.sha256.bin`.
    let filter_path = active_dir.join("trained_inputs.sha256.bin");
    if !report.row_leakage_checked && filter_path.exists() && !test_items.is_empty() {
        let filter = TrainedInputsFilter::read(&filter_path)?;
        let check = filter.check_items(test_items);
        report.row_leakage_checked = true;
        if check.has_leakage() {
            for id in &check.exact_leaked_ids {
                report.leaked_item_ids.push(format!("{id} (exact)"));
            }
            for id in &check.normalized_leaked_ids {
                report.leaked_item_ids.push(format!("{id} (normalized)"));
            }
            report.errors.push(format!(
                "ROW-LEVEL DATA LEAKAGE DETECTED: {} exact and {} normalized evaluation items were seen during fine-tuning!",
                check.exact_leaked_ids.len(),
                check.normalized_leaked_ids.len()
            ));
            report.valid = false;
        }
    }

    if !report.valid && !allow_leakage {
        bail!(
            "Refusing to evaluate contaminated or invalid fine-tuned artifact '{}' (pass --allow-leakage to override):\n  - {}",
            active_dir.display(),
            report.errors.join("\n  - ")
        );
    }
    Ok(Some(report))
}

/// Format an `ArtifactVerificationReport` as a human-readable table.
pub fn format_verification_table(rep: &ArtifactVerificationReport) -> String {
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Verification Layer").fg(Color::Cyan),
        Cell::new("Status").fg(Color::White),
        Cell::new("Details").fg(Color::DarkGrey),
    ]);

    let badge = |ok: Option<bool>| match ok {
        Some(true) => ("PASS", Color::Green),
        Some(false) => ("FAIL", Color::Red),
        None => ("SKIP", Color::Yellow),
    };

    let (w_txt, w_col) = badge(Some(rep.weights_verified));
    table.add_row(vec![
        Cell::new("1. Adapter / Weights SHA-256"),
        Cell::new(w_txt).fg(w_col),
        Cell::new(format!("Run {} ({})", rep.run_id, rep.method)),
    ]);

    let (b_txt, b_col) = badge(rep.base_header_verified);
    table.add_row(vec![
        Cell::new("2. Base Safetensors Header"),
        Cell::new(b_txt).fg(b_col),
        Cell::new(&rep.base_checkpoint),
    ]);

    let (s_txt, s_col) = badge(rep.schema_verified);
    table.add_row(vec![
        Cell::new("3. Schema Digest Match"),
        Cell::new(s_txt).fg(s_col),
        Cell::new("Question & option schema compatibility"),
    ]);

    let (sp_txt, sp_col) = badge(rep.split_hashes_verified);
    table.add_row(vec![
        Cell::new("4. Split File Hash Guard"),
        Cell::new(sp_txt).fg(sp_col),
        Cell::new("Test file SHA-256 disjoint from train/val/cal"),
    ]);

    let (r_txt, r_col) = if rep.row_leakage_checked {
        badge(Some(rep.leaked_item_ids.is_empty()))
    } else {
        badge(None)
    };
    table.add_row(vec![
        Cell::new("5. Row-Level State Filter"),
        Cell::new(r_txt).fg(r_col),
        Cell::new(if rep.row_leakage_checked {
            format!("{} leaked item(s)", rep.leaked_item_ids.len())
        } else {
            "Pass --test-file or --split-dir to verify eval rows".to_string()
        }),
    ]);

    let mut out = format!("{table}\n");
    for w in &rep.warnings {
        out.push_str(&format!("⚠️  {w}\n"));
    }
    for e in &rep.errors {
        out.push_str(&format!("❌ {e}\n"));
    }
    out
}

/// Planned external shell command for `finetune push` or `finetune pull`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportPlan {
    pub program: String,
    pub args: Vec<String>,
    pub description: String,
}

impl TransportPlan {
    pub fn command_line(&self) -> String {
        let mut parts = vec![self.program.clone()];
        for a in &self.args {
            if a.contains(' ') || a.contains('*') {
                parts.push(format!("'{a}'"));
            } else {
                parts.push(a.clone());
            }
        }
        parts.join(" ")
    }
}

/// Build or execute `lemmata finetune push` (`<run_dir>/bundle/` -> `<remote>/bundle/`).
pub fn push_bundle(run_dir: &Path, remote: &str, dry_run: bool) -> Result<TransportPlan> {
    let bundle_dir = if run_dir.ends_with("bundle") {
        run_dir.to_path_buf()
    } else {
        run_dir.join("bundle")
    };
    if !bundle_dir.join("train.py").exists() {
        bail!(
            "Bundle directory {} does not contain train.py. Run `lemmata finetune prepare` first.",
            bundle_dir.display()
        );
    }
    if bundle_dir.join("data/test.jsonl").exists() {
        bail!(
            "Refusing to push bundle {}: data/test.jsonl must never be uploaded to training host!",
            bundle_dir.display()
        );
    }

    let remote_trimmed = remote.trim().trim_end_matches('/');
    if remote_trimmed.is_empty() {
        bail!("Remote destination cannot be empty");
    }

    let plan = if remote_trimmed.starts_with("s3://") {
        TransportPlan {
            program: "aws".to_string(),
            args: vec![
                "s3".to_string(),
                "sync".to_string(),
                bundle_dir.display().to_string(),
                format!("{remote_trimmed}/bundle/"),
                "--exclude".to_string(),
                "*test.jsonl*".to_string(),
            ],
            description: format!("Sync training bundle to {remote_trimmed}/bundle/"),
        }
    } else {
        let mut src = bundle_dir.display().to_string();
        if !src.ends_with('/') {
            src.push('/');
        }
        TransportPlan {
            program: "rsync".to_string(),
            args: vec![
                "-avz".to_string(),
                "--exclude".to_string(),
                "*test.jsonl*".to_string(),
                src,
                format!("{remote_trimmed}/bundle/"),
            ],
            description: format!("Sync training bundle via SSH/rsync to {remote_trimmed}/bundle/"),
        }
    };

    if !dry_run {
        let status = Command::new(&plan.program)
            .args(&plan.args)
            .status()
            .with_context(|| format!("Failed to execute `{}`", plan.program))?;
        if !status.success() {
            bail!(
                "`{}` exited with status {:?}",
                plan.command_line(),
                status.code()
            );
        }
    }

    Ok(plan)
}

/// Build or execute `lemmata finetune pull` (`<remote>/artifacts/` -> `<run_dir>/artifacts/`),
/// explicitly excluding multi-GB resumable optimizer `checkpoints/`.
pub fn pull_artifacts(run_dir: &Path, remote: &str, dry_run: bool) -> Result<TransportPlan> {
    let artifacts_dir = if run_dir.ends_with("artifacts") {
        run_dir.to_path_buf()
    } else {
        run_dir.join("artifacts")
    };
    let remote_trimmed = remote.trim().trim_end_matches('/');
    if remote_trimmed.is_empty() {
        bail!("Remote source cannot be empty");
    }

    let remote_artifacts = if remote_trimmed.ends_with("/artifacts") {
        format!("{remote_trimmed}/")
    } else {
        format!("{remote_trimmed}/artifacts/")
    };

    let plan = if remote_trimmed.starts_with("s3://") {
        TransportPlan {
            program: "aws".to_string(),
            args: vec![
                "s3".to_string(),
                "sync".to_string(),
                remote_artifacts,
                artifacts_dir.display().to_string(),
                "--exclude".to_string(),
                "checkpoints/*".to_string(),
                "--exclude".to_string(),
                "*.pt".to_string(),
            ],
            description: format!(
                "Pull lightweight fine-tuned artifacts from {remote_trimmed} (skipping optimizer checkpoints)"
            ),
        }
    } else {
        let mut dst = artifacts_dir.display().to_string();
        if !dst.ends_with('/') {
            dst.push('/');
        }
        TransportPlan {
            program: "rsync".to_string(),
            args: vec![
                "-avz".to_string(),
                "--exclude".to_string(),
                "checkpoints/".to_string(),
                "--exclude".to_string(),
                "*.pt".to_string(),
                remote_artifacts,
                dst,
            ],
            description: format!(
                "Pull lightweight fine-tuned artifacts via SSH/rsync from {remote_trimmed} (skipping optimizer checkpoints)"
            ),
        }
    };

    if !dry_run {
        std::fs::create_dir_all(&artifacts_dir)?;
        let status = Command::new(&plan.program)
            .args(&plan.args)
            .status()
            .with_context(|| format!("Failed to execute `{}`", plan.program))?;
        if !status.success() {
            bail!(
                "`{}` exited with status {:?}",
                plan.command_line(),
                status.code()
            );
        }
    }

    Ok(plan)
}

/// Merge a fine-tuned PEFT adapter directory into a standalone checkpoint directory (`finetune merge`).
pub fn merge_finetuned_adapter(
    adapter_dir: &Path,
    checkpoint_override: Option<&str>,
    output_dir: &Path,
    overwrite: bool,
) -> Result<MergeReport> {
    merge_finetuned_adapter_with_options(
        adapter_dir,
        checkpoint_override,
        output_dir,
        overwrite,
        false,
    )
}

/// Merge a fine-tuned PEFT adapter directory into a standalone checkpoint directory with explicit
/// `overwrite` (`--force`) and `force_f32` (`--f32`) controls.
pub fn merge_finetuned_adapter_with_options(
    adapter_dir: &Path,
    checkpoint_override: Option<&str>,
    output_dir: &Path,
    overwrite: bool,
    force_f32: bool,
) -> Result<MergeReport> {
    if output_dir.join("model.safetensors").exists() && !overwrite {
        bail!(
            "Destination directory '{}' already contains model.safetensors. Pass --force to overwrite.",
            output_dir.display()
        );
    }
    let ckpt_arg = checkpoint_override.unwrap_or("english");
    let spec = ModelSpec::resolve(ckpt_arg, Some(adapter_dir))?;
    let adir = spec
        .adapter_dir
        .as_deref()
        .ok_or_else(|| anyhow!("No valid PEFT adapter found at {}", adapter_dir.display()))?;
    merge_adapter_to_dir(&spec.base_weights_dir(), adir, output_dir, force_f32)
}

/// Count JSONL non-empty lines in a file.
pub fn count_jsonl_rows(path: &Path) -> Result<usize> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut n = 0usize;
    for line in reader.lines() {
        if !line?.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}
