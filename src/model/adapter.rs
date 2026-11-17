//! Native Hugging Face PEFT (`LoRA`, `rsLoRA`, `DoRA`, `modules_to_save`) adapter loader,
//! standalone weight merger, and unified `ModelSpec` resolution for `candle`.
//!
//! `PeftBackend` implements `candle_nn::var_builder::SimpleBackend` and folds low-rank and
//! weight-decomposed updates into base weights in `F32` at load time, adding zero runtime
//! latency overhead during inference.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_nn::var_builder::SimpleBackend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::checkpoint::Checkpoint;
use super::head_dispatch::head_paradigm_for;
use super::registry::inspect_model_directory;
use super::safetensors::{inspect_safetensors, safetensors_header_sha256};

/// Canonical schema version for `training_manifest.json`.
pub const TRAINING_MANIFEST_VERSION: &str = "lemmata-ft-2026.1";

fn default_lora_r() -> usize {
    8
}

fn default_lora_alpha() -> f64 {
    8.0
}

fn default_bias_mode() -> String {
    "none".to_string()
}

/// Hugging Face PEFT `adapter_config.json` specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    #[serde(default)]
    pub peft_type: Option<String>,
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
    #[serde(default = "default_lora_r")]
    pub r: usize,
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: f64,
    #[serde(default)]
    pub use_rslora: bool,
    #[serde(default)]
    pub use_dora: bool,
    #[serde(default)]
    pub fan_in_fan_out: bool,
    #[serde(default)]
    pub lora_dropout: f64,
    #[serde(default = "default_bias_mode")]
    pub bias: String,
    #[serde(default)]
    pub target_modules: Option<serde_json::Value>,
    #[serde(default)]
    pub modules_to_save: Option<Vec<String>>,
    #[serde(default)]
    pub rank_pattern: HashMap<String, usize>,
    #[serde(default)]
    pub alpha_pattern: HashMap<String, f64>,
    #[serde(default)]
    pub task_type: Option<String>,
}

impl AdapterConfig {
    /// Load `adapter_config.json` from an adapter directory or direct file path.
    pub fn load(path: &Path) -> Result<Self> {
        let cfg_path = if path.is_dir() {
            path.join("adapter_config.json")
        } else {
            path.to_path_buf()
        };
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("Failed to read PEFT config {}", cfg_path.display()))?;
        let cfg: Self = serde_json::from_str(&raw)
            .with_context(|| format!("Malformed PEFT config in {}", cfg_path.display()))?;
        if let Some(pt) = &cfg.peft_type {
            let up = pt.to_ascii_uppercase();
            if up != "LORA" && up != "DORA" {
                bail!(
                    "Unsupported PEFT adapter type '{}' in {} (supported: LORA, DORA)",
                    pt,
                    cfg_path.display()
                );
            }
        }
        Ok(cfg)
    }

    /// True when Weight-Decomposed Low-Rank Adaptation (DoRA) is enabled.
    pub fn is_dora(&self) -> bool {
        self.use_dora
            || self
                .peft_type
                .as_deref()
                .is_some_and(|p| p.eq_ignore_ascii_case("DORA"))
    }

    /// Effective `lora_alpha` for `module_name`, honoring per-layer `alpha_pattern` overrides.
    pub fn alpha_for(&self, module_name: &str) -> f64 {
        if let Some(&exact) = self.alpha_pattern.get(module_name) {
            return exact;
        }
        let mut best: Option<(usize, f64)> = None;
        for (pat, &val) in &self.alpha_pattern {
            let matches = module_name == pat
                || module_name
                    .strip_suffix(pat.as_str())
                    .is_some_and(|prefix| prefix.ends_with('.'));
            if matches && best.is_none_or(|(len, _)| pat.len() > len) {
                best = Some((pat.len(), val));
            }
        }
        best.map_or(self.lora_alpha, |(_, v)| v)
    }

    /// Effective scaling factor (`alpha / sqrt(r)` for rsLoRA, `alpha / r` for standard LoRA).
    pub fn scaling_for(&self, module_name: &str, rank: usize) -> Result<f64> {
        if rank == 0 {
            bail!("LoRA rank cannot be 0 for module '{module_name}'");
        }
        let alpha = self.alpha_for(module_name);
        if self.use_rslora {
            Ok(alpha / (rank as f64).sqrt())
        } else {
            Ok(alpha / (rank as f64))
        }
    }
}

/// Base model provenance recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaseModelProvenance {
    #[serde(default)]
    pub checkpoint_slug: String,
    #[serde(default)]
    pub repo_id: String,
    #[serde(default)]
    pub hf_revision: Option<String>,
    #[serde(default)]
    pub paradigm: String,
    #[serde(default)]
    pub local_base_header_sha256: Option<String>,
    #[serde(default)]
    pub remote_base_header_sha256: Option<String>,
    #[serde(default)]
    pub max_len: Option<usize>,
    #[serde(default)]
    pub head_max_len: Option<usize>,
}

/// Schema provenance recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaProvenance {
    #[serde(default)]
    pub schema_sha256: String,
    #[serde(default)]
    pub questions: BTreeMap<String, serde_json::Value>,
}

/// Per-split digest provenance recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitProvenance {
    #[serde(default)]
    pub rows: usize,
    #[serde(default)]
    pub file_sha256: String,
    #[serde(default)]
    pub state_merkle_sha256: String,
    #[serde(default)]
    pub item_set_digest: Option<String>,
}

/// Test-split anti-leakage metadata recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LockedTestGuard {
    #[serde(default)]
    pub expected_test_file_sha256: Option<String>,
    #[serde(default)]
    pub trained_inputs_filter_file: Option<String>,
    #[serde(default)]
    pub trained_unique_exact_hashes: usize,
    #[serde(default)]
    pub trained_unique_norm_hashes: usize,
}

/// Dataset split provenance recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetProvenance {
    #[serde(default)]
    pub split_id: Option<String>,
    #[serde(default)]
    pub train: SplitProvenance,
    #[serde(default)]
    pub val: Option<SplitProvenance>,
    #[serde(default)]
    pub cal: Option<SplitProvenance>,
    #[serde(default)]
    pub locked_test_guard: LockedTestGuard,
}

/// Fine-tuning method metadata recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MethodProvenance {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub peft_type: Option<String>,
    #[serde(default)]
    pub r: Option<usize>,
    #[serde(default)]
    pub lora_alpha: Option<f64>,
    #[serde(default)]
    pub use_rslora: bool,
    #[serde(default)]
    pub use_dora: bool,
    #[serde(default)]
    pub lora_dropout: f64,
    #[serde(default)]
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub modules_to_save: Vec<String>,
}

/// Emitted artifact digests recorded inside `training_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactDigests {
    #[serde(default)]
    pub adapter_safetensors_sha256: Option<String>,
    #[serde(default)]
    pub merged_safetensors_header_sha256: Option<String>,
    #[serde(default)]
    pub adapter_bytes: Option<u64>,
    #[serde(default)]
    pub calibration_sha256: Option<String>,
}

/// Cryptographic provenance, hyperparameters, and metrics emitted by `train.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingManifest {
    pub manifest_version: String,
    pub run_id: String,
    pub status: String,
    #[serde(default)]
    pub created_by_lemmata_version: String,
    #[serde(default)]
    pub git_commit: Option<String>,
    #[serde(default)]
    pub base_model: BaseModelProvenance,
    #[serde(default)]
    pub schema: SchemaProvenance,
    #[serde(default)]
    pub dataset_provenance: DatasetProvenance,
    #[serde(default)]
    pub method: MethodProvenance,
    #[serde(default)]
    pub hyperparameters: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub results: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub environment: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub artifacts: ArtifactDigests,
}

impl TrainingManifest {
    /// Load `training_manifest.json` from a directory or direct JSON file path.
    pub fn load(path: &Path) -> Result<Self> {
        let p = if path.is_dir() {
            path.join("training_manifest.json")
        } else {
            path.to_path_buf()
        };
        let raw = std::fs::read_to_string(&p)
            .with_context(|| format!("Failed to read {}", p.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("Malformed training manifest in {}", p.display()))
    }

    /// Load `training_manifest.json` if present and valid in `dir`.
    pub fn load_opt(dir: &Path) -> Option<Self> {
        Self::load(dir).ok()
    }
}

/// Compute streaming SHA-256 hex digest of a file on disk.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("Failed to open file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Unified model target combining a canonical `Checkpoint` architecture archetype with
/// an optional standalone local checkpoint directory and/or PEFT adapter directory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelSpec {
    pub family: Checkpoint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_dir: Option<PathBuf>,
}

impl ModelSpec {
    pub fn from_checkpoint(family: Checkpoint) -> Self {
        Self {
            family,
            model_dir: None,
            adapter_dir: None,
        }
    }

    /// True when pointing at a custom local model directory or a PEFT adapter.
    pub fn is_custom_or_adapted(&self) -> bool {
        self.model_dir.is_some() || self.adapter_dir.is_some()
    }

    /// Directory holding the base `model.safetensors`.
    pub fn base_weights_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| self.family.local_dir())
    }

    /// Directory holding the active calibration / `rl_agent_config.json` / manifest metadata.
    pub fn active_config_dir(&self) -> PathBuf {
        self.adapter_dir
            .clone()
            .or_else(|| self.model_dir.clone())
            .unwrap_or_else(|| self.family.local_dir())
    }

    /// Stable human-readable and receipt-safe model identifier.
    ///
    /// For custom local checkpoints and PEFT adapters, appends `@<12-hex-sha256>` from
    /// [`Self::portable_fingerprint`] so `DecisionResult.model` and signed `DecisionReceipt`s
    /// bind cryptographically to the exact base header and adapter weights.
    pub fn display_model_id(&self) -> String {
        if !self.is_custom_or_adapted() {
            return self.family.repo_id().to_string();
        }
        let short_sha = self
            .portable_fingerprint()
            .ok()
            .map(|fp| format!("@{}", &fp[..12.min(fp.len())]))
            .unwrap_or_default();
        let active_dir = self.active_config_dir();
        if let Some(manifest) = TrainingManifest::load_opt(&active_dir)
            && !manifest.run_id.is_empty()
        {
            let base = if !manifest.base_model.repo_id.is_empty() {
                manifest.base_model.repo_id
            } else {
                self.family.repo_id().to_string()
            };
            return format!("{base}+ft:{}{short_sha}", manifest.run_id);
        }
        match (&self.model_dir, &self.adapter_dir) {
            (None, None) => self.family.repo_id().to_string(),
            (Some(mdir), None) => {
                let name = mdir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("custom");
                format!("local:{name}{short_sha}")
            }
            (_, Some(adir)) => {
                let base = match &self.model_dir {
                    Some(mdir) => format!(
                        "local:{}",
                        mdir.file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("custom")
                    ),
                    None => self.family.repo_id().to_string(),
                };
                let name = adir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("adapter");
                format!("{base}+adapter:{name}{short_sha}")
            }
        }
    }

    /// Machine-independent SHA-256 fingerprint over base safetensors header and adapter weights.
    pub fn portable_fingerprint(&self) -> Result<String> {
        let mut h = Sha256::new();
        h.update(self.family.repo_id().as_bytes());
        let base_st = self.base_weights_dir().join("model.safetensors");
        if base_st.exists() {
            h.update(b"|base_hdr:");
            h.update(safetensors_header_sha256(&base_st)?.as_bytes());
        } else if let Some(mdir) = &self.model_dir {
            h.update(b"|model_dir:");
            h.update(mdir.to_string_lossy().as_bytes());
        }
        if let Some(adir) = &self.adapter_dir {
            let cfg_bytes = std::fs::read(adir.join("adapter_config.json"))
                .with_context(|| format!("Missing adapter_config.json in {}", adir.display()))?;
            h.update(b"|adapter_cfg:");
            h.update(&cfg_bytes);
            let adapter_st = adir.join("adapter_model.safetensors");
            h.update(b"|adapter_sha:");
            h.update(sha256_file(&adapter_st)?.as_bytes());
        }
        Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
    }

    /// Resolve CLI `--checkpoint <slug_or_dir_or_base+adapter>` and optional `--adapter <dir>`.
    pub fn resolve(checkpoint_arg: &str, adapter_arg: Option<&Path>) -> Result<Self> {
        let trimmed = checkpoint_arg.trim();

        // Support inline `base+adapter_dir` specification (e.g. in `compare` or `experiment`).
        if adapter_arg.is_none()
            && let Some((lhs, rhs)) = trimmed.split_once('+')
            && !lhs.is_empty()
            && !rhs.is_empty()
        {
            let rhs_path = Path::new(rhs.trim());
            if rhs_path.exists() || rhs.contains('/') || rhs.contains('\\') || rhs.starts_with('.')
            {
                return Self::resolve(lhs.trim(), Some(rhs_path));
            }
        }

        // First try canonical checkpoint slug if it doesn't look like an explicit filesystem path.
        let looks_like_path = trimmed.contains('/')
            || trimmed.contains('\\')
            || trimmed.starts_with('.')
            || Path::new(trimmed).exists();

        let mut model_dir: Option<PathBuf> = None;
        let mut inline_adapter_dir: Option<PathBuf> = None;

        let parsed_ckpt = if let Ok(ckpt) = Checkpoint::from_str(trimmed) {
            Some(ckpt)
        } else if looks_like_path {
            let raw_path = Path::new(trimmed);
            let dir = if raw_path.is_file() {
                raw_path
                    .parent()
                    .map(Path::to_path_buf)
                    .ok_or_else(|| anyhow!("Invalid model file path: {}", raw_path.display()))?
            } else {
                raw_path.to_path_buf()
            };
            if !dir.exists() {
                bail!(
                    "Model or adapter directory does not exist: {}",
                    dir.display()
                );
            }
            if dir.join("adapter_config.json").exists() && !dir.join("model.safetensors").exists() {
                inline_adapter_dir = Some(dir);
                None
            } else {
                let detected = detect_checkpoint_family(&dir)?;
                model_dir = Some(dir);
                Some(detected)
            }
        } else {
            return Err(anyhow!("{}", Checkpoint::from_str(trimmed).unwrap_err()));
        };

        let effective_adapter = match (inline_adapter_dir, adapter_arg) {
            (Some(a), Some(b)) if a != b => bail!(
                "Conflicting adapter directories: '{}' in --checkpoint vs '{}' in --adapter",
                a.display(),
                b.display()
            ),
            (Some(a), _) => Some(a),
            (None, Some(b)) => Some(normalize_adapter_dir(b)?),
            (None, None) => None,
        };

        if let Some(ref adir) = effective_adapter {
            validate_adapter_directory(adir)?;
            let inferred = infer_base_checkpoint_from_adapter(adir);
            let family = match (parsed_ckpt, inferred) {
                (None, Some(inf)) => inf,
                (None, None) => Checkpoint::English,
                (Some(ckpt), Some(inf)) => {
                    // When CLI `--checkpoint` kept its default `"english"` and `--adapter` points
                    // at a non-English checkpoint, honor the adapter's recorded base checkpoint.
                    if model_dir.is_none()
                        && trimmed.eq_ignore_ascii_case("english")
                        && inf != Checkpoint::English
                    {
                        inf
                    } else if head_paradigm_for(ckpt) != head_paradigm_for(inf)
                        || ckpt.backbone() != inf.backbone()
                    {
                        bail!(
                            "Adapter in '{}' was trained on {} ({}), which is incompatible with -c {} ({})",
                            adir.display(),
                            inf.repo_id(),
                            inf.backbone(),
                            ckpt.repo_id(),
                            ckpt.backbone()
                        );
                    } else {
                        ckpt
                    }
                }
                (Some(ckpt), None) => ckpt,
            };
            if family == Checkpoint::AutoRoute {
                bail!("Cannot attach a PEFT --adapter to dynamic meta-router checkpoint 'route'");
            }
            Ok(Self {
                family,
                model_dir,
                adapter_dir: Some(adir.clone()),
            })
        } else {
            let family = parsed_ckpt.unwrap_or(Checkpoint::English);
            Ok(Self {
                family,
                model_dir,
                adapter_dir: None,
            })
        }
    }

    /// Parse comma-separated model specs (e.g. `"english,english+runs/ft1/artifacts,models/custom"`).
    pub fn parse_list(csv_specs: &str) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for raw in csv_specs.split(',') {
            let s = raw.trim();
            if s.is_empty() {
                continue;
            }
            out.push(Self::resolve(s, None)?);
        }
        if out.is_empty() {
            bail!("At least one checkpoint or model directory must be specified");
        }
        Ok(out)
    }
}

fn normalize_adapter_dir(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        path.parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow!("Invalid adapter file path: {}", path.display()))
    } else {
        let nested = path.join("artifacts");
        if !path.join("adapter_config.json").exists() && nested.join("adapter_config.json").exists()
        {
            Ok(nested)
        } else {
            Ok(path.to_path_buf())
        }
    }
}

/// Validate that an adapter directory contains `adapter_config.json` and `adapter_model.safetensors`.
pub fn validate_adapter_directory(dir: &Path) -> Result<()> {
    if !dir.exists() {
        bail!("Adapter directory does not exist: {}", dir.display());
    }
    let cfg = dir.join("adapter_config.json");
    if !cfg.exists() {
        bail!(
            "Missing adapter_config.json in adapter directory {}",
            dir.display()
        );
    }
    let st = dir.join("adapter_model.safetensors");
    if !st.exists() {
        bail!(
            "Missing adapter_model.safetensors in adapter directory {}",
            dir.display()
        );
    }
    let _ = AdapterConfig::load(dir)?;
    let _ = inspect_safetensors(&st)?;
    Ok(())
}

/// Infer the canonical `Checkpoint` base from `training_manifest.json` or `adapter_config.json`.
pub fn infer_base_checkpoint_from_adapter(adapter_dir: &Path) -> Option<Checkpoint> {
    if let Some(manifest) = TrainingManifest::load_opt(adapter_dir) {
        if let Ok(c) = Checkpoint::from_str(&manifest.base_model.checkpoint_slug) {
            return Some(c);
        }
        if let Ok(c) = Checkpoint::from_str(&manifest.base_model.repo_id) {
            return Some(c);
        }
    }
    let cfg = AdapterConfig::load(adapter_dir).ok()?;
    let base = cfg.base_model_name_or_path?;
    if let Ok(c) = Checkpoint::from_str(&base) {
        return Some(c);
    }
    let lower = base.to_lowercase();
    if lower.contains("laya-multilingual") || lower.contains("mmbert") {
        Some(Checkpoint::Multilingual)
    } else if lower.contains("typed-decisions") {
        Some(Checkpoint::TypedDecisions)
    } else if lower.contains("laya") {
        Some(Checkpoint::English)
    } else if lower.contains("deberta") {
        Some(Checkpoint::DebertaV3ZeroShot)
    } else if lower.contains("modernbert") && lower.contains("nli") {
        Some(Checkpoint::ModernBertNli)
    } else if lower.contains("modernbert") {
        Some(Checkpoint::ModernBertZeroShot)
    } else if lower.contains("eurobert") {
        Some(Checkpoint::EuroBert)
    } else if lower.contains("neobert") {
        Some(Checkpoint::NeoBert)
    } else {
        None
    }
}

/// Detect the native `Checkpoint` loader family for an arbitrary local model directory.
pub fn detect_checkpoint_family(dir: &Path) -> Result<Checkpoint> {
    if let Some(manifest) = TrainingManifest::load_opt(dir) {
        if let Ok(c) = Checkpoint::from_str(&manifest.base_model.checkpoint_slug) {
            return Ok(c);
        }
        if let Ok(c) = Checkpoint::from_str(&manifest.base_model.repo_id) {
            return Ok(c);
        }
    }

    if dir.join("rl_agent_config.json").exists() || dir.join("encoder/config.json").exists() {
        let rl_val = std::fs::read_to_string(dir.join("rl_agent_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let enc_val = std::fs::read_to_string(dir.join("encoder/config.json"))
            .or_else(|_| std::fs::read_to_string(dir.join("config.json")))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let vocab = enc_val
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let max_len = rl_val.get("max_len").and_then(|v| v.as_u64()).unwrap_or(0);
        if vocab >= 200_000 {
            return Ok(Checkpoint::Multilingual);
        } else if max_len >= 1024 {
            return Ok(Checkpoint::TypedDecisions);
        } else {
            return Ok(Checkpoint::English);
        }
    }

    let spec = inspect_model_directory(dir)?;
    if let Ok(c) = Checkpoint::from_str(&spec.model_id) {
        return Ok(c);
    }
    let mt = spec.model_type.to_lowercase();
    if mt == "laya" || spec.detected_paradigm == super::head_dispatch::HeadParadigm::JointSinglePass
    {
        if spec.vocab_size >= 200_000 {
            Ok(Checkpoint::Multilingual)
        } else {
            Ok(Checkpoint::English)
        }
    } else if mt == "deberta-v2" || mt.contains("deberta") {
        Ok(Checkpoint::DebertaV3ZeroShot)
    } else if mt == "modernbert" {
        let cfg_raw = std::fs::read_to_string(dir.join("config.json")).unwrap_or_default();
        if cfg_raw.to_lowercase().contains("neutral") {
            Ok(Checkpoint::ModernBertNli)
        } else {
            Ok(Checkpoint::ModernBertZeroShot)
        }
    } else if mt == "eurobert" {
        Ok(Checkpoint::EuroBert)
    } else if mt == "neobert" {
        Ok(Checkpoint::NeoBert)
    } else {
        bail!(
            "Unsupported model_type '{}' in {}",
            spec.model_type,
            dir.display()
        )
    }
}

#[derive(Debug, Clone)]
struct LoraEntry {
    module_name: String,
    a_key: String,
    b_key: String,
    mag_key: Option<String>,
    is_embedding: bool,
}

#[derive(Debug)]
enum ParsedAdapterKey {
    LoraA {
        target_weight: String,
        module_name: String,
        is_embedding: bool,
    },
    LoraB {
        target_weight: String,
        module_name: String,
        is_embedding: bool,
    },
    DoraMag {
        target_weight: String,
        module_name: String,
    },
    BaseLayer {
        target_param: String,
    },
    FullOverride {
        target_param: String,
    },
    IgnoredShadow,
}

fn strip_peft_prefixes(mut key: &str) -> &str {
    while let Some(rest) = key.strip_prefix("base_model.model.") {
        key = rest;
    }
    key
}

fn strip_adapter_infix<'a>(suffix: &'a str, marker: &str) -> Option<&'a str> {
    let rest = suffix.strip_prefix(marker)?;
    if rest.is_empty() || rest == "weight" {
        return Some("");
    }
    if let Some(after_dot) = rest.strip_prefix('.') {
        if after_dot == "weight" {
            return Some("");
        }
        // `<adapter_name>.weight` or `<adapter_name>`
        if let Some((_, leaf)) = after_dot.split_once('.') {
            if leaf == "weight" {
                return Some("");
            }
        } else {
            return Some("");
        }
    }
    None
}

fn parse_adapter_key(raw_key: &str) -> ParsedAdapterKey {
    let s = strip_peft_prefixes(raw_key);

    if s.contains(".original_module.") {
        return ParsedAdapterKey::IgnoredShadow;
    }

    if let Some((module, rest)) = s.split_once(".modules_to_save.") {
        // Format: `<module>.modules_to_save.<adapter_name>.<param_leaf>`
        if let Some((_, leaf)) = rest.split_once('.') {
            return ParsedAdapterKey::FullOverride {
                target_param: format!("{module}.{leaf}"),
            };
        }
    }

    if let Some((module, rest)) = s.split_once(".base_layer.") {
        return ParsedAdapterKey::BaseLayer {
            target_param: format!("{module}.{rest}"),
        };
    }

    // Check LoRA linear & embedding markers
    for (marker, is_a, is_emb) in [
        (".lora_A", true, false),
        (".lora_B", false, false),
        (".lora_embedding_A", true, true),
        (".lora_embedding_B", false, true),
    ] {
        if let Some(idx) = s.find(marker) {
            let module = &s[..idx];
            let tail = &s[idx..];
            if strip_adapter_infix(tail, marker).is_some() {
                let target_weight = format!("{module}.weight");
                return if is_a {
                    ParsedAdapterKey::LoraA {
                        target_weight,
                        module_name: module.to_string(),
                        is_embedding: is_emb,
                    }
                } else {
                    ParsedAdapterKey::LoraB {
                        target_weight,
                        module_name: module.to_string(),
                        is_embedding: is_emb,
                    }
                };
            }
        }
    }

    if let Some(idx) = s.find(".lora_magnitude_vector") {
        let module = &s[..idx];
        let tail = &s[idx..];
        if strip_adapter_infix(tail, ".lora_magnitude_vector").is_some() {
            return ParsedAdapterKey::DoraMag {
                target_weight: format!("{module}.weight"),
                module_name: module.to_string(),
            };
        }
    }

    ParsedAdapterKey::FullOverride {
        target_param: s.to_string(),
    }
}

fn is_optional_unqueried_key(name: &str) -> bool {
    name.starts_with("act_head.")
        || name.contains(".act_head.")
        || name == "temperature"
        || name.ends_with(".temperature")
}

/// Zero-overhead `SimpleBackend` merging PEFT `LoRA` / `rsLoRA` / `DoRA` and `modules_to_save`
/// tensors into base Safetensors weights in `F32` at model load time.
pub struct PeftBackend {
    base: MmapedSafetensors,
    adapter: MmapedSafetensors,
    cfg: AdapterConfig,
    lora: HashMap<String, LoraEntry>,
    base_layer_overrides: HashMap<String, String>,
    overrides: HashMap<String, String>,
    consumed_lora: Mutex<HashSet<String>>,
    consumed_overrides: Mutex<HashSet<String>>,
}

impl PeftBackend {
    /// Open `base_safetensors_path` and `adapter_dir`, indexing and validating all PEFT keys.
    pub fn open(base_safetensors_path: &Path, adapter_dir: &Path) -> Result<Self> {
        let cfg = AdapterConfig::load(adapter_dir)?;
        let adapter_st_path = if adapter_dir.is_dir() {
            adapter_dir.join("adapter_model.safetensors")
        } else {
            adapter_dir.to_path_buf()
        };
        let adapter_report = inspect_safetensors(&adapter_st_path)?;

        // SAFETY: Both safetensors files are memory-mapped read-only for model loading.
        let base = unsafe { MmapedSafetensors::new(base_safetensors_path) }.with_context(|| {
            format!(
                "Failed to mmap base weights {}",
                base_safetensors_path.display()
            )
        })?;
        let adapter = unsafe { MmapedSafetensors::new(&adapter_st_path) }.with_context(|| {
            format!(
                "Failed to mmap adapter weights {}",
                adapter_st_path.display()
            )
        })?;

        #[derive(Default)]
        struct PartialLora {
            module_name: String,
            a_key: Option<String>,
            b_key: Option<String>,
            mag_key: Option<String>,
            is_embedding: bool,
        }

        let mut partials: BTreeMap<String, PartialLora> = BTreeMap::new();
        let mut base_layer_overrides = HashMap::new();
        let mut overrides = HashMap::new();

        for raw_key in adapter_report.tensor_shapes.keys() {
            match parse_adapter_key(raw_key) {
                ParsedAdapterKey::LoraA {
                    target_weight,
                    module_name,
                    is_embedding,
                } => {
                    let entry = partials.entry(target_weight).or_default();
                    entry.module_name = module_name;
                    entry.a_key = Some(raw_key.clone());
                    entry.is_embedding = is_embedding;
                }
                ParsedAdapterKey::LoraB {
                    target_weight,
                    module_name,
                    is_embedding,
                } => {
                    let entry = partials.entry(target_weight).or_default();
                    if entry.module_name.is_empty() {
                        entry.module_name = module_name;
                    }
                    entry.b_key = Some(raw_key.clone());
                    entry.is_embedding |= is_embedding;
                }
                ParsedAdapterKey::DoraMag {
                    target_weight,
                    module_name,
                } => {
                    let entry = partials.entry(target_weight).or_default();
                    if entry.module_name.is_empty() {
                        entry.module_name = module_name;
                    }
                    entry.mag_key = Some(raw_key.clone());
                }
                ParsedAdapterKey::BaseLayer { target_param } => {
                    base_layer_overrides.insert(target_param, raw_key.clone());
                }
                ParsedAdapterKey::FullOverride { target_param } => {
                    overrides.insert(target_param, raw_key.clone());
                }
                ParsedAdapterKey::IgnoredShadow => {}
            }
        }

        let mut lora = HashMap::with_capacity(partials.len());
        for (target_weight, p) in partials {
            let a_key = p.a_key.ok_or_else(|| {
                anyhow!(
                    "Incomplete LoRA adapter in {}: missing lora_A for '{}'",
                    adapter_st_path.display(),
                    target_weight
                )
            })?;
            let b_key = p.b_key.ok_or_else(|| {
                anyhow!(
                    "Incomplete LoRA adapter in {}: missing lora_B for '{}'",
                    adapter_st_path.display(),
                    target_weight
                )
            })?;
            if cfg.is_dora() && p.mag_key.is_none() {
                bail!(
                    "DoRA adapter in {} is missing lora_magnitude_vector for '{}'",
                    adapter_st_path.display(),
                    target_weight
                );
            }
            lora.insert(
                target_weight,
                LoraEntry {
                    module_name: p.module_name,
                    a_key,
                    b_key,
                    mag_key: p.mag_key,
                    is_embedding: p.is_embedding,
                },
            );
        }

        if lora.is_empty() && overrides.is_empty() {
            bail!(
                "Adapter {} contains no recognizable LoRA or modules_to_save tensors",
                adapter_st_path.display()
            );
        }

        Ok(Self {
            base,
            adapter,
            cfg,
            lora,
            base_layer_overrides,
            overrides,
            consumed_lora: Mutex::new(HashSet::new()),
            consumed_overrides: Mutex::new(HashSet::new()),
        })
    }

    /// Construct a `VarBuilder` backed by an `Arc<PeftBackend>` so the caller can run
    /// `verify_all_consumed()` after model construction completes.
    pub fn into_var_builder(self: &Arc<Self>, dtype: DType, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(
            Box::new(SharedPeftBackend(Arc::clone(self))),
            dtype,
            dev.clone(),
        )
    }

    /// Number of LoRA/DoRA adapted weight matrices in this adapter.
    pub fn lora_count(&self) -> usize {
        self.lora.len()
    }

    /// Number of full-tensor `modules_to_save` / bias overrides in this adapter.
    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    /// Verify that every LoRA module and `modules_to_save` tensor in `adapter_model.safetensors`
    /// was actually consumed when loading the target model.
    pub fn verify_all_consumed(&self) -> Result<()> {
        let used_lora = self
            .consumed_lora
            .lock()
            .map_err(|_| anyhow!("PeftBackend consumed_lora lock poisoned"))?;
        let used_ov = self
            .consumed_overrides
            .lock()
            .map_err(|_| anyhow!("PeftBackend consumed_overrides lock poisoned"))?;

        let mut unused: Vec<&str> = self
            .lora
            .keys()
            .filter(|k| !used_lora.contains(*k) && !is_optional_unqueried_key(k))
            .map(String::as_str)
            .chain(
                self.overrides
                    .keys()
                    .filter(|k| {
                        !self.lora.contains_key(*k)
                            && !used_ov.contains(*k)
                            && !is_optional_unqueried_key(k)
                    })
                    .map(String::as_str),
            )
            .collect();
        unused.sort_unstable();

        if !unused.is_empty() {
            let preview: Vec<&str> = unused.iter().copied().take(8).collect();
            bail!(
                "Adapter architecture mismatch: {} adapter tensor(s) were not matched by the base model (e.g. {:?}). Verify that `--checkpoint` matches the base model the adapter was trained on.",
                unused.len(),
                preview
            );
        }
        Ok(())
    }

    fn load_merged(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        // 1. Direct full-tensor override (`modules_to_save` or trained bias) when no LoRA targets `name`.
        if !self.lora.contains_key(name)
            && let Some(raw_ov) = self.overrides.get(name)
        {
            if let Ok(mut guard) = self.consumed_overrides.lock() {
                guard.insert(name.to_string());
            }
            return self.adapter.load(raw_ov, dev)?.to_dtype(dtype);
        }

        // 2. Load base weight (preferring `base_layer` in adapter if present, else base safetensors).
        let base_tensor = if let Some(raw_base) = self.base_layer_overrides.get(name) {
            self.adapter.load(raw_base, dev)?
        } else if let Some(raw_ov) = self.overrides.get(name) {
            if let Ok(mut guard) = self.consumed_overrides.lock() {
                guard.insert(name.to_string());
            }
            self.adapter.load(raw_ov, dev)?
        } else {
            self.base.load(name, dev)?
        };

        let Some(entry) = self.lora.get(name) else {
            return base_tensor.to_dtype(dtype);
        };

        if let Ok(mut guard) = self.consumed_lora.lock() {
            guard.insert(name.to_string());
        }

        // Perform all LoRA / rsLoRA / DoRA arithmetic in F32 for numerical parity with PyTorch.
        let w = base_tensor.to_dtype(DType::F32)?;
        let a = self.adapter.load(&entry.a_key, dev)?.to_dtype(DType::F32)?;
        let b = self.adapter.load(&entry.b_key, dev)?.to_dtype(DType::F32)?;
        let r = a.dim(0)?;
        let scale = self
            .cfg
            .scaling_for(&entry.module_name, r)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let use_dora = self.cfg.is_dora() || entry.mag_key.is_some();

        let merged_f32 = if entry.is_embedding {
            // Embedding w: (vocab, dim), lora_embedding_A: (r, vocab), lora_embedding_B: (dim, r)
            let delta = (a.t()?.contiguous()?.matmul(&b.t()?.contiguous()?)? * scale)?;
            let v = (&w + &delta)?;
            if use_dora {
                let mag_key = entry.mag_key.as_deref().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "DoRA missing lora_magnitude_vector for {name}"
                    ))
                })?;
                let m = self
                    .adapter
                    .load(mag_key, dev)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?;
                // Embedding output features correspond to dim=1 (`dim`), or vocab rows if m matches vocab.
                if m.elem_count() == v.dim(0)? {
                    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12f32, f32::MAX)?;
                    let factor = m.reshape((v.dim(0)?, 1))?.broadcast_div(&norm)?;
                    v.broadcast_mul(&factor)?
                } else {
                    let norm = v.sqr()?.sum_keepdim(0)?.sqrt()?.clamp(1e-12f32, f32::MAX)?;
                    let factor = m.reshape((1, v.dim(1)?))?.broadcast_div(&norm)?;
                    v.broadcast_mul(&factor)?
                }
            } else {
                v
            }
        } else {
            // Linear w: (d_out, d_in) when !fan_in_fan_out; (d_in, d_out) when fan_in_fan_out.
            // B @ A is always (d_out, d_in).
            let ba_scaled = (b.matmul(&a)? * scale)?;
            let w_oriented = if self.cfg.fan_in_fan_out {
                w.t()?.contiguous()?
            } else {
                w
            };
            let v = (&w_oriented + &ba_scaled)?;
            let v_final = if use_dora {
                let mag_key = entry.mag_key.as_deref().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "DoRA missing lora_magnitude_vector for {name}"
                    ))
                })?;
                let m = self
                    .adapter
                    .load(mag_key, dev)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?;
                let d_out = v.dim(0)?;
                if m.elem_count() != d_out {
                    return Err(candle_core::Error::Msg(format!(
                        "DoRA magnitude vector length {} does not match out_features {} for '{}'",
                        m.elem_count(),
                        d_out,
                        name
                    )));
                }
                // PyTorch `dora.py`: `weight_norm = torch.linalg.norm(weight, dim=1)`
                let row_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12f32, f32::MAX)?;
                let factor = m.reshape((d_out, 1))?.broadcast_div(&row_norm)?;
                v.broadcast_mul(&factor)?
            } else {
                v
            };
            if self.cfg.fan_in_fan_out {
                v_final.t()?.contiguous()?
            } else {
                v_final
            }
        };

        merged_f32.to_dtype(dtype)
    }
}

impl SimpleBackend for PeftBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_merged(name, dtype, dev)?;
        if tensor.shape() != &s {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_merged(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.overrides.contains_key(name)
            || self.base_layer_overrides.contains_key(name)
            || self.base.get(name).is_ok()
    }
}

/// Newtype wrapper satisfying Rust's orphan rule for `Arc<PeftBackend>`.
#[derive(Clone)]
pub struct SharedPeftBackend(pub Arc<PeftBackend>);

impl SimpleBackend for SharedPeftBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.0.get(s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.0.get_unchecked(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }
}

/// Summary report emitted by `merge_adapter_to_dir`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeReport {
    pub base_dir: PathBuf,
    pub adapter_dir: PathBuf,
    pub output_dir: PathBuf,
    pub total_tensors_written: usize,
    pub lora_tensors_merged: usize,
    pub override_tensors_applied: usize,
    pub merged_file_bytes: u64,
    pub merged_header_sha256: String,
}

fn parse_safetensors_dtype(dtype_str: &str) -> DType {
    match dtype_str.to_ascii_uppercase().as_str() {
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F64" => DType::F64,
        "I64" => DType::I64,
        "U32" => DType::U32,
        "U8" => DType::U8,
        _ => DType::F32,
    }
}

fn copy_if_exists(src: &Path, dst: &Path) -> Result<()> {
    if src.is_file() {
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(src, dst)
            .with_context(|| format!("Failed to copy {} to {}", src.display(), dst.display()))?;
    }
    Ok(())
}

fn copy_subdir_files_if_exists(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    if !src_dir.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(dst_dir)?;
    for entry in std::fs::read_dir(src_dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() {
            let dst = dst_dir.join(entry.file_name());
            std::fs::copy(&p, &dst)?;
        }
    }
    Ok(())
}

/// Merge a PEFT adapter (`adapter_dir`) into the standalone base checkpoint (`base_dir`) and
/// write a complete standalone checkpoint directory to `output_dir`.
pub fn merge_adapter_to_dir(
    base_dir: &Path,
    adapter_dir: &Path,
    output_dir: &Path,
    force_f32: bool,
) -> Result<MergeReport> {
    let base_st_path = base_dir.join("model.safetensors");
    if !base_st_path.exists() {
        bail!(
            "Base checkpoint weights not found at {}. Run `lemmata download` first.",
            base_st_path.display()
        );
    }
    validate_adapter_directory(adapter_dir)?;

    let base_report = inspect_safetensors(&base_st_path)?;
    let adapter_report = inspect_safetensors(&adapter_dir.join("adapter_model.safetensors"))?;
    let peft = PeftBackend::open(&base_st_path, adapter_dir)?;
    let dev = Device::Cpu;

    let mut merged: HashMap<String, Tensor> =
        HashMap::with_capacity(base_report.total_tensors + peft.override_count());

    for (name, dtype_str) in &base_report.tensor_dtypes {
        let out_dtype = if force_f32 {
            DType::F32
        } else {
            parse_safetensors_dtype(dtype_str)
        };
        let t = peft.get_unchecked(name, out_dtype, &dev)?;
        merged.insert(name.clone(), t);
    }

    // Include any new tensors introduced solely in `modules_to_save` (e.g. a newly added head).
    for (target_name, raw_key) in &peft.overrides {
        if !merged.contains_key(target_name) {
            let out_dtype = if force_f32 {
                DType::F32
            } else {
                adapter_report
                    .tensor_dtypes
                    .get(raw_key)
                    .map(|d| parse_safetensors_dtype(d))
                    .unwrap_or(DType::F32)
            };
            let t = peft.get_unchecked(target_name, out_dtype, &dev)?;
            merged.insert(target_name.clone(), t);
        }
    }

    peft.verify_all_consumed()?;

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory {}", output_dir.display()))?;
    let out_st = output_dir.join("model.safetensors");
    let tmp_st = output_dir.join("model.safetensors.tmp");
    candle_core::safetensors::save(&merged, &tmp_st).with_context(|| {
        format!(
            "Failed to serialize merged safetensors to {}",
            tmp_st.display()
        )
    })?;
    std::fs::rename(&tmp_st, &out_st)?;

    // Copy config & tokenizer metadata from base_dir (covering both HF flat and Laya nested layouts).
    for filename in [
        "config.json",
        "gliclass_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",
        "vocab.json",
        "merges.txt",
    ] {
        copy_if_exists(&base_dir.join(filename), &output_dir.join(filename))?;
    }
    // Strip stale zero-shot temperatures when copying base `rl_agent_config.json` into a fine-tuned dir.
    let base_rl_cfg = base_dir.join("rl_agent_config.json");
    if base_rl_cfg.is_file() {
        if let Ok(raw) = std::fs::read_to_string(&base_rl_cfg)
            && let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&raw)
        {
            if let Some(obj) = val.as_object_mut() {
                obj.remove("temperature");
                obj.remove("temperature_by_options");
            }
            std::fs::write(
                output_dir.join("rl_agent_config.json"),
                serde_json::to_string_pretty(&val)?,
            )?;
        } else {
            copy_if_exists(&base_rl_cfg, &output_dir.join("rl_agent_config.json"))?;
        }
    }
    copy_subdir_files_if_exists(&base_dir.join("encoder"), &output_dir.join("encoder"))?;
    copy_subdir_files_if_exists(&base_dir.join("tokenizer"), &output_dir.join("tokenizer"))?;

    // Overlay updated configs / calibration / provenance from adapter_dir.
    for filename in [
        "rl_agent_config.json",
        "lemmata_calibration.json",
        "config.json",
    ] {
        copy_if_exists(&adapter_dir.join(filename), &output_dir.join(filename))?;
    }

    let merged_bytes = std::fs::metadata(&out_st)?.len();
    let merged_header_sha256 = safetensors_header_sha256(&out_st)?;

    if let Some(mut manifest) = TrainingManifest::load_opt(adapter_dir) {
        manifest.artifacts.merged_safetensors_header_sha256 = Some(merged_header_sha256.clone());
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(output_dir.join("training_manifest.json"), json)?;
    }
    copy_if_exists(
        &adapter_dir.join("trained_inputs.sha256.bin"),
        &output_dir.join("trained_inputs.sha256.bin"),
    )?;

    Ok(MergeReport {
        base_dir: base_dir.to_path_buf(),
        adapter_dir: adapter_dir.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        total_tensors_written: merged.len(),
        lora_tensors_merged: peft.lora_count(),
        override_tensors_applied: peft.override_count(),
        merged_file_bytes: merged_bytes,
        merged_header_sha256,
    })
}
