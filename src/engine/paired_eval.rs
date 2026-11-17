//! Statistically rigorous paired model experimentation (`base` vs `candidate` / fine-tuned):
//! 5-layer zero-leakage provenance guard, fair post-hoc `cal`-split calibration, Lancaster
//! McNemar mid-p & exact binomial tests, group-aware paired BCa/percentile bootstrap CIs,
//! Phipson-Smyth sign-flip permutation tests, Holm-Bonferroni multiplicity correction,
//! log-softmax NLL, top-label ECE & equal-mass ACE, AURC/E-AURC, conformal & deferral
//! efficiency deltas, per-label F1 deltas, and 2x2 prediction flip matrices.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, Color, ContentArrangement, Table};
use serde::{Deserialize, Serialize};

use crate::datasource::DataItem;
use crate::engine::cache::compute_schema_hash;
use crate::engine::calibrate::{DEFAULT_MIN_BUCKET_SAMPLES, MAX_TEMPERATURE, MIN_TEMPERATURE};
use crate::engine::cascade::{CascadeRouter, fit_samples};
use crate::engine::classifier::{ClassifierConfig, LayaClassifier};
use crate::engine::dataset::{
    DatasetManifest, DeterministicRng, TrainedInputsFilter, compute_item_set_digest,
    compute_state_merkle_sha256, file_sha256_hex,
};
use crate::engine::metrics::resolve_ground_truth_for_question;
use crate::engine::probe::choice_labels;
use crate::engine::slicing::{SliceBy, extract_item_slice};
use crate::model::{DecisionAnswer, DecisionResult, Question, QuestionType};

/// Calibration comparison protocol for `PairedExperimentRunner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CalibrationMode {
    /// Evaluate `base` and `candidate` with their as-configured temperatures only.
    Raw,
    /// Fit fresh post-hoc temperatures, conformal quantiles, and deferral thresholds on `cal_data` for both models.
    Calibrated,
    /// Run both `Raw` and `Calibrated` modes when `cal_data` is supplied (falls back to `Raw` otherwise).
    #[default]
    Both,
}

impl std::str::FromStr for CalibrationMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "raw" | "uncalibrated" => Ok(Self::Raw),
            "calibrated" | "fair" | "cal" => Ok(Self::Calibrated),
            "both" | "all" => Ok(Self::Both),
            other => bail!("Unknown calibration mode '{other}'. Options: raw, calibrated, both"),
        }
    }
}

/// Automated promotion gate verdict emitted by `PairedExperimentRunner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExperimentVerdict {
    Promote,
    PromoteWithCalibration,
    Inconclusive,
    Regressed,
}

impl ExperimentVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Promote => "PROMOTE",
            Self::PromoteWithCalibration => "PROMOTE_WITH_CALIBRATION",
            Self::Inconclusive => "INCONCLUSIVE",
            Self::Regressed => "REGRESSED",
        }
    }
}

/// One model participating in a paired head-to-head experiment.
pub struct ExperimentParticipant {
    pub label: String,
    pub classifier: LayaClassifier,
    /// Optional model/adapter artifact directory containing `training_manifest.json` and `trained_inputs.sha256.bin`.
    pub artifact_dir: Option<PathBuf>,
}

impl ExperimentParticipant {
    pub fn new(label: impl Into<String>, classifier: LayaClassifier) -> Self {
        Self {
            label: label.into(),
            classifier,
            artifact_dir: None,
        }
    }

    pub fn with_artifact_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.artifact_dir = Some(dir.into());
        self
    }
}

/// Configuration for `PairedExperimentRunner`.
#[derive(Debug, Clone)]
pub struct PairedExperimentConfig {
    pub calibration_mode: CalibrationMode,
    pub bootstrap_samples: usize,
    pub permutations: usize,
    /// Confidence level for bootstrap CIs and deferral certification (default `0.95`).
    pub confidence: f64,
    /// Significance threshold `alpha` for conformal sets and hypothesis tests (default `0.05`).
    pub alpha: f64,
    /// Target accuracy for System 1 / System 2 deferral policy fitting (default `0.95`).
    pub target_accuracy: f64,
    pub system2_cost: f64,
    pub seed: u64,
    pub slice_by: Option<SliceBy>,
    pub warmup_runs: usize,
    pub timing_iterations: usize,
    /// Override hard failure when train/val/cal leakage is detected in `test_items`.
    pub allow_leakage: bool,
    pub top_regressions: usize,
}

impl Default for PairedExperimentConfig {
    fn default() -> Self {
        Self {
            calibration_mode: CalibrationMode::Both,
            bootstrap_samples: 2000,
            permutations: 5000,
            confidence: 0.95,
            alpha: 0.05,
            target_accuracy: 0.95,
            system2_cost: 0.005,
            seed: 42,
            slice_by: None,
            warmup_runs: 1,
            timing_iterations: 1,
            allow_leakage: false,
            top_regressions: 5,
        }
    }
}

// ============================================================================
// Provenance & Zero-Leakage Guard (`training_manifest.json` + `dataset_manifest.json`)
// ============================================================================

#[derive(Debug, Clone, Default, Deserialize)]
struct TrainingManifestSummary {
    #[serde(default)]
    manifest_version: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    base_model: Option<TrainingBaseModel>,
    #[serde(default)]
    base_checkpoint: Option<String>,
    #[serde(default)]
    schema: Option<TrainingSchemaSection>,
    #[serde(default)]
    schema_digest: Option<String>,
    #[serde(default)]
    dataset_provenance: Option<TrainingDatasetProvenance>,
    #[serde(default)]
    train_split_sha256: Option<String>,
    #[serde(default)]
    val_split_sha256: Option<String>,
    #[serde(default)]
    cal_split_sha256: Option<String>,
    #[serde(default)]
    artifacts: Option<TrainingArtifactsSection>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TrainingBaseModel {
    #[serde(default)]
    checkpoint_slug: Option<String>,
    #[serde(default)]
    repo_id: Option<String>,
    #[serde(default)]
    remote_base_header_sha256: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TrainingSchemaSection {
    #[serde(default)]
    schema_sha256: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TrainingSplitHashEntry {
    #[serde(default)]
    file_sha256: Option<String>,
    #[serde(default)]
    state_merkle_sha256: Option<String>,
    #[serde(default)]
    item_set_digest: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct LockedTestGuard {
    #[serde(default)]
    expected_test_file_sha256: Option<String>,
    #[serde(default)]
    trained_inputs_filter_file: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TrainingDatasetProvenance {
    #[serde(default)]
    split_id: Option<String>,
    #[serde(default)]
    train: Option<TrainingSplitHashEntry>,
    #[serde(default)]
    val: Option<TrainingSplitHashEntry>,
    #[serde(default)]
    cal: Option<TrainingSplitHashEntry>,
    #[serde(default)]
    locked_test_guard: Option<LockedTestGuard>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TrainingArtifactsSection {
    #[serde(default)]
    adapter_safetensors_sha256: Option<String>,
    #[serde(default)]
    merged_safetensors_header_sha256: Option<String>,
    #[serde(default)]
    calibration_sha256: Option<String>,
}

/// Cryptographic provenance and leakage verification report for a paired experiment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceAuditReport {
    pub verified: bool,
    pub leakage_detected: bool,
    pub leakage_overridden: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_id: Option<String>,
    pub test_item_set_digest: String,
    pub test_state_merkle_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_file_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub leaked_item_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Auto-infer the base checkpoint slug or repo ID from `<artifact_dir>/training_manifest.json`
/// or `<artifact_dir>/adapter_config.json`.
pub fn infer_base_checkpoint_from_artifact(artifact_dir: &Path) -> Option<String> {
    let tm_path = artifact_dir.join("training_manifest.json");
    if let Ok(text) = std::fs::read_to_string(&tm_path)
        && let Ok(tm) = serde_json::from_str::<TrainingManifestSummary>(&text)
    {
        if let Some(bm) = tm.base_model {
            if let Some(slug) = bm.checkpoint_slug.filter(|s| !s.is_empty()) {
                return Some(slug);
            }
            if let Some(repo) = bm.repo_id.filter(|s| !s.is_empty()) {
                return Some(repo);
            }
        }
        if let Some(bc) = tm.base_checkpoint.filter(|s| !s.is_empty()) {
            return Some(bc);
        }
    }
    let peft_path = artifact_dir.join("adapter_config.json");
    if let Ok(text) = std::fs::read_to_string(&peft_path)
        && let Ok(val) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(base) = val
            .get("base_model_name_or_path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    {
        return Some(base.to_string());
    }
    None
}

/// Run the 5-layer provenance & zero-leakage verification before paired evaluation.
pub fn verify_experiment_provenance(
    test_items: &[DataItem],
    test_path: Option<&Path>,
    questions: &BTreeMap<String, Question>,
    base_checkpoint_id: Option<&str>,
    dataset_manifest_path: Option<&Path>,
    candidate_artifact_dir: Option<&Path>,
    allow_leakage: bool,
) -> Result<ProvenanceAuditReport> {
    verify_experiment_provenance_with_base_dir(
        test_items,
        test_path,
        questions,
        base_checkpoint_id,
        dataset_manifest_path,
        candidate_artifact_dir,
        None,
        allow_leakage,
    )
}

/// Run the 5-layer provenance & zero-leakage verification before paired evaluation, including
/// Layer 4 base Safetensors header verification against `candidate_base_dir`.
#[allow(clippy::too_many_arguments)]
pub fn verify_experiment_provenance_with_base_dir(
    test_items: &[DataItem],
    test_path: Option<&Path>,
    questions: &BTreeMap<String, Question>,
    base_checkpoint_id: Option<&str>,
    dataset_manifest_path: Option<&Path>,
    candidate_artifact_dir: Option<&Path>,
    candidate_base_dir: Option<&Path>,
    allow_leakage: bool,
) -> Result<ProvenanceAuditReport> {
    let mut report = ProvenanceAuditReport {
        verified: true,
        leakage_detected: false,
        leakage_overridden: false,
        test_item_set_digest: compute_item_set_digest(test_items),
        test_state_merkle_sha256: compute_state_merkle_sha256(test_items),
        test_file_sha256: test_path.and_then(|p| file_sha256_hex(p).ok()),
        ..Default::default()
    };

    let schema_sha = compute_schema_hash(questions);
    let mut hard_errors = Vec::new();
    let mut leaked_ids_set: BTreeSet<String> = BTreeSet::new();

    // Resolve optional `dataset_manifest.json`
    let resolved_dm_path = dataset_manifest_path.map(Path::to_path_buf).or_else(|| {
        test_path
            .and_then(|p| p.parent())
            .map(|dir| dir.join("dataset_manifest.json"))
            .filter(|p| p.exists())
    });

    if let Some(dm_path) = &resolved_dm_path
        && dm_path.exists()
    {
        let dm: DatasetManifest = serde_json::from_reader(
            File::open(dm_path).with_context(|| format!("Failed to open {}", dm_path.display()))?,
        )?;
        report.split_id = Some(dm.split_id.clone());
        if dm.schema_digest != schema_sha {
            report.warnings.push(format!(
                "Question schema digest differs from dataset_manifest.json ({} vs {})",
                &schema_sha[..12],
                &dm.schema_digest[..12.min(dm.schema_digest.len())]
            ));
        }
        for forbidden in ["train", "val", "cal"] {
            if let Some(meta) = dm.splits.get(forbidden)
                && (report.test_file_sha256.as_deref() == Some(meta.sha256.as_str())
                    || report.test_item_set_digest == meta.item_set_digest
                    || report.test_state_merkle_sha256 == meta.state_merkle_sha256)
            {
                report.leakage_detected = true;
                hard_errors.push(format!(
                    "Test dataset matches '{forbidden}' split in {} (split_id={})!",
                    dm_path.display(),
                    dm.split_id
                ));
            }
        }
        // Check row-level overlap against sibling train/val/cal files if present in split dir
        if let Some(split_dir) = dm_path.parent() {
            let mut non_test_items = Vec::new();
            for forbidden in ["train", "val", "cal"] {
                let fp = split_dir.join(format!("{forbidden}.jsonl"));
                if fp.exists()
                    && test_path.is_none_or(|tp| tp != fp)
                    && let Ok(its) = crate::engine::dataset::read_dataset_items(&fp)
                {
                    non_test_items.extend(its);
                }
            }
            if !non_test_items.is_empty() {
                let filter = TrainedInputsFilter::from_items(&non_test_items);
                let ov = filter.check_overlap(test_items);
                if ov.has_overlap() {
                    report.leakage_detected = true;
                    for id in ov
                        .exact_overlap_ids
                        .into_iter()
                        .chain(ov.normalized_overlap_ids)
                    {
                        leaked_ids_set.insert(id);
                    }
                    hard_errors.push(format!(
                        "Test items overlap with sibling train/val/cal splits in {}",
                        split_dir.display()
                    ));
                }
            }
        }
    }

    // Check candidate `training_manifest.json` + `trained_inputs.sha256.bin`
    if let Some(art_dir) = candidate_artifact_dir {
        let tm_path = art_dir.join("training_manifest.json");
        if tm_path.exists() {
            let tm: TrainingManifestSummary = serde_json::from_reader(
                File::open(&tm_path)
                    .with_context(|| format!("Failed to open {}", tm_path.display()))?,
            )?;
            let _ = &tm.manifest_version;
            report.run_id = tm.run_id.clone();

            if let Some(status) = tm.status.as_deref().filter(|s| !s.is_empty())
                && status != "completed"
            {
                bail!(
                    "Artifact integrity check failed: training run status is '{}' (expected 'completed') in {}",
                    status,
                    tm_path.display()
                );
            }

            // Layer 1: Adapter / merged weights & calibration SHA-256 tamper checks
            if let Some(expected_sha) = tm
                .artifacts
                .as_ref()
                .and_then(|a| a.adapter_safetensors_sha256.as_deref())
                .filter(|s| !s.is_empty())
            {
                let adapter_st = art_dir.join("adapter_model.safetensors");
                if !adapter_st.exists() {
                    bail!(
                        "Artifact integrity check failed: missing {} required by training_manifest.json",
                        adapter_st.display()
                    );
                }
                let actual_sha = file_sha256_hex(&adapter_st)?;
                if !actual_sha.eq_ignore_ascii_case(expected_sha) {
                    bail!(
                        "Artifact integrity check failed: adapter_model.safetensors SHA-256 ({}) != training_manifest.json ({})",
                        &actual_sha[..16],
                        &expected_sha[..16.min(expected_sha.len())]
                    );
                }
            }
            if let Some(expected_hdr) = tm
                .artifacts
                .as_ref()
                .and_then(|a| a.merged_safetensors_header_sha256.as_deref())
                .filter(|s| !s.is_empty())
            {
                let merged_st = art_dir.join("model.safetensors");
                if !merged_st.exists() {
                    bail!(
                        "Artifact integrity check failed: missing merged weights {} required by training_manifest.json",
                        merged_st.display()
                    );
                }
                let actual_hdr = crate::model::safetensors_header_sha256(&merged_st)?;
                if !actual_hdr.eq_ignore_ascii_case(expected_hdr) {
                    bail!(
                        "Artifact integrity check failed: merged model.safetensors header SHA-256 ({}) != training_manifest.json ({})",
                        &actual_hdr[..16.min(actual_hdr.len())],
                        &expected_hdr[..16.min(expected_hdr.len())]
                    );
                }
            }
            if let Some(expected_cal_sha) = tm
                .artifacts
                .as_ref()
                .and_then(|a| a.calibration_sha256.as_deref())
                .filter(|s| !s.is_empty())
            {
                let cal_path = art_dir.join("lemmata_calibration.json");
                if !cal_path.exists() {
                    bail!(
                        "Artifact integrity check failed: missing {} required by training_manifest.json",
                        cal_path.display()
                    );
                }
                let actual_cal_sha = file_sha256_hex(&cal_path)?;
                if !actual_cal_sha.eq_ignore_ascii_case(expected_cal_sha) {
                    bail!(
                        "Artifact integrity check failed: lemmata_calibration.json SHA-256 ({}) != training_manifest.json ({})",
                        &actual_cal_sha[..16],
                        &expected_cal_sha[..16.min(expected_cal_sha.len())]
                    );
                }
            }

            // Layer 4: Base Safetensors header SHA-256 compatibility check + schema/checkpoint warnings
            if let Some(remote_hdr) = tm
                .base_model
                .as_ref()
                .and_then(|b| b.remote_base_header_sha256.as_deref())
                .filter(|s| !s.is_empty())
            {
                let fallback_base = tm.base_model.as_ref().and_then(|bm| {
                    bm.checkpoint_slug
                        .as_deref()
                        .and_then(|s| s.parse::<crate::model::Checkpoint>().ok())
                        .or_else(|| {
                            bm.repo_id
                                .as_deref()
                                .and_then(|r| r.parse::<crate::model::Checkpoint>().ok())
                        })
                        .map(|c| c.local_dir())
                });
                let resolved_base_dir = candidate_base_dir.map(Path::to_path_buf).or(fallback_base);
                if let Some(bdir) = resolved_base_dir
                    && (art_dir.join("adapter_model.safetensors").exists() || bdir != art_dir)
                {
                    let base_st = bdir.join("model.safetensors");
                    if base_st.exists() {
                        let local_hdr = crate::model::safetensors_header_sha256(&base_st)?;
                        if !local_hdr.eq_ignore_ascii_case(remote_hdr) {
                            bail!(
                                "Base checkpoint header mismatch: candidate adapter was trained against safetensors header {}, but local base {} has header {}",
                                &remote_hdr[..16.min(remote_hdr.len())],
                                base_st.display(),
                                &local_hdr[..16.min(local_hdr.len())]
                            );
                        }
                    }
                }
            }
            if let Some(tm_schema_sha) = tm
                .schema
                .as_ref()
                .and_then(|s| s.schema_sha256.as_deref())
                .or(tm.schema_digest.as_deref())
                .filter(|s| !s.is_empty())
                && !tm_schema_sha.eq_ignore_ascii_case(&schema_sha)
            {
                report.warnings.push(format!(
                    "Evaluation schema SHA-256 ({}) differs from training_manifest.json ({})",
                    &schema_sha[..12],
                    &tm_schema_sha[..12.min(tm_schema_sha.len())]
                ));
            }
            if let Some(base_id) = base_checkpoint_id
                && let Some(bm) = &tm.base_model
            {
                let matches_slug = bm
                    .checkpoint_slug
                    .as_deref()
                    .is_some_and(|s| s.eq_ignore_ascii_case(base_id));
                let matches_repo = bm
                    .repo_id
                    .as_deref()
                    .is_some_and(|r| r.eq_ignore_ascii_case(base_id));
                if (bm.checkpoint_slug.is_some() || bm.repo_id.is_some())
                    && !matches_slug
                    && !matches_repo
                {
                    report.warnings.push(format!(
                        "Base model '{base_id}' differs from candidate's training base ({:?} / {:?})",
                        bm.checkpoint_slug, bm.repo_id
                    ));
                }
            }

            // Layer 2: Split file & state merkle guards against `dataset_provenance`
            if let Some(dp) = &tm.dataset_provenance {
                if report.split_id.is_none() {
                    report.split_id = dp.split_id.clone();
                }
                for (name, entry_opt) in [("train", &dp.train), ("val", &dp.val), ("cal", &dp.cal)]
                {
                    if let Some(entry) = entry_opt {
                        if let (Some(tf), Some(ef)) = (&report.test_file_sha256, &entry.file_sha256)
                            && tf.eq_ignore_ascii_case(ef)
                        {
                            report.leakage_detected = true;
                            hard_errors.push(format!(
                                "Test file SHA-256 matches fine-tuning '{name}' split in training_manifest.json!"
                            ));
                        }
                        if let Some(em) = &entry.state_merkle_sha256
                            && em.eq_ignore_ascii_case(&report.test_state_merkle_sha256)
                        {
                            report.leakage_detected = true;
                            hard_errors.push(format!(
                                "Test state Merkle digest matches fine-tuning '{name}' split!"
                            ));
                        }
                        if let Some(eid) = &entry.item_set_digest
                            && eid.eq_ignore_ascii_case(&report.test_item_set_digest)
                        {
                            report.leakage_detected = true;
                            hard_errors.push(format!(
                                "Test item-set digest matches fine-tuning '{name}' split!"
                            ));
                        }
                    }
                }
                if let Some(ltg) = &dp.locked_test_guard
                    && let (Some(expected_test), Some(actual_test)) = (
                        ltg.expected_test_file_sha256
                            .as_deref()
                            .filter(|s| !s.is_empty()),
                        report.test_file_sha256.as_deref(),
                    )
                    && !expected_test.eq_ignore_ascii_case(actual_test)
                {
                    report.warnings.push(format!(
                        "Test file SHA-256 ({}) differs from locked_test_guard ({})",
                        &actual_test[..12.min(actual_test.len())],
                        &expected_test[..12.min(expected_test.len())]
                    ));
                }
            }

            for (name, flat_sha) in [
                ("train", &tm.train_split_sha256),
                ("val", &tm.val_split_sha256),
                ("cal", &tm.cal_split_sha256),
            ] {
                if let (Some(tf), Some(fs)) = (&report.test_file_sha256, flat_sha)
                    && tf.eq_ignore_ascii_case(fs)
                {
                    report.leakage_detected = true;
                    hard_errors.push(format!(
                        "Test file SHA-256 matches `{name}_split_sha256` in training_manifest.json!"
                    ));
                }
            }

            // Layer 3: Binary prefix filter (`trained_inputs.sha256.bin`)
            let filter_rel = tm
                .dataset_provenance
                .as_ref()
                .and_then(|d| d.locked_test_guard.as_ref())
                .and_then(|g| g.trained_inputs_filter_file.as_deref())
                .unwrap_or("trained_inputs.sha256.bin");
            let filter_path = art_dir.join(filter_rel);
            if filter_path.exists() {
                let filter = TrainedInputsFilter::load_from_file(&filter_path)?;
                let ov = filter.check_overlap(test_items);
                if ov.has_overlap() {
                    report.leakage_detected = true;
                    let cnt = ov.total_overlapping_items();
                    for id in ov
                        .exact_overlap_ids
                        .into_iter()
                        .chain(ov.normalized_overlap_ids)
                    {
                        leaked_ids_set.insert(id);
                    }
                    hard_errors.push(format!(
                        "{cnt} test item(s) matched trained_inputs.sha256.bin (train/val/cal contamination)!"
                    ));
                }
            }
        } else {
            // Even without training_manifest.json, check trained_inputs.sha256.bin if present
            let filter_path = art_dir.join("trained_inputs.sha256.bin");
            if filter_path.exists() {
                let filter = TrainedInputsFilter::load_from_file(&filter_path)?;
                let ov = filter.check_overlap(test_items);
                if ov.has_overlap() {
                    report.leakage_detected = true;
                    let cnt = ov.total_overlapping_items();
                    for id in ov
                        .exact_overlap_ids
                        .into_iter()
                        .chain(ov.normalized_overlap_ids)
                    {
                        leaked_ids_set.insert(id);
                    }
                    hard_errors.push(format!(
                        "{cnt} test item(s) matched {}!",
                        filter_path.display()
                    ));
                }
            }
        }
    }

    report.leaked_item_ids = leaked_ids_set.into_iter().take(25).collect();

    if report.leakage_detected {
        if allow_leakage {
            report.verified = false;
            report.leakage_overridden = true;
            for err in hard_errors {
                report.warnings.push(format!("LEAKAGE OVERRIDDEN: {err}"));
            }
        } else {
            let sample_str = if report.leaked_item_ids.is_empty() {
                String::new()
            } else {
                format!(" Offending item IDs: {:?}", report.leaked_item_ids)
            };
            bail!(
                "Refusing to run paired experiment due to train/test data leakage: {}{sample_str} (pass `--allow-leakage` only for debugging)",
                hard_errors.join("; ")
            );
        }
    }

    Ok(report)
}

// ============================================================================
// Core Statistical Primitives (McNemar Mid-p, BCa Bootstrap, Permutation, Holm)
// ============================================================================

/// Contingency counts and exact / Lancaster mid-p statistics for paired binary correctness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McNemarStats {
    pub n11_both_correct: usize,
    pub n10_broken: usize,
    pub n01_fixed: usize,
    pub n00_both_wrong: usize,
    pub discordant_pairs: usize,
    pub p_exact: f64,
    pub p_mid: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_mid_holm: Option<f64>,
    pub haldane_anscombe_or: f64,
}

fn ln_factorials(n: usize) -> Vec<f64> {
    let mut table = vec![0.0; n + 1];
    for i in 1..=n {
        table[i] = table[i - 1] + (i as f64).ln();
    }
    table
}

/// Exact two-sided binomial McNemar test and Lancaster's McNemar mid-p adjustment.
///
/// `b_broken` (`n10`): Base correct, Candidate wrong.
/// `c_fixed` (`n01`): Base wrong, Candidate correct.
pub fn mcnemar_test(
    n11_both_correct: usize,
    b_broken: usize,
    c_fixed: usize,
    n00_both_wrong: usize,
) -> McNemarStats {
    let nd = b_broken + c_fixed;
    let or = (c_fixed as f64 + 0.5) / (b_broken as f64 + 0.5);
    if nd == 0 {
        return McNemarStats {
            n11_both_correct,
            n10_broken: b_broken,
            n01_fixed: c_fixed,
            n00_both_wrong,
            discordant_pairs: 0,
            p_exact: 1.0,
            p_mid: 1.0,
            p_mid_holm: None,
            haldane_anscombe_or: round6(or),
        };
    }

    let k = b_broken.min(c_fixed);
    let lnf = ln_factorials(nd);
    let ln2 = std::f64::consts::LN_2;
    let prob_at = |j: usize| -> f64 { (lnf[nd] - lnf[j] - lnf[nd - j] - (nd as f64) * ln2).exp() };

    let mut lower_strict_sum = 0.0;
    for j in 0..k {
        lower_strict_sum += prob_at(j);
    }
    let pk = prob_at(k);
    let p_exact = (2.0 * (lower_strict_sum + pk)).clamp(0.0, 1.0);
    let p_mid = if b_broken == c_fixed {
        1.0
    } else {
        (2.0 * lower_strict_sum + pk).clamp(0.0, 1.0)
    };

    McNemarStats {
        n11_both_correct,
        n10_broken: b_broken,
        n01_fixed: c_fixed,
        n00_both_wrong,
        discordant_pairs: nd,
        p_exact: round6(p_exact),
        p_mid: round6(p_mid),
        p_mid_holm: None,
        haldane_anscombe_or: round6(or),
    }
}

/// Holm-Bonferroni step-down multiplicity correction across a family of `p_values`.
pub fn holm_bonferroni(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]));

    let mut adjusted = vec![1.0; m];
    let mut running_max: f64 = 0.0;
    for (rank, &orig_idx) in order.iter().enumerate() {
        let mult = (m - rank) as f64;
        let scaled = (p_values[orig_idx].clamp(0.0, 1.0) * mult).min(1.0);
        running_max = running_max.max(scaled);
        adjusted[orig_idx] = round6(running_max);
    }
    adjusted
}

/// Standard normal CDF `\Phi(z)` (Abramowitz & Stegun 7.1.26, error `< 1.5e-7`).
pub fn normal_cdf(z: f64) -> f64 {
    if !z.is_finite() {
        return if z.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t)
        * (-x * x).exp();
    let erf = 1.0 - poly;
    if z >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

/// Inverse standard normal CDF `\Phi^{-1}(p)` (Peter Acklam rational approximation, error `< 1.15e-9`).
pub fn inv_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = p.floor() as usize;
    let hi = p.ceil() as usize;
    let w = p - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// Compute BCa (Bias-Corrected and Accelerated) or Percentile CI from sorted bootstrap replicates.
fn compute_bootstrap_ci(
    observed: f64,
    sorted_replicates: &[f64],
    jackknife_loo: Option<&[f64]>,
    confidence: f64,
) -> (f64, f64, &'static str) {
    if sorted_replicates.is_empty() {
        return (observed, observed, "point");
    }
    let alpha = (1.0 - confidence).clamp(1e-4, 0.5);
    let first = sorted_replicates[0];
    let last = sorted_replicates[sorted_replicates.len() - 1];
    if (last - first).abs() < 1e-12 {
        return (observed, observed, "constant");
    }

    if let Some(jack) = jackknife_loo
        && jack.len() >= 4
    {
        let b = sorted_replicates.len() as f64;
        let less = sorted_replicates
            .iter()
            .filter(|&&v| v < observed - 1e-12)
            .count() as f64;
        let eq = sorted_replicates
            .iter()
            .filter(|&&v| (v - observed).abs() <= 1e-12)
            .count() as f64;
        let prop = (less + 0.5 * eq) / b;
        if prop > 1e-4 && prop < 1.0 - 1e-4 {
            let z0 = inv_normal_cdf(prop);
            let u_len = jack.len() as f64;
            let mean_j = jack.iter().sum::<f64>() / u_len;
            let mut num = 0.0;
            let mut den_sq = 0.0;
            for &jv in jack {
                let d = mean_j - jv;
                num += d.powi(3);
                den_sq += d.powi(2);
            }
            let accel = if den_sq > 1e-18 {
                (num / (6.0 * den_sq.powf(1.5))).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let adj_q = |p: f64| -> Option<f64> {
                let z_alpha = inv_normal_cdf(p);
                let denom = 1.0 - accel * (z0 + z_alpha);
                if denom.abs() < 1e-6 {
                    return None;
                }
                Some(normal_cdf(z0 + (z0 + z_alpha) / denom))
            };
            if let (Some(q_lo), Some(q_hi)) = (adj_q(alpha / 2.0), adj_q(1.0 - alpha / 2.0)) {
                return (
                    sorted_quantile(sorted_replicates, q_lo),
                    sorted_quantile(sorted_replicates, q_hi),
                    "bca",
                );
            }
        }
    }

    (
        sorted_quantile(sorted_replicates, alpha / 2.0),
        sorted_quantile(sorted_replicates, 1.0 - alpha / 2.0),
        "percentile",
    )
}

#[inline]
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

#[inline]
fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

// ============================================================================
// Report Data Structures
// ============================================================================

/// Paired metric comparison with bootstrap CI and paired sign-flip permutation p-value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDeltaCi {
    pub base: f64,
    pub candidate: f64,
    /// `candidate - base`
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub ci_method: String,
    pub p_perm: f64,
    /// `true` when `higher_is_better` (`accuracy`, `f1`, `coverage`), `false` when `lower_is_better` (`nll`, `ece`, `brier`, `aurc`, `mae`).
    pub higher_is_better: bool,
}

/// Per-label F1 delta and win/regression breakdown on a `Choice` question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelDeltaRow {
    pub label: String,
    pub support: usize,
    pub base_f1: f64,
    pub candidate_f1: f64,
    pub delta_f1: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub fixed_count: usize,
    pub broken_count: usize,
}

/// Kind of prediction flip between `base` and `candidate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlipKind {
    Fixed,
    Broken,
    ShiftedError,
}

/// Aggregated transition `(truth: base_pred -> candidate_pred)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipTransition {
    pub truth: String,
    pub base_pred: String,
    pub candidate_pred: String,
    pub count: usize,
    pub kind: FlipKind,
}

/// One concrete regression (`base` right, `candidate` wrong) ranked by candidate confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionCase {
    pub item_id: String,
    pub question_id: String,
    pub snippet: String,
    pub truth: String,
    pub base_pred: String,
    pub base_prob: f64,
    pub candidate_pred: String,
    pub candidate_prob: f64,
}

/// Per-question paired evaluation breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionPairedReport {
    pub question_id: String,
    pub question_type: String,
    pub evaluated_samples: usize,
    pub accuracy: MetricDeltaCi,
    pub nll: MetricDeltaCi,
    pub brier: MetricDeltaCi,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ece: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macro_f1: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weighted_f1: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_label_f1: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_mae: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_rps: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_perm_nll_holm: Option<f64>,
    pub mcnemar: McNemarStats,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<LabelDeltaRow>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_transitions: Vec<FlipTransition>,
}

/// Per-slice paired accuracy and NLL breakdown (`--slice-by`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedSliceRow {
    pub slice: String,
    pub samples: usize,
    pub base_accuracy: f64,
    pub candidate_accuracy: f64,
    pub delta_accuracy: f64,
    pub base_nll: f64,
    pub candidate_nll: f64,
    pub delta_nll: f64,
    pub fixed: usize,
    pub broken: usize,
}

/// Conformal and System 1/2 deferral comparison fitted on `cal_data` and evaluated on `test_data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveRiskSummary {
    pub alpha: f64,
    pub base_conformal_q_hat: f64,
    pub candidate_conformal_q_hat: f64,
    pub base_conformal_coverage: f64,
    pub candidate_conformal_coverage: f64,
    pub conformal_set_size: MetricDeltaCi,
    pub conformal_singleton_rate: MetricDeltaCi,
    pub target_accuracy: f64,
    pub base_defer_threshold: Option<f64>,
    pub candidate_defer_threshold: Option<f64>,
    pub base_defer_kept_accuracy: f64,
    pub candidate_defer_kept_accuracy: f64,
    pub defer_coverage: MetricDeltaCi,
    pub base_defer_cost_per_1000: f64,
    pub candidate_defer_cost_per_1000: f64,
}

/// Complete paired evaluation summary under one calibration regime (`raw` or `calibrated`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedComparisonSummary {
    pub regime: String,
    pub total_items: usize,
    pub total_groups: usize,
    pub total_decisions: usize,
    pub accuracy: MetricDeltaCi,
    pub nll: MetricDeltaCi,
    pub brier: MetricDeltaCi,
    pub ece: MetricDeltaCi,
    pub ace: MetricDeltaCi,
    pub aurc: MetricDeltaCi,
    pub e_aurc: MetricDeltaCi,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macro_f1: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weighted_f1: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_label_f1: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_mae: Option<MetricDeltaCi>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_rps: Option<MetricDeltaCi>,
    pub overall_mcnemar: McNemarStats,
    pub item_exact_match_mcnemar: McNemarStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selective_risk: Option<SelectiveRiskSummary>,
    pub questions: BTreeMap<String, QuestionPairedReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slices: Vec<PairedSliceRow>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_regressions: Vec<RegressionCase>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub base_fitted_temperatures: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub candidate_fitted_temperatures: BTreeMap<String, f64>,
}

/// Paired ABBA timing and throughput summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedLatencyStats {
    pub warmup_runs: usize,
    pub timing_iterations: usize,
    pub base_p50_ms: f64,
    pub base_p95_ms: f64,
    pub base_p99_ms: f64,
    pub base_items_per_sec: f64,
    pub candidate_p50_ms: f64,
    pub candidate_p95_ms: f64,
    pub candidate_p99_ms: f64,
    pub candidate_items_per_sec: f64,
    /// `base_p50_ms / candidate_p50_ms` (`> 1.0` means candidate is faster).
    pub speedup_ratio: f64,
}

/// Top-level report returned by `PairedExperimentRunner`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedExperimentReport {
    pub created_at: String,
    pub base_label: String,
    pub candidate_label: String,
    pub active_regime: String,
    pub verdict: ExperimentVerdict,
    pub verdict_reasons: Vec<String>,
    pub provenance: ProvenanceAuditReport,
    pub primary: PairedComparisonSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_comparison: Option<PairedComparisonSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated_comparison: Option<PairedComparisonSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency: Option<PairedLatencyStats>,
}

// ============================================================================
// Internal Per-Decision Observation & Post-Hoc Calibration Fitting
// ============================================================================

#[derive(Debug, Clone)]
struct DecisionObs {
    pred_label: String,
    truth_label: String,
    is_correct: bool,
    /// Top-label winning probability (`p_win`) used for top-label ECE, ACE, and AURC.
    p_win: f64,
    /// True-class NLL from temperature-scaled logits (`log_softmax` or `softplus` BCE).
    nll: f64,
    brier: f64,
    /// Renormalized discrete probability distribution (sorted descending for APS conformal).
    sorted_probs_desc: Vec<(String, f64)>,
    /// Multi-label binary `(tp, fp, fn)` per option when `multi_label` is active.
    ml_counts: Option<(usize, usize, usize)>,
    score_abs_err: Option<f64>,
    score_rps: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct FittedPostHocCalibration {
    bucket_temperatures: BTreeMap<String, f64>,
    /// Per-question symmetric APS quantile `\hat{q}_{1-\alpha}` on `cal`.
    conformal_quantiles: BTreeMap<String, f64>,
    /// Certified deferral risk threshold `\tau^*` on `cal` (`None` if uncertifiable).
    defer_threshold: Option<f64>,
}

/// Pre-computed calibration split bundle: `(cal_items, base_cal_results, candidate_cal_results)`.
pub type PrecomputedCalBundle<'a> = (
    &'a [DataItem],
    &'a [(String, DecisionResult)],
    &'a [(String, DecisionResult)],
);

pub struct PairedExperimentRunner;

impl PairedExperimentRunner {
    /// Run a complete end-to-end paired experiment comparing `base` vs `candidate` on `test_items`,
    /// optionally fitting fair post-hoc calibration on `cal_items` for both models.
    #[allow(clippy::too_many_arguments)]
    pub fn run_paired(
        config: &PairedExperimentConfig,
        base: &ExperimentParticipant,
        candidate: &ExperimentParticipant,
        test_items: &[DataItem],
        cal_items: Option<&[DataItem]>,
        questions: &BTreeMap<String, Question>,
        test_path: Option<&Path>,
        dataset_manifest_path: Option<&Path>,
    ) -> Result<PairedExperimentReport> {
        if test_items.is_empty() {
            bail!("Test dataset cannot be empty for paired experimentation");
        }
        if questions.is_empty() {
            bail!("Question schema cannot be empty for paired experimentation");
        }

        // 1. 5-layer provenance & train/test leakage guard
        let cand_spec = candidate.classifier.config().model_spec();
        let inferred_art_dir = cand_spec
            .adapter_dir
            .clone()
            .or_else(|| cand_spec.model_dir.clone());
        let effective_art_dir = candidate
            .artifact_dir
            .as_deref()
            .or(inferred_art_dir.as_deref());
        let cand_base_dir = cand_spec.base_weights_dir();
        let provenance = verify_experiment_provenance_with_base_dir(
            test_items,
            test_path,
            questions,
            Some(base.classifier.config().checkpoint.repo_id()),
            dataset_manifest_path,
            effective_art_dir,
            Some(&cand_base_dir),
            config.allow_leakage,
        )?;

        // 2. Execute raw test passes + ABBA timing passes
        let (base_raw_results, cand_raw_results, latency) =
            run_paired_inference_and_timing(config, base, candidate, test_items, questions)?;

        let want_raw = matches!(
            config.calibration_mode,
            CalibrationMode::Raw | CalibrationMode::Both
        ) || cal_items.is_none();
        let want_cal = matches!(
            config.calibration_mode,
            CalibrationMode::Calibrated | CalibrationMode::Both
        ) && cal_items.is_some_and(|c| !c.is_empty());

        let raw_comp = if want_raw {
            Some(evaluate_paired_decisions(
                "raw",
                config,
                base.classifier.config(),
                candidate.classifier.config(),
                &base_raw_results,
                &cand_raw_results,
                test_items,
                questions,
                None,
                None,
            )?)
        } else {
            None
        };

        let cal_comp = if want_cal {
            let cal_slice = cal_items.unwrap();
            let base_cal_results = base.classifier.classify_batch(cal_slice, questions)?;
            crate::model::native::release_models();
            let cand_cal_results = candidate.classifier.classify_batch(cal_slice, questions)?;
            crate::model::native::release_models();

            let base_fit = fit_posthoc_on_cal(
                base.classifier.config(),
                &base.classifier,
                cal_slice,
                &base_cal_results,
                questions,
                config.alpha,
                config.target_accuracy,
                config.confidence,
                config.system2_cost,
            )?;
            let cand_fit = fit_posthoc_on_cal(
                candidate.classifier.config(),
                &candidate.classifier,
                cal_slice,
                &cand_cal_results,
                questions,
                config.alpha,
                config.target_accuracy,
                config.confidence,
                config.system2_cost,
            )?;

            let mut base_cal_cfg = base.classifier.config().clone();
            base_cal_cfg.temperature = 1.0;
            base_cal_cfg.temperature_by_options = base_fit.bucket_temperatures.clone();

            let mut cand_cal_cfg = candidate.classifier.config().clone();
            cand_cal_cfg.temperature = 1.0;
            cand_cal_cfg.temperature_by_options = cand_fit.bucket_temperatures.clone();

            Some(evaluate_paired_decisions(
                "calibrated",
                config,
                &base_cal_cfg,
                &cand_cal_cfg,
                &base_raw_results,
                &cand_raw_results,
                test_items,
                questions,
                Some(&base_fit),
                Some(&cand_fit),
            )?)
        } else {
            None
        };

        let (active_regime, primary) = match (&cal_comp, &raw_comp) {
            (Some(c), _) => ("calibrated".to_string(), c.clone()),
            (None, Some(r)) => ("raw".to_string(), r.clone()),
            (None, None) => bail!("No comparison regime executed"),
        };

        let (verdict, verdict_reasons) =
            determine_verdict(config, raw_comp.as_ref(), cal_comp.as_ref(), &primary);

        Ok(PairedExperimentReport {
            created_at: crate::engine::metrics::format_utc_iso8601(),
            base_label: base.label.clone(),
            candidate_label: candidate.label.clone(),
            active_regime,
            verdict,
            verdict_reasons,
            provenance,
            primary,
            raw_comparison: raw_comp,
            calibrated_comparison: cal_comp,
            latency: Some(latency),
        })
    }

    /// Evaluate pre-computed `(base_decisions, candidate_decisions)` directly (offline tests or
    /// custom pipelines), optionally with pre-computed calibration split decisions.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_precomputed_pair(
        config: &PairedExperimentConfig,
        base_label: impl Into<String>,
        candidate_label: impl Into<String>,
        base_cfg: &ClassifierConfig,
        candidate_cfg: &ClassifierConfig,
        base_test_results: &[(String, DecisionResult)],
        candidate_test_results: &[(String, DecisionResult)],
        test_items: &[DataItem],
        cal_bundle: Option<PrecomputedCalBundle<'_>>,
        questions: &BTreeMap<String, Question>,
    ) -> Result<PairedExperimentReport> {
        let provenance = verify_experiment_provenance(
            test_items,
            None,
            questions,
            Some(base_cfg.checkpoint.repo_id()),
            None,
            None,
            config.allow_leakage,
        )?;

        let want_raw = matches!(
            config.calibration_mode,
            CalibrationMode::Raw | CalibrationMode::Both
        ) || cal_bundle.is_none();
        let want_cal = matches!(
            config.calibration_mode,
            CalibrationMode::Calibrated | CalibrationMode::Both
        ) && cal_bundle.is_some();

        let raw_comp = if want_raw {
            Some(evaluate_paired_decisions(
                "raw",
                config,
                base_cfg,
                candidate_cfg,
                base_test_results,
                candidate_test_results,
                test_items,
                questions,
                None,
                None,
            )?)
        } else {
            None
        };

        let cal_comp =
            if let (true, Some((cal_items, base_cal_res, cand_cal_res))) = (want_cal, cal_bundle) {
                let base_cls = LayaClassifier::new(base_cfg.clone());
                let cand_cls = LayaClassifier::new(candidate_cfg.clone());
                let base_fit = fit_posthoc_on_cal(
                    base_cfg,
                    &base_cls,
                    cal_items,
                    base_cal_res,
                    questions,
                    config.alpha,
                    config.target_accuracy,
                    config.confidence,
                    config.system2_cost,
                )?;
                let cand_fit = fit_posthoc_on_cal(
                    candidate_cfg,
                    &cand_cls,
                    cal_items,
                    cand_cal_res,
                    questions,
                    config.alpha,
                    config.target_accuracy,
                    config.confidence,
                    config.system2_cost,
                )?;
                let mut b_cfg = base_cfg.clone();
                b_cfg.temperature = 1.0;
                b_cfg.temperature_by_options = base_fit.bucket_temperatures.clone();
                let mut c_cfg = candidate_cfg.clone();
                c_cfg.temperature = 1.0;
                c_cfg.temperature_by_options = cand_fit.bucket_temperatures.clone();

                Some(evaluate_paired_decisions(
                    "calibrated",
                    config,
                    &b_cfg,
                    &c_cfg,
                    base_test_results,
                    candidate_test_results,
                    test_items,
                    questions,
                    Some(&base_fit),
                    Some(&cand_fit),
                )?)
            } else {
                None
            };

        let (active_regime, primary) = match (&cal_comp, &raw_comp) {
            (Some(c), _) => ("calibrated".to_string(), c.clone()),
            (None, Some(r)) => ("raw".to_string(), r.clone()),
            (None, None) => bail!("No comparison regime executed"),
        };

        let (verdict, verdict_reasons) =
            determine_verdict(config, raw_comp.as_ref(), cal_comp.as_ref(), &primary);

        Ok(PairedExperimentReport {
            created_at: crate::engine::metrics::format_utc_iso8601(),
            base_label: base_label.into(),
            candidate_label: candidate_label.into(),
            active_regime,
            verdict,
            verdict_reasons,
            provenance,
            primary,
            raw_comparison: raw_comp,
            calibrated_comparison: cal_comp,
            latency: None,
        })
    }

    /// Render the paired experiment report as a rich ANSI terminal table.
    pub fn format_table(report: &PairedExperimentReport) -> String {
        let p = &report.primary;
        let verdict_badge = match report.verdict {
            ExperimentVerdict::Promote => "✅ PROMOTE",
            ExperimentVerdict::PromoteWithCalibration => "✅ PROMOTE (WITH CALIBRATION)",
            ExperimentVerdict::Inconclusive => "⚖️  INCONCLUSIVE",
            ExperimentVerdict::Regressed => "❌ REGRESSED",
        };
        let mut out = format!(
            "\n🧪 Lemmata Paired Experiment — {verdict_badge}\nBase      : {}\nCandidate : {}\nRegime    : {} ({} items, {} groups, {} decisions)\n",
            report.base_label,
            report.candidate_label,
            report.active_regime,
            p.total_items,
            p.total_groups,
            p.total_decisions
        );
        for reason in &report.verdict_reasons {
            out.push_str(&format!("  • {reason}\n"));
        }
        for w in &report.provenance.warnings {
            out.push_str(&format!("  ⚠️  {w}\n"));
        }

        let mut summary_tbl = Table::new();
        summary_tbl
            .load_style(comfy_table::presets::UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Metric").fg(Color::Cyan),
                Cell::new("Base").fg(Color::White),
                Cell::new("Candidate").fg(Color::Green),
                Cell::new("Delta [95% CI]").fg(Color::Yellow),
                Cell::new("p-value").fg(Color::Magenta),
            ]);

        let fmt_row = |name: &str, m: &MetricDeltaCi, pct: bool| -> Vec<Cell> {
            let (b_s, c_s, d_s) = if pct {
                (
                    format!("{:.2}%", m.base * 100.0),
                    format!("{:.2}%", m.candidate * 100.0),
                    format!(
                        "{:+.2}% [{:+.2}%, {:+.2}%]",
                        m.delta * 100.0,
                        m.ci_low * 100.0,
                        m.ci_high * 100.0
                    ),
                )
            } else {
                (
                    format!("{:.4}", m.base),
                    format!("{:.4}", m.candidate),
                    format!("{:+.4} [{:+.4}, {:+.4}]", m.delta, m.ci_low, m.ci_high),
                )
            };
            vec![
                Cell::new(name),
                Cell::new(b_s),
                Cell::new(c_s),
                Cell::new(d_s),
                Cell::new(format!("p_perm={:.4}", m.p_perm)),
            ]
        };

        summary_tbl.add_row(fmt_row("Accuracy", &p.accuracy, true));
        if let Some(mf1) = &p.macro_f1 {
            summary_tbl.add_row(fmt_row("Macro-F1", mf1, false));
        }
        if let Some(wf1) = &p.weighted_f1 {
            summary_tbl.add_row(fmt_row("Weighted-F1", wf1, false));
        }
        if let Some(mlf1) = &p.multi_label_f1 {
            summary_tbl.add_row(fmt_row("Multi-Label Micro-F1", mlf1, false));
        }
        summary_tbl.add_row(fmt_row("NLL (Log-Softmax)", &p.nll, false));
        summary_tbl.add_row(fmt_row("Top-Label ECE", &p.ece, false));
        summary_tbl.add_row(fmt_row("Adaptive ECE (ACE)", &p.ace, false));
        summary_tbl.add_row(fmt_row("Brier Score", &p.brier, false));
        summary_tbl.add_row(fmt_row("AURC", &p.aurc, false));
        summary_tbl.add_row(fmt_row("Excess-AURC", &p.e_aurc, false));
        if let Some(mae) = &p.score_mae {
            summary_tbl.add_row(fmt_row("Score MAE", mae, false));
        }
        if let Some(rps) = &p.score_rps {
            summary_tbl.add_row(fmt_row("Ordinal Score RPS", rps, false));
        }
        if let Some(sr) = &p.selective_risk {
            summary_tbl.add_row(fmt_row("Conformal Set Size", &sr.conformal_set_size, false));
            summary_tbl.add_row(fmt_row(
                "Conformal Singleton %",
                &sr.conformal_singleton_rate,
                true,
            ));
            summary_tbl.add_row(fmt_row("Defer System-1 Coverage", &sr.defer_coverage, true));
        }
        out.push('\n');
        out.push_str(&summary_tbl.to_string());
        out.push('\n');

        let mc = &p.overall_mcnemar;
        out.push_str(&format!(
            "\n🔀 2x2 Flip Matrix (All Decisions): Both ✓ = {} | +Fixed (Base ✗ -> Cand ✓) = {} | -Broken (Base ✓ -> Cand ✗) = {} | Persistent ✗ = {}\n   McNemar mid-p = {:.4} (exact p = {:.4}, Haldane-Anscombe Win/Loss OR = {:.2}x)\n",
            mc.n11_both_correct,
            mc.n01_fixed,
            mc.n10_broken,
            mc.n00_both_wrong,
            mc.p_mid,
            mc.p_exact,
            mc.haldane_anscombe_or
        ));

        for (qid, qr) in &p.questions {
            out.push_str(&format!(
                "\n📋 Question `{qid}` ({}, n={}): ΔAcc={:+.2}% (McNemar mid-p={:.4}, Holm={:.4}) | ΔNLL={:+.4}\n",
                qr.question_type,
                qr.evaluated_samples,
                qr.accuracy.delta * 100.0,
                qr.mcnemar.p_mid,
                qr.mcnemar.p_mid_holm.unwrap_or(qr.mcnemar.p_mid),
                qr.nll.delta
            ));
            if !qr.labels.is_empty() {
                let mut l_tbl = Table::new();
                l_tbl
                    .load_style(comfy_table::presets::UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_header(vec![
                        "Label",
                        "Support",
                        "Base F1",
                        "Cand F1",
                        "ΔF1 [95% CI]",
                        "+Fixed",
                        "-Broken",
                    ]);
                for l in &qr.labels {
                    l_tbl.add_row(vec![
                        l.label.clone(),
                        l.support.to_string(),
                        format!("{:.3}", l.base_f1),
                        format!("{:.3}", l.candidate_f1),
                        format!("{:+.3} [{:+.3}, {:+.3}]", l.delta_f1, l.ci_low, l.ci_high),
                        format!("+{}", l.fixed_count),
                        format!("-{}", l.broken_count),
                    ]);
                }
                out.push_str(&l_tbl.to_string());
                out.push('\n');
            }
        }

        if !p.slices.is_empty() {
            let mut s_tbl = Table::new();
            s_tbl
                .load_style(comfy_table::presets::UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_header(vec![
                    "Slice",
                    "Samples",
                    "Base Acc",
                    "Cand Acc",
                    "ΔAcc",
                    "ΔNLL",
                    "+Fixed / -Broken",
                ]);
            for s in &p.slices {
                s_tbl.add_row(vec![
                    s.slice.clone(),
                    s.samples.to_string(),
                    format!("{:.1}%", s.base_accuracy * 100.0),
                    format!("{:.1}%", s.candidate_accuracy * 100.0),
                    format!("{:+.1}%", s.delta_accuracy * 100.0),
                    format!("{:+.4}", s.delta_nll),
                    format!("+{} / -{}", s.fixed, s.broken),
                ]);
            }
            out.push_str("\n🧩 Slice Breakdown:\n");
            out.push_str(&s_tbl.to_string());
            out.push('\n');
        }

        if !p.top_regressions.is_empty() {
            out.push_str("\n🚨 Top High-Confidence Regressions (Base ✓ -> Candidate ✗):\n");
            for reg in &p.top_regressions {
                out.push_str(&format!(
                    "  • [{}:{}] truth='{}' | base='{}' ({:.1}%) -> cand='{}' ({:.1}%) | \"{}\"\n",
                    reg.item_id,
                    reg.question_id,
                    reg.truth,
                    reg.base_pred,
                    reg.base_prob * 100.0,
                    reg.candidate_pred,
                    reg.candidate_prob * 100.0,
                    reg.snippet
                ));
            }
        }

        if let Some(lat) = &report.latency {
            out.push_str(&format!(
                "\n⚡ Latency (ABBA interleaved): Base p50={:.2}ms ({:.1} it/s) | Cand p50={:.2}ms ({:.1} it/s) | Speedup={:.2}x\n",
                lat.base_p50_ms,
                lat.base_items_per_sec,
                lat.candidate_p50_ms,
                lat.candidate_items_per_sec,
                lat.speedup_ratio
            ));
        }

        out
    }

    /// Render the paired experiment report as GitHub-flavored Markdown (PR / model-card ready).
    pub fn format_markdown(report: &PairedExperimentReport) -> String {
        let p = &report.primary;
        let esc = |s: &str| s.replace('|', "\\|");
        let mut md = format!(
            "# Lemmata Paired Experiment: `{}` vs `{}`\n\n**Verdict:** `{}` (regime: `{}`, {} test items, {} groups)\n\n",
            esc(&report.base_label),
            esc(&report.candidate_label),
            report.verdict.as_str(),
            report.active_regime,
            p.total_items,
            p.total_groups
        );
        for reason in &report.verdict_reasons {
            md.push_str(&format!("- {}\n", reason));
        }
        md.push_str(
            "\n## Aggregate Paired Metrics\n\n| Metric | Base | Candidate | Delta [95% CI] | p_perm |\n|---|---:|---:|---:|---:|\n",
        );

        let add_md_row = |md: &mut String, name: &str, m: &MetricDeltaCi, pct: bool| {
            if pct {
                md.push_str(&format!(
                    "| {} | {:.2}% | {:.2}% | {:+.2}% [{:+.2}%, {:+.2}%] | {:.4} |\n",
                    name,
                    m.base * 100.0,
                    m.candidate * 100.0,
                    m.delta * 100.0,
                    m.ci_low * 100.0,
                    m.ci_high * 100.0,
                    m.p_perm
                ));
            } else {
                md.push_str(&format!(
                    "| {} | {:.4} | {:.4} | {:+.4} [{:+.4}, {:+.4}] | {:.4} |\n",
                    name, m.base, m.candidate, m.delta, m.ci_low, m.ci_high, m.p_perm
                ));
            }
        };

        add_md_row(&mut md, "Accuracy", &p.accuracy, true);
        if let Some(mf1) = &p.macro_f1 {
            add_md_row(&mut md, "Macro-F1", mf1, false);
        }
        add_md_row(&mut md, "NLL (Log-Softmax)", &p.nll, false);
        add_md_row(&mut md, "Top-Label ECE", &p.ece, false);
        add_md_row(&mut md, "Adaptive ECE (ACE)", &p.ace, false);
        add_md_row(&mut md, "Brier Score", &p.brier, false);
        add_md_row(&mut md, "AURC", &p.aurc, false);
        add_md_row(&mut md, "Excess-AURC", &p.e_aurc, false);
        if let Some(mae) = &p.score_mae {
            add_md_row(&mut md, "Score MAE", mae, false);
        }

        let mc = &p.overall_mcnemar;
        md.push_str(&format!(
            "\n## 2x2 Prediction Flip Summary\n\n- **Both Correct (`n11`)**: {}\n- **Wins / Fixed (`n01`, Base ✗ → Cand ✓)**: {}\n- **Regressions / Broken (`n10`, Base ✓ → Cand ✗)**: {}\n- **Persistent Errors (`n00`, Both ✗)**: {}\n- **Lancaster McNemar mid-p**: `{:.4}` (exact binomial p = `{:.4}`, OR = `{:.2}x`)\n",
            mc.n11_both_correct,
            mc.n01_fixed,
            mc.n10_broken,
            mc.n00_both_wrong,
            mc.p_mid,
            mc.p_exact,
            mc.haldane_anscombe_or
        ));

        for (qid, qr) in &p.questions {
            md.push_str(&format!(
                "\n### Question `{}` (`{}`)\n\nMcNemar mid-p = `{:.4}` (Holm-adjusted = `{:.4}`), ΔAcc = `{:+.2}%`, ΔNLL = `{:+.4}`\n",
                esc(qid),
                qr.question_type,
                qr.mcnemar.p_mid,
                qr.mcnemar.p_mid_holm.unwrap_or(qr.mcnemar.p_mid),
                qr.accuracy.delta * 100.0,
                qr.nll.delta
            ));
            if !qr.labels.is_empty() {
                md.push_str(
                    "\n| Label | Support | Base F1 | Cand F1 | ΔF1 [95% CI] | +Fixed | -Broken |\n|---|---:|---:|---:|---:|---:|---:|\n",
                );
                for l in &qr.labels {
                    md.push_str(&format!(
                        "| `{}` | {} | {:.3} | {:.3} | {:+.3} [{:+.3}, {:+.3}] | +{} | -{} |\n",
                        esc(&l.label),
                        l.support,
                        l.base_f1,
                        l.candidate_f1,
                        l.delta_f1,
                        l.ci_low,
                        l.ci_high,
                        l.fixed_count,
                        l.broken_count
                    ));
                }
            }
        }

        if !p.top_regressions.is_empty() {
            md.push_str("\n## Top High-Confidence Regressions\n\n");
            for reg in &p.top_regressions {
                md.push_str(&format!(
                    "- **`{}:{}`** (truth `{}`): base `{}` ({:.1}%) → cand `{}` ({:.1}%) — _{}_\n",
                    esc(&reg.item_id),
                    esc(&reg.question_id),
                    esc(&reg.truth),
                    esc(&reg.base_pred),
                    reg.base_prob * 100.0,
                    esc(&reg.candidate_pred),
                    reg.candidate_prob * 100.0,
                    esc(&reg.snippet)
                ));
            }
        }
        md
    }

    pub fn format_json(report: &PairedExperimentReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    pub fn export_report(report: &PairedExperimentReport, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let content = match ext.as_str() {
            "md" | "markdown" => Self::format_markdown(report),
            _ => Self::format_json(report)?,
        };
        std::fs::write(path, content)
            .with_context(|| format!("Failed to write experiment report to {}", path.display()))
    }
}

type DecisionStream = Vec<(String, DecisionResult)>;
type SinglePassTiming = (DecisionStream, Vec<f64>, f64);

fn run_paired_inference_and_timing(
    config: &PairedExperimentConfig,
    base: &ExperimentParticipant,
    candidate: &ExperimentParticipant,
    test_items: &[DataItem],
    questions: &BTreeMap<String, Question>,
) -> Result<(DecisionStream, DecisionStream, PairedLatencyStats)> {
    let warmup_slice = &test_items[..test_items.len().min(5)];
    for _ in 0..config.warmup_runs {
        for item in warmup_slice {
            let _ = base.classifier.classify(&item.state, questions);
            let _ = candidate.classifier.classify(&item.state, questions);
        }
    }

    let iters = config.timing_iterations.max(1);
    let mut base_latencies = Vec::with_capacity(test_items.len() * iters);
    let mut cand_latencies = Vec::with_capacity(test_items.len() * iters);
    let mut base_elapsed_total = 0.0;
    let mut cand_elapsed_total = 0.0;
    let mut base_final = Vec::new();
    let mut cand_final = Vec::new();

    // ABBA interleaved passes so CPU thermal/cache state treats both models symmetrically
    for round in 0..iters {
        let base_first = round % 2 == 0;
        let run_one = |cls: &LayaClassifier| -> Result<SinglePassTiming> {
            let mut lats = Vec::with_capacity(test_items.len());
            let mut out = Vec::with_capacity(test_items.len());
            let t_batch = Instant::now();
            for item in test_items {
                let t0 = Instant::now();
                let dec = cls.classify(&item.state, questions)?;
                lats.push(t0.elapsed().as_secs_f64() * 1000.0);
                out.push((item.id.clone(), dec));
            }
            let elapsed = t_batch.elapsed().as_secs_f64();
            Ok((out, lats, elapsed))
        };

        if base_first {
            let (b_out, b_lat, b_el) = run_one(&base.classifier)?;
            let (c_out, c_lat, c_el) = run_one(&candidate.classifier)?;
            base_latencies.extend(b_lat);
            cand_latencies.extend(c_lat);
            base_elapsed_total += b_el;
            cand_elapsed_total += c_el;
            if round + 1 == iters {
                base_final = b_out;
                cand_final = c_out;
            }
        } else {
            let (c_out, c_lat, c_el) = run_one(&candidate.classifier)?;
            let (b_out, b_lat, b_el) = run_one(&base.classifier)?;
            base_latencies.extend(b_lat);
            cand_latencies.extend(c_lat);
            base_elapsed_total += b_el;
            cand_elapsed_total += c_el;
            if round + 1 == iters {
                base_final = b_out;
                cand_final = c_out;
            }
        }
    }

    crate::model::native::release_models();

    base_latencies.sort_by(|a, b| a.total_cmp(b));
    cand_latencies.sort_by(|a, b| a.total_cmp(b));

    let base_p50 = sorted_quantile(&base_latencies, 0.50);
    let cand_p50 = sorted_quantile(&cand_latencies, 0.50);
    let total_processed = (test_items.len() * iters) as f64;

    let latency = PairedLatencyStats {
        warmup_runs: config.warmup_runs,
        timing_iterations: iters,
        base_p50_ms: round4(base_p50),
        base_p95_ms: round4(sorted_quantile(&base_latencies, 0.95)),
        base_p99_ms: round4(sorted_quantile(&base_latencies, 0.99)),
        base_items_per_sec: round4(total_processed / base_elapsed_total.max(1e-9)),
        candidate_p50_ms: round4(cand_p50),
        candidate_p95_ms: round4(sorted_quantile(&cand_latencies, 0.95)),
        candidate_p99_ms: round4(sorted_quantile(&cand_latencies, 0.99)),
        candidate_items_per_sec: round4(total_processed / cand_elapsed_total.max(1e-9)),
        speedup_ratio: round4(base_p50 / cand_p50.max(1e-6)),
    };

    Ok((base_final, cand_final, latency))
}

/// Fit per-bucket post-hoc temperatures `T* \in [0.5, 5.0]`, symmetric APS conformal quantiles,
/// and Learn-then-Test deferral threshold on `cal_items` using raw logits from `cal_results`.
#[allow(clippy::too_many_arguments)]
fn fit_posthoc_on_cal(
    orig_cfg: &ClassifierConfig,
    cls_helper: &LayaClassifier,
    cal_items: &[DataItem],
    cal_results: &[(String, DecisionResult)],
    questions: &BTreeMap<String, Question>,
    alpha: f64,
    target_accuracy: f64,
    confidence: f64,
    system2_cost: f64,
) -> Result<FittedPostHocCalibration> {
    struct CalBucketSample {
        logits: Vec<f64>,
        target_idx: usize,
    }

    let mut bucket_samples: BTreeMap<String, Vec<CalBucketSample>> = BTreeMap::new();

    for (item, (_, dec)) in cal_items.iter().zip(cal_results.iter()) {
        for (qid, q) in questions {
            let Some(tv) = resolve_ground_truth_for_question(
                item.ground_truth.as_ref(),
                qid,
                q,
                questions.len(),
            ) else {
                continue;
            };
            let Some(ans) = dec.answers.get(qid) else {
                continue;
            };
            if orig_cfg.multi_label && q.qtype == QuestionType::Choice {
                continue;
            }
            let raw_logits = extract_raw_logits_from_answer(ans, q);
            if raw_logits.len() < 2 {
                continue;
            }
            let Some(t_idx) = resolve_discrete_target_index(tv, q, raw_logits.len()) else {
                continue;
            };
            let bkey = cls_helper.get_temperature_bucket(q.qtype, raw_logits.len());
            bucket_samples
                .entry(bkey)
                .or_default()
                .push(CalBucketSample {
                    logits: raw_logits,
                    target_idx: t_idx,
                });
        }
    }

    let mut fitted_temps = orig_cfg.temperature_by_options.clone();
    for (bkey, samples) in &bucket_samples {
        let bucket_qtype = if bkey.starts_with("score:") {
            QuestionType::Score
        } else if bkey.starts_with("noul:") {
            QuestionType::Noul
        } else {
            QuestionType::Choice
        };
        let prior_t = resolve_effective_temperature(orig_cfg, bkey, bucket_qtype);
        let min_req = DEFAULT_MIN_BUCKET_SAMPLES.min(10); // allow fitting on compact cal splits >= 10 while staying regularized
        if samples.len() < min_req {
            fitted_temps.insert(bkey.clone(), prior_t);
            continue;
        }
        let eval_t = |t: f64| -> (f64, f64) {
            let mut nll_sum = 0.0;
            let mut p_wins = Vec::with_capacity(samples.len());
            let mut corrs = Vec::with_capacity(samples.len());
            for s in samples {
                let (probs, log_probs) = softmax_and_log_softmax(&s.logits, t);
                nll_sum -= log_probs[s.target_idx];
                let (pred_idx, &p_win) = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap_or((0, &1.0));
                p_wins.push(p_win);
                corrs.push(pred_idx == s.target_idx);
            }
            let ece = compute_equal_width_ece(&p_wins, &corrs, 10);
            (nll_sum / samples.len() as f64, ece)
        };

        let (_, prior_ece) = eval_t(prior_t);
        let grid_n = 48usize;
        let ln_lo = MIN_TEMPERATURE.ln();
        let ln_hi = MAX_TEMPERATURE.ln();
        let mut cands: Vec<f64> = (0..grid_n)
            .map(|i| (ln_lo + (ln_hi - ln_lo) * (i as f64 / (grid_n - 1) as f64)).exp())
            .collect();
        cands.push(1.0);
        cands.push(prior_t);
        cands.sort_by(|a, b| a.total_cmp(b));

        let mut best_idx = 0usize;
        let mut best_nll = f64::INFINITY;
        for (i, &t) in cands.iter().enumerate() {
            let (nll, _) = eval_t(t);
            if nll < best_nll {
                best_nll = nll;
                best_idx = i;
            }
        }
        // Golden-section refinement around best grid candidate
        let mut a = cands[best_idx.saturating_sub(1)];
        let mut b = cands[(best_idx + 1).min(cands.len() - 1)];
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        for _ in 0..24 {
            let c = b - phi * (b - a);
            let d = a + phi * (b - a);
            if eval_t(c).0 < eval_t(d).0 {
                b = d;
            } else {
                a = c;
            }
        }
        cands.push(0.5 * (a + b));

        let mut chosen_t = prior_t;
        let mut chosen_nll = eval_t(prior_t).0;
        for t in cands {
            let t_r = (t * 1000.0).round() / 1000.0;
            let (nll, ece) = eval_t(t_r);
            if nll < chosen_nll - 1e-6 && ece <= prior_ece + 0.03 {
                chosen_nll = nll;
                chosen_t = t_r;
            }
        }
        fitted_temps.insert(bkey.clone(), chosen_t);
    }

    let mut cal_cfg = orig_cfg.clone();
    cal_cfg.temperature = 1.0;
    cal_cfg.temperature_by_options = fitted_temps.clone();

    // Fit symmetric APS conformal quantiles per question and CascadeRouter threshold on calibrated `cal`
    let mut per_q_scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut defer_samples: Vec<(f64, bool)> = Vec::new();

    for (item, (_, dec)) in cal_items.iter().zip(cal_results.iter()) {
        let mut all_q_correct = true;
        let mut min_top_p = 1.0_f64;
        let mut matched_any = false;

        for (qid, q) in questions {
            let Some(tv) = resolve_ground_truth_for_question(
                item.ground_truth.as_ref(),
                qid,
                q,
                questions.len(),
            ) else {
                continue;
            };
            let Some(ans) = dec.answers.get(qid) else {
                continue;
            };
            if let Some(obs) = score_single_decision(ans, q, qid, tv, &cal_cfg, cls_helper) {
                matched_any = true;
                if !obs.is_correct {
                    all_q_correct = false;
                }
                min_top_p = min_top_p.min(obs.p_win);
                if let Some(s_xy) =
                    aps_nonconformity_score(&obs.sorted_probs_desc, &obs.truth_label)
                {
                    per_q_scores.entry(qid.clone()).or_default().push(s_xy);
                }
            }
        }
        if matched_any {
            let risk = ((1.0 - min_top_p).clamp(0.0, 1.0) * 1_000_000.0).round() / 1_000_000.0;
            defer_samples.push((risk, all_q_correct));
        }
    }

    let mut conformal_quantiles = BTreeMap::new();
    for (qid, mut scores) in per_q_scores {
        scores.sort_by(|a, b| a.total_cmp(b));
        let n = scores.len();
        let rank = (((n + 1) as f64 * (1.0 - alpha) - 1e-10).ceil().max(0.0)) as usize;
        let q_hat = if rank == 0 {
            0.0
        } else if rank > n {
            1.0
        } else {
            scores[rank - 1]
        };
        conformal_quantiles.insert(qid, q_hat);
    }

    let defer_threshold = if !defer_samples.is_empty() {
        fit_samples(
            &defer_samples,
            target_accuracy,
            confidence,
            system2_cost,
            orig_cfg.checkpoint.repo_id(),
        )
        .policy
        .map(|p| p.threshold)
    } else {
        None
    };

    // Ensure CascadeRouter symbol stays linked cleanly
    let _ = CascadeRouter::risk;

    Ok(FittedPostHocCalibration {
        bucket_temperatures: fitted_temps,
        conformal_quantiles,
        defer_threshold,
    })
}

/// Strict prefix probability mass before `truth_label` in descending-sorted renormalized probabilities:
/// `y \in C(x) \iff s(x, y) <= q_hat`, guaranteeing non-empty sets (`s(x, top1) == 0.0`) and 100%
/// singleton efficiency whenever top-1 accuracy on `cal` exceeds `1 - alpha`.
fn aps_nonconformity_score(sorted_probs_desc: &[(String, f64)], truth_label: &str) -> Option<f64> {
    let mut prefix_mass = 0.0;
    for (lbl, p) in sorted_probs_desc {
        if lbl.eq_ignore_ascii_case(truth_label) {
            return Some(prefix_mass);
        }
        prefix_mass += *p;
    }
    None
}

fn aps_predict_set(sorted_probs_desc: &[(String, f64)], q_hat: f64) -> Vec<String> {
    let mut set = Vec::new();
    let mut prefix_mass = 0.0;
    for (lbl, p) in sorted_probs_desc {
        if prefix_mass <= q_hat + 1e-12 || set.is_empty() {
            set.push(lbl.clone());
            prefix_mass += *p;
        } else {
            break;
        }
    }
    set
}

fn extract_raw_logits_from_answer(ans: &DecisionAnswer, q: &Question) -> Vec<f64> {
    if let Some(logits) = &ans.logits
        && !logits.is_empty()
    {
        return logits.clone();
    }
    match q.qtype {
        QuestionType::Choice => {
            let keys = choice_labels(q);
            keys.iter()
                .map(|k| {
                    ans.probabilities
                        .iter()
                        .find(|(pk, _)| pk.eq_ignore_ascii_case(k))
                        .map(|(_, &v)| v.max(1e-9).ln())
                        .unwrap_or(-20.0)
                })
                .collect()
        }
        QuestionType::Score => {
            let k = q.render_options().len().max(2);
            (0..k)
                .map(|i| {
                    ans.probabilities
                        .get(&i.to_string())
                        .copied()
                        .unwrap_or(1.0 / k as f64)
                        .max(1e-9)
                        .ln()
                })
                .collect()
        }
        QuestionType::Noul => {
            let p_true = ans.noul.unwrap_or(0.5).clamp(1e-6, 1.0 - 1e-6);
            vec![(1.0 - p_true).ln(), p_true.ln()]
        }
    }
}

fn resolve_discrete_target_index(
    tv: &serde_json::Value,
    q: &Question,
    num_opts: usize,
) -> Option<usize> {
    match q.qtype {
        QuestionType::Choice => {
            let s = tv.as_str()?;
            let keys = choice_labels(q);
            keys.iter().position(|k| k.eq_ignore_ascii_case(s))
        }
        QuestionType::Score => {
            let v = tv
                .as_f64()
                .or_else(|| tv.as_str().and_then(|s| s.trim().parse::<f64>().ok()))?;
            let idx = v.round() as isize;
            if idx < 0 || idx as usize >= num_opts {
                None
            } else {
                Some(idx as usize)
            }
        }
        QuestionType::Noul => {
            let b = tv
                .as_bool()
                .or_else(|| {
                    tv.as_str()
                        .and_then(|s| match s.trim().to_lowercase().as_str() {
                            "true" | "1" => Some(true),
                            "false" | "0" => Some(false),
                            _ => None,
                        })
                })
                .or_else(|| tv.as_i64().map(|n| n != 0))?;
            Some(usize::from(b))
        }
    }
}

fn resolve_effective_temperature(
    cfg: &ClassifierConfig,
    bucket_key: &str,
    qtype: QuestionType,
) -> f64 {
    if cfg.probe.is_some() {
        return 1.0;
    }
    let has_explicit = (cfg.temperature - 1.0).abs() > 1e-6;
    if let Some(&bt) = cfg.temperature_by_options.get(bucket_key) {
        let is_default = cfg
            .checkpoint
            .default_temperature_by_options()
            .get(bucket_key)
            .is_some_and(|&d| (bt - d).abs() <= 1e-6);
        if has_explicit && is_default {
            return cfg.temperature.max(0.01);
        }
        return bt.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE);
    }
    if has_explicit || cfg.is_custom_or_adapted() {
        cfg.temperature.max(0.01)
    } else {
        let idx = match qtype {
            QuestionType::Choice => 0,
            QuestionType::Score => 1,
            QuestionType::Noul => 2,
        };
        cfg.checkpoint.default_temperatures()[idx].max(0.01)
    }
}

fn softmax_and_log_softmax(raw_logits: &[f64], temp: f64) -> (Vec<f64>, Vec<f64>) {
    let t = temp.max(0.01);
    let scaled: Vec<f64> = raw_logits.iter().map(|&z| z / t).collect();
    let max_z = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = scaled.iter().map(|&z| (z - max_z).exp()).sum();
    let lse = max_z + sum_exp.max(1e-30).ln();
    let log_probs: Vec<f64> = scaled.iter().map(|&z| z - lse).collect();
    let probs: Vec<f64> = log_probs.iter().map(|&lp| lp.exp()).collect();
    (probs, log_probs)
}

fn score_single_decision(
    ans: &DecisionAnswer,
    q: &Question,
    qid: &str,
    tv: &serde_json::Value,
    cfg: &ClassifierConfig,
    cls_helper: &LayaClassifier,
) -> Option<DecisionObs> {
    let raw_logits = extract_raw_logits_from_answer(ans, q);
    let k = raw_logits.len().max(2);
    let bkey = cls_helper.get_temperature_bucket(q.qtype, k);
    let temp = resolve_effective_temperature(cfg, &bkey, q.qtype);

    match q.qtype {
        QuestionType::Choice => {
            let keys = choice_labels(q);
            if keys.len() != raw_logits.len() || keys.is_empty() {
                return None;
            }
            if cfg.multi_label || tv.is_array() {
                let mut truth_set: BTreeMap<usize, bool> = BTreeMap::new();
                match tv {
                    serde_json::Value::Array(arr) => {
                        for elem in arr {
                            if let Some(s) = elem.as_str()
                                && let Some(pos) =
                                    keys.iter().position(|k| k.eq_ignore_ascii_case(s))
                            {
                                truth_set.insert(pos, true);
                            }
                        }
                    }
                    serde_json::Value::String(s) => {
                        let pos = keys.iter().position(|k| k.eq_ignore_ascii_case(s))?;
                        truth_set.insert(pos, true);
                    }
                    _ => return None,
                }

                let t = temp.max(0.01);
                let mut bce_sum = 0.0;
                let mut brier_sum = 0.0;
                let mut pred_indices = Vec::new();
                let mut best_idx = 0usize;
                let mut best_sig = -1.0_f64;
                let mut sigs = Vec::with_capacity(keys.len());

                for (i, &z_raw) in raw_logits.iter().enumerate() {
                    let z = z_raw / t;
                    let sig = 1.0 / (1.0 + (-z).exp());
                    sigs.push(sig);
                    let y = if truth_set.contains_key(&i) { 1.0 } else { 0.0 };
                    // Numerically stable softplus BCE: max(z, 0) - z*y + ln(1 + exp(-|z|))
                    bce_sum += z.max(0.0) - z * y + ((-z.abs()).exp()).ln_1p();
                    brier_sum += (sig - y).powi(2);
                    if sig >= cfg.multi_label_threshold {
                        pred_indices.push(i);
                    }
                    if sig > best_sig {
                        best_sig = sig;
                        best_idx = i;
                    }
                }
                if pred_indices.is_empty() {
                    pred_indices.push(best_idx);
                }
                let pred_set: HashSet<usize> = pred_indices.iter().copied().collect();
                let mut tp = 0usize;
                let mut fp = 0usize;
                let mut fneg = 0usize;
                for i in 0..keys.len() {
                    match (pred_set.contains(&i), truth_set.contains_key(&i)) {
                        (true, true) => tp += 1,
                        (true, false) => fp += 1,
                        (false, true) => fneg += 1,
                        (false, false) => {}
                    }
                }
                let is_correct = fp == 0 && fneg == 0;
                let mut pred_names: Vec<String> = pred_indices
                    .iter()
                    .map(|&i| keys[i].to_lowercase())
                    .collect();
                pred_names.sort();
                let mut truth_names: Vec<String> =
                    truth_set.keys().map(|&i| keys[i].to_lowercase()).collect();
                truth_names.sort();

                let sum_sig: f64 = sigs.iter().sum::<f64>().max(1e-12);
                let mut sorted_desc: Vec<(String, f64)> = keys
                    .iter()
                    .zip(sigs.iter())
                    .map(|(k_name, &s)| (k_name.to_lowercase(), s / sum_sig))
                    .collect();
                sorted_desc.sort_by(|a, b| b.1.total_cmp(&a.1));

                Some(DecisionObs {
                    pred_label: pred_names.join(";"),
                    truth_label: truth_names.join(";"),
                    is_correct,
                    p_win: best_sig.clamp(0.0, 1.0),
                    nll: bce_sum / keys.len() as f64,
                    brier: brier_sum / keys.len() as f64,
                    sorted_probs_desc: sorted_desc,
                    ml_counts: Some((tp, fp, fneg)),
                    score_abs_err: None,
                    score_rps: None,
                })
            } else {
                let target_s = tv.as_str()?.trim();
                let target_idx = keys.iter().position(|k| k.eq_ignore_ascii_case(target_s))?;
                let (probs, log_probs) = softmax_and_log_softmax(&raw_logits, temp);

                let mut best_idx = 0usize;
                let mut best_weighted = f64::NEG_INFINITY;
                for (i, (&p, key)) in probs.iter().zip(keys.iter()).enumerate() {
                    let w = cfg
                        .cost_weights
                        .get(&key.to_lowercase())
                        .or_else(|| cfg.cost_weights.get(key))
                        .copied()
                        .unwrap_or(1.0);
                    if p * w > best_weighted {
                        best_weighted = p * w;
                        best_idx = i;
                    }
                }
                let p_win = probs[best_idx];
                let nll = -log_probs[target_idx];
                let brier: f64 = probs
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (p - if i == target_idx { 1.0 } else { 0.0 }).powi(2))
                    .sum();

                let mut sorted_desc: Vec<(String, f64)> = keys
                    .iter()
                    .zip(probs.iter())
                    .map(|(k_name, &p)| (k_name.to_lowercase(), p))
                    .collect();
                sorted_desc.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

                Some(DecisionObs {
                    pred_label: keys[best_idx].to_lowercase(),
                    truth_label: keys[target_idx].to_lowercase(),
                    is_correct: best_idx == target_idx,
                    p_win,
                    nll,
                    brier,
                    sorted_probs_desc: sorted_desc,
                    ml_counts: None,
                    score_abs_err: None,
                    score_rps: None,
                })
            }
        }
        QuestionType::Score => {
            let target_num = tv
                .as_f64()
                .or_else(|| tv.as_str().and_then(|s| s.trim().parse::<f64>().ok()))?;
            let num_levels = raw_logits.len();
            let target_idx =
                (target_num.round() as isize).clamp(0, num_levels as isize - 1) as usize;
            let (probs, log_probs) = softmax_and_log_softmax(&raw_logits, temp);
            let pred_score: f64 = probs.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
            let abs_err = (pred_score - target_num).abs();
            let is_correct = abs_err < 0.5;

            // Soft linear-interpolated cross-entropy if target_num is fractional, exact discrete NLL otherwise
            let lo_idx = (target_num.floor() as isize).clamp(0, num_levels as isize - 1) as usize;
            let hi_idx = (target_num.ceil() as isize).clamp(0, num_levels as isize - 1) as usize;
            let frac = (target_num - lo_idx as f64).clamp(0.0, 1.0);
            let nll = -((1.0 - frac) * log_probs[lo_idx] + frac * log_probs[hi_idx]);

            let brier: f64 = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (p - if i == target_idx { 1.0 } else { 0.0 }).powi(2))
                .sum();

            // Ordinal Ranked Probability Score (RPS)
            let mut cum_p = 0.0;
            let mut rps_sum = 0.0;
            for (level, &prob_l) in probs.iter().enumerate().take(num_levels.saturating_sub(1)) {
                cum_p += prob_l;
                let cum_y = if target_num <= level as f64 + 1e-9 {
                    1.0
                } else {
                    0.0
                };
                rps_sum += (cum_p - cum_y).powi(2);
            }
            let rps = rps_sum / (num_levels.saturating_sub(1).max(1) as f64);

            let mut sorted_desc: Vec<(String, f64)> = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i.to_string(), p))
                .collect();
            sorted_desc.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let pred_idx = (pred_score.round() as isize).clamp(0, num_levels as isize - 1) as usize;
            let p_win = probs.get(pred_idx).copied().unwrap_or(1.0);

            Some(DecisionObs {
                pred_label: format!("{}", pred_score.round() as i64),
                truth_label: format!("{}", target_idx),
                is_correct,
                p_win,
                nll,
                brier,
                sorted_probs_desc: sorted_desc,
                ml_counts: None,
                score_abs_err: Some(abs_err),
                score_rps: Some(rps),
            })
        }
        QuestionType::Noul => {
            let target_bool = tv
                .as_bool()
                .or_else(|| {
                    tv.as_str()
                        .and_then(|s| match s.trim().to_lowercase().as_str() {
                            "true" | "1" => Some(true),
                            "false" | "0" => Some(false),
                            _ => None,
                        })
                })
                .or_else(|| tv.as_i64().map(|n| n != 0))?;

            let (probs, log_probs) = softmax_and_log_softmax(&raw_logits, temp);
            let raw_p_true = probs.get(1).copied().unwrap_or(0.5);
            let w_true = cfg
                .cost_weights
                .get("true")
                .or_else(|| cfg.cost_weights.get(qid))
                .copied()
                .unwrap_or(1.0);
            let w_false = cfg.cost_weights.get("false").copied().unwrap_or(1.0);
            let has_cost = (w_true - 1.0).abs() > 1e-9 || (w_false - 1.0).abs() > 1e-9;

            let (p_true, nll) = if has_cost {
                let denom = (w_true * raw_p_true + w_false * (1.0 - raw_p_true)).max(1e-12);
                let pt = (w_true * raw_p_true / denom).clamp(1e-12, 1.0 - 1e-12);
                let loss = if target_bool {
                    -pt.ln()
                } else {
                    -(1.0 - pt).ln()
                };
                (pt, loss)
            } else {
                let loss = -log_probs[usize::from(target_bool)];
                (raw_p_true, loss)
            };

            let pred_bool = p_true >= 0.5;
            let p_win = if pred_bool { p_true } else { 1.0 - p_true };
            let y_f = if target_bool { 1.0 } else { 0.0 };
            let brier = 2.0 * (p_true - y_f).powi(2);

            let mut sorted_desc = vec![
                ("false".to_string(), 1.0 - p_true),
                ("true".to_string(), p_true),
            ];
            sorted_desc.sort_by(|a, b| b.1.total_cmp(&a.1));

            Some(DecisionObs {
                pred_label: pred_bool.to_string(),
                truth_label: target_bool.to_string(),
                is_correct: pred_bool == target_bool,
                p_win,
                nll,
                brier,
                sorted_probs_desc: sorted_desc,
                ml_counts: None,
                score_abs_err: None,
                score_rps: None,
            })
        }
    }
}

// ============================================================================
// Calibration & Selective Metrics Helpers
// ============================================================================

fn compute_equal_width_ece(p_wins: &[f64], correctness: &[bool], bins: usize) -> f64 {
    if p_wins.is_empty() || p_wins.len() != correctness.len() {
        return 0.0;
    }
    let n = p_wins.len() as f64;
    let mut counts = vec![0usize; bins];
    let mut corr_sums = vec![0usize; bins];
    let mut conf_sums = vec![0.0; bins];

    for (&p, &c) in p_wins.iter().zip(correctness.iter()) {
        let b = ((p.clamp(0.0, 1.0) * bins as f64).floor() as usize).min(bins - 1);
        counts[b] += 1;
        conf_sums[b] += p;
        if c {
            corr_sums[b] += 1;
        }
    }

    let mut ece = 0.0;
    for b in 0..bins {
        if counts[b] > 0 {
            let acc = corr_sums[b] as f64 / counts[b] as f64;
            let avg_p = conf_sums[b] as f64 / counts[b] as f64;
            ece += (counts[b] as f64 / n) * (acc - avg_p).abs();
        }
    }
    ece
}

/// Equal-mass Adaptive ECE (ACE): partitions sorted winning probabilities into equal-count bins
/// so overconfident fine-tuned models clustering near `0.99` do not leave 9 of 10 bins empty.
fn compute_equal_mass_ace(p_wins: &[f64], correctness: &[bool], max_bins: usize) -> f64 {
    let n = p_wins.len();
    if n == 0 || n != correctness.len() {
        return 0.0;
    }
    let b = max_bins.min(n).max(1);
    let mut pairs: Vec<(f64, bool)> = p_wins
        .iter()
        .copied()
        .zip(correctness.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let base_sz = n / b;
    let rem = n % b;
    let mut start = 0usize;
    let mut ace = 0.0;
    for bin_idx in 0..b {
        let sz = base_sz + usize::from(bin_idx < rem);
        if sz == 0 {
            continue;
        }
        let slice = &pairs[start..start + sz];
        let conf_mean = slice.iter().map(|(p, _)| *p).sum::<f64>() / sz as f64;
        let acc_mean = slice.iter().filter(|(_, c)| *c).count() as f64 / sz as f64;
        ace += (sz as f64 / n as f64) * (acc_mean - conf_mean).abs();
        start += sz;
    }
    ace
}

/// Returns `(aurc, excess_aurc)` where `excess_aurc` subtracts the exact finite-sample discrete
/// oracle risk-coverage area (`>= 0.0` for all `N >= 1`).
fn compute_aurc_and_eaurc(p_wins: &[f64], correctness: &[bool]) -> (f64, f64) {
    let n = p_wins.len();
    if n == 0 || n != correctness.len() {
        return (0.0, 0.0);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p_wins[b].total_cmp(&p_wins[a]).then_with(|| a.cmp(&b)));

    let mut corr_running = 0usize;
    let mut risk_sum = 0.0;
    for (rank_idx, &i) in order.iter().enumerate() {
        if correctness[i] {
            corr_running += 1;
        }
        let acc_k = corr_running as f64 / (rank_idx + 1) as f64;
        risk_sum += 1.0 - acc_k;
    }
    let aurc = risk_sum / n as f64;

    // Exact finite-sample oracle ranks all `total_corr` correct items ahead of errors
    let total_corr = corr_running;
    let mut oracle_risk_sum = 0.0;
    for k in (total_corr + 1)..=n {
        oracle_risk_sum += (k - total_corr) as f64 / k as f64;
    }
    let oracle_aurc = oracle_risk_sum / n as f64;
    (aurc, (aurc - oracle_aurc).max(0.0))
}

fn macro_and_weighted_f1_from_pairs(pairs: &[(String, String)]) -> (Option<f64>, Option<f64>) {
    if pairs.is_empty() {
        return (None, None);
    }
    let mut counts: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for (t, p) in pairs {
        counts.entry(t.as_str()).or_default().0 += 1;
        counts.entry(p.as_str()).or_default().1 += 1;
        if t == p {
            counts.entry(t.as_str()).or_default().2 += 1;
        }
    }
    let mut f1_sum = 0.0;
    let mut weighted_sum = 0.0;
    let mut present_classes = 0usize;
    let mut total_support = 0usize;
    for &(sup, pred, tp) in counts.values() {
        if sup == 0 {
            continue;
        }
        present_classes += 1;
        total_support += sup;
        let denom = sup + pred;
        let f1 = if denom > 0 {
            (2 * tp) as f64 / denom as f64
        } else {
            0.0
        };
        f1_sum += f1;
        weighted_sum += f1 * sup as f64;
    }
    if present_classes == 0 || total_support == 0 {
        return (None, None);
    }
    (
        Some(f1_sum / present_classes as f64),
        Some(weighted_sum / total_support as f64),
    )
}

// ============================================================================
// Core Paired Statistical Evaluation Engine
// ============================================================================

#[derive(Clone)]
struct PairedDecisionRecord {
    item_idx: usize,
    qid: String,
    qtype: QuestionType,
    base: DecisionObs,
    cand: DecisionObs,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_paired_decisions(
    regime: &str,
    config: &PairedExperimentConfig,
    base_cfg: &ClassifierConfig,
    cand_cfg: &ClassifierConfig,
    base_results: &[(String, DecisionResult)],
    cand_results: &[(String, DecisionResult)],
    test_items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    base_cal: Option<&FittedPostHocCalibration>,
    cand_cal: Option<&FittedPostHocCalibration>,
) -> Result<PairedComparisonSummary> {
    if base_results.len() != test_items.len() || cand_results.len() != test_items.len() {
        bail!(
            "Mismatched result lengths in paired evaluation (items={}, base={}, candidate={})",
            test_items.len(),
            base_results.len(),
            cand_results.len()
        );
    }

    let base_helper = LayaClassifier::new(base_cfg.clone());
    let cand_helper = LayaClassifier::new(cand_cfg.clone());

    let mut records: Vec<PairedDecisionRecord> = Vec::new();
    let mut item_all_correct: Vec<Option<(bool, bool)>> = vec![None; test_items.len()];
    let mut item_min_p_win: Vec<Option<(f64, f64)>> = vec![None; test_items.len()];

    for (item_idx, item) in test_items.iter().enumerate() {
        let (_, base_dec) = &base_results[item_idx];
        let (_, cand_dec) = &cand_results[item_idx];
        let mut all_b = true;
        let mut all_c = true;
        let mut min_pb = 1.0_f64;
        let mut min_pc = 1.0_f64;
        let mut any_q = false;

        for (qid, q) in questions {
            let Some(tv) = resolve_ground_truth_for_question(
                item.ground_truth.as_ref(),
                qid,
                q,
                questions.len(),
            ) else {
                continue;
            };
            let (Some(b_ans), Some(c_ans)) = (base_dec.answers.get(qid), cand_dec.answers.get(qid))
            else {
                continue;
            };
            let (Some(b_obs), Some(c_obs)) = (
                score_single_decision(b_ans, q, qid, tv, base_cfg, &base_helper),
                score_single_decision(c_ans, q, qid, tv, cand_cfg, &cand_helper),
            ) else {
                continue;
            };
            any_q = true;
            if !b_obs.is_correct {
                all_b = false;
            }
            if !c_obs.is_correct {
                all_c = false;
            }
            min_pb = min_pb.min(b_obs.p_win);
            min_pc = min_pc.min(c_obs.p_win);
            records.push(PairedDecisionRecord {
                item_idx,
                qid: qid.clone(),
                qtype: q.qtype,
                base: b_obs,
                cand: c_obs,
            });
        }
        if any_q {
            item_all_correct[item_idx] = Some((all_b, all_c));
            item_min_p_win[item_idx] = Some((min_pb, min_pc));
        }
    }

    if records.is_empty() {
        bail!("No labeled ground-truth decisions matched between test dataset and schema");
    }

    // Group test items into atomic resampling units (honoring `leakage_group_id` if present)
    let mut group_key_to_unit: BTreeMap<String, usize> = BTreeMap::new();
    let mut unit_items: Vec<Vec<usize>> = Vec::new();
    let mut item_to_unit = vec![0usize; test_items.len()];
    for (idx, item) in test_items.iter().enumerate() {
        let gkey = item
            .metadata
            .get("leakage_group_id")
            .and_then(|v| v.as_str())
            .map(|s| format!("g:{s}"))
            .unwrap_or_else(|| format!("i:{idx}"));
        let u_idx = if let Some(&u) = group_key_to_unit.get(&gkey) {
            u
        } else {
            let next = unit_items.len();
            group_key_to_unit.insert(gkey, next);
            unit_items.push(Vec::new());
            next
        };
        unit_items[u_idx].push(idx);
        item_to_unit[idx] = u_idx;
    }
    let num_units = unit_items.len();

    let mut unit_records: Vec<Vec<usize>> = vec![Vec::new(); num_units];
    for (rec_idx, rec) in records.iter().enumerate() {
        unit_records[item_to_unit[rec.item_idx]].push(rec_idx);
    }

    type UnitMetricFn<'a> = &'a dyn Fn(&[usize], Option<&[bool]>) -> Option<(f64, f64)>;

    // Helper: paired bootstrap CI + sign-flip permutation p-value over arbitrary unit-indexed metric
    let eval_stat_pair = |seed_salt: u64,
                          higher_is_better: bool,
                          compute_pair: UnitMetricFn<'_>|
     -> Option<MetricDeltaCi> {
        let all_units: Vec<usize> = (0..num_units).collect();
        let (base_obs, cand_obs) = compute_pair(&all_units, None)?;
        let delta_obs = cand_obs - base_obs;

        let b_iters = config.bootstrap_samples;
        let mut rng = DeterministicRng::new(config.seed ^ seed_salt);
        let mut boot_deltas = Vec::with_capacity(b_iters);
        let mut sampled_units = vec![0usize; num_units];
        for _ in 0..b_iters {
            for slot in &mut sampled_units {
                *slot = rng.gen_range(num_units);
            }
            if let Some((b_v, c_v)) = compute_pair(&sampled_units, None) {
                boot_deltas.push(c_v - b_v);
            }
        }
        boot_deltas.sort_by(|a, b| a.total_cmp(b));

        // Jackknife LOO for BCa when num_units <= 500 (bounded to keep complex non-additive metrics fast)
        let jack_vec = if (4..=500).contains(&num_units) && b_iters > 0 {
            let mut j_vals = Vec::with_capacity(num_units);
            let mut loo_units = Vec::with_capacity(num_units - 1);
            for omit in 0..num_units {
                loo_units.clear();
                for u in 0..num_units {
                    if u != omit {
                        loo_units.push(u);
                    }
                }
                if let Some((b_v, c_v)) = compute_pair(&loo_units, None) {
                    j_vals.push(c_v - b_v);
                }
            }
            (j_vals.len() == num_units).then_some(j_vals)
        } else {
            None
        };

        let (ci_lo, ci_hi, ci_method) = compute_bootstrap_ci(
            delta_obs,
            &boot_deltas,
            jack_vec.as_deref(),
            config.confidence,
        );

        let r_perms = config.permutations;
        let p_perm = if r_perms == 0 || delta_obs.abs() < 1e-12 {
            1.0
        } else {
            let mut perm_rng =
                DeterministicRng::new(config.seed ^ seed_salt ^ 0x5045_524D_5445_5354);
            let mut flips = vec![false; num_units];
            let mut extreme = 0usize;
            for _ in 0..r_perms {
                for fl in &mut flips {
                    *fl = (perm_rng.next_u64() & 1) == 1;
                }
                if let Some((pb, pc)) = compute_pair(&all_units, Some(&flips))
                    && (pc - pb).abs() >= delta_obs.abs() - 1e-12
                {
                    extreme += 1;
                }
            }
            (1.0 + extreme as f64) / (r_perms as f64 + 1.0)
        };

        Some(MetricDeltaCi {
            base: round4(base_obs),
            candidate: round4(cand_obs),
            delta: round4(delta_obs),
            ci_low: round4(ci_lo),
            ci_high: round4(ci_hi),
            ci_method: ci_method.to_string(),
            p_perm: round6(p_perm),
            higher_is_better,
        })
    };

    // Fast additive per-unit metric evaluator (exact O(U) BCa jackknife + O(U) bootstrap/permutation!)
    let eval_additive_metric =
        |seed_salt: u64,
         higher_is_better: bool,
         rec_filter: &dyn Fn(&PairedDecisionRecord) -> Option<(f64, f64)>|
         -> Option<MetricDeltaCi> {
            let mut unit_sums = vec![(0.0_f64, 0.0_f64, 0usize); num_units];
            let mut tot_b = 0.0;
            let mut tot_c = 0.0;
            let mut tot_n = 0usize;
            for rec in &records {
                if let Some((bv, cv)) = rec_filter(rec) {
                    let u = item_to_unit[rec.item_idx];
                    unit_sums[u].0 += bv;
                    unit_sums[u].1 += cv;
                    unit_sums[u].2 += 1;
                    tot_b += bv;
                    tot_c += cv;
                    tot_n += 1;
                }
            }
            if tot_n == 0 {
                return None;
            }
            let base_obs = tot_b / tot_n as f64;
            let cand_obs = tot_c / tot_n as f64;
            let delta_obs = cand_obs - base_obs;

            let b_iters = config.bootstrap_samples;
            let mut rng = DeterministicRng::new(config.seed ^ seed_salt);
            let mut boot_deltas = Vec::with_capacity(b_iters);
            for _ in 0..b_iters {
                let mut sb = 0.0;
                let mut sc = 0.0;
                let mut sn = 0usize;
                for _ in 0..num_units {
                    let u = rng.gen_range(num_units);
                    sb += unit_sums[u].0;
                    sc += unit_sums[u].1;
                    sn += unit_sums[u].2;
                }
                if sn > 0 {
                    boot_deltas.push((sc - sb) / sn as f64);
                }
            }
            boot_deltas.sort_by(|a, b| a.total_cmp(b));

            // Exact O(U) leave-one-unit-out jackknife for BCa when num_units <= 2000!
            let jack_vec = if (4..=2000).contains(&num_units) && b_iters > 0 {
                let mut jv = Vec::with_capacity(num_units);
                for &(ub, uc, un) in unit_sums.iter().take(num_units) {
                    let rem_n = tot_n.saturating_sub(un);
                    if rem_n > 0 {
                        let lb = (tot_b - ub) / rem_n as f64;
                        let lc = (tot_c - uc) / rem_n as f64;
                        jv.push(lc - lb);
                    }
                }
                (jv.len() == num_units).then_some(jv)
            } else {
                None
            };

            let (ci_lo, ci_hi, ci_method) = compute_bootstrap_ci(
                delta_obs,
                &boot_deltas,
                jack_vec.as_deref(),
                config.confidence,
            );

            let r_perms = config.permutations;
            let p_perm = if r_perms == 0 || delta_obs.abs() < 1e-12 {
                1.0
            } else {
                let mut perm_rng =
                    DeterministicRng::new(config.seed ^ seed_salt ^ 0x5045_524D_5445_5354);
                let mut extreme = 0usize;
                for _ in 0..r_perms {
                    let mut diff_sum = 0.0;
                    for &(ub, uc, un) in &unit_sums {
                        if un == 0 {
                            continue;
                        }
                        let sign = if (perm_rng.next_u64() & 1) == 1 {
                            -1.0
                        } else {
                            1.0
                        };
                        diff_sum += sign * (uc - ub);
                    }
                    let perm_delta = diff_sum / tot_n as f64;
                    if perm_delta.abs() >= delta_obs.abs() - 1e-12 {
                        extreme += 1;
                    }
                }
                (1.0 + extreme as f64) / (r_perms as f64 + 1.0)
            };

            Some(MetricDeltaCi {
                base: round4(base_obs),
                candidate: round4(cand_obs),
                delta: round4(delta_obs),
                ci_low: round4(ci_lo),
                ci_high: round4(ci_hi),
                ci_method: ci_method.to_string(),
                p_perm: round6(p_perm),
                higher_is_better,
            })
        };

    // 1. Overall Additive Metrics (Accuracy, NLL, Brier, Score MAE, Score RPS)
    let accuracy = eval_additive_metric(0x0101, true, &|r| {
        Some((
            if r.base.is_correct { 1.0 } else { 0.0 },
            if r.cand.is_correct { 1.0 } else { 0.0 },
        ))
    })
    .unwrap();
    let nll = eval_additive_metric(0x0102, false, &|r| Some((r.base.nll, r.cand.nll))).unwrap();
    let brier =
        eval_additive_metric(0x0103, false, &|r| Some((r.base.brier, r.cand.brier))).unwrap();
    let score_mae = eval_additive_metric(0x0104, false, &|r| {
        r.base.score_abs_err.zip(r.cand.score_abs_err)
    });
    let score_rps =
        eval_additive_metric(0x0105, false, &|r| r.base.score_rps.zip(r.cand.score_rps));

    // 2. Non-additive Calibration & Selective Classification Metrics (ECE, ACE, AURC, E-AURC)
    // Exclude continuous `Score` expected-value predictions from top-label ECE/ACE when Choice/Noul exist, matching #10 High #1
    let has_discrete_cls = records
        .iter()
        .any(|r| r.qtype != QuestionType::Score && r.base.ml_counts.is_none());

    let collect_pwin_corr = |units: &[usize],
                             flips: Option<&[bool]>,
                             discrete_only: bool|
     -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<bool>) {
        let mut bp = Vec::new();
        let mut bc = Vec::new();
        let mut cp = Vec::new();
        let mut cc = Vec::new();
        for &u in units {
            let swapped = flips.is_some_and(|fl| fl[u]);
            for &ri in &unit_records[u] {
                let rec = &records[ri];
                if discrete_only
                    && (rec.qtype == QuestionType::Score || rec.base.ml_counts.is_some())
                {
                    continue;
                }
                let (b_o, c_o) = if swapped {
                    (&rec.cand, &rec.base)
                } else {
                    (&rec.base, &rec.cand)
                };
                bp.push(b_o.p_win);
                bc.push(b_o.is_correct);
                cp.push(c_o.p_win);
                cc.push(c_o.is_correct);
            }
        }
        (bp, bc, cp, cc)
    };

    let ece = eval_stat_pair(0x0201, false, &|units, flips| {
        let (bp, bc, cp, cc) = collect_pwin_corr(units, flips, has_discrete_cls);
        (!bp.is_empty()).then(|| {
            (
                compute_equal_width_ece(&bp, &bc, 10),
                compute_equal_width_ece(&cp, &cc, 10),
            )
        })
    })
    .unwrap();

    let ace = eval_stat_pair(0x0202, false, &|units, flips| {
        let (bp, bc, cp, cc) = collect_pwin_corr(units, flips, has_discrete_cls);
        (!bp.is_empty()).then(|| {
            (
                compute_equal_mass_ace(&bp, &bc, 10),
                compute_equal_mass_ace(&cp, &cc, 10),
            )
        })
    })
    .unwrap();

    let aurc = eval_stat_pair(0x0203, false, &|units, flips| {
        let (bp, bc, cp, cc) = collect_pwin_corr(units, flips, false);
        (!bp.is_empty()).then(|| {
            (
                compute_aurc_and_eaurc(&bp, &bc).0,
                compute_aurc_and_eaurc(&cp, &cc).0,
            )
        })
    })
    .unwrap();

    let e_aurc = eval_stat_pair(0x0204, false, &|units, flips| {
        let (bp, bc, cp, cc) = collect_pwin_corr(units, flips, false);
        (!bp.is_empty()).then(|| {
            (
                compute_aurc_and_eaurc(&bp, &bc).1,
                compute_aurc_and_eaurc(&cp, &cc).1,
            )
        })
    })
    .unwrap();

    // 3. Aggregate Macro-F1 / Weighted-F1 (averaged across single-label Choice questions) & Multi-Label Micro-F1
    let choice_qids: Vec<String> = questions
        .iter()
        .filter(|(_, q)| q.qtype == QuestionType::Choice)
        .map(|(qid, _)| qid.clone())
        .collect();

    let has_single_choice = records
        .iter()
        .any(|r| r.qtype == QuestionType::Choice && r.base.ml_counts.is_none());
    let has_multi_label = records.iter().any(|r| r.base.ml_counts.is_some());

    let (macro_f1, weighted_f1) = if has_single_choice {
        let compute_f1s =
            |units: &[usize], flips: Option<&[bool]>, weighted: bool| -> Option<(f64, f64)> {
                let mut b_per_q: BTreeMap<&str, Vec<(String, String)>> = BTreeMap::new();
                let mut c_per_q: BTreeMap<&str, Vec<(String, String)>> = BTreeMap::new();
                for &u in units {
                    let swapped = flips.is_some_and(|fl| fl[u]);
                    for &ri in &unit_records[u] {
                        let rec = &records[ri];
                        if rec.qtype != QuestionType::Choice || rec.base.ml_counts.is_some() {
                            continue;
                        }
                        let (bo, co) = if swapped {
                            (&rec.cand, &rec.base)
                        } else {
                            (&rec.base, &rec.cand)
                        };
                        b_per_q
                            .entry(rec.qid.as_str())
                            .or_default()
                            .push((bo.truth_label.clone(), bo.pred_label.clone()));
                        c_per_q
                            .entry(rec.qid.as_str())
                            .or_default()
                            .push((co.truth_label.clone(), co.pred_label.clone()));
                    }
                }
                let mut b_sum = 0.0;
                let mut c_sum = 0.0;
                let mut q_cnt = 0usize;
                for qid in &choice_qids {
                    if let (Some(bp), Some(cp)) =
                        (b_per_q.get(qid.as_str()), c_per_q.get(qid.as_str()))
                    {
                        let (bm, bw) = macro_and_weighted_f1_from_pairs(bp);
                        let (cm, cw) = macro_and_weighted_f1_from_pairs(cp);
                        let bv = if weighted { bw } else { bm };
                        let cv = if weighted { cw } else { cm };
                        if let (Some(b_val), Some(c_val)) = (bv, cv) {
                            b_sum += b_val;
                            c_sum += c_val;
                            q_cnt += 1;
                        }
                    }
                }
                (q_cnt > 0).then(|| (b_sum / q_cnt as f64, c_sum / q_cnt as f64))
            };
        (
            eval_stat_pair(0x0301, true, &|u, fl| compute_f1s(u, fl, false)),
            eval_stat_pair(0x0302, true, &|u, fl| compute_f1s(u, fl, true)),
        )
    } else {
        (None, None)
    };

    let multi_label_f1 = if has_multi_label {
        eval_stat_pair(0x0303, true, &|units, flips| {
            let (mut b_tp, mut b_fp, mut b_fn) = (0usize, 0usize, 0usize);
            let (mut c_tp, mut c_fp, mut c_fn) = (0usize, 0usize, 0usize);
            let mut any = false;
            for &u in units {
                let swapped = flips.is_some_and(|fl| fl[u]);
                for &ri in &unit_records[u] {
                    let rec = &records[ri];
                    let (Some(bm), Some(cm)) = (rec.base.ml_counts, rec.cand.ml_counts) else {
                        continue;
                    };
                    any = true;
                    let ((tp0, fp0, fn0), (tp1, fp1, fn1)) =
                        if swapped { (cm, bm) } else { (bm, cm) };
                    b_tp += tp0;
                    b_fp += fp0;
                    b_fn += fn0;
                    c_tp += tp1;
                    c_fp += fp1;
                    c_fn += fn1;
                }
            }
            if !any {
                return None;
            }
            let bf1 = if 2 * b_tp + b_fp + b_fn > 0 {
                (2 * b_tp) as f64 / (2 * b_tp + b_fp + b_fn) as f64
            } else {
                1.0
            };
            let cf1 = if 2 * c_tp + c_fp + c_fn > 0 {
                (2 * c_tp) as f64 / (2 * c_tp + c_fp + c_fn) as f64
            } else {
                1.0
            };
            Some((bf1, cf1))
        })
    } else {
        None
    };

    // 4. Overall & Item-Level Exact-Match McNemar Tests
    let overall_mcnemar = compute_mcnemar_over_records(
        records
            .iter()
            .map(|r| (r.base.is_correct, r.cand.is_correct)),
    );
    let item_exact_match_mcnemar =
        compute_mcnemar_over_records(item_all_correct.iter().filter_map(|&x| x));

    // 5. Selective Risk: Conformal Set Size / Singleton Rate & Deferral Coverage (when `cal` was fitted)
    let selective_risk = if let (Some(bf), Some(cf)) = (base_cal, cand_cal) {
        let mut b_cov_hits = 0usize;
        let mut c_cov_hits = 0usize;
        let mut cov_total = 0usize;
        type ConformalRecPair = ((f64, f64), (f64, f64));
        let mut per_rec_conformal: Vec<Option<ConformalRecPair>> =
            Vec::with_capacity(records.len());

        for rec in &records {
            if let (Some(&bq), Some(&cq)) = (
                bf.conformal_quantiles.get(&rec.qid),
                cf.conformal_quantiles.get(&rec.qid),
            ) && rec.base.ml_counts.is_none()
            {
                let b_set = aps_predict_set(&rec.base.sorted_probs_desc, bq);
                let c_set = aps_predict_set(&rec.cand.sorted_probs_desc, cq);
                cov_total += 1;
                if b_set
                    .iter()
                    .any(|s| s.eq_ignore_ascii_case(&rec.base.truth_label))
                {
                    b_cov_hits += 1;
                }
                if c_set
                    .iter()
                    .any(|s| s.eq_ignore_ascii_case(&rec.cand.truth_label))
                {
                    c_cov_hits += 1;
                }
                per_rec_conformal.push(Some((
                    (b_set.len() as f64, c_set.len() as f64),
                    (
                        if b_set.len() == 1 { 1.0 } else { 0.0 },
                        if c_set.len() == 1 { 1.0 } else { 0.0 },
                    ),
                )));
            } else {
                per_rec_conformal.push(None);
            }
        }

        let set_size_ci = eval_stat_pair(0x0401, false, &|units, flips| {
            let (mut sb, mut sc, mut cnt) = (0.0, 0.0, 0usize);
            for &u in units {
                let swapped = flips.is_some_and(|fl| fl[u]);
                for &ri in &unit_records[u] {
                    if let Some(((bsz, csz), _)) = per_rec_conformal[ri] {
                        let (x, y) = if swapped { (csz, bsz) } else { (bsz, csz) };
                        sb += x;
                        sc += y;
                        cnt += 1;
                    }
                }
            }
            (cnt > 0).then(|| (sb / cnt as f64, sc / cnt as f64))
        });

        let singleton_ci = eval_stat_pair(0x0402, true, &|units, flips| {
            let (mut sb, mut sc, mut cnt) = (0.0, 0.0, 0usize);
            for &u in units {
                let swapped = flips.is_some_and(|fl| fl[u]);
                for &ri in &unit_records[u] {
                    if let Some((_, (bsg, csg))) = per_rec_conformal[ri] {
                        let (x, y) = if swapped { (csg, bsg) } else { (bsg, csg) };
                        sb += x;
                        sc += y;
                        cnt += 1;
                    }
                }
            }
            (cnt > 0).then(|| (sb / cnt as f64, sc / cnt as f64))
        });

        // Deferral coverage & test kept accuracy
        let b_tau = bf.defer_threshold;
        let c_tau = cf.defer_threshold;
        let mut item_defer_kept: Vec<Option<(bool, bool, bool, bool)>> =
            vec![None; test_items.len()];
        let (mut b_kept_n, mut b_kept_ok) = (0usize, 0usize);
        let (mut c_kept_n, mut c_kept_ok) = (0usize, 0usize);

        for idx in 0..test_items.len() {
            if let (Some((all_b, all_c)), Some((min_pb, min_pc))) =
                (item_all_correct[idx], item_min_p_win[idx])
            {
                let b_risk = ((1.0 - min_pb).clamp(0.0, 1.0) * 1_000_000.0).round() / 1_000_000.0;
                let c_risk = ((1.0 - min_pc).clamp(0.0, 1.0) * 1_000_000.0).round() / 1_000_000.0;
                let bk = b_tau.is_some_and(|t| b_risk <= t);
                let ck = c_tau.is_some_and(|t| c_risk <= t);
                if bk {
                    b_kept_n += 1;
                    if all_b {
                        b_kept_ok += 1;
                    }
                }
                if ck {
                    c_kept_n += 1;
                    if all_c {
                        c_kept_ok += 1;
                    }
                }
                item_defer_kept[idx] = Some((bk, ck, all_b, all_c));
            }
        }

        let defer_cov_ci = eval_stat_pair(0x0403, true, &|units, flips| {
            let (mut bk_sum, mut ck_sum, mut tot) = (0.0, 0.0, 0usize);
            for &u in units {
                let swapped = flips.is_some_and(|fl| fl[u]);
                for &it_idx in &unit_items[u] {
                    if let Some((bk, ck, _, _)) = item_defer_kept[it_idx] {
                        let b_val = if bk { 1.0 } else { 0.0 };
                        let c_val = if ck { 1.0 } else { 0.0 };
                        let (x, y) = if swapped {
                            (c_val, b_val)
                        } else {
                            (b_val, c_val)
                        };
                        bk_sum += x;
                        ck_sum += y;
                        tot += 1;
                    }
                }
            }
            (tot > 0).then(|| (bk_sum / tot as f64, ck_sum / tot as f64))
        });

        if let (Some(ss_ci), Some(sg_ci), Some(dc_ci)) = (set_size_ci, singleton_ci, defer_cov_ci) {
            let avg_bq = if bf.conformal_quantiles.is_empty() {
                1.0
            } else {
                bf.conformal_quantiles.values().sum::<f64>() / bf.conformal_quantiles.len() as f64
            };
            let avg_cq = if cf.conformal_quantiles.is_empty() {
                1.0
            } else {
                cf.conformal_quantiles.values().sum::<f64>() / cf.conformal_quantiles.len() as f64
            };
            Some(SelectiveRiskSummary {
                alpha: config.alpha,
                base_conformal_q_hat: round4(avg_bq),
                candidate_conformal_q_hat: round4(avg_cq),
                base_conformal_coverage: round4(b_cov_hits as f64 / cov_total.max(1) as f64),
                candidate_conformal_coverage: round4(c_cov_hits as f64 / cov_total.max(1) as f64),
                conformal_set_size: ss_ci,
                conformal_singleton_rate: sg_ci,
                target_accuracy: config.target_accuracy,
                base_defer_threshold: b_tau,
                candidate_defer_threshold: c_tau,
                base_defer_kept_accuracy: if b_kept_n > 0 {
                    round4(b_kept_ok as f64 / b_kept_n as f64)
                } else {
                    0.0
                },
                candidate_defer_kept_accuracy: if c_kept_n > 0 {
                    round4(c_kept_ok as f64 / c_kept_n as f64)
                } else {
                    0.0
                },
                base_defer_cost_per_1000: round4((1.0 - dc_ci.base) * 1000.0 * config.system2_cost),
                candidate_defer_cost_per_1000: round4(
                    (1.0 - dc_ci.candidate) * 1000.0 * config.system2_cost,
                ),
                defer_coverage: dc_ci,
            })
        } else {
            None
        }
    } else {
        None
    };

    // 6. Per-Question Paired Reports + Holm-Bonferroni Multiplicity Adjustment
    let mut q_reports_vec: Vec<(String, QuestionPairedReport)> = Vec::new();
    for (q_idx, (qid, q)) in questions.iter().enumerate() {
        let q_rec_indices: Vec<usize> = records
            .iter()
            .enumerate()
            .filter_map(|(ri, r)| (r.qid == *qid).then_some(ri))
            .collect();
        if q_rec_indices.is_empty() {
            continue;
        }
        let salt_base = 0x1000 + (q_idx as u64) * 0x100;

        let q_acc = eval_additive_metric(salt_base + 1, true, &|r| {
            (r.qid == *qid).then_some((
                if r.base.is_correct { 1.0 } else { 0.0 },
                if r.cand.is_correct { 1.0 } else { 0.0 },
            ))
        })
        .unwrap();
        let q_nll = eval_additive_metric(salt_base + 2, false, &|r| {
            (r.qid == *qid).then_some((r.base.nll, r.cand.nll))
        })
        .unwrap();
        let q_brier = eval_additive_metric(salt_base + 3, false, &|r| {
            (r.qid == *qid).then_some((r.base.brier, r.cand.brier))
        })
        .unwrap();

        let q_ece = (q.qtype != QuestionType::Score)
            .then(|| {
                eval_stat_pair(salt_base + 4, false, &|units, flips| {
                    let mut bp = Vec::new();
                    let mut bc = Vec::new();
                    let mut cp = Vec::new();
                    let mut cc = Vec::new();
                    for &u in units {
                        let swapped = flips.is_some_and(|fl| fl[u]);
                        for &ri in &unit_records[u] {
                            let rec = &records[ri];
                            if rec.qid != *qid {
                                continue;
                            }
                            let (bo, co) = if swapped {
                                (&rec.cand, &rec.base)
                            } else {
                                (&rec.base, &rec.cand)
                            };
                            bp.push(bo.p_win);
                            bc.push(bo.is_correct);
                            cp.push(co.p_win);
                            cc.push(co.is_correct);
                        }
                    }
                    (!bp.is_empty()).then(|| {
                        (
                            compute_equal_width_ece(&bp, &bc, 10),
                            compute_equal_width_ece(&cp, &cc, 10),
                        )
                    })
                })
            })
            .flatten();

        let q_mae = eval_additive_metric(salt_base + 5, false, &|r| {
            (r.qid == *qid)
                .then_some(r.base.score_abs_err.zip(r.cand.score_abs_err))
                .flatten()
        });
        let q_rps = eval_additive_metric(salt_base + 6, false, &|r| {
            (r.qid == *qid)
                .then_some(r.base.score_rps.zip(r.cand.score_rps))
                .flatten()
        });

        let is_q_ml = q_rec_indices
            .iter()
            .any(|&ri| records[ri].base.ml_counts.is_some());

        let (q_mf1, q_wf1, q_ml_f1, label_rows) = if q.qtype == QuestionType::Choice && !is_q_ml {
            let mf1 = eval_stat_pair(salt_base + 7, true, &|units, flips| {
                let (bp, cp) = collect_q_pairs(units, flips, &unit_records, &records, qid);
                let bm = macro_and_weighted_f1_from_pairs(&bp).0?;
                let cm = macro_and_weighted_f1_from_pairs(&cp).0?;
                Some((bm, cm))
            });
            let wf1 = eval_stat_pair(salt_base + 8, true, &|units, flips| {
                let (bp, cp) = collect_q_pairs(units, flips, &unit_records, &records, qid);
                let bw = macro_and_weighted_f1_from_pairs(&bp).1?;
                let cw = macro_and_weighted_f1_from_pairs(&cp).1?;
                Some((bw, cw))
            });
            let l_rows = compute_per_label_deltas(
                q,
                qid,
                &q_rec_indices,
                &records,
                &unit_records,
                num_units,
                config,
                salt_base + 20,
            );
            (mf1, wf1, None, l_rows)
        } else if is_q_ml {
            let mlf1 = eval_stat_pair(salt_base + 9, true, &|units, flips| {
                let (mut b_tp, mut b_fp, mut b_fn) = (0usize, 0usize, 0usize);
                let (mut c_tp, mut c_fp, mut c_fn) = (0usize, 0usize, 0usize);
                let mut any = false;
                for &u in units {
                    let swapped = flips.is_some_and(|fl| fl[u]);
                    for &ri in &unit_records[u] {
                        let rec = &records[ri];
                        if rec.qid != *qid {
                            continue;
                        }
                        if let (Some(bm), Some(cm)) = (rec.base.ml_counts, rec.cand.ml_counts) {
                            any = true;
                            let ((tp0, fp0, fn0), (tp1, fp1, fn1)) =
                                if swapped { (cm, bm) } else { (bm, cm) };
                            b_tp += tp0;
                            b_fp += fp0;
                            b_fn += fn0;
                            c_tp += tp1;
                            c_fp += fp1;
                            c_fn += fn1;
                        }
                    }
                }
                if !any {
                    return None;
                }
                let bf1 = (2 * b_tp) as f64 / (2 * b_tp + b_fp + b_fn).max(1) as f64;
                let cf1 = (2 * c_tp) as f64 / (2 * c_tp + c_fp + c_fn).max(1) as f64;
                Some((bf1, cf1))
            });
            (None, None, mlf1, Vec::new())
        } else {
            (None, None, None, Vec::new())
        };

        let q_mcnemar = compute_mcnemar_over_records(
            q_rec_indices
                .iter()
                .map(|&ri| (records[ri].base.is_correct, records[ri].cand.is_correct)),
        );

        let top_transitions = compute_flip_transitions(&q_rec_indices, &records, 8);

        q_reports_vec.push((
            qid.clone(),
            QuestionPairedReport {
                question_id: qid.clone(),
                question_type: match q.qtype {
                    QuestionType::Choice => "choice",
                    QuestionType::Score => "score",
                    QuestionType::Noul => "noul",
                }
                .to_string(),
                evaluated_samples: q_rec_indices.len(),
                accuracy: q_acc,
                nll: q_nll,
                brier: q_brier,
                ece: q_ece,
                macro_f1: q_mf1,
                weighted_f1: q_wf1,
                multi_label_f1: q_ml_f1,
                score_mae: q_mae,
                score_rps: q_rps,
                p_perm_nll_holm: None,
                mcnemar: q_mcnemar,
                labels: label_rows,
                top_transitions,
            },
        ));
    }

    // Apply Holm-Bonferroni step-down multiplicity correction across questions
    let raw_mids: Vec<f64> = q_reports_vec.iter().map(|(_, r)| r.mcnemar.p_mid).collect();
    let raw_nll_ps: Vec<f64> = q_reports_vec.iter().map(|(_, r)| r.nll.p_perm).collect();
    let adj_mids = holm_bonferroni(&raw_mids);
    let adj_nlls = holm_bonferroni(&raw_nll_ps);
    for (idx, (_, qr)) in q_reports_vec.iter_mut().enumerate() {
        qr.mcnemar.p_mid_holm = Some(adj_mids[idx]);
        qr.p_perm_nll_holm = Some(adj_nlls[idx]);
    }
    let questions_map: BTreeMap<String, QuestionPairedReport> = q_reports_vec.into_iter().collect();

    #[derive(Default)]
    struct SliceAcc {
        cnt: usize,
        bc: usize,
        cc: usize,
        bn: f64,
        cn: f64,
        fixed: usize,
        broken: usize,
    }

    // 7. Diagnostic Slice Breakdown (`--slice-by`, honoring `item.metadata` first!)
    let slices = if let Some(slice_mode) = &config.slice_by {
        let mut by_slice: BTreeMap<String, SliceAcc> = BTreeMap::new();
        for rec in &records {
            let item = &test_items[rec.item_idx];
            let s_label = resolve_slice_with_metadata(item, slice_mode);
            let entry = by_slice.entry(s_label).or_default();
            entry.cnt += 1;
            if rec.base.is_correct {
                entry.bc += 1;
            }
            if rec.cand.is_correct {
                entry.cc += 1;
            }
            entry.bn += rec.base.nll;
            entry.cn += rec.cand.nll;
            if !rec.base.is_correct && rec.cand.is_correct {
                entry.fixed += 1;
            } else if rec.base.is_correct && !rec.cand.is_correct {
                entry.broken += 1;
            }
        }
        by_slice
            .into_iter()
            .map(|(slice, a)| {
                let n_f = a.cnt.max(1) as f64;
                let ba = a.bc as f64 / n_f;
                let ca = a.cc as f64 / n_f;
                let bnll = a.bn / n_f;
                let cnll = a.cn / n_f;
                PairedSliceRow {
                    slice,
                    samples: a.cnt,
                    base_accuracy: round4(ba),
                    candidate_accuracy: round4(ca),
                    delta_accuracy: round4(ca - ba),
                    base_nll: round4(bnll),
                    candidate_nll: round4(cnll),
                    delta_nll: round4(cnll - bnll),
                    fixed: a.fixed,
                    broken: a.broken,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // 8. Top High-Confidence Regressions (`Base ✓ -> Candidate ✗`, sorted by candidate p_win desc)
    let mut regressions: Vec<RegressionCase> = records
        .iter()
        .filter(|r| r.base.is_correct && !r.cand.is_correct)
        .map(|r| {
            let item = &test_items[r.item_idx];
            RegressionCase {
                item_id: item.id.clone(),
                question_id: r.qid.clone(),
                snippet: item.preview_text(64),
                truth: r.base.truth_label.clone(),
                base_pred: r.base.pred_label.clone(),
                base_prob: round4(r.base.p_win),
                candidate_pred: r.cand.pred_label.clone(),
                candidate_prob: round4(r.cand.p_win),
            }
        })
        .collect();
    regressions.sort_by(|a, b| {
        b.candidate_prob
            .total_cmp(&a.candidate_prob)
            .then_with(|| a.item_id.cmp(&b.item_id))
    });
    regressions.truncate(config.top_regressions);

    Ok(PairedComparisonSummary {
        regime: regime.to_string(),
        total_items: test_items.len(),
        total_groups: num_units,
        total_decisions: records.len(),
        accuracy,
        nll,
        brier,
        ece,
        ace,
        aurc,
        e_aurc,
        macro_f1,
        weighted_f1,
        multi_label_f1,
        score_mae,
        score_rps,
        overall_mcnemar,
        item_exact_match_mcnemar,
        selective_risk,
        questions: questions_map,
        slices,
        top_regressions: regressions,
        base_fitted_temperatures: base_cal
            .map(|c| c.bucket_temperatures.clone())
            .unwrap_or_default(),
        candidate_fitted_temperatures: cand_cal
            .map(|c| c.bucket_temperatures.clone())
            .unwrap_or_default(),
    })
}

fn compute_mcnemar_over_records(iter: impl Iterator<Item = (bool, bool)>) -> McNemarStats {
    let (mut n11, mut n10, mut n01, mut n00) = (0usize, 0usize, 0usize, 0usize);
    for (b_ok, c_ok) in iter {
        match (b_ok, c_ok) {
            (true, true) => n11 += 1,
            (true, false) => n10 += 1,
            (false, true) => n01 += 1,
            (false, false) => n00 += 1,
        }
    }
    mcnemar_test(n11, n10, n01, n00)
}

type LabelPairVec = Vec<(String, String)>;

fn collect_q_pairs<T: HasQidObs>(
    units: &[usize],
    flips: Option<&[bool]>,
    unit_records: &[Vec<usize>],
    records: &[T],
    qid: &str,
) -> (LabelPairVec, LabelPairVec) {
    let mut bp = Vec::new();
    let mut cp = Vec::new();
    for &u in units {
        let swapped = flips.is_some_and(|fl| fl[u]);
        for &ri in &unit_records[u] {
            let rec = &records[ri];
            if rec.qid_str() != qid {
                continue;
            }
            let (bo, co) = if swapped {
                (rec.cand_obs(), rec.base_obs())
            } else {
                (rec.base_obs(), rec.cand_obs())
            };
            bp.push((bo.truth_label.clone(), bo.pred_label.clone()));
            cp.push((co.truth_label.clone(), co.pred_label.clone()));
        }
    }
    (bp, cp)
}

trait HasQidObs {
    fn qid_str(&self) -> &str;
    fn base_obs(&self) -> &DecisionObs;
    fn cand_obs(&self) -> &DecisionObs;
}

#[allow(clippy::too_many_arguments)]
fn compute_per_label_deltas<T: HasQidObs>(
    q: &Question,
    qid: &str,
    q_rec_indices: &[usize],
    records: &[T],
    unit_records: &[Vec<usize>],
    num_units: usize,
    config: &PairedExperimentConfig,
    seed_salt: u64,
) -> Vec<LabelDeltaRow> {
    let mut label_order: Vec<String> = choice_labels(q)
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect();
    for &ri in q_rec_indices {
        let rec = &records[ri];
        for lbl in [
            &rec.base_obs().truth_label,
            &rec.base_obs().pred_label,
            &rec.cand_obs().pred_label,
        ] {
            if !lbl.is_empty() && !label_order.contains(lbl) {
                label_order.push(lbl.clone());
            }
        }
    }

    // Precompute per-unit `(sup, b_pred, b_tp, c_pred, c_tp)` for each label so bootstrap is O(U)!
    let mut rows = Vec::with_capacity(label_order.len());
    for (l_idx, lbl) in label_order.iter().enumerate() {
        let mut unit_counts = vec![(0usize, 0usize, 0usize, 0usize, 0usize); num_units];
        let mut tot_sup = 0usize;
        let mut tot_bp = 0usize;
        let mut tot_btp = 0usize;
        let mut tot_cp = 0usize;
        let mut tot_ctp = 0usize;
        let mut fixed_count = 0usize;
        let mut broken_count = 0usize;

        for (u, u_recs) in unit_records.iter().enumerate() {
            for &ri in u_recs {
                let rec = &records[ri];
                if rec.qid_str() != qid {
                    continue;
                }
                let bo = rec.base_obs();
                let co = rec.cand_obs();
                let is_true = bo.truth_label == *lbl;
                let b_p = bo.pred_label == *lbl;
                let c_p = co.pred_label == *lbl;
                if is_true {
                    unit_counts[u].0 += 1;
                    tot_sup += 1;
                    if !b_p && c_p {
                        fixed_count += 1;
                    } else if b_p && !c_p {
                        broken_count += 1;
                    }
                }
                if b_p {
                    unit_counts[u].1 += 1;
                    tot_bp += 1;
                    if is_true {
                        unit_counts[u].2 += 1;
                        tot_btp += 1;
                    }
                }
                if c_p {
                    unit_counts[u].3 += 1;
                    tot_cp += 1;
                    if is_true {
                        unit_counts[u].4 += 1;
                        tot_ctp += 1;
                    }
                }
            }
        }

        if tot_sup == 0 && tot_bp == 0 && tot_cp == 0 {
            continue;
        }

        let calc_f1 = |sup: usize, pred: usize, tp: usize| -> f64 {
            if sup + pred > 0 {
                (2 * tp) as f64 / (sup + pred) as f64
            } else {
                0.0
            }
        };
        let base_f1 = calc_f1(tot_sup, tot_bp, tot_btp);
        let cand_f1 = calc_f1(tot_sup, tot_cp, tot_ctp);
        let obs_delta = cand_f1 - base_f1;

        let b_iters = config.bootstrap_samples.min(1000);
        let mut rng = DeterministicRng::new(config.seed ^ seed_salt ^ (l_idx as u64));
        let mut boot = Vec::with_capacity(b_iters);
        for _ in 0..b_iters {
            let (mut s, mut bp, mut btp, mut cp, mut ctp) = (0, 0, 0, 0, 0);
            for _ in 0..num_units {
                let u = rng.gen_range(num_units);
                let c = unit_counts[u];
                s += c.0;
                bp += c.1;
                btp += c.2;
                cp += c.3;
                ctp += c.4;
            }
            if s > 0 || bp > 0 || cp > 0 {
                boot.push(calc_f1(s, cp, ctp) - calc_f1(s, bp, btp));
            }
        }
        boot.sort_by(|a, b| a.total_cmp(b));
        let (ci_lo, ci_hi, _) = compute_bootstrap_ci(obs_delta, &boot, None, config.confidence);

        rows.push(LabelDeltaRow {
            label: lbl.clone(),
            support: tot_sup,
            base_f1: round4(base_f1),
            candidate_f1: round4(cand_f1),
            delta_f1: round4(obs_delta),
            ci_low: round4(ci_lo),
            ci_high: round4(ci_hi),
            fixed_count,
            broken_count,
        });
    }
    rows
}

fn compute_flip_transitions<T: HasQidObs>(
    q_rec_indices: &[usize],
    records: &[T],
    top_k: usize,
) -> Vec<FlipTransition> {
    let mut counts: BTreeMap<(String, String, String), (usize, FlipKind)> = BTreeMap::new();
    for &ri in q_rec_indices {
        let bo = records[ri].base_obs();
        let co = records[ri].cand_obs();
        if bo.pred_label == co.pred_label {
            continue;
        }
        let kind = match (bo.is_correct, co.is_correct) {
            (false, true) => FlipKind::Fixed,
            (true, false) => FlipKind::Broken,
            _ => FlipKind::ShiftedError,
        };
        let entry = counts
            .entry((
                bo.truth_label.clone(),
                bo.pred_label.clone(),
                co.pred_label.clone(),
            ))
            .or_insert((0, kind));
        entry.0 += 1;
    }
    let mut list: Vec<FlipTransition> = counts
        .into_iter()
        .map(
            |((truth, base_pred, candidate_pred), (count, kind))| FlipTransition {
                truth,
                base_pred,
                candidate_pred,
                count,
                kind,
            },
        )
        .collect();
    list.sort_by_key(|t| std::cmp::Reverse(t.count));
    list.truncate(top_k);
    list
}

fn resolve_slice_with_metadata(item: &DataItem, mode: &SliceBy) -> String {
    if let SliceBy::Field(key) = mode
        && let Some(val) = item.metadata.get(key)
    {
        match val {
            serde_json::Value::String(s) if !s.trim().is_empty() => return s.trim().to_string(),
            serde_json::Value::Bool(b) => return b.to_string(),
            serde_json::Value::Number(n) => return n.to_string(),
            _ => {}
        }
    }
    extract_item_slice(item, mode)
}

fn determine_verdict(
    config: &PairedExperimentConfig,
    raw: Option<&PairedComparisonSummary>,
    cal: Option<&PairedComparisonSummary>,
    primary: &PairedComparisonSummary,
) -> (ExperimentVerdict, Vec<String>) {
    let sig = config.alpha;
    let mut reasons = Vec::new();

    // Check severe regressions first on primary regime
    let acc_sig_drop = primary.accuracy.delta < -1e-6 && primary.overall_mcnemar.p_mid < sig;
    let f1_ci_drop = primary.macro_f1.as_ref().is_some_and(|m| m.ci_high < -1e-6);
    let nll_ci_worse = primary.nll.ci_low > 0.02 && primary.accuracy.delta <= 0.0;
    let ece_severe_worse = primary.ece.delta > 0.05;

    if acc_sig_drop || f1_ci_drop || nll_ci_worse || ece_severe_worse {
        if acc_sig_drop {
            reasons.push(format!(
                "Accuracy dropped significantly by {:.2}% (McNemar mid-p = {:.4} < {:.2})",
                primary.accuracy.delta * 100.0,
                primary.overall_mcnemar.p_mid,
                sig
            ));
        }
        if let Some(mf1) = &primary.macro_f1
            && mf1.ci_high < -1e-6
        {
            reasons.push(format!(
                "Macro-F1 95% CI [{:+.4}, {:+.4}] lies strictly below zero",
                mf1.ci_low, mf1.ci_high
            ));
        }
        if nll_ci_worse {
            reasons.push(format!(
                "Log-softmax NLL degraded by {:+.4} (95% CI [{:+.4}, {:+.4}])",
                primary.nll.delta, primary.nll.ci_low, primary.nll.ci_high
            ));
        }
        if ece_severe_worse {
            reasons.push(format!(
                "Top-label ECE degraded by {:+.4} ({:.4} -> {:.4})",
                primary.ece.delta, primary.ece.base, primary.ece.candidate
            ));
        }
        return (ExperimentVerdict::Regressed, reasons);
    }

    // Check `PROMOTE_WITH_CALIBRATION` when raw had fine-tuning overconfidence that `cal` cured!
    if let (Some(r), Some(c)) = (raw, cal) {
        let raw_ece_inflated = r.ece.delta > 0.02
            || r.nll.delta > 0.02
            || (r.ece.candidate > 0.08 && (r.ece.candidate - c.ece.candidate) >= 0.025)
            || (r.nll.candidate - c.nll.candidate) >= 0.10;
        let cal_cured = c.ece.delta <= 0.02 && c.nll.delta <= 0.01;
        let quality_gain = (c.accuracy.delta > 0.0 && c.overall_mcnemar.p_mid < sig)
            || c.macro_f1.as_ref().is_some_and(|m| m.ci_low > 0.0)
            || (c.nll.ci_high < 0.0 && c.accuracy.delta >= 0.0);

        if raw_ece_inflated && cal_cured && quality_gain {
            reasons.push(format!(
                "Candidate improves task quality (ΔAcc={:+.2}%, mid-p={:.4}), while raw fine-tuning overconfidence (raw Cand ECE={:.4} -> cal Cand ECE={:.4}, raw Cand NLL={:.4} -> cal Cand NLL={:.4}) is resolved by post-hoc calibration",
                c.accuracy.delta * 100.0,
                c.overall_mcnemar.p_mid,
                r.ece.candidate,
                c.ece.candidate,
                r.nll.candidate,
                c.nll.candidate
            ));
            return (ExperimentVerdict::PromoteWithCalibration, reasons);
        }
    }

    let sig_acc_gain = primary.accuracy.delta > 0.0 && primary.overall_mcnemar.p_mid < sig;
    let sig_f1_gain = primary.macro_f1.as_ref().is_some_and(|m| m.ci_low > 0.0);
    let sig_nll_gain = primary.nll.ci_high < 0.0 && primary.accuracy.delta >= -1e-6;
    let sig_mae_gain = primary.score_mae.as_ref().is_some_and(|m| m.ci_high < 0.0);
    let ece_ok = primary.ece.delta <= 0.02;

    if (sig_acc_gain || sig_f1_gain || sig_nll_gain || sig_mae_gain) && ece_ok {
        if sig_acc_gain {
            reasons.push(format!(
                "Statistically significant accuracy gain {:+.2}% (McNemar mid-p = {:.4} < {:.2})",
                primary.accuracy.delta * 100.0,
                primary.overall_mcnemar.p_mid,
                sig
            ));
        }
        if let Some(mf1) = &primary.macro_f1
            && mf1.ci_low > 0.0
        {
            reasons.push(format!(
                "Macro-F1 improved by {:+.4} with positive 95% CI [{:+.4}, {:+.4}]",
                mf1.delta, mf1.ci_low, mf1.ci_high
            ));
        }
        if sig_nll_gain {
            reasons.push(format!(
                "Proper scoring NLL improved by {:+.4} (95% CI [{:+.4}, {:+.4}])",
                primary.nll.delta, primary.nll.ci_low, primary.nll.ci_high
            ));
        }
        if sig_mae_gain && let Some(mae) = &primary.score_mae {
            reasons.push(format!(
                "Score MAE reduced by {:+.4} (95% CI [{:+.4}, {:+.4}])",
                mae.delta, mae.ci_low, mae.ci_high
            ));
        }
        return (ExperimentVerdict::Promote, reasons);
    }

    if (sig_acc_gain || sig_f1_gain) && !ece_ok {
        reasons.push(format!(
            "Accuracy/F1 improved (ΔAcc={:+.2}%), but uncalibrated top-label ECE degraded by {:+.4}; supply `--cal-data` (`lemmata calibrate`) before promotion",
            primary.accuracy.delta * 100.0,
            primary.ece.delta
        ));
        return (ExperimentVerdict::PromoteWithCalibration, reasons);
    }

    reasons.push(format!(
        "Observed differences are within statistical noise (ΔAcc={:+.2}%, McNemar mid-p={:.4}, ΔNLL={:+.4} [{:+.4}, {:+.4}])",
        primary.accuracy.delta * 100.0,
        primary.overall_mcnemar.p_mid,
        primary.nll.delta,
        primary.nll.ci_low,
        primary.nll.ci_high
    ));
    (ExperimentVerdict::Inconclusive, reasons)
}

impl HasQidObs for PairedDecisionRecord {
    fn qid_str(&self) -> &str {
        &self.qid
    }
    fn base_obs(&self) -> &DecisionObs {
        &self.base
    }
    fn cand_obs(&self) -> &DecisionObs {
        &self.cand
    }
}
