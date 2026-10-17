use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::datasource::DataItem;
use crate::engine::calibrate::{CalibratedAgentConfig, CalibrationReport, CalibrationTuner};
use crate::engine::classifier::ClassifierConfig;
use crate::model::{Checkpoint, Question};

/// Canonical filename for per-checkpoint persisted calibration profiles.
pub const MODEL_CALIBRATION_FILENAME: &str = "lemmata_calibration.json";

/// Return the default local calibration file path `<ckpt.local_dir()>/lemmata_calibration.json`.
pub fn calibration_path_for(ckpt: Checkpoint) -> PathBuf {
    ckpt.local_dir().join(MODEL_CALIBRATION_FILENAME)
}

/// Persist a calibrated configuration to the checkpoint's default local directory.
pub fn save_checkpoint_calibration(
    ckpt: Checkpoint,
    cfg: &CalibratedAgentConfig,
) -> Result<PathBuf> {
    save_checkpoint_calibration_at(&ckpt.local_dir(), cfg)
}

/// Persist a calibrated configuration into `<dir>/lemmata_calibration.json`.
pub fn save_checkpoint_calibration_at(dir: &Path, cfg: &CalibratedAgentConfig) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create calibration directory: {}", dir.display()))?;
    let path = dir.join(MODEL_CALIBRATION_FILENAME);
    let payload = serde_json::to_string_pretty(cfg)
        .context("Failed to serialize CalibratedAgentConfig to JSON")?;
    std::fs::write(&path, payload)
        .with_context(|| format!("Failed to write calibration file: {}", path.display()))?;
    Ok(path)
}

/// Load a persisted calibration configuration from the checkpoint's default local directory, if present.
pub fn load_checkpoint_calibration(ckpt: Checkpoint) -> Option<CalibratedAgentConfig> {
    load_checkpoint_calibration_from(&ckpt.local_dir())
}

/// Load a persisted calibration configuration from `<dir>/lemmata_calibration.json` (or directly if `dir` is a JSON file).
pub fn load_checkpoint_calibration_from(dir: &Path) -> Option<CalibratedAgentConfig> {
    let path = if dir.is_file() {
        dir.to_path_buf()
    } else {
        dir.join(MODEL_CALIBRATION_FILENAME)
    };
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Apply a persisted `CalibratedAgentConfig` onto a `ClassifierConfig`.
pub fn apply_persisted_calibration(config: &mut ClassifierConfig, cal: &CalibratedAgentConfig) {
    config.temperature = cal.temperature;
    config.temperature_by_options.extend(
        cal.temperature_by_options
            .iter()
            .map(|(k, &v)| (k.clone(), v)),
    );
    config.qats_profile = cal.qats.clone();
    config
        .cost_weights
        .extend(cal.cost_weights.iter().map(|(k, &v)| (k.clone(), v)));
}

/// Fit post-hoc temperature scaling + optional Q-ATS for each checkpoint and persist `lemmata_calibration.json`.
pub fn calibrate_and_persist_checkpoints(
    checkpoints: &[Checkpoint],
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
    enable_qats: bool,
    min_bucket_samples: usize,
    base_dir: Option<&Path>,
) -> Result<BTreeMap<String, CalibrationReport>> {
    let mut reports = BTreeMap::new();
    for &ckpt in checkpoints {
        let (report, cfg) = CalibrationTuner::new(ckpt)
            .with_qats(enable_qats, 0.30)
            .with_min_bucket_samples(min_bucket_samples)
            .fit(items, questions, "golden_benchmark")?;
        let target_dir = match base_dir {
            Some(base) => base.join(ckpt.short_name()),
            None => ckpt.local_dir(),
        };
        save_checkpoint_calibration_at(&target_dir, &cfg)?;
        reports.insert(ckpt.repo_id().to_string(), report);
    }
    Ok(reports)
}
