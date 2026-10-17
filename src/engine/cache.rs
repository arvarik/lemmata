//! Exact decision cache, saved on disk between runs.
//!
//! A decision is reused only for the same input text, the same questions, and the same model
//! fingerprint, so a cache hit returns the decision that the model would return.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::engine::attest::to_hex;
use crate::engine::classifier::LayaClassifier;
use crate::model::{Checkpoint, DecisionResult, Question};

/// Lookups and insertions in one run.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
}

impl CacheStats {
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        match self.total_lookups() {
            0 => 0.0,
            total => self.hits as f64 / total as f64,
        }
    }
}

/// Decisions for one model fingerprint, keyed by the exact input text and questions.
pub struct DecisionCache {
    path: Option<PathBuf>,
    entries: HashMap<String, DecisionResult>,
    dirty: bool,
    stats: CacheStats,
}

impl DecisionCache {
    /// A cache that lives only in memory, for one run.
    pub fn in_memory() -> Self {
        Self {
            path: None,
            entries: HashMap::new(),
            dirty: false,
            stats: CacheStats::default(),
        }
    }

    /// Open the saved cache for `classifier`. A missing file gives an empty cache.
    pub fn open(classifier: &LayaClassifier) -> Result<Self> {
        let path = cache_dir().join(format!("{}.json", fingerprint(classifier)?));
        let entries = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).with_context(|| {
                format!(
                    "Corrupt cache file {}. Run `lemmata cache clear`.",
                    path.display()
                )
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
            Err(e) => return Err(e).with_context(|| format!("Failed to read {}", path.display())),
        };
        Ok(Self {
            path: Some(path),
            entries,
            dirty: false,
            stats: CacheStats::default(),
        })
    }

    /// The key for `text` under the questions with hash `schema_hash`. The text is not normalized:
    /// the models are case-sensitive, so "REFUND" and "refund" can get different decisions.
    pub fn key(text: &str, schema_hash: &str) -> String {
        to_hex(&Sha256::digest(format!("{schema_hash}\n{text}").as_bytes()))
    }

    pub fn lookup(&mut self, text: &str, schema_hash: &str) -> Option<DecisionResult> {
        match self.entries.get(&Self::key(text, schema_hash)) {
            Some(result) => {
                self.stats.hits += 1;
                let mut result = result.clone();
                // A hit runs no model.
                result.latency_ms = 0.0;
                Some(result)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    pub fn insert(&mut self, text: &str, schema_hash: &str, result: &DecisionResult) {
        self.entries
            .insert(Self::key(text, schema_hash), result.clone());
        self.stats.insertions += 1;
        self.dirty = true;
    }

    /// Write new decisions to disk. A temporary file and a rename keep the old file intact if
    /// the write fails.
    pub fn save(&self) -> Result<()> {
        let Some(path) = self.path.as_ref().filter(|_| self.dirty) else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&self.entries)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// `$LEMMATA_CACHE_DIR`, else `$XDG_CACHE_HOME/lemmata`, else `~/.cache/lemmata`.
pub fn cache_dir() -> PathBuf {
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    var("LEMMATA_CACHE_DIR")
        .or_else(|| var("XDG_CACHE_HOME").map(|d| d.join("lemmata")))
        .or_else(|| var("HOME").map(|d| d.join(".cache").join("lemmata")))
        .unwrap_or_else(|| PathBuf::from(".lemmata-cache"))
}

/// A SHA-256 over everything that changes a decision: the lemmata version, the backend, the
/// device, the model settings (saved calibration included), the probe, and the weights files.
pub fn fingerprint(classifier: &LayaClassifier) -> Result<String> {
    use crate::model::native;
    let cfg = classifier.config();
    let mut h = Sha256::new();
    h.update(env!("CARGO_PKG_VERSION"));
    h.update(format!(
        "{:?}|gpu={}",
        native::backend(),
        native::runs_on_gpu()
    ));
    h.update(serde_json::to_vec(cfg)?);
    if let Some(probe) = &cfg.probe {
        h.update(serde_json::to_vec(probe.as_ref())?);
    }
    let checkpoints = match cfg.checkpoint {
        Checkpoint::AutoRoute => Checkpoint::laya_checkpoints().to_vec(),
        c => vec![c],
    };
    for ckpt in checkpoints {
        let file = match &cfg.local_weights_path {
            Some(path) if ckpt == cfg.checkpoint => path.clone(),
            _ => ckpt.local_dir().join("model.safetensors"),
        };
        // ponytail: size and modified time stand in for a hash of the weights, which takes seconds.
        if let Ok(meta) = std::fs::metadata(&file) {
            h.update(format!(
                "{}|{}|{:?}",
                file.display(),
                meta.len(),
                meta.modified().ok()
            ));
        }
    }
    Ok(to_hex(&h.finalize()))
}

/// One saved cache file: its model fingerprint, decision count, and size in bytes.
#[derive(Debug, Clone, Serialize)]
pub struct CacheFileSummary {
    pub fingerprint: String,
    pub entries: usize,
    pub bytes: u64,
}

/// The cache files in `dir`, largest first.
pub fn cache_summary(dir: &Path) -> Result<Vec<CacheFileSummary>> {
    let mut files = Vec::new();
    for path in cache_files(dir)? {
        let bytes = std::fs::metadata(&path)?.len();
        let entries: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&path)?).unwrap_or_default();
        let fingerprint = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        files.push(CacheFileSummary {
            fingerprint,
            entries: entries.len(),
            bytes,
        });
    }
    files.sort_by_key(|f| std::cmp::Reverse(f.bytes));
    Ok(files)
}

/// Delete the cache files in `dir` and return how many were deleted. Other files stay.
pub fn clear_cache(dir: &Path) -> Result<usize> {
    let files = cache_files(dir)?;
    for path in &files {
        std::fs::remove_file(path)?;
    }
    Ok(files.len())
}

/// Files named `<64 hex digits>.json`: the only files that lemmata writes in the cache folder.
fn cache_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut files = Vec::new();
    for entry in read {
        let path = entry?.path();
        let is_cache_file = path.extension().is_some_and(|e| e == "json")
            && path.file_stem().is_some_and(|s| {
                s.len() == 64 && s.to_string_lossy().chars().all(|c| c.is_ascii_hexdigit())
            });
        if is_cache_file {
            files.push(path);
        }
    }
    Ok(files)
}

/// A SHA-256 over the full question schema, stable across runs and Rust versions.
pub fn compute_schema_hash(questions: &BTreeMap<String, Question>) -> String {
    to_hex(&Sha256::digest(
        serde_json::to_vec(questions).unwrap_or_default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DecisionAnswer, QuestionType};

    fn decision(choice: &str) -> DecisionResult {
        let mut answers = BTreeMap::new();
        answers.insert(
            "q".to_string(),
            DecisionAnswer {
                qtype: QuestionType::Choice,
                choice: Some(choice.to_string()),
                selected_labels: None,
                score: None,
                noul: None,
                confidence: 0.9,
                probabilities: BTreeMap::new(),
                act_probability: None,
                logits: None,
            },
        );
        DecisionResult {
            model: "mock:english".to_string(),
            answers,
            input_tokens: 3,
            latency_ms: 12.0,
        }
    }

    #[test]
    fn hits_only_the_exact_text_and_schema() {
        let mut cache = DecisionCache::in_memory();
        cache.insert("refund me", "s1", &decision("billing"));
        let hit = cache
            .lookup("refund me", "s1")
            .expect("same text and schema hit");
        assert_eq!(hit.answers["q"].choice.as_deref(), Some("billing"));
        assert_eq!(hit.latency_ms, 0.0);
        assert!(cache.lookup("Refund me", "s1").is_none(), "case matters");
        assert!(cache.lookup("refund me ", "s1").is_none(), "spaces matter");
        assert!(
            cache.lookup("refund me", "s2").is_none(),
            "schemas are separate"
        );
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 3);
    }

    #[test]
    fn saves_reloads_and_clears() {
        let dir = std::env::temp_dir().join(format!("lemmata_cache_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join(format!("{}.json", "a".repeat(64)));
        let mut cache = DecisionCache {
            path: Some(path.clone()),
            ..DecisionCache::in_memory()
        };
        cache.insert("refund me", "s1", &decision("billing"));
        cache.save().unwrap();

        let reloaded: HashMap<String, DecisionResult> =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(reloaded.len(), 1);
        std::fs::write(dir.join("notes.txt"), "keep me").unwrap();
        let summary = cache_summary(&dir).unwrap();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].entries, 1);
        assert_eq!(clear_cache(&dir).unwrap(), 1);
        assert!(
            dir.join("notes.txt").exists(),
            "clear only deletes cache files"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
