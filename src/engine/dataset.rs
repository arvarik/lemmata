//! Dataset lifecycle management: canonical ingestion (`import`), deterministic leakage-safe
//! multi-split partitioning (`split`), and 5-stage ground-truth & tokenizer preflight (`verify`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail};
use comfy_table::{Cell, Color, ContentArrangement, Table};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::datasource::DataItem;
use crate::engine::attest::to_hex;
use crate::engine::cache::compute_schema_hash;
use crate::engine::calibrate::DEFAULT_MIN_BUCKET_SAMPLES;
use crate::engine::cascade::RISK_GRID;
use crate::engine::metrics::{fnv1a64, format_utc_iso8601, resolve_ground_truth_for_question};
use crate::engine::probe::choice_labels;
use crate::model::{
    Checkpoint, ChunkStrategy, Question, QuestionType, clean_email_body, estimate_tokens,
};

/// Schema version recorded in `dataset_manifest.json`.
pub const DATASET_MANIFEST_VERSION: &str = "lemmata-dataset-v1";

/// 8-byte header magic for dual exact+normalized `trained_inputs.sha256.bin` filters.
pub const TRAINED_FILTER_MAGIC: &[u8; 8] = b"LMFT1\0\0\0";

/// Upstream Laya per-option token cap (`src/model/native.rs`).
const MAX_LAYA_OPTION_TOKENS: usize = 48;

/// MinHash signature length (number of deterministic 64-bit permutations).
const MINHASH_NUM_PERM: usize = 128;
/// Number of LSH bands over the 128 permutations (`16 * 8 = 128`).
const LSH_BANDS: usize = 16;
/// Rows per LSH band.
const LSH_ROWS_PER_BAND: usize = 8;

// ============================================================================
// Deterministic PRNG (`SplitMix64` + `Xoshiro256**`)
// ============================================================================

/// Fast 64-bit SplitMix64 generator used to expand a single 64-bit seed into
/// `Xoshiro256**` state or universal hash coefficients.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Platform-independent `Xoshiro256**` deterministic pseudo-random generator.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    s: [u64; 4],
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Self {
            s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in `[0.0, 1.0)` with 53 bits of precision.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Unbiased uniform integer in `0..upper` via Lemire's rejection method.
    pub fn gen_range(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be > 0");
        let bound = upper as u64;
        if bound.is_power_of_two() {
            return (self.next_u64() & (bound - 1)) as usize;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            let prod = (r as u128) * (bound as u128);
            let low = prod as u64;
            if low >= threshold {
                return (prod >> 64) as usize;
            }
        }
    }

    /// Deterministic Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}

// ============================================================================
// Shared Hashing, State Normalization & Binary Overlap Filter
// ============================================================================

/// Compute lowercase hex SHA-256 digest of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

/// Stream a file from disk and return its lowercase hex SHA-256 digest.
pub fn file_sha256_hex(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex(&hasher.finalize()))
}

/// Return the first 8 bytes of `SHA-256(bytes)` decoded as a little-endian `u64`.
///
/// Equivalent in Python:
/// `int.from_bytes(hashlib.sha256(b).digest()[:8], "little")`
pub fn sha256_u64_prefix(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(arr)
}

/// Canonical whitespace-and-case state normalization shared across `dataset split`,
/// `finetune prepare`, `train.py`, `finetune verify`, and `paired_eval`.
///
/// Strips UTF-8 BOM (`\u{FEFF}`) and NULs, lowercases Unicode characters, and collapses
/// runs of Unicode whitespace into single ASCII spaces.
///
/// Exact 1-line Python equivalent:
/// `" ".join(text.replace("\ufeff", "").replace("\x00", "").lower().split())`
pub fn normalize_state_for_leakage(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '\u{FEFF}' || ch == '\0' {
            continue;
        }
        for lower in ch.to_lowercase() {
            cleaned.push(lower);
        }
    }
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Stronger template-masking normalization for Stage 2 split grouping: masks URLs, emails,
/// hex/UUID identifiers, and digit runs (`#INV-9921` -> `<num>`) so boilerplate retry
/// tickets and system logs differing only in numbers/timestamps group into the same split.
pub fn normalize_template_for_grouping(text: &str) -> String {
    let base = normalize_state_for_leakage(text);
    let mut masked_tokens = Vec::new();
    for raw_tok in base.split_whitespace() {
        let tok = raw_tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != ':');
        if tok.is_empty() {
            continue;
        }
        if tok.starts_with("http://") || tok.starts_with("https://") || tok.starts_with("www.") {
            masked_tokens.push("<url>".to_string());
            continue;
        }
        if tok.contains('@')
            && let Some((lhs, rhs)) = tok.split_once('@')
            && !lhs.is_empty()
            && rhs.contains('.')
        {
            masked_tokens.push("<email>".to_string());
            continue;
        }
        let has_digit = tok.chars().any(|c| c.is_ascii_digit());
        let all_hex_or_sep = tok
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-' || c == '_');
        if has_digit && all_hex_or_sep && tok.len() >= 8 {
            masked_tokens.push("<id>".to_string());
            continue;
        }
        // Replace contiguous numeric/currency/date/version runs (`$1,250.00`, `inv-9921`) -> `inv <num>`
        let chars: Vec<char> = tok.chars().collect();
        let mut sub = String::new();
        let mut in_digits = false;
        for (idx, &ch) in chars.iter().enumerate() {
            let next_is_digit = chars.get(idx + 1).is_some_and(|c| c.is_ascii_digit());
            if ch.is_ascii_digit() {
                if !in_digits {
                    if !sub.ends_with(' ') && !sub.is_empty() {
                        sub.push(' ');
                    }
                    sub.push_str("<num> ");
                    in_digits = true;
                }
            } else if in_digits && matches!(ch, ',' | '.' | ':' | '/' | '-') && next_is_digit {
                // Keep thousand-grouped / decimal / date / time spans inside the active `<num>` token
            } else {
                in_digits = false;
                if ch.is_alphanumeric() {
                    sub.push(ch);
                } else {
                    sub.push(' ');
                }
            }
        }
        for piece in sub.split_whitespace() {
            if piece == "<num>" && masked_tokens.last().map(String::as_str) == Some("<num>") {
                continue;
            }
            masked_tokens.push(piece.to_string());
        }
    }
    masked_tokens.join(" ")
}

/// Compute a canonical per-item SHA-256 over `(id, state_text, ground_truth)`.
pub fn compute_item_digest(item: &DataItem) -> String {
    let state_txt = item.state_text();
    let state_sha = sha256_hex(state_txt.as_bytes());
    let gt_bytes = serde_json::to_vec(&item.ground_truth).unwrap_or_default();
    let gt_sha = sha256_hex(&gt_bytes);
    let mut h = Sha256::new();
    h.update(item.id.as_bytes());
    h.update([0x1f]);
    h.update(state_sha.as_bytes());
    h.update([0x1f]);
    h.update(gt_sha.as_bytes());
    to_hex(&h.finalize())
}

/// Order-invariant SHA-256 digest of a slice of `DataItem`s (invariant to line shuffling).
pub fn compute_item_set_digest(items: &[DataItem]) -> String {
    let mut digests: Vec<String> = items.iter().map(compute_item_digest).collect();
    digests.sort_unstable();
    sha256_hex(digests.join("\n").as_bytes())
}

/// Order-invariant Merkle-style SHA-256 over exact model-visible `state_text()` values.
pub fn compute_state_merkle_sha256(items: &[DataItem]) -> String {
    let mut hashes: Vec<String> = items
        .iter()
        .map(|it| sha256_hex(it.state_text().as_bytes()))
        .collect();
    hashes.sort_unstable();
    sha256_hex(hashes.join("\n").as_bytes())
}

/// Overlap result from probing test items against a `TrainedInputsFilter`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LeakageOverlapReport {
    pub exact_overlap_ids: Vec<String>,
    pub normalized_overlap_ids: Vec<String>,
}

impl LeakageOverlapReport {
    pub fn has_overlap(&self) -> bool {
        !self.exact_overlap_ids.is_empty() || !self.normalized_overlap_ids.is_empty()
    }

    pub fn total_overlapping_items(&self) -> usize {
        let mut unique: BTreeSet<&str> = BTreeSet::new();
        for id in &self.exact_overlap_ids {
            unique.insert(id.as_str());
        }
        for id in &self.normalized_overlap_ids {
            unique.insert(id.as_str());
        }
        unique.len()
    }
}

/// Compact sorted `u64` SHA-256 prefix set for `<1 ms` row-level train/val/cal vs test
/// leakage detection (`trained_inputs.sha256.bin`).
///
/// Supports both:
/// 1. Dual headered layout: `b"LMFT1\0\0\0" || exact_len_u64_le || norm_len_u64_le || exact_u64s || norm_u64s`
/// 2. Flat raw layout: sorted `[u64_le; N]` containing exact and/or normalized prefixes.
#[derive(Debug, Clone, Default)]
pub struct TrainedInputsFilter {
    pub exact_prefixes: Vec<u64>,
    pub normalized_prefixes: Vec<u64>,
}

impl TrainedInputsFilter {
    pub fn from_items(items: &[DataItem]) -> Self {
        let mut exact = Vec::with_capacity(items.len());
        let mut norm = Vec::with_capacity(items.len());
        for item in items {
            let raw = item.state_text();
            exact.push(sha256_u64_prefix(raw.as_bytes()));
            let n = normalize_state_for_leakage(&raw);
            if !n.is_empty() {
                norm.push(sha256_u64_prefix(n.as_bytes()));
            }
        }
        exact.sort_unstable();
        exact.dedup();
        norm.sort_unstable();
        norm.dedup();
        Self {
            exact_prefixes: exact,
            normalized_prefixes: norm,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let total_bytes = 24 + 8 * (self.exact_prefixes.len() + self.normalized_prefixes.len());
        let mut out = Vec::with_capacity(total_bytes);
        out.extend_from_slice(TRAINED_FILTER_MAGIC);
        out.extend_from_slice(&(self.exact_prefixes.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.normalized_prefixes.len() as u64).to_le_bytes());
        for &v in &self.exact_prefixes {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.normalized_prefixes {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() >= 24 && (&bytes[..8] == TRAINED_FILTER_MAGIC || &bytes[..8] == b"LMFTFLT1")
        {
            let exact_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
            let norm_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            let expected = 24usize
                .checked_add(
                    exact_len
                        .checked_add(norm_len)
                        .and_then(|n| n.checked_mul(8))
                        .ok_or_else(|| anyhow!("trained_inputs.sha256.bin length overflow"))?,
                )
                .ok_or_else(|| anyhow!("trained_inputs.sha256.bin header overflow"))?;
            if bytes.len() != expected {
                bail!(
                    "Malformed trained_inputs.sha256.bin: expected {} bytes from header, got {}",
                    expected,
                    bytes.len()
                );
            }
            let mut exact = Vec::with_capacity(exact_len);
            let mut offset = 24;
            for _ in 0..exact_len {
                exact.push(u64::from_le_bytes(
                    bytes[offset..offset + 8].try_into().unwrap(),
                ));
                offset += 8;
            }
            let mut norm = Vec::with_capacity(norm_len);
            for _ in 0..norm_len {
                norm.push(u64::from_le_bytes(
                    bytes[offset..offset + 8].try_into().unwrap(),
                ));
                offset += 8;
            }
            exact.sort_unstable();
            norm.sort_unstable();
            return Ok(Self {
                exact_prefixes: exact,
                normalized_prefixes: norm,
            });
        }

        if !bytes.len().is_multiple_of(8) {
            bail!(
                "Malformed trained_inputs.sha256.bin: byte length {} is not a multiple of 8",
                bytes.len()
            );
        }
        let mut hashes = Vec::with_capacity(bytes.len() / 8);
        for &chunk in bytes.as_chunks::<8>().0 {
            hashes.push(u64::from_le_bytes(chunk));
        }
        hashes.sort_unstable();
        hashes.dedup();
        Ok(Self {
            exact_prefixes: hashes.clone(),
            normalized_prefixes: hashes,
        })
    }

    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_bytes())
            .with_context(|| format!("Failed to write {}", path.display()))
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read filter {}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn check_overlap(&self, test_items: &[DataItem]) -> LeakageOverlapReport {
        let mut report = LeakageOverlapReport::default();
        for item in test_items {
            let raw = item.state_text();
            let exact_h = sha256_u64_prefix(raw.as_bytes());
            if self.exact_prefixes.binary_search(&exact_h).is_ok() {
                report.exact_overlap_ids.push(item.id.clone());
                continue;
            }
            let norm = normalize_state_for_leakage(&raw);
            if !norm.is_empty() {
                let norm_h = sha256_u64_prefix(norm.as_bytes());
                if self.normalized_prefixes.binary_search(&norm_h).is_ok() {
                    report.normalized_overlap_ids.push(item.id.clone());
                }
            }
        }
        report
    }
}

// ============================================================================
// Canonical Dataset File I/O
// ============================================================================

/// Write canonical `DataItem` records as deterministic newline-delimited JSON.
pub fn write_dataset_jsonl(path: &Path, items: &[DataItem]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)
        .with_context(|| format!("Failed to create dataset file {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for item in items {
        serde_json::to_writer(&mut writer, item)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Read canonical or raw `DataItem` records from `.jsonl`, `.json`, `.csv`, `.tsv`, or `.parquet`.
pub fn read_dataset_items(path: &Path) -> Result<Vec<DataItem>> {
    let cfg = ImportConfig::default();
    let (items, _) = DatasetImporter::import(&cfg, &ImportSource::File(path.to_path_buf()), None)?;
    Ok(items)
}

// ============================================================================
// Section 1: Dataset Import & Canonicalization (`DatasetImporter`)
// ============================================================================

/// Input data origin for `lemmata dataset import`.
#[derive(Debug, Clone)]
pub enum ImportSource {
    File(PathBuf),
    Directory {
        path: PathBuf,
        recursive: bool,
        extensions: Option<Vec<String>>,
    },
    HuggingFace {
        repo_id: String,
        subset: Option<String>,
        split: Option<String>,
        revision: Option<String>,
        cache_dir: Option<PathBuf>,
        dry_run: bool,
    },
}

/// Configuration for `lemmata dataset import`.
#[derive(Debug, Clone, Default)]
pub struct ImportConfig {
    pub schema: Option<BTreeMap<String, Question>>,
    pub id_col: Option<String>,
    pub text_col: Option<String>,
    pub state_cols: Vec<String>,
    pub meta_cols: Vec<String>,
    /// Explicit mapping `question_id -> source_column_name`.
    pub label_map: BTreeMap<String, String>,
    /// Explicit value remapping `question_id -> (lowercase_raw_value -> canonical_value)`.
    pub value_map: BTreeMap<String, BTreeMap<String, String>>,
    /// Map integer `ClassLabel` indices (`0..K-1`) to schema choice option keys in schema order.
    pub int_labels_from_schema: bool,
    /// Additive integer/float offset applied to `score` ground-truth targets (e.g. `-1` for `1..5 -> 0..4`).
    pub score_offset: i64,
    /// Delimiter splitting string multi-label targets into JSON arrays (e.g. `';'` or `'|'`).
    pub multi_label_sep: Option<char>,
    /// Normalize line endings, strip BOM/NULs, and compose common decomposed accents.
    pub normalize_nfc: bool,
    /// Strip quoted email reply headers, `>` quote lines, signatures, and legal disclaimers.
    pub clean_email: bool,
    /// Drop records whose model-visible `state` is blank.
    pub drop_empty: bool,
    /// Fail immediately on malformed rows, duplicate IDs, or out-of-domain schema targets.
    pub strict: bool,
}

/// Summary report emitted by `DatasetImporter::import`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportReport {
    pub source_description: String,
    pub output_path: Option<String>,
    pub output_sha256: Option<String>,
    pub item_set_digest: String,
    pub total_rows_read: usize,
    pub imported_items: usize,
    pub labeled_items: usize,
    pub dropped_empty: usize,
    pub skipped_malformed: usize,
    pub deduplicated_ids: usize,
    pub dry_run_command: Option<String>,
    pub warnings: Vec<String>,
}

/// Parse repeatable `KEY=VALUE` CLI flags (e.g., `--label-map department=category`).
pub fn parse_key_val_pairs(specs: &[String]) -> Result<BTreeMap<String, String>> {
    let mut map = BTreeMap::new();
    for raw in specs {
        let Some((k, v)) = raw.split_once('=') else {
            bail!("Invalid mapping '{raw}'. Expected format: <question_id>=<column_name>");
        };
        let k = k.trim();
        let v = v.trim();
        if k.is_empty() || v.is_empty() {
            bail!("Invalid mapping '{raw}': key and column name must be non-empty");
        }
        map.insert(k.to_string(), v.to_string());
    }
    Ok(map)
}

/// Parse repeatable `QID:RAW=CANONICAL` CLI flags (e.g., `--value-map department:fin=billing`).
pub fn parse_value_map_specs(
    specs: &[String],
) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    let mut out: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for raw in specs {
        let Some((lhs, canon)) = raw.split_once('=') else {
            bail!("Invalid --value-map '{raw}'. Expected format: <question_id>:<raw>=<canonical>");
        };
        let Some((qid, raw_val)) = lhs.split_once(':') else {
            bail!("Invalid --value-map '{raw}'. Expected '<question_id>:<raw>=<canonical>'");
        };
        let qid = qid.trim();
        let raw_val = raw_val.trim().to_lowercase();
        let canon = canon.trim();
        if qid.is_empty() || raw_val.is_empty() || canon.is_empty() {
            bail!("Invalid --value-map '{raw}': components must be non-empty");
        }
        out.entry(qid.to_string())
            .or_default()
            .insert(raw_val, canon.to_string());
    }
    Ok(out)
}

pub struct DatasetImporter;

impl DatasetImporter {
    /// Import and canonicalize records from `source` according to `config`, optionally writing
    /// canonical JSONL to `output_path`.
    pub fn import(
        config: &ImportConfig,
        source: &ImportSource,
        output_path: Option<&Path>,
    ) -> Result<(Vec<DataItem>, ImportReport)> {
        let mut report = ImportReport::default();
        let mut hf_class_labels: BTreeMap<String, Vec<String>> = BTreeMap::new();

        let raw_files: Vec<(PathBuf, Option<String>)> = match source {
            ImportSource::File(p) => {
                if !p.exists() {
                    bail!("Input file does not exist: {}", p.display());
                }
                report.source_description = p.display().to_string();
                vec![(p.clone(), None)]
            }
            ImportSource::Directory {
                path,
                recursive,
                extensions,
            } => {
                if !path.exists() || !path.is_dir() {
                    bail!("Input directory does not exist: {}", path.display());
                }
                report.source_description = path.display().to_string();
                let allowed = extensions.clone().unwrap_or_else(|| {
                    vec![
                        "jsonl".into(),
                        "json".into(),
                        "csv".into(),
                        "tsv".into(),
                        "parquet".into(),
                    ]
                });
                let files = collect_sorted_files(path, *recursive, &allowed);
                files
                    .into_iter()
                    .map(|f| {
                        let prefix = f
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string());
                        (f, prefix)
                    })
                    .collect()
            }
            ImportSource::HuggingFace {
                repo_id,
                subset,
                split,
                revision,
                cache_dir,
                dry_run,
            } => {
                report.source_description = format!("hf://datasets/{repo_id}");
                let (files, class_map, dry_cmd) = download_hf_dataset(
                    repo_id,
                    subset.as_deref(),
                    split.as_deref(),
                    revision.as_deref(),
                    cache_dir.as_deref(),
                    *dry_run,
                    &mut report.warnings,
                )?;
                if let Some(cmd_str) = dry_cmd {
                    report.dry_run_command = Some(cmd_str);
                    return Ok((Vec::new(), report));
                }
                hf_class_labels = class_map;
                files
            }
        };

        let mut items = Vec::new();
        let mut seen_ids: HashMap<String, usize> = HashMap::new();
        let multi_file = raw_files.len() > 1;

        for (file_path, split_prefix) in &raw_files {
            let (raw_records, file_class_labels) =
                read_raw_records_from_file(file_path, config.strict, &mut report)?;
            for (k, v) in file_class_labels {
                hf_class_labels.entry(k).or_insert(v);
            }

            let file_stem = split_prefix
                .clone()
                .or_else(|| {
                    file_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| "item".to_string());
            let uri_str = if report.source_description.starts_with("hf://") {
                format!("{}/{}", report.source_description, file_stem)
            } else {
                file_path.display().to_string()
            };

            for (row_idx, raw_obj) in raw_records.into_iter().enumerate() {
                report.total_rows_read += 1;
                let default_id = format!("{file_stem}:{row_idx}");
                match canonicalize_record(
                    raw_obj,
                    &default_id,
                    multi_file.then_some(file_stem.as_str()),
                    &uri_str,
                    config,
                    &hf_class_labels,
                    &mut report.warnings,
                ) {
                    Ok(Some(mut item)) => {
                        if let Some(count) = seen_ids.get_mut(&item.id) {
                            if config.strict {
                                bail!(
                                    "Duplicate item id '{}' encountered in {}",
                                    item.id,
                                    file_path.display()
                                );
                            }
                            *count += 1;
                            report.deduplicated_ids += 1;
                            item.id = format!("{}#{}", item.id, *count);
                        } else {
                            seen_ids.insert(item.id.clone(), 0);
                        }

                        if item.ground_truth.as_ref().is_some_and(|gt| !gt.is_empty()) {
                            report.labeled_items += 1;
                        }
                        items.push(item);
                    }
                    Ok(None) => {
                        report.dropped_empty += 1;
                    }
                    Err(e) => {
                        if config.strict {
                            return Err(e.context(format!(
                                "Strict import failed at {} row {}",
                                file_path.display(),
                                row_idx + 1
                            )));
                        }
                        report.skipped_malformed += 1;
                        if report.warnings.len() < 20 {
                            report.warnings.push(format!(
                                "{}:{}: skipped row ({e})",
                                file_path.display(),
                                row_idx + 1
                            ));
                        }
                    }
                }
            }
        }

        report.imported_items = items.len();
        report.item_set_digest = compute_item_set_digest(&items);

        if let Some(out) = output_path {
            write_dataset_jsonl(out, &items)?;
            report.output_path = Some(out.display().to_string());
            report.output_sha256 = Some(file_sha256_hex(out)?);
        }

        Ok((items, report))
    }

    pub fn format_table(report: &ImportReport) -> String {
        let mut table = Table::new();
        table
            .load_style(comfy_table::presets::UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Import Metric").fg(Color::Cyan),
                Cell::new("Value").fg(Color::Green),
            ]);
        table.add_row(vec!["Source", &report.source_description]);
        if let Some(cmd) = &report.dry_run_command {
            table.add_row(vec!["Dry-Run Command", cmd]);
            return format!("\n📥 Dataset Import (Dry Run)\n{table}\n");
        }
        table.add_row(vec!["Rows Read", &report.total_rows_read.to_string()]);
        table.add_row(vec![
            "Canonical Items Imported",
            &report.imported_items.to_string(),
        ]);
        table.add_row(vec!["Labeled Items", &report.labeled_items.to_string()]);
        if report.dropped_empty > 0 {
            table.add_row(vec![
                "Dropped Empty States",
                &report.dropped_empty.to_string(),
            ]);
        }
        if report.skipped_malformed > 0 {
            table.add_row(vec![
                "Skipped Malformed Rows",
                &report.skipped_malformed.to_string(),
            ]);
        }
        if report.deduplicated_ids > 0 {
            table.add_row(vec![
                "Re-keyed Duplicate IDs",
                &report.deduplicated_ids.to_string(),
            ]);
        }
        table.add_row(vec!["Item-Set Digest", &report.item_set_digest]);
        if let Some(out) = &report.output_path {
            table.add_row(vec!["Output File", out]);
        }
        let mut out = format!("\n📥 Lemmata Dataset Import Summary\n{table}\n");
        if !report.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in report.warnings.iter().take(10) {
                out.push_str(&format!("  ⚠️  {w}\n"));
            }
        }
        out
    }
}

fn collect_sorted_files(dir: &Path, recursive: bool, allowed_exts: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s.starts_with('.') || name_s == "target" || name_s == "dataset_manifest.json" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                out.extend(collect_sorted_files(&path, true, allowed_exts));
            }
        } else if path.is_file()
            && let Some(ext) = path.extension().and_then(|e| e.to_str())
            && allowed_exts.iter().any(|a| a.eq_ignore_ascii_case(ext))
        {
            out.push(path);
        }
    }
    out
}

fn detect_hf_cli() -> Result<String> {
    for bin in ["hf", "huggingface-cli"] {
        if Command::new(bin)
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Ok(bin.to_string());
        }
    }
    bail!(
        "Neither `hf` nor `huggingface-cli` was found in PATH. Install `huggingface_hub` CLI to use `--hf <repo>`."
    )
}

type ClassLabelMap = BTreeMap<String, Vec<String>>;
type HfDownloadResult = (
    Vec<(PathBuf, Option<String>)>,
    ClassLabelMap,
    Option<String>,
);

fn download_hf_dataset(
    repo_id: &str,
    subset: Option<&str>,
    split: Option<&str>,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
    dry_run: bool,
    warnings: &mut Vec<String>,
) -> Result<HfDownloadResult> {
    let sanitized = repo_id.replace(['/', '\\', ':'], "__");
    let target_dir = match cache_dir {
        Some(d) => d.join(&sanitized),
        None => crate::engine::cache::cache_dir()
            .join("hf_datasets")
            .join(&sanitized),
    };
    let effective_rev = revision.unwrap_or("refs/convert/parquet");
    let include_glob = match (subset, split) {
        (Some(sub), Some(sp)) if !sp.eq_ignore_ascii_case("all") => {
            Some(format!("{sub}/{sp}/*.parquet"))
        }
        (Some(sub), _) => Some(format!("{sub}/**/*.parquet")),
        (None, Some(sp)) if !sp.eq_ignore_ascii_case("all") => Some(format!("*{sp}*")),
        _ => None,
    };

    if dry_run {
        let bin = detect_hf_cli().unwrap_or_else(|_| "hf".to_string());
        let mut cmd = format!(
            "{bin} download {repo_id} --repo-type dataset --revision {effective_rev} --local-dir {}",
            target_dir.display()
        );
        if let Some(inc) = &include_glob {
            cmd.push_str(&format!(" --include \"{inc}\""));
        }
        return Ok((Vec::new(), BTreeMap::new(), Some(cmd)));
    }

    let bin = detect_hf_cli()?;
    std::fs::create_dir_all(&target_dir)?;

    let run_dl = |rev: &str, inc: Option<&str>| -> bool {
        let mut cmd = Command::new(&bin);
        cmd.args([
            "download",
            repo_id,
            "--repo-type",
            "dataset",
            "--revision",
            rev,
            "--local-dir",
            &target_dir.display().to_string(),
        ]);
        if let Some(pattern) = inc {
            cmd.args(["--include", pattern]);
        }
        cmd.status().is_ok_and(|s| s.success())
    };

    let ok = if let Some(explicit_rev) = revision {
        run_dl(explicit_rev, include_glob.as_deref())
    } else if run_dl("refs/convert/parquet", include_glob.as_deref()) {
        true
    } else {
        warnings.push(format!(
            "Parquet branch `refs/convert/parquet` unavailable for {repo_id}; falling back to `main`"
        ));
        let fallback_inc = split
            .filter(|s| !s.eq_ignore_ascii_case("all"))
            .map(|s| format!("*{s}*"));
        run_dl("main", fallback_inc.as_deref())
    };

    if !ok {
        bail!("Failed to download Hugging Face dataset '{repo_id}' via `{bin} download`");
    }

    let class_labels = extract_hf_dataset_infos_labels(&target_dir);
    let allowed = vec![
        "parquet".into(),
        "jsonl".into(),
        "json".into(),
        "csv".into(),
        "tsv".into(),
    ];
    let all_files = collect_sorted_files(&target_dir, true, &allowed);
    let filtered: Vec<(PathBuf, Option<String>)> = all_files
        .into_iter()
        .filter(|p| {
            let fname = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            fname != "dataset_infos.json" && fname != "dataset_info.json"
        })
        .filter(|p| {
            let rel = p
                .strip_prefix(&target_dir)
                .unwrap_or(p)
                .to_string_lossy()
                .to_lowercase();
            let sub_ok = subset.is_none_or(|sub| rel.contains(&sub.to_lowercase()));
            let split_ok = split
                .filter(|s| !s.eq_ignore_ascii_case("all"))
                .is_none_or(|sp| rel.contains(&sp.to_lowercase()));
            sub_ok && split_ok
        })
        .map(|p| {
            let prefix = infer_split_tag_from_path(&p);
            (p, prefix)
        })
        .collect();

    if filtered.is_empty() {
        bail!(
            "Downloaded '{repo_id}' into {}, but no matching data files were found",
            target_dir.display()
        );
    }

    Ok((filtered, class_labels, None))
}

fn infer_split_tag_from_path(path: &Path) -> Option<String> {
    for comp in path.components().rev() {
        let s = comp.as_os_str().to_string_lossy().to_lowercase();
        for known in ["train", "validation", "val", "dev", "test", "cal"] {
            if s == known
                || s.starts_with(&format!("{known}-"))
                || s.starts_with(&format!("{known}_"))
            {
                return Some(known.to_string());
            }
        }
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
}

fn extract_hf_dataset_infos_labels(dir: &Path) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::new();
    for candidate in ["dataset_infos.json", "dataset_info.json"] {
        let p = dir.join(candidate);
        if let Ok(text) = std::fs::read_to_string(&p)
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(&text)
        {
            collect_class_label_names(&val, &mut out);
        }
    }
    out
}

fn collect_class_label_names(val: &serde_json::Value, out: &mut BTreeMap<String, Vec<String>>) {
    if let serde_json::Value::Object(map) = val {
        for (k, v) in map {
            if let serde_json::Value::Object(inner) = v
                && inner.get("_type").and_then(|t| t.as_str()) == Some("ClassLabel")
                && let Some(serde_json::Value::Array(names)) = inner.get("names")
            {
                let labels: Vec<String> = names
                    .iter()
                    .filter_map(|n| n.as_str().map(|s| s.to_string()))
                    .collect();
                if !labels.is_empty() {
                    out.insert(k.clone(), labels);
                }
            }
            collect_class_label_names(v, out);
        }
    }
}

type RawRecord = serde_json::Map<String, serde_json::Value>;
type ParsedRawFile = (Vec<RawRecord>, ClassLabelMap);

fn read_raw_records_from_file(
    path: &Path,
    strict: bool,
    report: &mut ImportReport,
) -> Result<ParsedRawFile> {
    if let Ok(meta) = std::fs::metadata(path)
        && meta.len() == 0
    {
        return Ok((Vec::new(), BTreeMap::new()));
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => Ok((
            read_delimited_records(path, b',', strict, report)?,
            BTreeMap::new(),
        )),
        "tsv" => Ok((
            read_delimited_records(path, b'\t', strict, report)?,
            BTreeMap::new(),
        )),
        "jsonl" | "ndjson" => Ok((read_jsonl_records(path, strict, report)?, BTreeMap::new())),
        "json" => Ok((read_json_records(path)?, BTreeMap::new())),
        "parquet" | "pq" => read_parquet_records(path),
        _ => {
            if let Ok(recs) = read_jsonl_records(path, true, report) {
                Ok((recs, BTreeMap::new()))
            } else {
                Ok((read_json_records(path)?, BTreeMap::new()))
            }
        }
    }
}

fn strip_utf8_bom(s: &str) -> &str {
    s.strip_prefix('\u{FEFF}').unwrap_or(s)
}

fn read_delimited_records(
    path: &Path,
    delimiter: u8,
    strict: bool,
    report: &mut ImportReport,
) -> Result<Vec<RawRecord>> {
    let raw_bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read delimited file {}", path.display()))?;
    let body = if raw_bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &raw_bytes[3..]
    } else {
        &raw_bytes[..]
    };
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(!strict)
        .from_reader(body);

    let headers: Vec<String> = rdr
        .headers()
        .with_context(|| format!("Failed to parse headers in {}", path.display()))?
        .iter()
        .map(|h| strip_utf8_bom(h).trim().to_string())
        .collect();

    let mut out = Vec::new();
    for (idx, rec_res) in rdr.records().enumerate() {
        match rec_res {
            Ok(rec) => {
                let mut obj = serde_json::Map::new();
                for (h, val) in headers.iter().zip(rec.iter()) {
                    if !h.is_empty() {
                        obj.insert(h.clone(), serde_json::Value::String(val.to_string()));
                    }
                }
                out.push(obj);
            }
            Err(e) => {
                if strict {
                    bail!(
                        "Malformed delimited row {} in {}: {e}",
                        idx + 1,
                        path.display()
                    );
                }
                report.skipped_malformed += 1;
                if report.warnings.len() < 20 {
                    report.warnings.push(format!(
                        "{}:{}: malformed CSV row ({e})",
                        path.display(),
                        idx + 1
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn read_jsonl_records(
    path: &Path,
    strict: bool,
    report: &mut ImportReport,
) -> Result<Vec<RawRecord>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        let cleaned = if idx == 0 {
            strip_utf8_bom(&line).trim()
        } else {
            line.trim()
        };
        if cleaned.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(cleaned) {
            Ok(serde_json::Value::Object(map)) => out.push(map),
            Ok(serde_json::Value::String(s)) => {
                let mut m = serde_json::Map::new();
                m.insert("text".to_string(), serde_json::Value::String(s));
                out.push(m);
            }
            Ok(_) => {
                if strict {
                    bail!(
                        "JSONL line {} in {} is not a JSON object",
                        idx + 1,
                        path.display()
                    );
                }
                report.skipped_malformed += 1;
            }
            Err(e) => {
                if strict {
                    bail!("Malformed JSONL at {}:{} ({e})", path.display(), idx + 1);
                }
                report.skipped_malformed += 1;
                if report.warnings.len() < 20 {
                    report.warnings.push(format!(
                        "{}:{}: skipped invalid JSON ({e})",
                        path.display(),
                        idx + 1
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn read_json_records(path: &Path) -> Result<Vec<RawRecord>> {
    let raw = std::fs::read_to_string(path)?;
    let val: serde_json::Value = serde_json::from_str(strip_utf8_bom(&raw))
        .with_context(|| format!("Malformed JSON in {}", path.display()))?;
    match val {
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .enumerate()
            .map(|(i, v)| match v {
                serde_json::Value::Object(m) => Ok(m),
                serde_json::Value::String(s) => {
                    let mut m = serde_json::Map::new();
                    m.insert("text".to_string(), serde_json::Value::String(s));
                    Ok(m)
                }
                _ => bail!(
                    "JSON array element {i} in {} must be an object",
                    path.display()
                ),
            })
            .collect(),
        serde_json::Value::Object(mut map) => {
            for wrapper_key in ["rows", "data", "items", "records"] {
                if let Some(serde_json::Value::Array(arr)) = map.get(wrapper_key)
                    && arr.iter().all(|v| v.is_object())
                {
                    let Some(serde_json::Value::Array(owned)) = map.remove(wrapper_key) else {
                        unreachable!()
                    };
                    return Ok(owned
                        .into_iter()
                        .filter_map(|v| match v {
                            serde_json::Value::Object(m) => Some(m),
                            _ => None,
                        })
                        .collect());
                }
            }
            Ok(vec![map])
        }
        _ => bail!(
            "JSON root in {} must be an object or array of objects",
            path.display()
        ),
    }
}

/// Stream-convert a `.parquet` file to JSONL records via `python3` (`pyarrow`/`polars`/`duckdb`),
/// `duckdb` CLI, or `uv run --no-project --with pyarrow`, keeping `Cargo.toml` free of `arrow` bloat.
fn read_parquet_records(path: &Path) -> Result<ParsedRawFile> {
    const PY_SCRIPT: &str = r#"
import json, sys
path = sys.argv[1]
meta_labels = {}
try:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata or {}
    if b"huggingface" in md:
        hf = json.loads(md[b"huggingface"].decode("utf-8", "ignore"))
        feats = hf.get("info", {}).get("features", {})
        for k, v in feats.items():
            if isinstance(v, dict) and v.get("_type") == "ClassLabel" and "names" in v:
                meta_labels[k] = v["names"]
    print("__LEMMATA_HF_META__" + json.dumps(meta_labels), flush=True)
    for batch in pf.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            print(json.dumps(row, default=str))
    sys.exit(0)
except ImportError:
    pass
try:
    import duckdb
    print("__LEMMATA_HF_META__{}", flush=True)
    rel = duckdb.read_parquet(path)
    cols = rel.columns
    for row in rel.fetchall():
        print(json.dumps(dict(zip(cols, row)), default=str))
    sys.exit(0)
except ImportError:
    sys.exit(42)
"#;

    let path_str = path.display().to_string();

    // 1. Try local python3 first (zero network access if pyarrow or duckdb is already present)
    if let Ok(res) = run_parquet_stream_cmd("python3", &["-c", PY_SCRIPT, &path_str]) {
        return Ok(res);
    }

    // 2. Try standalone duckdb CLI if installed
    if let Ok(out) = Command::new("duckdb")
        .args([
            "-json",
            "-c",
            &format!(
                "SELECT * FROM read_parquet('{}');",
                path_str.replace('\'', "''")
            ),
        ])
        .output()
        && out.status.success()
        && let Ok(serde_json::Value::Array(arr)) = serde_json::from_slice(&out.stdout)
    {
        let recs = arr
            .into_iter()
            .filter_map(|v| match v {
                serde_json::Value::Object(m) => Some(m),
                _ => None,
            })
            .collect();
        return Ok((recs, BTreeMap::new()));
    }

    // 3. Try ephemeral `uv run --no-project --with pyarrow` (never touches conda base!)
    if let Ok(res) = run_parquet_stream_cmd(
        "uv",
        &[
            "run",
            "--no-project",
            "--with",
            "pyarrow",
            "python3",
            "-c",
            PY_SCRIPT,
            &path_str,
        ],
    ) {
        return Ok(res);
    }

    bail!(
        "Cannot read Parquet file '{}': no Parquet bridge (`uv`, `duckdb`, or `python3` with `pyarrow`) was available. Install `uv` (`https://astral.sh/uv`) or `duckdb`, or export the dataset as `.jsonl`/`.csv`.",
        path.display()
    )
}

fn run_parquet_stream_cmd(bin: &str, args: &[&str]) -> Result<ParsedRawFile> {
    let mut child = Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("missing stdout pipe"))?;
    let reader = BufReader::new(stdout);
    let mut records = Vec::new();
    let mut class_labels = BTreeMap::new();
    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("__LEMMATA_HF_META__") {
            if let Ok(map) = serde_json::from_str::<BTreeMap<String, Vec<String>>>(rest) {
                class_labels = map;
            }
            continue;
        }
        if let Ok(serde_json::Value::Object(m)) = serde_json::from_str(trimmed) {
            records.push(m);
        }
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("{bin} exited with {status}");
    }
    Ok((records, class_labels))
}

fn value_to_id_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn sanitize_string(raw: &str, normalize_nfc: bool, clean_mail: bool) -> String {
    let mut s = raw
        .replace("\u{FEFF}", "")
        .replace('\0', "")
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    if normalize_nfc {
        s = compose_common_combining_marks(&s);
    }
    if clean_mail {
        s = clean_email_body(&s, 32_768);
    }
    s
}

/// Lightweight pure-Rust NFC composition for common decomposed Latin accent sequences
/// (`e` + `\u{0301}` -> `é`, etc.) without pulling external Unicode tables.
fn compose_common_combining_marks(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(base) = chars.next() {
        if let Some(&comb) = chars.peek()
            && let Some(composed) = try_compose_pair(base, comb)
        {
            chars.next();
            out.push(composed);
            continue;
        }
        out.push(base);
    }
    out
}

fn try_compose_pair(base: char, combining: char) -> Option<char> {
    match (base, combining) {
        ('a', '\u{0300}') => Some('à'),
        ('a', '\u{0301}') => Some('á'),
        ('a', '\u{0302}') => Some('â'),
        ('a', '\u{0303}') => Some('ã'),
        ('a', '\u{0308}') => Some('ä'),
        ('c', '\u{0327}') => Some('ç'),
        ('e', '\u{0300}') => Some('è'),
        ('e', '\u{0301}') => Some('é'),
        ('e', '\u{0302}') => Some('ê'),
        ('e', '\u{0308}') => Some('ë'),
        ('i', '\u{0300}') => Some('ì'),
        ('i', '\u{0301}') => Some('í'),
        ('i', '\u{0302}') => Some('î'),
        ('i', '\u{0308}') => Some('ï'),
        ('n', '\u{0303}') => Some('ñ'),
        ('o', '\u{0300}') => Some('ò'),
        ('o', '\u{0301}') => Some('ó'),
        ('o', '\u{0302}') => Some('ô'),
        ('o', '\u{0303}') => Some('õ'),
        ('o', '\u{0308}') => Some('ö'),
        ('u', '\u{0300}') => Some('ù'),
        ('u', '\u{0301}') => Some('ú'),
        ('u', '\u{0302}') => Some('û'),
        ('u', '\u{0308}') => Some('ü'),
        ('A', '\u{0301}') => Some('Á'),
        ('E', '\u{0301}') => Some('É'),
        ('I', '\u{0301}') => Some('Í'),
        ('N', '\u{0303}') => Some('Ñ'),
        ('O', '\u{0301}') => Some('Ó'),
        ('O', '\u{0308}') => Some('Ö'),
        ('U', '\u{0301}') => Some('Ú'),
        ('U', '\u{0308}') => Some('Ü'),
        _ => None,
    }
}

fn sanitize_json_value(
    v: serde_json::Value,
    normalize_nfc: bool,
    clean_mail: bool,
) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => {
            serde_json::Value::String(sanitize_string(&s, normalize_nfc, clean_mail))
        }
        serde_json::Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, val) in map {
                sorted.insert(k, sanitize_json_value(val, normalize_nfc, clean_mail));
            }
            let mut out = serde_json::Map::new();
            for (k, val) in sorted {
                out.insert(k, val);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(|x| sanitize_json_value(x, normalize_nfc, clean_mail))
                .collect(),
        ),
        other => other,
    }
}

fn lookup_nested<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    col_spec: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(v) = map.get(col_spec) {
        return Some(v);
    }
    let mut cur: &serde_json::Value = return_nested_step(map, col_spec)?;
    for part in col_spec.split('.').skip(1) {
        cur = cur.as_object()?.get(part)?;
    }
    Some(cur)
}

fn return_nested_step<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    col_spec: &str,
) -> Option<&'a serde_json::Value> {
    let first = col_spec.split('.').next()?;
    map.get(first)
}

fn match_schema_qid(raw_qid: &str, schema: Option<&BTreeMap<String, Question>>) -> String {
    if let Some(questions) = schema {
        if questions.contains_key(raw_qid) {
            return raw_qid.to_string();
        }
        if let Some(found) = questions.keys().find(|k| k.eq_ignore_ascii_case(raw_qid)) {
            return found.clone();
        }
    }
    raw_qid.to_string()
}

fn strip_gt_prefix_preserve_case(key: &str) -> Option<&str> {
    for prefix in ["gt_", "gt:", "label_", "label:", "target_", "target:"] {
        if key.len() > prefix.len() && key[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return Some(&key[prefix.len()..]);
        }
    }
    None
}

fn is_nullish_str(s: &str) -> bool {
    let t = s.trim();
    t.is_empty()
        || t.eq_ignore_ascii_case("null")
        || t.eq_ignore_ascii_case("none")
        || t.eq_ignore_ascii_case("nan")
        || t.eq_ignore_ascii_case("n/a")
}

fn canonicalize_gt_value(
    qid: &str,
    src_col_hint: Option<&str>,
    raw_val: serde_json::Value,
    config: &ImportConfig,
    hf_class_labels: &BTreeMap<String, Vec<String>>,
    warnings: &mut Vec<String>,
) -> Result<Option<serde_json::Value>> {
    if raw_val.is_null() {
        return Ok(None);
    }
    if let serde_json::Value::String(ref s) = raw_val
        && is_nullish_str(s)
    {
        return Ok(None);
    }

    let vmap = config.value_map.get(qid);
    let question = config.schema.as_ref().and_then(|s| s.get(qid));

    let hf_names = src_col_hint
        .and_then(|c| hf_class_labels.get(c))
        .or_else(|| hf_class_labels.get(qid))
        .or_else(|| hf_class_labels.get("label"));

    let Some(q) = question else {
        // Schema-less canonicalization: still apply value_map, multi_label_sep, and HF ClassLabel names
        if let Some(sep) = config.multi_label_sep
            && let serde_json::Value::String(s) = &raw_val
        {
            let parts: Vec<serde_json::Value> = s
                .split(sep)
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .map(|p| {
                    let mapped = vmap
                        .and_then(|m| m.get(&p.to_lowercase()))
                        .cloned()
                        .unwrap_or_else(|| p.to_string());
                    serde_json::Value::String(mapped)
                })
                .collect();
            return Ok(Some(serde_json::Value::Array(parts)));
        }
        if let Some(names) = hf_names
            && let Some(idx) = raw_val
                .as_u64()
                .or_else(|| raw_val.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
            && let Some(name) = names.get(idx as usize)
        {
            let mapped = vmap
                .and_then(|m| m.get(&name.to_lowercase()))
                .cloned()
                .unwrap_or_else(|| name.clone());
            return Ok(Some(serde_json::Value::String(mapped)));
        }
        if let serde_json::Value::String(s) = &raw_val
            && let Some(mapped) = vmap.and_then(|m| m.get(&s.trim().to_lowercase()))
        {
            return Ok(Some(serde_json::Value::String(mapped.clone())));
        }
        return Ok(Some(raw_val));
    };

    match q.qtype {
        QuestionType::Choice => {
            let schema_keys = choice_labels(q);
            let mut map_single_choice = |val: &serde_json::Value| -> Result<String> {
                // 1. Check integer ClassLabel mapping first
                if let Some(idx) = val.as_u64().or_else(|| {
                    val.as_str().and_then(|s| {
                        let t = s.trim();
                        (!schema_keys.iter().any(|k| k.eq_ignore_ascii_case(t)))
                            .then(|| t.parse::<u64>().ok())
                            .flatten()
                    })
                }) {
                    if let Some(names) = hf_names
                        && let Some(lbl) = names.get(idx as usize)
                    {
                        let after_vmap = vmap
                            .and_then(|m| m.get(&lbl.to_lowercase()))
                            .map(String::as_str)
                            .unwrap_or(lbl.as_str());
                        if let Some(canon) = schema_keys
                            .iter()
                            .find(|k| k.eq_ignore_ascii_case(after_vmap))
                        {
                            return Ok(canon.clone());
                        }
                        return Ok(after_vmap.to_string());
                    }
                    if config.int_labels_from_schema {
                        if let Some(canon) = schema_keys.get(idx as usize) {
                            return Ok(canon.clone());
                        }
                        bail!(
                            "Integer label index {idx} out of bounds for question '{qid}' ({} options)",
                            schema_keys.len()
                        );
                    }
                }

                let raw_s = match val {
                    serde_json::Value::String(s) => s.trim().to_string(),
                    other => other.to_string(),
                };
                let mapped = vmap
                    .and_then(|m| m.get(&raw_s.to_lowercase()))
                    .cloned()
                    .unwrap_or(raw_s);
                if let Some(canon) = schema_keys.iter().find(|k| k.eq_ignore_ascii_case(&mapped)) {
                    return Ok(canon.clone());
                }
                if config.strict {
                    bail!(
                        "Label '{}' for question '{}' is not in schema options {:?}",
                        mapped,
                        qid,
                        schema_keys
                    );
                }
                if warnings.len() < 20 {
                    warnings.push(format!(
                        "Question '{qid}': ground-truth label '{mapped}' not in schema options {:?}",
                        schema_keys
                    ));
                }
                Ok(mapped)
            };

            if let serde_json::Value::Array(arr) = &raw_val {
                let mut out_arr = Vec::new();
                let mut seen = BTreeSet::new();
                for elem in arr {
                    let c = map_single_choice(elem)?;
                    if seen.insert(c.to_lowercase()) {
                        out_arr.push(serde_json::Value::String(c));
                    }
                }
                return Ok(Some(serde_json::Value::Array(out_arr)));
            }
            if let Some(sep) = config.multi_label_sep
                && let serde_json::Value::String(s) = &raw_val
            {
                let mut out_arr = Vec::new();
                let mut seen = BTreeSet::new();
                for piece in s.split(sep).map(str::trim).filter(|p| !p.is_empty()) {
                    let c = map_single_choice(&serde_json::Value::String(piece.to_string()))?;
                    if seen.insert(c.to_lowercase()) {
                        out_arr.push(serde_json::Value::String(c));
                    }
                }
                return Ok(Some(serde_json::Value::Array(out_arr)));
            }
            let canon = map_single_choice(&raw_val)?;
            Ok(Some(serde_json::Value::String(canon)))
        }
        QuestionType::Score => {
            let raw_str = raw_val.as_str().map(|s| s.trim().to_lowercase());
            let mapped_val = if let Some(ref rs) = raw_str
                && let Some(m) = vmap.and_then(|vm| vm.get(rs))
            {
                m.parse::<f64>()
                    .map_err(|_| anyhow!("Mapped score '{m}' for '{qid}' is not numeric"))?
            } else if let Some(n) = raw_val.as_f64() {
                n
            } else if let Some(ref rs) = raw_str {
                rs.parse::<f64>().map_err(|_| {
                    anyhow!("Score value '{rs}' for question '{qid}' is not a valid number")
                })?
            } else {
                bail!("Score value '{raw_val}' for question '{qid}' must be numeric");
            };

            let shifted = mapped_val + (config.score_offset as f64);
            let max_level = q.render_options().len().saturating_sub(1) as f64;
            if config.strict && (shifted < -1e-9 || shifted > max_level + 1e-9) {
                bail!(
                    "Score {shifted} for question '{qid}' is outside rubric bounds [0, {max_level}]"
                );
            }
            if (shifted - shifted.round()).abs() < 1e-9 {
                Ok(Some(serde_json::json!(shifted.round() as i64)))
            } else {
                let num = serde_json::Number::from_f64(shifted)
                    .ok_or_else(|| anyhow!("Non-finite score for '{qid}'"))?;
                Ok(Some(serde_json::Value::Number(num)))
            }
        }
        QuestionType::Noul => {
            let raw_s = match &raw_val {
                serde_json::Value::Bool(b) => return Ok(Some(serde_json::Value::Bool(*b))),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        match i {
                            1 => return Ok(Some(serde_json::Value::Bool(true))),
                            0 => return Ok(Some(serde_json::Value::Bool(false))),
                            _ => bail!(
                                "Invalid numeric noul target {i} for '{qid}' (expected 0 or 1)"
                            ),
                        }
                    }
                    n.to_string()
                }
                serde_json::Value::String(s) => s.trim().to_lowercase(),
                other => bail!("Invalid noul target '{other}' for '{qid}'"),
            };
            let mapped = vmap
                .and_then(|m| m.get(&raw_s))
                .map(|s| s.to_lowercase())
                .unwrap_or(raw_s);
            match mapped.as_str() {
                "true" | "yes" | "1" | "t" | "y" => Ok(Some(serde_json::Value::Bool(true))),
                "false" | "no" | "0" | "f" | "n" => Ok(Some(serde_json::Value::Bool(false))),
                other => {
                    if config.strict {
                        bail!("Invalid boolean target '{other}' for noul question '{qid}'");
                    }
                    warnings.push(format!(
                        "Question '{qid}': unrecognized boolean target '{other}'"
                    ));
                    Ok(Some(serde_json::Value::String(other.to_string())))
                }
            }
        }
    }
}

fn canonicalize_record(
    mut map: RawRecord,
    default_id: &str,
    split_prefix: Option<&str>,
    source_uri: &str,
    config: &ImportConfig,
    hf_class_labels: &BTreeMap<String, Vec<String>>,
    warnings: &mut Vec<String>,
) -> Result<Option<DataItem>> {
    let mut consumed_cols: HashSet<String> = HashSet::new();

    // 1. Resolve ID
    let mut resolved_id = None;
    if let Some(id_col) = &config.id_col {
        consumed_cols.insert(id_col.clone());
        if let Some(v) = lookup_nested(&map, id_col) {
            resolved_id = value_to_id_string(v);
        }
    } else {
        for cand in ["id", "item_id"] {
            if let Some(v) = map.get(cand) {
                consumed_cols.insert(cand.to_string());
                if let Some(s) = value_to_id_string(v) {
                    resolved_id = Some(s);
                    break;
                }
            }
        }
    }
    let id = match (resolved_id, split_prefix) {
        (Some(raw_id), Some(pref))
            if raw_id.chars().all(|c| c.is_ascii_digit()) && !raw_id.starts_with(pref) =>
        {
            format!("{pref}:{raw_id}")
        }
        (Some(raw_id), _) => raw_id,
        (None, _) => default_id.to_string(),
    };

    // Preserve existing `source_uri` if re-importing a canonical DataItem
    let final_uri = map
        .get("source_uri")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| source_uri.to_string());
    consumed_cols.insert("source_uri".to_string());

    // 2. Resolve Ground Truth
    let mut gt_map: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    // 2a. Explicit `--label-map qid=src_col`
    for (raw_qid, src_col) in &config.label_map {
        consumed_cols.insert(src_col.clone());
        let qid = match_schema_qid(raw_qid, config.schema.as_ref());
        if let Some(val) = lookup_nested(&map, src_col).cloned()
            && let Some(canon) = canonicalize_gt_value(
                &qid,
                Some(src_col.as_str()),
                val,
                config,
                hf_class_labels,
                warnings,
            )?
        {
            gt_map.insert(qid, canon);
        }
    }

    // 2b. Existing `ground_truth` / `target` / `label` fields (if not already mapped as state/meta)
    let state_or_meta = |col: &str| {
        config.text_col.as_deref() == Some(col)
            || config.state_cols.iter().any(|c| c == col)
            || config.meta_cols.iter().any(|c| c == col)
    };
    let single_schema_qid = config
        .schema
        .as_ref()
        .filter(|s| s.len() == 1)
        .and_then(|s| s.keys().next().cloned());

    for std_gt_col in ["ground_truth", "target", "label"] {
        if state_or_meta(std_gt_col) || consumed_cols.contains(std_gt_col) {
            continue;
        }
        let Some(val) = map.get(std_gt_col).cloned() else {
            continue;
        };
        consumed_cols.insert(std_gt_col.to_string());
        match val {
            serde_json::Value::Object(obj) => {
                for (k, v) in obj {
                    let qid = match_schema_qid(&k, config.schema.as_ref());
                    if gt_map.contains_key(&qid) {
                        continue;
                    }
                    if let Some(canon) =
                        canonicalize_gt_value(&qid, Some(&k), v, config, hf_class_labels, warnings)?
                    {
                        gt_map.insert(qid, canon);
                    }
                }
            }
            scalar_or_arr => {
                let target_qid = single_schema_qid
                    .clone()
                    .unwrap_or_else(|| "default".to_string());
                if !gt_map.contains_key(&target_qid)
                    && let Some(canon) = canonicalize_gt_value(
                        &target_qid,
                        Some(std_gt_col),
                        scalar_or_arr,
                        config,
                        hf_class_labels,
                        warnings,
                    )?
                {
                    gt_map.insert(target_qid, canon);
                }
            }
        }
    }

    // 2c. Prefixed target columns (`gt_<qid>`, `label_<qid>`, `target_<qid>`) preserving `qid` case!
    let all_keys: Vec<String> = map.keys().cloned().collect();
    for key in all_keys {
        if state_or_meta(&key) || consumed_cols.contains(&key) {
            continue;
        }
        if let Some(raw_qid) = strip_gt_prefix_preserve_case(&key) {
            consumed_cols.insert(key.clone());
            let qid = match_schema_qid(raw_qid, config.schema.as_ref());
            if !gt_map.contains_key(&qid)
                && let Some(val) = map.get(&key).cloned()
                && let Some(canon) =
                    canonicalize_gt_value(&qid, Some(&key), val, config, hf_class_labels, warnings)?
            {
                gt_map.insert(qid, canon);
            }
        }
    }

    // 2d. Direct schema `qid` column fallback when `label_map` wasn't manually provided
    if config.label_map.is_empty()
        && let Some(questions) = &config.schema
    {
        for qid in questions.keys() {
            if gt_map.contains_key(qid) || state_or_meta(qid) || consumed_cols.contains(qid) {
                continue;
            }
            if let Some(matched_col) = map
                .keys()
                .find(|k| k.eq_ignore_ascii_case(qid) && !state_or_meta(k))
                .cloned()
            {
                consumed_cols.insert(matched_col.clone());
                if let Some(val) = map.get(&matched_col).cloned()
                    && let Some(canon) = canonicalize_gt_value(
                        qid,
                        Some(&matched_col),
                        val,
                        config,
                        hf_class_labels,
                        warnings,
                    )?
                {
                    gt_map.insert(qid.clone(), canon);
                }
            }
        }
    }

    // 3. Resolve `metadata` (strictly kept out of model-visible `state`)
    let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    if let Some(serde_json::Value::Object(existing_meta)) = map.remove("metadata") {
        for (k, v) in existing_meta {
            metadata.insert(k, v);
        }
    }
    for meta_col in &config.meta_cols {
        consumed_cols.insert(meta_col.clone());
        if let Some(val) = lookup_nested(&map, meta_col).cloned()
            && !val.is_null()
        {
            metadata.insert(meta_col.clone(), val);
        }
    }

    // 4. Resolve model-visible `state`
    let raw_state = if let Some(text_col) = &config.text_col {
        let v = lookup_nested(&map, text_col).cloned().ok_or_else(|| {
            anyhow!(
                "Configured --text-col '{text_col}' not found in record keys {:?}",
                map.keys().collect::<Vec<_>>()
            )
        })?;
        match v {
            serde_json::Value::String(s) => serde_json::Value::String(s),
            serde_json::Value::Null => serde_json::Value::String(String::new()),
            other => serde_json::Value::String(other.to_string()),
        }
    } else if !config.state_cols.is_empty() {
        let mut st_obj = serde_json::Map::new();
        let mut sorted_cols = config.state_cols.clone();
        sorted_cols.sort();
        for col in &sorted_cols {
            let v = lookup_nested(&map, col).cloned().ok_or_else(|| {
                anyhow!(
                    "Configured state column '{col}' not found in record keys {:?}",
                    map.keys().collect::<Vec<_>>()
                )
            })?;
            st_obj.insert(col.clone(), v);
        }
        serde_json::Value::Object(st_obj)
    } else if let Some(explicit_state) = map.remove("state") {
        // Any remaining unconsumed columns when `state` was explicit go into `metadata` so they never vanish or leak
        for (k, v) in map {
            if !consumed_cols.contains(&k) && !v.is_null() {
                metadata.entry(k).or_insert(v);
            }
        }
        explicit_state
    } else {
        let mut remaining: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        for (k, v) in map {
            if !consumed_cols.contains(&k) {
                remaining.insert(k, v);
            }
        }
        if remaining.len() == 1
            && let Some(serde_json::Value::String(s)) = remaining.remove("text")
        {
            serde_json::Value::String(s)
        } else {
            let mut st_obj = serde_json::Map::new();
            for (k, v) in remaining {
                st_obj.insert(k, v);
            }
            serde_json::Value::Object(st_obj)
        }
    };

    let clean_state = sanitize_json_value(raw_state, config.normalize_nfc, config.clean_email);
    let is_blank = match &clean_state {
        serde_json::Value::String(s) => s.trim().is_empty(),
        serde_json::Value::Null => true,
        serde_json::Value::Object(m) => {
            m.is_empty()
                || m.values().all(|v| match v {
                    serde_json::Value::String(s) => s.trim().is_empty(),
                    serde_json::Value::Null => true,
                    _ => false,
                })
        }
        _ => false,
    };

    if is_blank {
        if config.drop_empty {
            return Ok(None);
        }
        if config.strict {
            bail!("Record '{id}' has an empty model-visible state");
        }
    }

    let mut item = DataItem::new(id, clean_state, final_uri);
    if !gt_map.is_empty() {
        item = item.with_ground_truth(gt_map);
    }
    if !metadata.is_empty() {
        item = item.with_metadata(metadata);
    }
    Ok(Some(item))
}

// ============================================================================
// Section 2: Deterministic Leakage-Safe Partitioning (`DatasetSplitter`)
// ============================================================================

/// Target proportions across `train`, `val`, `cal`, and `test`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitRatios {
    pub train: f64,
    pub val: f64,
    pub cal: f64,
    pub test: f64,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 0.70,
            val: 0.10,
            cal: 0.10,
            test: 0.10,
        }
    }
}

impl SplitRatios {
    /// Standard 3-way `80/10/0/10` ratios when `--no-cal` is requested.
    pub fn three_way_no_cal() -> Self {
        Self {
            train: 0.80,
            val: 0.10,
            cal: 0.0,
            test: 0.10,
        }
    }

    /// Parse `"0.70,0.10,0.10,0.10"` (4-way) or `"0.80,0.10,0.10"` (3-way `train,val,test`).
    pub fn from_csv_str(spec: &str) -> Result<Self> {
        let parts: Vec<f64> = spec
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<f64>()
                    .map_err(|_| anyhow!("Invalid numeric split ratio in '{spec}'"))
            })
            .collect::<Result<_>>()?;
        let raw = match parts.as_slice() {
            [tr, va, ca, te] => Self {
                train: *tr,
                val: *va,
                cal: *ca,
                test: *te,
            },
            [tr, va, te] => Self {
                train: *tr,
                val: *va,
                cal: 0.0,
                test: *te,
            },
            _ => bail!(
                "Expected 4 ratios `train,val,cal,test` (or 3 ratios `train,val,test`), got '{spec}'"
            ),
        };
        raw.validated()
    }

    pub fn validated(self) -> Result<Self> {
        for (name, r) in [
            ("train", self.train),
            ("val", self.val),
            ("cal", self.cal),
            ("test", self.test),
        ] {
            if !r.is_finite() || r < 0.0 {
                bail!("Split ratio '{name}' must be non-negative and finite (got {r})");
            }
        }
        if self.train <= 0.0 || self.test <= 0.0 {
            bail!("Both `train` and `test` split ratios must be strictly positive");
        }
        let sum = self.train + self.val + self.cal + self.test;
        if (sum - 1.0).abs() > 1e-3 {
            bail!("Split ratios must sum to 1.0 (got {sum:.4})");
        }
        Ok(Self {
            train: self.train / sum,
            val: self.val / sum,
            cal: self.cal / sum,
            test: self.test / sum,
        })
    }
}

/// How exact duplicates should be handled prior to split partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DedupAction {
    /// Keep all items, locking exact/template/MinHash near-duplicate clusters into the same split.
    #[default]
    Group,
    /// Drop exact `(normalized_state, ground_truth)` duplicates before grouping & splitting.
    DropExact,
}

impl std::str::FromStr for DedupAction {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "group" | "cluster" => Ok(Self::Group),
            "drop-exact" | "drop_exact" | "drop" => Ok(Self::DropExact),
            other => bail!("Invalid dedup action '{other}'. Options: group, drop-exact"),
        }
    }
}

/// Configuration for `DatasetSplitter`.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    pub ratios: SplitRatios,
    pub seed: u64,
    pub group_col: Option<String>,
    /// MinHash Jaccard threshold in `(0.0, 1.0]` for near-duplicate clustering (default `0.85`).
    pub dedup_sim: f64,
    pub dedup_action: DedupAction,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            ratios: SplitRatios::default(),
            seed: 42,
            group_col: None,
            dedup_sim: 0.85,
            dedup_action: DedupAction::Group,
        }
    }
}

/// Per-split digest & item count entry in `dataset_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitFileMeta {
    pub file: String,
    pub sha256: String,
    pub item_set_digest: String,
    pub state_merkle_sha256: String,
    pub items: usize,
    pub groups: usize,
}

/// Deduplication & leakage-cluster diagnostics in `dataset_manifest.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DedupSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_col: Option<String>,
    pub threshold: f64,
    pub action: DedupAction,
    pub dropped_exact_duplicates: usize,
    pub clusters_found: usize,
    pub grouped_items: usize,
    pub max_cluster_size: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Cryptographic provenance and support manifest written alongside `{train,val,cal,test}.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub manifest_version: String,
    pub split_id: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_sha256: Option<String>,
    pub source_item_set_digest: String,
    pub schema_digest: String,
    pub seed: u64,
    pub ratios: SplitRatios,
    pub dedup: DedupSummary,
    pub splits: BTreeMap<String, SplitFileMeta>,
    /// `question_id -> label -> split_name -> support_count`
    pub per_question_support: BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>>,
}

/// In-memory result returned by `DatasetSplitter::split_items`.
#[derive(Debug, Clone)]
pub struct SplitOutput {
    pub splits: BTreeMap<String, Vec<DataItem>>,
    pub manifest: DatasetManifest,
}

/// 128-permutation MinHash + `16 x 8` Banded LSH engine for $O(N)$ near-duplicate detection.
///
/// Strings with fewer than 4 whitespace-delimited words return `None` for signatures so tiny
/// generic utterances (`"Thanks"`, `"Help me"`) never form giant false-positive LSH hubs.
pub struct MinHashIndex {
    coeffs: [(u64, u64); MINHASH_NUM_PERM],
}

impl MinHashIndex {
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed ^ 0x4D49_4E48_4153_4831);
        let mut coeffs = [(0u64, 0u64); MINHASH_NUM_PERM];
        for pair in &mut coeffs {
            pair.0 = sm.next_u64() | 1; // odd multiplier in Z_{2^64}
            pair.1 = sm.next_u64();
        }
        Self { coeffs }
    }

    /// Compute 128-permutation MinHash over character 5-grams of `normalized_text`.
    /// Returns `None` if `normalized_text` has fewer than 4 words.
    pub fn signature(&self, normalized_text: &str) -> Option<[u64; MINHASH_NUM_PERM]> {
        if normalized_text.split_whitespace().count() < 4 {
            return None;
        }
        let chars: Vec<char> = normalized_text.chars().collect();
        let mut shingle_hashes = Vec::new();
        if chars.len() <= 5 {
            shingle_hashes.push(fnv1a64(normalized_text.as_bytes()));
        } else {
            let mut buf = String::with_capacity(20);
            for win in chars.windows(5) {
                buf.clear();
                for &c in win {
                    buf.push(c);
                }
                shingle_hashes.push(fnv1a64(buf.as_bytes()));
            }
            shingle_hashes.sort_unstable();
            shingle_hashes.dedup();
        }
        if shingle_hashes.is_empty() {
            return None;
        }

        let mut sig = [u64::MAX; MINHASH_NUM_PERM];
        for &sh in &shingle_hashes {
            for (k, &(a, b)) in self.coeffs.iter().enumerate() {
                let h = a.wrapping_mul(sh).wrapping_add(b);
                if h < sig[k] {
                    sig[k] = h;
                }
            }
        }
        Some(sig)
    }

    #[inline]
    pub fn jaccard_estimate(a: &[u64; MINHASH_NUM_PERM], b: &[u64; MINHASH_NUM_PERM]) -> f64 {
        let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        matches as f64 / MINHASH_NUM_PERM as f64
    }

    pub fn band_hash(sig: &[u64; MINHASH_NUM_PERM], band: usize) -> u64 {
        let start = band * LSH_ROWS_PER_BAND;
        let mut bytes = [0u8; LSH_ROWS_PER_BAND * 8];
        for r in 0..LSH_ROWS_PER_BAND {
            let le = sig[start + r].to_le_bytes();
            bytes[r * 8..(r + 1) * 8].copy_from_slice(&le);
        }
        fnv1a64(&bytes)
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

pub struct DatasetSplitter;

impl DatasetSplitter {
    /// Partition `items` into leakage-free, group-aware stratified splits and write them to `out_dir`.
    pub fn split_to_dir(
        config: &SplitConfig,
        input_path: &Path,
        questions: &BTreeMap<String, Question>,
        out_dir: &Path,
    ) -> Result<DatasetManifest> {
        let items = read_dataset_items(input_path)?;
        let src_sha = file_sha256_hex(input_path).ok();
        let mut output = Self::split_items(config, &items, questions, src_sha)?;
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("Failed to create split dir {}", out_dir.display()))?;

        for (split_name, split_items) in &output.splits {
            let fname = format!("{split_name}.jsonl");
            let fpath = out_dir.join(&fname);
            write_dataset_jsonl(&fpath, split_items)?;
            let f_sha = file_sha256_hex(&fpath)?;
            if let Some(meta) = output.manifest.splits.get_mut(split_name) {
                meta.sha256 = f_sha;
            }
        }

        let manifest_path = out_dir.join("dataset_manifest.json");
        serde_json::to_writer_pretty(File::create(&manifest_path)?, &output.manifest)?;
        Ok(output.manifest)
    }

    /// Partition in-memory `items` deterministically into `train` / `val` / (`cal`) / `test`.
    pub fn split_items(
        config: &SplitConfig,
        raw_items: &[DataItem],
        questions: &BTreeMap<String, Question>,
        source_sha256: Option<String>,
    ) -> Result<SplitOutput> {
        let ratios = config.ratios.validated()?;
        if raw_items.len() < 2 {
            bail!(
                "Dataset must contain at least 2 items to split (got {})",
                raw_items.len()
            );
        }

        let source_item_set_digest = compute_item_set_digest(raw_items);

        // Optional Stage 0: Drop exact `(normalized_state, canonical_gt)` duplicates
        let mut dropped_exact_duplicates = 0usize;
        let items: Vec<DataItem> = if config.dedup_action == DedupAction::DropExact {
            let mut seen_exact = HashSet::new();
            let mut kept = Vec::with_capacity(raw_items.len());
            for item in raw_items {
                let norm = normalize_state_for_leakage(&item.state_text());
                let gt_bytes = serde_json::to_vec(&item.ground_truth).unwrap_or_default();
                let key = (norm, gt_bytes);
                if seen_exact.insert(key) {
                    kept.push(item.clone());
                } else {
                    dropped_exact_duplicates += 1;
                }
            }
            kept
        } else {
            raw_items.to_vec()
        };

        let n = items.len();
        if n < 2 {
            bail!("Fewer than 2 unique items remain after exact deduplication");
        }

        let mut uf = DisjointSet::new(n);

        // Stage 1: Explicit `--group-col` grouping (checks `metadata` first, then structured `state`)
        if let Some(gcol) = &config.group_col {
            let mut by_group: BTreeMap<String, usize> = BTreeMap::new();
            for (idx, item) in items.iter().enumerate() {
                let val = item
                    .metadata
                    .get(gcol)
                    .or_else(|| item.state.as_object().and_then(|o| o.get(gcol)));
                if let Some(gv) = val.and_then(value_to_id_string) {
                    if let Some(&first_idx) = by_group.get(&gv) {
                        uf.union(first_idx, idx);
                    } else {
                        by_group.insert(gv, idx);
                    }
                }
            }
        }

        // Stage 2: Normalized Template SHA-256 grouping (masks emails/URLs/UUIDs/digit runs)
        if config.dedup_sim > 0.0 {
            let mut by_template: HashMap<String, usize> = HashMap::with_capacity(n);
            for (idx, item) in items.iter().enumerate() {
                let tmpl = normalize_template_for_grouping(&item.state_text());
                if !tmpl.is_empty() {
                    let digest = sha256_hex(tmpl.as_bytes());
                    if let Some(&first_idx) = by_template.get(&digest) {
                        uf.union(first_idx, idx);
                    } else {
                        by_template.insert(digest, idx);
                    }
                }
            }
        }

        // Stage 3: 128-hash MinHash + 16x8 Banded LSH near-duplicate clustering (skips <4-word texts)
        if config.dedup_sim > 0.0 && config.dedup_sim < 1.0 {
            let mh = MinHashIndex::new(config.seed);
            let sigs: Vec<Option<[u64; MINHASH_NUM_PERM]>> = items
                .iter()
                .map(|it| mh.signature(&normalize_state_for_leakage(&it.state_text())))
                .collect();

            let mut buckets: BTreeMap<(usize, u64), Vec<usize>> = BTreeMap::new();
            for (idx, maybe_sig) in sigs.iter().enumerate() {
                if let Some(sig) = maybe_sig {
                    for b in 0..LSH_BANDS {
                        buckets
                            .entry((b, MinHashIndex::band_hash(sig, b)))
                            .or_default()
                            .push(idx);
                    }
                }
            }

            for members in buckets.values() {
                if members.len() < 2 {
                    continue;
                }
                for (pos, &i) in members.iter().enumerate() {
                    let Some(sig_i) = &sigs[i] else { continue };
                    // Bound per-member window in huge buckets to stay strictly O(N)
                    let window_end = (pos + 64).min(members.len());
                    for &j in &members[pos + 1..window_end] {
                        if uf.find(i) == uf.find(j) {
                            continue;
                        }
                        if let Some(sig_j) = &sigs[j]
                            && MinHashIndex::jaccard_estimate(sig_i, sig_j) >= config.dedup_sim
                        {
                            uf.union(i, j);
                        }
                    }
                }
            }
        }

        // Materialize atomic leakage groups
        let mut comp_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for idx in 0..n {
            comp_map.entry(uf.find(idx)).or_default().push(idx);
        }

        let mut clusters_found = 0usize;
        let mut grouped_items = 0usize;
        let mut max_cluster_size = 0usize;
        for members in comp_map.values() {
            max_cluster_size = max_cluster_size.max(members.len());
            if members.len() > 1 {
                clusters_found += 1;
                grouped_items += members.len();
            }
        }
        let mut dedup_warnings = Vec::new();
        if max_cluster_size > (n / 20).max(5) {
            dedup_warnings.push(format!(
                "Largest near-duplicate/entity group spans {max_cluster_size} items ({:.1}% of dataset)",
                (max_cluster_size as f64 / n as f64) * 100.0
            ));
        }

        // Build multi-question / multi-label binary stratification indicators
        let (item_strata, num_strata) = extract_stratification_indicators(&items, questions);

        struct GroupUnit {
            group_id: String,
            det_hash: u64,
            member_indices: Vec<usize>,
            counts: BTreeMap<usize, usize>,
        }

        let mut groups: Vec<GroupUnit> = comp_map
            .into_values()
            .map(|members| {
                let min_id = members
                    .iter()
                    .map(|&i| items[i].id.as_str())
                    .min()
                    .unwrap_or("");
                let mut seed_buf = Vec::with_capacity(8 + min_id.len());
                seed_buf.extend_from_slice(&config.seed.to_le_bytes());
                seed_buf.extend_from_slice(min_id.as_bytes());
                let det_hash = sha256_u64_prefix(&seed_buf);
                let group_id = format!("grp_{:012x}", det_hash & 0xFFFF_FFFF_FFFF);
                let mut counts = BTreeMap::new();
                for &i in &members {
                    for &s_idx in &item_strata[i] {
                        *counts.entry(s_idx).or_insert(0) += 1;
                    }
                }
                GroupUnit {
                    group_id,
                    det_hash,
                    member_indices: members,
                    counts,
                }
            })
            .collect();

        // Sort deterministically so group indices are 100% platform-invariant
        groups.sort_by_key(|g| (g.det_hash, g.group_id.clone()));

        let mut active_splits: Vec<(&'static str, f64, usize)> = vec![("train", ratios.train, 0)];
        if ratios.val > 0.0 {
            active_splits.push(("val", ratios.val, 3));
        }
        if ratios.cal > 0.0 {
            active_splits.push(("cal", ratios.cal, 2));
        }
        active_splits.push(("test", ratios.test, 1));
        let num_splits = active_splits.len();
        let train_split_idx = 0usize;
        let test_split_idx = active_splits
            .iter()
            .position(|(name, _, _)| *name == "test")
            .unwrap_or(num_splits - 1);

        let desired_items: Vec<f64> = active_splits
            .iter()
            .map(|(_, r, _)| *r * n as f64)
            .collect();
        let mut total_per_stratum = vec![0usize; num_strata];
        let mut groups_per_stratum = vec![0usize; num_strata];
        for g in &groups {
            for (&l, &c) in &g.counts {
                total_per_stratum[l] += c;
                groups_per_stratum[l] += 1;
            }
        }

        let mut desired_stratum: Vec<Vec<f64>> = active_splits
            .iter()
            .map(|(_, r, _)| {
                total_per_stratum
                    .iter()
                    .map(|&tot| *r * tot as f64)
                    .collect()
            })
            .collect();

        let mut assigned_split: Vec<Option<usize>> = vec![None; groups.len()];
        let mut assigned_items = vec![0usize; num_splits];
        let mut assigned_stratum = vec![vec![0usize; num_strata]; num_splits];

        // Iterative Rarest-Stratum-First Group Assignment
        loop {
            let mut rem_per_stratum = vec![0usize; num_strata];
            for (g_idx, g) in groups.iter().enumerate() {
                if assigned_split[g_idx].is_none() {
                    for (&l, &c) in &g.counts {
                        rem_per_stratum[l] += c;
                    }
                }
            }
            let Some((rarest_l, _)) = rem_per_stratum
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, rem)| rem > 0)
                .min_by_key(|&(l, rem)| (rem, total_per_stratum[l], l))
            else {
                break;
            };

            let mut cand_groups: Vec<usize> = (0..groups.len())
                .filter(|&gi| {
                    assigned_split[gi].is_none() && groups[gi].counts.contains_key(&rarest_l)
                })
                .collect();
            // Pack larger duplicate clusters first; tie-break by seeded hash
            cand_groups.sort_by_key(|&gi| {
                (
                    std::cmp::Reverse(groups[gi].member_indices.len()),
                    groups[gi].det_hash,
                )
            });

            for gi in cand_groups {
                if assigned_split[gi].is_some() {
                    continue;
                }
                let g = &groups[gi];
                let w = g.member_indices.len();

                // Rare-class train-first floor: never allow a class to land in eval with 0 in train!
                let chosen_s = if assigned_stratum[train_split_idx][rarest_l] == 0 {
                    train_split_idx
                } else if groups_per_stratum[rarest_l] < num_splits
                    && assigned_stratum[test_split_idx][rarest_l] == 0
                {
                    test_split_idx
                } else {
                    (0..num_splits)
                        .max_by(|&sa, &sb| {
                            let score_split = |s: usize| -> f64 {
                                let stratum_gain: f64 = g
                                    .counts
                                    .iter()
                                    .map(|(&l, &c)| {
                                        let denom = total_per_stratum[l].max(1) as f64;
                                        (desired_stratum[s][l] / denom) * (c as f64)
                                    })
                                    .sum();
                                let cap = desired_items[s].max(1.0);
                                let overflow = ((assigned_items[s] + w) as f64 - desired_items[s])
                                    .max(0.0)
                                    / cap;
                                let size_pull = (desired_items[s] - assigned_items[s] as f64) / cap;
                                stratum_gain - 0.80 * overflow + 1e-3 * size_pull
                            };
                            score_split(sa)
                                .total_cmp(&score_split(sb))
                                .then_with(|| active_splits[sb].2.cmp(&active_splits[sa].2))
                        })
                        .unwrap_or(train_split_idx)
                };

                assigned_split[gi] = Some(chosen_s);
                assigned_items[chosen_s] += w;
                for (&l, &c) in &g.counts {
                    desired_stratum[chosen_s][l] -= c as f64;
                    assigned_stratum[chosen_s][l] += c;
                }
            }
        }

        // Assign any completely unlabeled groups proportionally to remaining split deficits
        let mut unlabeled_groups: Vec<usize> = (0..groups.len())
            .filter(|&gi| assigned_split[gi].is_none())
            .collect();
        unlabeled_groups.sort_by_key(|&gi| {
            (
                std::cmp::Reverse(groups[gi].member_indices.len()),
                groups[gi].det_hash,
            )
        });
        for gi in unlabeled_groups {
            let w = groups[gi].member_indices.len();
            let chosen_s = (0..num_splits)
                .max_by(|&sa, &sb| {
                    let rem_a = (desired_items[sa] - assigned_items[sa] as f64)
                        / desired_items[sa].max(1.0);
                    let rem_b = (desired_items[sb] - assigned_items[sb] as f64)
                        / desired_items[sb].max(1.0);
                    rem_a
                        .total_cmp(&rem_b)
                        .then_with(|| active_splits[sb].2.cmp(&active_splits[sa].2))
                })
                .unwrap_or(train_split_idx);
            assigned_split[gi] = Some(chosen_s);
            assigned_items[chosen_s] += w;
        }

        // Guarantee non-empty train and test when n >= num_splits
        if assigned_items[test_split_idx] == 0
            && let Some(donor_gi) = (0..groups.len())
                .filter(|&gi| assigned_split[gi] == Some(train_split_idx))
                .min_by_key(|&gi| (groups[gi].member_indices.len(), groups[gi].det_hash))
            && assigned_items[train_split_idx] > groups[donor_gi].member_indices.len()
        {
            let w = groups[donor_gi].member_indices.len();
            assigned_split[donor_gi] = Some(test_split_idx);
            assigned_items[train_split_idx] -= w;
            assigned_items[test_split_idx] += w;
        }

        // Build per-split item vectors (deterministically shuffled by `--seed` within each split)
        let mut split_items_map: BTreeMap<String, Vec<DataItem>> = BTreeMap::new();
        let mut split_group_counts: BTreeMap<String, usize> = BTreeMap::new();
        for (name, _, _) in &active_splits {
            split_items_map.insert((*name).to_string(), Vec::new());
            split_group_counts.insert((*name).to_string(), 0);
        }

        for (gi, g) in groups.iter().enumerate() {
            let s_idx = assigned_split[gi].unwrap_or(train_split_idx);
            let s_name = active_splits[s_idx].0;
            *split_group_counts.entry(s_name.to_string()).or_insert(0) += 1;
            let record_group_tag = g.member_indices.len() > 1 || config.group_col.is_some();
            let bucket = split_items_map.get_mut(s_name).unwrap();
            for &item_idx in &g.member_indices {
                let mut item = items[item_idx].clone();
                if record_group_tag {
                    item.metadata.insert(
                        "leakage_group_id".to_string(),
                        serde_json::Value::String(g.group_id.clone()),
                    );
                }
                bucket.push(item);
            }
        }

        let mut rng = DeterministicRng::new(config.seed ^ 0x5350_4C49_545F_5348);
        for vec in split_items_map.values_mut() {
            rng.shuffle(vec);
        }

        // Compute per-split metadata and per-question support matrix
        let mut splits_meta = BTreeMap::new();
        for (name, _, _) in &active_splits {
            let s_items = &split_items_map[*name];
            let jsonl_bytes = serialize_items_jsonl_bytes(s_items)?;
            splits_meta.insert(
                (*name).to_string(),
                SplitFileMeta {
                    file: format!("{name}.jsonl"),
                    sha256: sha256_hex(&jsonl_bytes),
                    item_set_digest: compute_item_set_digest(s_items),
                    state_merkle_sha256: compute_state_merkle_sha256(s_items),
                    items: s_items.len(),
                    groups: split_group_counts.get(*name).copied().unwrap_or(0),
                },
            );
        }

        let per_question_support = compute_per_question_support(&split_items_map, questions);
        let schema_digest = compute_schema_hash(questions);
        let split_id_seed = format!(
            "{source_item_set_digest}|{schema_digest}|{}|{:?}",
            config.seed, ratios
        );
        let split_id = format!("sp_{}", &sha256_hex(split_id_seed.as_bytes())[..12]);

        let manifest = DatasetManifest {
            manifest_version: DATASET_MANIFEST_VERSION.to_string(),
            split_id,
            created_at: format_utc_iso8601(),
            source_sha256,
            source_item_set_digest,
            schema_digest,
            seed: config.seed,
            ratios,
            dedup: DedupSummary {
                group_col: config.group_col.clone(),
                threshold: config.dedup_sim,
                action: config.dedup_action,
                dropped_exact_duplicates,
                clusters_found,
                grouped_items,
                max_cluster_size,
                warnings: dedup_warnings,
            },
            splits: splits_meta,
            per_question_support,
        };

        Ok(SplitOutput {
            splits: split_items_map,
            manifest,
        })
    }

    pub fn format_table(manifest: &DatasetManifest) -> String {
        let mut table = Table::new();
        table
            .load_style(comfy_table::presets::UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Split").fg(Color::Cyan),
                Cell::new("File").fg(Color::White),
                Cell::new("Items").fg(Color::Green),
                Cell::new("Groups").fg(Color::Yellow),
                Cell::new("Item-Set SHA-256 Prefix").fg(Color::DarkGrey),
            ]);
        for name in ["train", "val", "cal", "test"] {
            if let Some(meta) = manifest.splits.get(name) {
                table.add_row(vec![
                    name.to_string(),
                    meta.file.clone(),
                    meta.items.to_string(),
                    meta.groups.to_string(),
                    meta.item_set_digest.chars().take(16).collect(),
                ]);
            }
        }
        format!(
            "\n✂️  Dataset Split Manifest ({}, seed={})\nNear-duplicate clusters grouped: {} ({} items, max cluster={})\n{table}\n",
            manifest.split_id,
            manifest.seed,
            manifest.dedup.clusters_found,
            manifest.dedup.grouped_items,
            manifest.dedup.max_cluster_size
        )
    }
}

fn serialize_items_jsonl_bytes(items: &[DataItem]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    for item in items {
        serde_json::to_writer(&mut buf, item)?;
        buf.push(b'\n');
    }
    Ok(buf)
}

fn extract_stratification_indicators(
    items: &[DataItem],
    questions: &BTreeMap<String, Question>,
) -> (Vec<Vec<usize>>, usize) {
    let mut key_to_idx: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut per_item: Vec<Vec<usize>> = Vec::with_capacity(items.len());

    let mut intern = |q: &str, lbl: String| -> usize {
        let next = key_to_idx.len();
        *key_to_idx.entry((q.to_string(), lbl)).or_insert(next)
    };

    for item in items {
        let mut tags = Vec::new();
        if questions.is_empty() {
            if let Some(gt) = &item.ground_truth {
                for (k, v) in gt {
                    for lbl in extract_canonical_labels_from_value(v, None) {
                        tags.push(intern(k, lbl));
                    }
                }
            }
        } else {
            for (qid, q) in questions {
                if let Some(val) = resolve_ground_truth_for_question(
                    item.ground_truth.as_ref(),
                    qid,
                    q,
                    questions.len(),
                ) {
                    for lbl in extract_canonical_labels_from_value(val, Some(q)) {
                        tags.push(intern(qid, lbl));
                    }
                }
            }
        }
        tags.sort_unstable();
        tags.dedup();
        per_item.push(tags);
    }
    (per_item, key_to_idx.len())
}

fn extract_canonical_labels_from_value(
    val: &serde_json::Value,
    question: Option<&Question>,
) -> Vec<String> {
    match question.map(|q| q.qtype) {
        Some(QuestionType::Choice) => match val {
            serde_json::Value::Array(arr) => {
                if arr.is_empty() {
                    vec!["__empty__".to_string()]
                } else {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(|s| s.trim().to_lowercase()))
                        .filter(|s| !s.is_empty())
                        .collect()
                }
            }
            serde_json::Value::String(s) if !s.trim().is_empty() => {
                vec![s.trim().to_lowercase()]
            }
            _ => Vec::new(),
        },
        Some(QuestionType::Score) => val
            .as_f64()
            .or_else(|| val.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
            .map(|n| vec![format!("{}", n.round() as i64)])
            .unwrap_or_default(),
        Some(QuestionType::Noul) => {
            let b = val
                .as_bool()
                .or_else(|| {
                    val.as_str()
                        .and_then(|s| match s.trim().to_lowercase().as_str() {
                            "true" | "1" | "yes" => Some(true),
                            "false" | "0" | "no" => Some(false),
                            _ => None,
                        })
                })
                .or_else(|| val.as_i64().map(|i| i != 0));
            b.map(|v| vec![v.to_string()]).unwrap_or_default()
        }
        None => match val {
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(value_to_id_string)
                .map(|s| s.to_lowercase())
                .collect(),
            other => value_to_id_string(other)
                .map(|s| vec![s.to_lowercase()])
                .unwrap_or_default(),
        },
    }
}

fn compute_per_question_support(
    splits: &BTreeMap<String, Vec<DataItem>>,
    questions: &BTreeMap<String, Question>,
) -> BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>> {
    let mut out: BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>> = BTreeMap::new();
    for (qid, q) in questions {
        let q_entry = out.entry(qid.clone()).or_default();
        let expected_keys: Vec<String> = match q.qtype {
            QuestionType::Choice => choice_labels(q)
                .into_iter()
                .map(|s| s.to_lowercase())
                .collect(),
            QuestionType::Score => (0..q.render_options().len())
                .map(|i| i.to_string())
                .collect(),
            QuestionType::Noul => vec!["false".to_string(), "true".to_string()],
        };
        for key in expected_keys {
            let split_counts = q_entry.entry(key).or_default();
            for s_name in splits.keys() {
                split_counts.insert(s_name.clone(), 0);
            }
        }
        for (s_name, items) in splits {
            for item in items {
                if let Some(val) = resolve_ground_truth_for_question(
                    item.ground_truth.as_ref(),
                    qid,
                    q,
                    questions.len(),
                ) {
                    for lbl in extract_canonical_labels_from_value(val, Some(q)) {
                        *q_entry
                            .entry(lbl)
                            .or_default()
                            .entry(s_name.clone())
                            .or_insert(0) += 1;
                    }
                }
            }
        }
    }
    out
}

// ============================================================================
// Section 3: 5-Stage Ground-Truth & Split Preflight (`DatasetVerifier`)
// ============================================================================

/// Preflight finding severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CheckSeverity {
    Pass,
    Warn,
    Fail,
}

/// One diagnostic preflight check outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightFinding {
    pub stage: String,
    pub code: String,
    pub severity: CheckSeverity,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remediation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sample_ids: Vec<String>,
}

/// Per-question ground-truth balance & ceiling metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionPreflightStats {
    pub question_id: String,
    pub question_type: String,
    pub labeled_count: usize,
    pub imbalance_ratio: f64,
    pub normalized_entropy: f64,
    pub empirical_bayes_ceiling: f64,
    pub per_split_class_support: BTreeMap<String, BTreeMap<String, usize>>,
}

/// Checkpoint tokenizer and context-budget diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerAuditStats {
    pub checkpoint: String,
    pub tokenizer_mode: String,
    pub max_len: usize,
    pub head_max_len: usize,
    pub per_question_prefix_tokens: BTreeMap<String, usize>,
    pub per_question_state_room: BTreeMap<String, usize>,
    pub min_state_room: usize,
    pub state_tokens_p50: usize,
    pub state_tokens_p90: usize,
    pub state_tokens_p95: usize,
    pub state_tokens_p99: usize,
    pub state_tokens_max: usize,
    pub truncated_items: usize,
    pub truncated_pct: f64,
    pub truncation_label_collisions: usize,
}

/// Configuration for `DatasetVerifier`.
#[derive(Debug, Clone)]
pub struct DatasetVerifyConfig {
    pub checkpoint: Checkpoint,
    pub chunk_strategy: ChunkStrategy,
    pub multi_label: bool,
    pub alpha: f64,
    pub target_accuracy: f64,
    pub defer_confidence: f64,
    pub dedup_sim: f64,
    pub allow_partial_gt: bool,
    pub require_tokenizer: bool,
    pub strict: bool,
}

impl Default for DatasetVerifyConfig {
    fn default() -> Self {
        Self {
            checkpoint: Checkpoint::English,
            chunk_strategy: ChunkStrategy::TruncatePrefix,
            multi_label: false,
            alpha: 0.05,
            target_accuracy: 0.95,
            defer_confidence: 0.95,
            dedup_sim: 0.85,
            allow_partial_gt: false,
            require_tokenizer: false,
            strict: false,
        }
    }
}

/// Complete 5-stage preflight verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetVerifyReport {
    pub ready: bool,
    pub strict: bool,
    pub fail_count: usize,
    pub warn_count: usize,
    pub split_item_counts: BTreeMap<String, usize>,
    pub questions: BTreeMap<String, QuestionPreflightStats>,
    pub tokenizer: TokenizerAuditStats,
    pub findings: Vec<PreflightFinding>,
}

pub struct DatasetVerifier;

impl DatasetVerifier {
    /// Verify a split directory (`{train,val,cal,test}.jsonl` + optional `dataset_manifest.json`).
    pub fn verify_dir(
        config: &DatasetVerifyConfig,
        split_dir: &Path,
        questions: &BTreeMap<String, Question>,
    ) -> Result<DatasetVerifyReport> {
        if !split_dir.exists() || !split_dir.is_dir() {
            bail!("Split directory does not exist: {}", split_dir.display());
        }
        let manifest_path = split_dir.join("dataset_manifest.json");
        let manifest: Option<DatasetManifest> = if manifest_path.exists() {
            Some(serde_json::from_reader(File::open(&manifest_path)?)?)
        } else {
            None
        };

        let mut splits = BTreeMap::new();
        for name in ["train", "val", "cal", "test"] {
            let p = split_dir.join(format!("{name}.jsonl"));
            if p.exists() {
                splits.insert(name.to_string(), read_dataset_items(&p)?);
            }
        }
        if splits.is_empty() {
            bail!(
                "No `{{train,val,cal,test}}.jsonl` split files found in {}",
                split_dir.display()
            );
        }
        Self::verify_splits(
            config,
            &splits,
            questions,
            manifest.as_ref(),
            Some(split_dir),
        )
    }

    /// Verify a single dataset file before splitting or evaluation.
    pub fn verify_file(
        config: &DatasetVerifyConfig,
        file_path: &Path,
        questions: &BTreeMap<String, Question>,
    ) -> Result<DatasetVerifyReport> {
        let items = read_dataset_items(file_path)?;
        let mut splits = BTreeMap::new();
        splits.insert("dataset".to_string(), items);
        Self::verify_splits(config, &splits, questions, None, None)
    }

    /// Run all 5 preflight verification stages on in-memory named splits.
    pub fn verify_splits(
        config: &DatasetVerifyConfig,
        splits: &BTreeMap<String, Vec<DataItem>>,
        questions: &BTreeMap<String, Question>,
        manifest: Option<&DatasetManifest>,
        split_dir: Option<&Path>,
    ) -> Result<DatasetVerifyReport> {
        if questions.is_empty() {
            bail!("Question schema cannot be empty for dataset verification");
        }

        let mut findings = Vec::new();
        let split_item_counts: BTreeMap<String, usize> =
            splits.iter().map(|(k, v)| (k.clone(), v.len())).collect();

        let is_multi_split = splits.contains_key("train") || splits.len() > 1;
        let primary_train_key = if splits.contains_key("train") {
            "train"
        } else {
            splits
                .keys()
                .next()
                .map(String::as_str)
                .unwrap_or("dataset")
        };

        // --------------------------------------------------------------------
        // Stage 1: Schema <-> Label Bijection, Noise & Empirical Bayes Ceiling
        // --------------------------------------------------------------------
        for (s_name, s_items) in splits {
            if s_items.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "EMPTY_SPLIT".into(),
                    severity: CheckSeverity::Fail,
                    message: format!("Split '{s_name}' contains 0 items"),
                    remediation: Some("Increase dataset size or adjust --ratios".into()),
                    sample_ids: vec![],
                });
            }
            let mut seen_ids = HashSet::new();
            let mut dup_ids = Vec::new();
            let mut blank_ids = Vec::new();
            for item in s_items {
                if !seen_ids.insert(item.id.as_str()) && dup_ids.len() < 5 {
                    dup_ids.push(item.id.clone());
                }
                if item.state_text().trim().is_empty() && blank_ids.len() < 5 {
                    blank_ids.push(item.id.clone());
                }
            }
            if !dup_ids.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "DUPLICATE_ITEM_IDS".into(),
                    severity: CheckSeverity::Fail,
                    message: format!("Split '{s_name}' has duplicate item IDs"),
                    remediation: Some(
                        "Run `lemmata dataset import` to canonicalize unique IDs".into(),
                    ),
                    sample_ids: dup_ids,
                });
            }
            if !blank_ids.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "EMPTY_STATE_TEXT".into(),
                    severity: CheckSeverity::Fail,
                    message: format!(
                        "Split '{s_name}' contains items with blank model-visible state"
                    ),
                    remediation: Some("Re-import with `--drop-empty`".into()),
                    sample_ids: blank_ids,
                });
            }
        }

        // Orphan ground-truth keys check
        let mut orphan_keys: BTreeSet<String> = BTreeSet::new();
        for s_items in splits.values() {
            for item in s_items {
                if let Some(gt) = &item.ground_truth {
                    for k in gt.keys() {
                        if k != "default" && !questions.keys().any(|qk| qk.eq_ignore_ascii_case(k))
                        {
                            orphan_keys.insert(k.clone());
                        }
                    }
                }
            }
        }
        if !orphan_keys.is_empty() {
            findings.push(PreflightFinding {
                stage: "schema_bijection".into(),
                code: "ORPHAN_GROUND_TRUTH_KEY".into(),
                severity: CheckSeverity::Warn,
                message: format!(
                    "Ground-truth contains keys absent from schema: {:?}",
                    orphan_keys
                ),
                remediation: Some("Map target keys with `--label-map <qid>=<col>`".into()),
                sample_ids: vec![],
            });
        }

        let all_items_total: usize = splits.values().map(Vec::len).sum();
        let support_map = compute_per_question_support(splits, questions);
        let mut q_stats_map = BTreeMap::new();

        for (qid, q) in questions {
            let valid_choices: Vec<String> = choice_labels(q);
            let valid_lower: HashSet<String> =
                valid_choices.iter().map(|s| s.to_lowercase()).collect();
            let num_score_levels = q.render_options().len();

            let mut labeled_total = 0usize;
            let mut oov_samples = Vec::new();
            let mut malformed_samples = Vec::new();
            let mut score_min = f64::INFINITY;
            let mut score_max = f64::NEG_INFINITY;
            let mut score_out_of_range_ids = Vec::new();
            let mut norm_state_labels: HashMap<String, BTreeMap<String, Vec<String>>> =
                HashMap::new();

            for s_items in splits.values() {
                for item in s_items {
                    let Some(tv) = resolve_ground_truth_for_question(
                        item.ground_truth.as_ref(),
                        qid,
                        q,
                        questions.len(),
                    ) else {
                        continue;
                    };
                    labeled_total += 1;

                    let canonical_repr = match q.qtype {
                        QuestionType::Choice => match tv {
                            serde_json::Value::String(s) => {
                                let lower = s.trim().to_lowercase();
                                if !valid_lower.contains(&lower) {
                                    if oov_samples.len() < 5 {
                                        oov_samples.push(format!("{}:'{}'", item.id, s));
                                    }
                                    None
                                } else {
                                    Some(lower)
                                }
                            }
                            serde_json::Value::Array(arr) => {
                                if !config.multi_label && malformed_samples.len() < 5 {
                                    malformed_samples
                                        .push(format!("{}:array_without_multi_label", item.id));
                                }
                                let mut elems = Vec::new();
                                for el in arr {
                                    if let Some(s) = el.as_str() {
                                        let lower = s.trim().to_lowercase();
                                        if !valid_lower.contains(&lower) {
                                            if oov_samples.len() < 5 {
                                                oov_samples.push(format!("{}:'{}'", item.id, s));
                                            }
                                        } else {
                                            elems.push(lower);
                                        }
                                    } else if malformed_samples.len() < 5 {
                                        malformed_samples.push(item.id.clone());
                                    }
                                }
                                elems.sort();
                                elems.dedup();
                                (!elems.is_empty()).then(|| elems.join("|"))
                            }
                            _ => {
                                if malformed_samples.len() < 5 {
                                    malformed_samples.push(item.id.clone());
                                }
                                None
                            }
                        },
                        QuestionType::Score => {
                            let parsed = tv
                                .as_f64()
                                .or_else(|| tv.as_str().and_then(|s| s.trim().parse::<f64>().ok()));
                            if let Some(v) = parsed
                                && v.is_finite()
                            {
                                score_min = score_min.min(v);
                                score_max = score_max.max(v);
                                if (v < -1e-9
                                    || v > (num_score_levels.saturating_sub(1) as f64) + 1e-9)
                                    && score_out_of_range_ids.len() < 5
                                {
                                    score_out_of_range_ids.push(format!("{}:{v}", item.id));
                                }
                                Some(format!("{}", v.round() as i64))
                            } else {
                                if malformed_samples.len() < 5 {
                                    malformed_samples.push(item.id.clone());
                                }
                                None
                            }
                        }
                        QuestionType::Noul => {
                            let b = tv.as_bool().or_else(|| {
                                tv.as_str()
                                    .and_then(|s| match s.trim().to_lowercase().as_str() {
                                        "true" | "1" => Some(true),
                                        "false" | "0" => Some(false),
                                        _ => None,
                                    })
                            });
                            if let Some(bv) = b {
                                Some(bv.to_string())
                            } else {
                                if malformed_samples.len() < 5 {
                                    malformed_samples.push(item.id.clone());
                                }
                                None
                            }
                        }
                    };

                    if let Some(c_lbl) = canonical_repr {
                        let norm_st = normalize_state_for_leakage(&item.state_text());
                        if !norm_st.is_empty() {
                            norm_state_labels
                                .entry(norm_st)
                                .or_default()
                                .entry(c_lbl)
                                .or_default()
                                .push(item.id.clone());
                        }
                    }
                }
            }

            if labeled_total == 0 {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "NO_GROUND_TRUTH_FOR_QUESTION".into(),
                    severity: CheckSeverity::Fail,
                    message: format!("Question '{qid}' has 0 matched ground-truth labels"),
                    remediation: Some(format!(
                        "Provide `ground_truth.{qid}` or `--label-map {qid}=<col>`"
                    )),
                    sample_ids: vec![],
                });
            } else if !config.allow_partial_gt && labeled_total < all_items_total {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "PARTIAL_GROUND_TRUTH".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' is labeled on {labeled_total}/{all_items_total} items ({:.1}%)",
                        (labeled_total as f64 / all_items_total.max(1) as f64) * 100.0
                    ),
                    remediation: Some("Pass `--allow-partial-gt` if unlabeled rows are expected".into()),
                    sample_ids: vec![],
                });
            }

            if !oov_samples.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "OOV_CHOICE_LABEL".into(),
                    severity: CheckSeverity::Fail,
                    message: format!(
                        "Question '{qid}' has ground-truth labels absent from schema options {:?}",
                        valid_choices
                    ),
                    remediation: Some("Use `--value-map` or update schema criteria keys".into()),
                    sample_ids: oov_samples,
                });
            }
            if !malformed_samples.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "MALFORMED_TARGET_VALUE".into(),
                    severity: CheckSeverity::Fail,
                    message: format!(
                        "Question '{qid}' has malformed or incompatible target values"
                    ),
                    remediation: Some(
                        "Run `lemmata dataset import --strict` to normalize types".into(),
                    ),
                    sample_ids: malformed_samples,
                });
            }
            if !score_out_of_range_ids.is_empty() {
                let is_one_indexed = (score_min - 1.0).abs() < 1e-9
                    && (score_max - num_score_levels as f64).abs() < 1e-9;
                let (code, hint) = if is_one_indexed {
                    (
                        "SCORE_ONE_INDEXED",
                        format!(
                            "Scores span 1..={num_score_levels} for a {num_score_levels}-level rubric (expected 0..={}). Re-import with `--score-offset -1`.",
                            num_score_levels.saturating_sub(1)
                        ),
                    )
                } else {
                    (
                        "SCORE_OUT_OF_RANGE",
                        format!(
                            "Ensure score targets fall within 0..={}",
                            num_score_levels.saturating_sub(1)
                        ),
                    )
                };
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: code.into(),
                    severity: CheckSeverity::Fail,
                    message: format!(
                        "Score question '{qid}' has targets outside [0, {}] (observed min={score_min}, max={score_max})",
                        num_score_levels.saturating_sub(1)
                    ),
                    remediation: Some(hint),
                    sample_ids: score_out_of_range_ids,
                });
            }

            // Conflicting duplicate labels & Empirical Bayes Ceiling
            let mut max_consistent = 0usize;
            let mut total_grouped = 0usize;
            let mut conflict_ids = Vec::new();
            for by_lbl in norm_state_labels.values() {
                let sum_here: usize = by_lbl.values().map(Vec::len).sum();
                let best_here: usize = by_lbl.values().map(Vec::len).max().unwrap_or(0);
                total_grouped += sum_here;
                max_consistent += best_here;
                if by_lbl.len() > 1 && conflict_ids.len() < 6 {
                    for ids in by_lbl.values() {
                        if let Some(first) = ids.first() {
                            conflict_ids.push(first.clone());
                        }
                    }
                }
            }
            let empirical_bayes_ceiling = if total_grouped > 0 {
                max_consistent as f64 / total_grouped as f64
            } else {
                1.0
            };
            if !conflict_ids.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "CONFLICTING_DUPLICATE_LABELS".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' has identical normalized states with conflicting ground-truth labels (empirical Bayes accuracy ceiling: {:.2}%)",
                        empirical_bayes_ceiling * 100.0
                    ),
                    remediation: Some("Audit contradictory annotations on identical text inputs".into()),
                    sample_ids: conflict_ids,
                });
            }

            // Dead schema options & unseen eval classes + imbalance metrics
            let q_sup = support_map.get(qid).cloned().unwrap_or_default();
            let mut class_totals = Vec::new();
            let mut dead_options = Vec::new();
            let mut unseen_eval = Vec::new();

            for (lbl, by_split) in &q_sup {
                let train_cnt = by_split.get(primary_train_key).copied().unwrap_or(0);
                let all_cnt: usize = by_split.values().sum();
                if all_cnt > 0 {
                    class_totals.push(all_cnt);
                }
                if train_cnt == 0 {
                    if is_multi_split && all_cnt > 0 {
                        unseen_eval.push(lbl.clone());
                    } else {
                        dead_options.push(lbl.clone());
                    }
                }
            }

            if !unseen_eval.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "UNSEEN_EVAL_CLASS".into(),
                    severity: CheckSeverity::Fail,
                    message: format!(
                        "Question '{qid}' has classes in val/cal/test with 0 training examples: {:?}",
                        unseen_eval
                    ),
                    remediation: Some("Re-partition with `lemmata dataset split` (train-first floor)".into()),
                    sample_ids: vec![],
                });
            }
            if !dead_options.is_empty() {
                findings.push(PreflightFinding {
                    stage: "schema_bijection".into(),
                    code: "DEAD_SCHEMA_OPTION".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' defines schema options with 0 training occurrences: {:?}",
                        dead_options
                    ),
                    remediation: Some(
                        "Collect training examples for dead options or prune unused options before fine-tuning".into(),
                    ),
                    sample_ids: vec![],
                });
            }

            let (imbalance_ratio, normalized_entropy) =
                compute_imbalance_and_entropy(&class_totals, q_sup.len().max(2));

            // ----------------------------------------------------------------
            // Stage 3 (per-question part): Support Floors & Imbalance Warnings
            // ----------------------------------------------------------------
            for (lbl, by_split) in &q_sup {
                let all_cnt: usize = by_split.values().sum();
                if all_cnt == 0 {
                    continue;
                }
                let tr = by_split.get(primary_train_key).copied().unwrap_or(0);
                if tr > 0 && tr < 2 {
                    findings.push(PreflightFinding {
                        stage: "support_and_imbalance".into(),
                        code: "TRAIN_CLASS_UNDER_2".into(),
                        severity: CheckSeverity::Fail,
                        message: format!(
                            "Question '{qid}' class '{lbl}' has only {tr} example in '{primary_train_key}' (minimum 2)"
                        ),
                        remediation: Some("Add examples or merge ultra-rare label into `other`".into()),
                        sample_ids: vec![],
                    });
                } else if (2..20).contains(&tr) {
                    findings.push(PreflightFinding {
                        stage: "support_and_imbalance".into(),
                        code: "TRAIN_CLASS_LOW_SUPPORT".into(),
                        severity: CheckSeverity::Warn,
                        message: format!(
                            "Question '{qid}' class '{lbl}' has low training support ({tr} < 20)"
                        ),
                        remediation: Some(
                            "Consider LoRA/probe or data augmentation for rare classes".into(),
                        ),
                        sample_ids: vec![],
                    });
                }
                if let Some(&va) = by_split.get("val")
                    && va < 5
                {
                    findings.push(PreflightFinding {
                        stage: "support_and_imbalance".into(),
                        code: "VAL_CLASS_LOW_SUPPORT".into(),
                        severity: CheckSeverity::Warn,
                        message: format!(
                            "Question '{qid}' class '{lbl}' has only {va} validation items (< 5)"
                        ),
                        remediation: Some(
                            "Early stopping macro-F1 may have high variance on this class".into(),
                        ),
                        sample_ids: vec![],
                    });
                }
            }

            if imbalance_ratio > 50.0 {
                findings.push(PreflightFinding {
                    stage: "support_and_imbalance".into(),
                    code: "SEVERE_CLASS_IMBALANCE".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' exhibits severe class imbalance (IR = {imbalance_ratio:.1}:1, H_norm = {normalized_entropy:.3})"
                    ),
                    remediation: Some("Enable class-weighted loss during fine-tuning and gate on Macro-F1".into()),
                    sample_ids: vec![],
                });
            } else if imbalance_ratio > 10.0 || normalized_entropy < 0.35 {
                findings.push(PreflightFinding {
                    stage: "support_and_imbalance".into(),
                    code: "MODERATE_CLASS_IMBALANCE".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' is imbalanced (IR = {imbalance_ratio:.1}:1, H_norm = {normalized_entropy:.3})"
                    ),
                    remediation: Some("Monitor per-label F1 deltas alongside accuracy".into()),
                    sample_ids: vec![],
                });
            }

            q_stats_map.insert(
                qid.clone(),
                QuestionPreflightStats {
                    question_id: qid.clone(),
                    question_type: match q.qtype {
                        QuestionType::Choice => "choice",
                        QuestionType::Score => "score",
                        QuestionType::Noul => "noul",
                    }
                    .to_string(),
                    labeled_count: labeled_total,
                    imbalance_ratio: (imbalance_ratio * 100.0).round() / 100.0,
                    normalized_entropy: (normalized_entropy * 10000.0).round() / 10000.0,
                    empirical_bayes_ceiling: (empirical_bayes_ceiling * 10000.0).round() / 10000.0,
                    per_split_class_support: q_sup,
                },
            );
        }

        // Calibration split exact downstream engine minimums (`calibrate`, `conformal`, `defer`)
        if let Some(cal_items) = splits.get("cal") {
            let n_cal = cal_items.len();
            let min_conformal =
                ((1.0 / config.alpha.clamp(1e-4, 0.5)).ceil() as usize).saturating_sub(1);
            if n_cal < min_conformal {
                findings.push(PreflightFinding {
                    stage: "support_and_imbalance".into(),
                    code: "CAL_CONFORMAL_TRIVIAL_FLOOR".into(),
                    severity: CheckSeverity::Fail,
                    message: format!(
                        "Calibration split has {n_cal} items, below conformal floor {min_conformal} for alpha={:.3} (quantile q_hat degenerates to 1.0)",
                        config.alpha
                    ),
                    remediation: Some("Increase `cal` split ratio or use 3-way `--no-cal`".into()),
                    sample_ids: vec![],
                });
            } else if n_cal < 100 {
                findings.push(PreflightFinding {
                    stage: "support_and_imbalance".into(),
                    code: "CAL_CONFORMAL_HIGH_VARIANCE".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Calibration split has {n_cal} items (< 100 recommended for tight conformal coverage concentration)"
                    ),
                    remediation: None,
                    sample_ids: vec![],
                });
            }

            let alpha_bonf =
                (1.0 - config.defer_confidence.clamp(0.01, 0.9999)) / RISK_GRID.len() as f64;
            let min_defer =
                (alpha_bonf.ln() / config.target_accuracy.clamp(0.01, 0.9999).ln()).ceil() as usize;
            if n_cal < min_defer {
                findings.push(PreflightFinding {
                    stage: "support_and_imbalance".into(),
                    code: "CAL_DEFER_UNCERTIFIABLE".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Calibration split ({n_cal} items) is below `lemmata defer fit` certification floor ({min_defer} items at {:.0}% target acc / {:.0}% confidence)",
                        config.target_accuracy * 100.0,
                        config.defer_confidence * 100.0
                    ),
                    remediation: Some(
                        "Allocate >= 120 items to `cal` if certifying System 1 -> System 2 deferral thresholds".into(),
                    ),
                    sample_ids: vec![],
                });
            }

            let dummy_cls = crate::engine::LayaClassifier::with_checkpoint(config.checkpoint);
            let mut bucket_counts: BTreeMap<String, usize> = BTreeMap::new();
            for item in cal_items {
                for (qid, q) in questions {
                    if resolve_ground_truth_for_question(
                        item.ground_truth.as_ref(),
                        qid,
                        q,
                        questions.len(),
                    )
                    .is_some()
                    {
                        let bkey = dummy_cls
                            .get_temperature_bucket(q.qtype, q.render_options().len().max(2));
                        *bucket_counts.entry(bkey).or_insert(0) += 1;
                    }
                }
            }
            for (bkey, bcnt) in bucket_counts {
                if bcnt < DEFAULT_MIN_BUCKET_SAMPLES {
                    findings.push(PreflightFinding {
                        stage: "support_and_imbalance".into(),
                        code: "CAL_BUCKET_UNDER_30".into(),
                        severity: CheckSeverity::Warn,
                        message: format!(
                            "Calibration bucket '{bkey}' has {bcnt} labeled decisions (< {DEFAULT_MIN_BUCKET_SAMPLES} minimum for temperature fitting)"
                        ),
                        remediation: Some("Bucket will retain prior temperature during `lemmata calibrate`".into()),
                        sample_ids: vec![],
                    });
                }
            }
        }

        // --------------------------------------------------------------------
        // Stage 2: Structured State -> Target Mutual Information Leakage Audit
        // --------------------------------------------------------------------
        audit_structured_state_leakage(splits, questions, &mut findings);

        // --------------------------------------------------------------------
        // Stage 4: Checkpoint Tokenizer, Head Budget & Truncation Collision Audit
        // --------------------------------------------------------------------
        let tokenizer_stats =
            audit_tokenizer_and_truncation(config, splits, questions, &mut findings)?;

        // --------------------------------------------------------------------
        // Stage 5: Cross-Split Contamination & Manifest Integrity Audit
        // --------------------------------------------------------------------
        if let (Some(man), Some(dir)) = (manifest, split_dir) {
            for meta in man.splits.values() {
                let p = dir.join(&meta.file);
                if p.exists()
                    && let Ok(actual_sha) = file_sha256_hex(&p)
                    && actual_sha != meta.sha256
                {
                    findings.push(PreflightFinding {
                        stage: "cross_split_leakage".into(),
                        code: "MANIFEST_DIGEST_MISMATCH".into(),
                        severity: CheckSeverity::Fail,
                        message: format!(
                            "Split file '{}' SHA-256 differs from dataset_manifest.json (expected {}..., got {}...)",
                            meta.file,
                            &meta.sha256[..12.min(meta.sha256.len())],
                            &actual_sha[..12.min(actual_sha.len())]
                        ),
                        remediation: Some("Re-run `lemmata dataset split` instead of editing split files in place".into()),
                        sample_ids: vec![],
                    });
                }
            }
        }

        if splits.len() > 1 {
            audit_cross_split_leakage(splits, config.dedup_sim, &mut findings);
        }

        let fail_count = findings
            .iter()
            .filter(|f| f.severity == CheckSeverity::Fail)
            .count();
        let warn_count = findings
            .iter()
            .filter(|f| f.severity == CheckSeverity::Warn)
            .count();
        let ready = fail_count == 0 && (!config.strict || warn_count == 0);

        Ok(DatasetVerifyReport {
            ready,
            strict: config.strict,
            fail_count,
            warn_count,
            split_item_counts,
            questions: q_stats_map,
            tokenizer: tokenizer_stats,
            findings,
        })
    }

    pub fn format_table(report: &DatasetVerifyReport) -> String {
        let badge = if report.ready {
            "✅ READY"
        } else {
            "❌ FAILED PREFLIGHT"
        };
        let mut out = format!(
            "\n🛡️  Lemmata Dataset Preflight Verification — {badge} (Fails: {}, Warnings: {})\n",
            report.fail_count, report.warn_count
        );

        let mut q_table = Table::new();
        q_table
            .load_style(comfy_table::presets::UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Question").fg(Color::Cyan),
                Cell::new("Type").fg(Color::White),
                Cell::new("Labeled").fg(Color::Green),
                Cell::new("Imbalance (IR)").fg(Color::Yellow),
                Cell::new("Entropy H~").fg(Color::White),
                Cell::new("Bayes Ceiling").fg(Color::Magenta),
                Cell::new("State Room").fg(Color::Blue),
            ]);
        for (qid, qs) in &report.questions {
            let room = report
                .tokenizer
                .per_question_state_room
                .get(qid)
                .copied()
                .unwrap_or(report.tokenizer.min_state_room);
            q_table.add_row(vec![
                qid.clone(),
                qs.question_type.clone(),
                qs.labeled_count.to_string(),
                format!("{:.1}:1", qs.imbalance_ratio),
                format!("{:.3}", qs.normalized_entropy),
                format!("{:.2}%", qs.empirical_bayes_ceiling * 100.0),
                format!("{room} tok"),
            ]);
        }
        out.push_str(&q_table.to_string());
        out.push('\n');

        let t = &report.tokenizer;
        out.push_str(&format!(
            "\n📏 Tokenizer Audit ({} via `{}`): max_len={}, head_budget={}, min_state_room={}\n   State tokens p50={} | p90={} | p95={} | p99={} | max={} | truncated={} ({:.1}%) | collisions={}\n",
            t.checkpoint,
            t.tokenizer_mode,
            t.max_len,
            t.head_max_len,
            t.min_state_room,
            t.state_tokens_p50,
            t.state_tokens_p90,
            t.state_tokens_p95,
            t.state_tokens_p99,
            t.state_tokens_max,
            t.truncated_items,
            t.truncated_pct,
            t.truncation_label_collisions
        ));

        if report.findings.is_empty() {
            out.push_str("\n✨ All 5 preflight stages passed with zero warnings.\n");
        } else {
            let mut f_table = Table::new();
            f_table
                .load_style(comfy_table::presets::UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_header(vec![
                    Cell::new("Severity").fg(Color::Cyan),
                    Cell::new("Stage / Code").fg(Color::White),
                    Cell::new("Finding & Remediation").fg(Color::Yellow),
                ]);
            for f in &report.findings {
                let sev_cell = match f.severity {
                    CheckSeverity::Fail => Cell::new("FAIL").fg(Color::Red),
                    CheckSeverity::Warn => Cell::new("WARN").fg(Color::Yellow),
                    CheckSeverity::Pass => Cell::new("PASS").fg(Color::Green),
                };
                let mut detail = f.message.clone();
                if !f.sample_ids.is_empty() {
                    detail.push_str(&format!(" [samples: {}]", f.sample_ids.join(", ")));
                }
                if let Some(rem) = &f.remediation {
                    detail.push_str(&format!("\n→ Fix: {rem}"));
                }
                f_table.add_row(vec![
                    sev_cell,
                    Cell::new(format!("{}\n{}", f.stage, f.code)),
                    Cell::new(detail),
                ]);
            }
            out.push('\n');
            out.push_str(&f_table.to_string());
            out.push('\n');
        }
        out
    }

    pub fn format_json(report: &DatasetVerifyReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }
}

fn compute_imbalance_and_entropy(counts: &[usize], num_classes: usize) -> (f64, f64) {
    if counts.is_empty() {
        return (1.0, 0.0);
    }
    let max_c = *counts.iter().max().unwrap_or(&1) as f64;
    let min_c = (*counts.iter().min().unwrap_or(&1)).max(1) as f64;
    let ir = max_c / min_c;
    let total: f64 = counts.iter().sum::<usize>().max(1) as f64;
    let ent: f64 = -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            p * p.ln()
        })
        .sum::<f64>();
    let denom = (num_classes.max(2) as f64).ln();
    (ir, (ent / denom).clamp(0.0, 1.0))
}

fn audit_structured_state_leakage(
    splits: &BTreeMap<String, Vec<DataItem>>,
    questions: &BTreeMap<String, Question>,
    findings: &mut Vec<PreflightFinding>,
) {
    let all_items: Vec<&DataItem> = splits.values().flat_map(|v| v.iter()).collect();
    let mut suspect_key_hits: BTreeSet<String> = BTreeSet::new();
    let mut field_values: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();

    for (idx, item) in all_items.iter().enumerate() {
        let Some(obj) = item.state.as_object() else {
            continue;
        };
        for (k, v) in obj {
            let k_low = k.to_ascii_lowercase();
            if k_low == "label"
                || k_low == "target"
                || k_low == "ground_truth"
                || strip_gt_prefix_preserve_case(k).is_some()
                || questions.keys().any(|qid| qid.eq_ignore_ascii_case(k))
            {
                suspect_key_hits.insert(k.clone());
            }
            if let Some(s) = value_to_id_string(v) {
                field_values.entry(k.clone()).or_default().push((idx, s));
            }
        }
    }

    if !suspect_key_hits.is_empty() {
        findings.push(PreflightFinding {
            stage: "feature_leakage".into(),
            code: "TARGET_KEY_IN_STATE".into(),
            severity: CheckSeverity::Fail,
            message: format!(
                "Model-visible `state` contains target-named keys: {:?}",
                suspect_key_hits
            ),
            remediation: Some(
                "Re-import with explicit `--text-col` / `--state-cols` or `--label-map` so labels do not leak into `state`".into(),
            ),
            sample_ids: vec![],
        });
    }

    // NMI check for categorical/discrete structured state fields against each question's ground truth
    for (field_name, indexed_vals) in &field_values {
        if indexed_vals.len() < 10 {
            continue;
        }
        for (qid, q) in questions {
            let mut pairs: Vec<(&str, String)> = Vec::new();
            let mut exact_label_matches = 0usize;
            for (item_idx, fval) in indexed_vals {
                let item = all_items[*item_idx];
                if let Some(tv) = resolve_ground_truth_for_question(
                    item.ground_truth.as_ref(),
                    qid,
                    q,
                    questions.len(),
                ) && let Some(lbl) = extract_canonical_labels_from_value(tv, Some(q))
                    .into_iter()
                    .next()
                {
                    if fval.trim().eq_ignore_ascii_case(&lbl) {
                        exact_label_matches += 1;
                    }
                    pairs.push((fval.as_str(), lbl));
                }
            }
            if pairs.len() < 10 {
                continue;
            }
            let n = pairs.len();
            let unique_x: HashSet<&str> = pairs.iter().map(|(x, _)| *x).collect();
            let unique_y: HashSet<&str> = pairs.iter().map(|(_, y)| y.as_str()).collect();
            if unique_x.len() < 2 || unique_y.len() < 2 {
                continue;
            }
            let exact_match_rate = exact_label_matches as f64 / n as f64;
            // Skip high-cardinality unique ID/free-text fields unless they literally match label strings
            if unique_x.len() > (unique_y.len() * 3).max(n / 2) && exact_match_rate < 0.80 {
                continue;
            }
            let nmi = normalized_mutual_information(&pairs);
            if nmi > 0.95 || exact_match_rate > 0.90 {
                findings.push(PreflightFinding {
                    stage: "feature_leakage".into(),
                    code: "HIGH_STATE_TARGET_NMI".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Structured state field '{field_name}' has near-deterministic correlation with '{qid}' (NMI={nmi:.3}, exact_match={:.1}%)",
                        exact_match_rate * 100.0
                    ),
                    remediation: Some(format!(
                        "Verify '{field_name}' is genuinely available at production inference time; otherwise move to `--meta-cols {field_name}`"
                    )),
                    sample_ids: vec![],
                });
            }
        }
    }
}

fn normalized_mutual_information(pairs: &[(&str, String)]) -> f64 {
    let n = pairs.len() as f64;
    let mut cx: BTreeMap<&str, usize> = BTreeMap::new();
    let mut cy: BTreeMap<&str, usize> = BTreeMap::new();
    let mut cxy: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for (x, y) in pairs {
        *cx.entry(*x).or_insert(0) += 1;
        *cy.entry(y.as_str()).or_insert(0) += 1;
        *cxy.entry((*x, y.as_str())).or_insert(0) += 1;
    }
    let hx: f64 = -cx
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p.ln()
        })
        .sum::<f64>();
    let hy: f64 = -cy
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            p * p.ln()
        })
        .sum::<f64>();
    if hx + hy < 1e-12 {
        return 0.0;
    }
    let mut mi = 0.0;
    for (&(x, y), &joint_c) in &cxy {
        let pxy = joint_c as f64 / n;
        let px = cx[x] as f64 / n;
        let py = cy[y] as f64 / n;
        mi += pxy * (pxy / (px * py)).ln();
    }
    ((2.0 * mi) / (hx + hy)).clamp(0.0, 1.0)
}

fn audit_tokenizer_and_truncation(
    config: &DatasetVerifyConfig,
    splits: &BTreeMap<String, Vec<DataItem>>,
    questions: &BTreeMap<String, Question>,
    findings: &mut Vec<PreflightFinding>,
) -> Result<TokenizerAuditStats> {
    let ckpt = config.checkpoint;
    let dir = ckpt.local_dir();
    let tok_path = [
        dir.join("tokenizer/tokenizer.json"),
        dir.join("tokenizer.json"),
    ]
    .into_iter()
    .find(|p| p.exists());

    let tokenizer = match tok_path {
        Some(ref p) => Some(crate::model::native::load_tokenizer(p)?),
        None => {
            if config.require_tokenizer {
                bail!(
                    "Tokenizer required (`--require-tokenizer`), but no tokenizer.json found in {}",
                    dir.display()
                );
            }
            None
        }
    };

    let (max_len, head_max_len) = resolve_ckpt_token_limits(ckpt, &dir);
    let tokenizer_mode = if tokenizer.is_some() {
        "checkpoint_tokenizer".to_string()
    } else {
        "heuristic_estimate".to_string()
    };

    let count_tokens = |text: &str| -> Result<Vec<u32>> {
        if let Some(tok) = &tokenizer {
            Ok(tok
                .encode(text, false)
                .map_err(|e| anyhow!(e))?
                .get_ids()
                .to_vec())
        } else {
            // Deterministic pseudo-token sequence matching `estimate_tokens` length for collision checks
            let mut ids = Vec::new();
            for word in text.split_whitespace() {
                let h = fnv1a64(word.as_bytes()) as u32;
                ids.push(h);
                if word.len() > 6 {
                    ids.push(h.wrapping_add(1));
                }
            }
            for ch in text.chars() {
                if ch as u32 > 0x024F && !ch.is_whitespace() {
                    ids.push(ch as u32);
                }
            }
            if ids.is_empty() && !text.trim().is_empty() {
                ids.push(fnv1a64(text.as_bytes()) as u32);
            }
            Ok(ids)
        }
    };

    let is_laya = Checkpoint::laya_checkpoints().contains(&ckpt) || ckpt == Checkpoint::AutoRoute;
    let mut per_question_prefix_tokens = BTreeMap::new();
    let mut per_question_state_room = BTreeMap::new();

    for (qid, q) in questions {
        let options = q.render_options();
        let qname = match q.qtype {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        };
        if is_laya {
            let head_ids = count_tokens(&format!("{qname} question: {}", q.instructions))?;
            let mut opt_lens = Vec::with_capacity(options.len());
            let mut clipped_opts = Vec::new();
            for opt in &options {
                let raw_len = if tokenizer.is_some() {
                    count_tokens(&format!(" {opt}"))?.len()
                } else {
                    estimate_tokens(opt)
                };
                if raw_len > MAX_LAYA_OPTION_TOKENS {
                    clipped_opts.push(opt.split(':').next().unwrap_or(opt).trim().to_string());
                }
                opt_lens.push(1 + raw_len.min(MAX_LAYA_OPTION_TOKENS));
            }
            if !clipped_opts.is_empty() {
                findings.push(PreflightFinding {
                    stage: "tokenizer_and_truncation".into(),
                    code: "OPTION_EXCEEDS_48_TOKENS".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' has options exceeding Laya's 48-token per-option cap: {:?}",
                        clipped_opts
                    ),
                    remediation: Some("Run `lemmata schema optimize` to compact verbose option criteria".into()),
                    sample_ids: vec![],
                });
            }
            let total_opt_tokens: usize = opt_lens.iter().sum();
            if total_opt_tokens + 16 > head_max_len {
                findings.push(PreflightFinding {
                    stage: "tokenizer_and_truncation".into(),
                    code: "HEAD_BUDGET_TOURNAMENT_FALLBACK".into(),
                    severity: CheckSeverity::Warn,
                    message: format!(
                        "Question '{qid}' options ({total_opt_tokens} tokens + 16 header > {head_max_len} budget) trigger multi-round tournament scoring"
                    ),
                    remediation: Some(
                        "Run `lemmata schema optimize` or switch to an NLI checkpoint for >20 options".into(),
                    ),
                    sample_ids: vec![],
                });
            }
            let avail_for_head = head_max_len.saturating_sub(total_opt_tokens).max(8);
            let head_len = head_ids.len().min(avail_for_head);
            // `[CLS]` + head + `[SEP]` + options + `[SEP]`
            let prefix_len = 1 + head_len + 1 + total_opt_tokens + 1;
            let state_room = max_len.saturating_sub(prefix_len + 1);
            per_question_prefix_tokens.insert(qid.clone(), prefix_len);
            per_question_state_room.insert(qid.clone(), state_room);
        } else {
            // NLI / MLM: `[CLS] premise [SEP] hypothesis [SEP]`
            let max_hyp = options
                .iter()
                .map(|o| count_tokens(o).map(|v| v.len()).unwrap_or(12))
                .max()
                .unwrap_or(12);
            let prefix_len = max_hyp + 3;
            let state_room = max_len.saturating_sub(prefix_len);
            per_question_prefix_tokens.insert(qid.clone(), prefix_len);
            per_question_state_room.insert(qid.clone(), state_room);
        }
    }

    let min_state_room = per_question_state_room
        .values()
        .copied()
        .min()
        .unwrap_or(max_len.saturating_sub(32))
        .max(1);

    let all_items: Vec<&DataItem> = splits.values().flat_map(|v| v.iter()).collect();
    let mut lengths = Vec::with_capacity(all_items.len());
    let mut truncated_items = 0usize;
    let mut truncated_sample_ids = Vec::new();

    // Map `visible_window_digest -> (full_state_norm_set, canonical_gt_map, sample_id)`
    let mut visible_buckets: HashMap<String, Vec<(String, String, String)>> = HashMap::new();

    for item in &all_items {
        let state_str = item.state_text();
        let ids = count_tokens(&state_str)?;
        let len = if tokenizer.is_some() {
            ids.len()
        } else {
            estimate_tokens(&state_str).max(ids.len())
        };
        lengths.push(len);

        if len > min_state_room {
            truncated_items += 1;
            if truncated_sample_ids.len() < 5 {
                truncated_sample_ids.push(format!("{}({len}t)", item.id));
            }
        }

        let windows =
            crate::model::native::state_windows(&ids, min_state_room, config.chunk_strategy);
        let mut vis_hasher = Sha256::new();
        for w in &windows {
            for &tok in w {
                vis_hasher.update(tok.to_le_bytes());
            }
            vis_hasher.update([0xff, 0x00, 0xff, 0x00]);
        }
        let vis_digest = to_hex(&vis_hasher.finalize());
        let full_norm = normalize_state_for_leakage(&state_str);
        let gt_canon = serde_json::to_string(&item.ground_truth).unwrap_or_default();
        visible_buckets
            .entry(vis_digest)
            .or_default()
            .push((full_norm, gt_canon, item.id.clone()));
    }

    let mut truncation_label_collisions = 0usize;
    let mut collision_ids = Vec::new();
    for group in visible_buckets.values() {
        if group.len() < 2 {
            continue;
        }
        let distinct_full: HashSet<&str> = group.iter().map(|(f, _, _)| f.as_str()).collect();
        let distinct_gt: HashSet<&str> = group.iter().map(|(_, g, _)| g.as_str()).collect();
        // Collision specifically induced by truncation/chunking: full texts differed, yet visible
        // truncated windows are identical and carry opposing labels!
        if distinct_full.len() > 1 && distinct_gt.len() > 1 {
            truncation_label_collisions += 1;
            if collision_ids.len() < 6 {
                for (_, _, id) in group.iter().take(2) {
                    collision_ids.push(id.clone());
                }
            }
        }
    }

    lengths.sort_unstable();
    let pct_idx = |p: f64| -> usize {
        if lengths.is_empty() {
            return 0;
        }
        let idx = ((lengths.len() - 1) as f64 * p).round() as usize;
        lengths[idx.min(lengths.len() - 1)]
    };

    let truncated_pct = if all_items.is_empty() {
        0.0
    } else {
        (truncated_items as f64 / all_items.len() as f64) * 100.0
    };

    if truncated_items > 0 && config.chunk_strategy == ChunkStrategy::TruncatePrefix {
        findings.push(PreflightFinding {
            stage: "tokenizer_and_truncation".into(),
            code: "STATE_TRUNCATION_DETECTED".into(),
            severity: CheckSeverity::Warn,
            message: format!(
                "{truncated_items}/{} items ({truncated_pct:.1}%) exceed checkpoint state_room ({min_state_room} tokens after prefix, max_len={max_len}) under `truncate-prefix`",
                all_items.len()
            ),
            remediation: Some(
                "Use `--chunk-strategy head-tail` or `sliding-window` so tail incidents are not clipped".into(),
            ),
            sample_ids: truncated_sample_ids,
        });
    }

    if truncation_label_collisions > 0 {
        findings.push(PreflightFinding {
            stage: "tokenizer_and_truncation".into(),
            code: "TRUNCATION_LABEL_COLLISION".into(),
            severity: CheckSeverity::Fail,
            message: format!(
                "{truncation_label_collisions} group(s) of distinct inputs collapse to identical visible tokens within {min_state_room} state_room while carrying opposing ground-truth labels"
            ),
            remediation: Some(
                "Switch to `--chunk-strategy head-tail` / `sliding-window` or a longer-context checkpoint (`multilingual` / `typed-decisions` / `neobert`)".into(),
            ),
            sample_ids: collision_ids,
        });
    }

    Ok(TokenizerAuditStats {
        checkpoint: ckpt.repo_id().to_string(),
        tokenizer_mode,
        max_len,
        head_max_len,
        per_question_prefix_tokens,
        per_question_state_room,
        min_state_room,
        state_tokens_p50: pct_idx(0.50),
        state_tokens_p90: pct_idx(0.90),
        state_tokens_p95: pct_idx(0.95),
        state_tokens_p99: pct_idx(0.99),
        state_tokens_max: lengths.last().copied().unwrap_or(0),
        truncated_items,
        truncated_pct: (truncated_pct * 100.0).round() / 100.0,
        truncation_label_collisions,
    })
}

fn resolve_ckpt_token_limits(ckpt: Checkpoint, dir: &Path) -> (usize, usize) {
    let agent_cfg_path = dir.join("rl_agent_config.json");
    if let Ok(text) = std::fs::read_to_string(&agent_cfg_path)
        && let Ok(val) = serde_json::from_str::<serde_json::Value>(&text)
    {
        let m = val
            .get("max_len")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| ckpt.context_window());
        let h = val
            .get("head_max_len")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| ckpt.head_budget());
        return (m, h);
    }
    // Both NLI zero-shot checkpoints were trained at 512 tokens (`DEFAULT_NLI_MAX_LEN` in `native.rs`)
    let max_len = match ckpt {
        Checkpoint::ModernBertZeroShot | Checkpoint::DebertaV3ZeroShot => 512,
        other => other.context_window(),
    };
    (max_len, ckpt.head_budget())
}

fn audit_cross_split_leakage(
    splits: &BTreeMap<String, Vec<DataItem>>,
    dedup_sim: f64,
    findings: &mut Vec<PreflightFinding>,
) {
    let split_names: Vec<&String> = splits.keys().collect();
    let mut id_leaks = Vec::new();
    let mut exact_leaks = Vec::new();
    let mut norm_leaks = Vec::new();

    // Map hashes to `(first_split, first_item_id)`
    let mut seen_ids: HashMap<&str, (&str, &str)> = HashMap::new();
    let mut seen_exact: HashMap<String, (&str, &str)> = HashMap::new();
    let mut seen_norm: HashMap<String, (&str, &str)> = HashMap::new();

    for s_name in &split_names {
        for item in &splits[*s_name] {
            if let Some(&(prev_s, _)) = seen_ids.get(item.id.as_str()) {
                if prev_s != s_name.as_str() && id_leaks.len() < 5 {
                    id_leaks.push(format!("{}({prev_s}<->{s_name})", item.id));
                }
            } else {
                seen_ids.insert(item.id.as_str(), (s_name.as_str(), item.id.as_str()));
            }

            let raw = item.state_text();
            let ex_h = sha256_hex(raw.as_bytes());
            if let Some(&(prev_s, prev_id)) = seen_exact.get(&ex_h) {
                if prev_s != s_name.as_str() && exact_leaks.len() < 5 {
                    exact_leaks.push(format!("{prev_s}:{prev_id}<->{s_name}:{}", item.id));
                }
            } else {
                seen_exact.insert(ex_h, (s_name.as_str(), item.id.as_str()));
            }

            let tmpl = normalize_template_for_grouping(&raw);
            if !tmpl.is_empty() {
                let tm_h = sha256_hex(tmpl.as_bytes());
                if let Some(&(prev_s, prev_id)) = seen_norm.get(&tm_h) {
                    if prev_s != s_name.as_str() && norm_leaks.len() < 5 {
                        norm_leaks.push(format!("{prev_s}:{prev_id}<->{s_name}:{}", item.id));
                    }
                } else {
                    seen_norm.insert(tm_h, (s_name.as_str(), item.id.as_str()));
                }
            }
        }
    }

    if !id_leaks.is_empty() {
        findings.push(PreflightFinding {
            stage: "cross_split_leakage".into(),
            code: "CROSS_SPLIT_ID_OVERLAP".into(),
            severity: CheckSeverity::Fail,
            message: "Identical item IDs appear in multiple splits".into(),
            remediation: Some("Re-partition dataset using `lemmata dataset split`".into()),
            sample_ids: id_leaks,
        });
    }
    if !exact_leaks.is_empty() {
        findings.push(PreflightFinding {
            stage: "cross_split_leakage".into(),
            code: "CROSS_SPLIT_EXACT_LEAKAGE".into(),
            severity: CheckSeverity::Fail,
            message: "Exact duplicate state texts straddle multiple splits".into(),
            remediation: Some(
                "Re-run `lemmata dataset split` so duplicate clusters stay in one split".into(),
            ),
            sample_ids: exact_leaks,
        });
    }
    if !norm_leaks.is_empty() {
        findings.push(PreflightFinding {
            stage: "cross_split_leakage".into(),
            code: "CROSS_SPLIT_NORMALIZED_LEAKAGE".into(),
            severity: CheckSeverity::Fail,
            message: "Normalized template duplicates (differing only in numbers/emails/URLs/casing) straddle splits".into(),
            remediation: Some("Re-run `lemmata dataset split` to group template variants together".into()),
            sample_ids: norm_leaks,
        });
    }

    if dedup_sim > 0.0 && dedup_sim <= 1.0 {
        let mh = MinHashIndex::new(42);
        let mut all_sigs: Vec<(&str, &str, [u64; MINHASH_NUM_PERM])> = Vec::new();
        for s_name in &split_names {
            for item in &splits[*s_name] {
                if let Some(sig) = mh.signature(&normalize_state_for_leakage(&item.state_text())) {
                    all_sigs.push((s_name.as_str(), item.id.as_str(), sig));
                }
            }
        }
        let mut buckets: BTreeMap<(usize, u64), Vec<usize>> = BTreeMap::new();
        for (idx, (_, _, sig)) in all_sigs.iter().enumerate() {
            for b in 0..LSH_BANDS {
                buckets
                    .entry((b, MinHashIndex::band_hash(sig, b)))
                    .or_default()
                    .push(idx);
            }
        }
        let mut seen_pair = HashSet::new();
        let mut near_leaks = Vec::new();
        for members in buckets.values() {
            if members.len() < 2 {
                continue;
            }
            for (pos, &i) in members.iter().enumerate() {
                let window_end = (pos + 64).min(members.len());
                for &j in &members[pos + 1..window_end] {
                    let (sa, ida, ref siga) = all_sigs[i];
                    let (sb, idb, ref sigb) = all_sigs[j];
                    if sa == sb {
                        continue;
                    }
                    let pair_key = if i < j { (i, j) } else { (j, i) };
                    if !seen_pair.insert(pair_key) {
                        continue;
                    }
                    let sim = MinHashIndex::jaccard_estimate(siga, sigb);
                    if sim >= dedup_sim && near_leaks.len() < 5 {
                        near_leaks.push(format!("{sa}:{ida}<->{sb}:{idb}(J={sim:.2})"));
                    }
                }
            }
        }
        if !near_leaks.is_empty() {
            findings.push(PreflightFinding {
                stage: "cross_split_leakage".into(),
                code: "CROSS_SPLIT_NEAR_DUPLICATE_LEAKAGE".into(),
                severity: CheckSeverity::Fail,
                message: format!(
                    "Near-duplicate texts (MinHash Jaccard >= {dedup_sim:.2}) straddle splits"
                ),
                remediation: Some(format!(
                    "Re-run `lemmata dataset split --dedup-sim {dedup_sim:.2}`"
                )),
                sample_ids: near_leaks,
            });
        }
    }
}
