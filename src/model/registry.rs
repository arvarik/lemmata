use anyhow::{Context, Result, bail};
use comfy_table::{Cell, Color, Table};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::head_dispatch::HeadParadigm;
use super::safetensors::inspect_safetensors;

/// Dynamically inspected Hugging Face / Laya checkpoint specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicModelSpec {
    pub model_id: String,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub detected_paradigm: HeadParadigm,
    pub total_parameters: usize,
    pub disk_usage_bytes: u64,
    pub is_valid: bool,
    pub local_path: PathBuf,
}

/// Non-recursively sum regular file sizes in `dir` so parent directories like `models/laya`
/// do not double-count nested checkpoint subdirectories (`multilingual`, `typed-decisions`).
pub fn compute_dir_size_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

fn read_json_opt(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

fn canonical_model_id(dir: &Path) -> Option<&'static str> {
    let norm = dir.to_string_lossy().replace('\\', "/");
    let trimmed = norm.trim_end_matches('/');
    if trimmed.ends_with("models/laya/multilingual") {
        Some("convaiinnovations/laya-multilingual")
    } else if trimmed.ends_with("models/laya/typed-decisions") {
        Some("convaiinnovations/laya-typed-decisions")
    } else if trimmed.ends_with("models/laya") {
        Some("convaiinnovations/laya")
    } else if trimmed.ends_with("models/modernbert-zeroshot") {
        Some("MoritzLaurer/ModernBERT-large-zeroshot-v2.0")
    } else if trimmed.ends_with("models/deberta-v3") {
        Some("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    } else if trimmed.ends_with("models/eurobert") {
        Some("EuroBERT/EuroBERT-210m")
    } else if trimmed.ends_with("models/neobert") {
        Some("chandar-lab/NeoBERT")
    } else {
        None
    }
}

/// Inspect a local checkpoint directory (`config.json`, `encoder/config.json`,
/// `rl_agent_config.json`, `gliclass_config.json`, and `model.safetensors`) and auto-detect
/// its classification head paradigm and architecture hyperparameters.
pub fn inspect_model_directory(dir: &Path) -> Result<DynamicModelSpec> {
    if !dir.exists() {
        bail!("Model directory does not exist: {}", dir.display());
    }

    let root_cfg_path = dir.join("config.json");
    let enc_cfg_path = dir.join("encoder/config.json");
    let rl_cfg_path = dir.join("rl_agent_config.json");
    let gliclass_cfg_path = dir.join("gliclass_config.json");

    let cfg = read_json_opt(&root_cfg_path)
        .or_else(|| read_json_opt(&enc_cfg_path))
        .unwrap_or(Value::Null);
    let rl_cfg = read_json_opt(&rl_cfg_path).unwrap_or(Value::Null);
    let gliclass_cfg = read_json_opt(&gliclass_cfg_path).unwrap_or(Value::Null);

    if cfg.is_null() && rl_cfg.is_null() && gliclass_cfg.is_null() {
        bail!(
            "No valid config.json, encoder/config.json, rl_agent_config.json, or gliclass_config.json in {}",
            dir.display()
        );
    }

    let get_val = |key: &str| -> Option<&Value> {
        cfg.get(key)
            .or_else(|| cfg.get("encoder_config").and_then(|v| v.get(key)))
            .or_else(|| gliclass_cfg.get(key))
            .or_else(|| gliclass_cfg.get("encoder_config").and_then(|v| v.get(key)))
            .or_else(|| rl_cfg.get(key))
    };

    let get_usize =
        |key: &str| -> Option<usize> { get_val(key).and_then(|v| v.as_u64()).map(|n| n as usize) };

    let get_str = |key: &str| -> Option<String> {
        get_val(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    };

    let model_type = get_str("model_type").unwrap_or_else(|| {
        if !gliclass_cfg.is_null() {
            "gliclass".to_string()
        } else if !rl_cfg.is_null() {
            "laya".to_string()
        } else {
            "unknown".to_string()
        }
    });

    let mut architectures: Vec<String> = get_val("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if architectures.is_empty() {
        if !gliclass_cfg.is_null() {
            architectures.push("GLiClassModel".to_string());
        } else if !rl_cfg.is_null() {
            architectures.push("LayaDecisionModel".to_string());
        }
    }

    let hidden_size = get_usize("hidden_size").unwrap_or(768);
    let num_hidden_layers = get_usize("num_hidden_layers")
        .or_else(|| get_usize("num_layers"))
        .unwrap_or(12);
    let num_attention_heads = get_usize("num_attention_heads")
        .or_else(|| get_usize("num_heads"))
        .unwrap_or(12);
    let max_position_embeddings = get_usize("max_position_embeddings")
        .or_else(|| get_usize("max_length"))
        .or_else(|| get_usize("max_len"))
        .unwrap_or(512);
    let vocab_size = get_usize("vocab_size").unwrap_or(30522);

    let st_report =
        if dir.join("model.safetensors").exists() {
            Some(inspect_safetensors(dir).with_context(|| {
                format!("Failed to inspect model.safetensors in {}", dir.display())
            })?)
        } else {
            None
        };

    let has_mask_scorer = st_report.as_ref().is_some_and(|s| {
        s.has_decision_scorer
            || s.has_act_head
            || s.is_laya_compatible
            || s.tensor_shapes.keys().any(|k| {
                k.contains("mask_scorer") || k.contains("gliclass") || k.contains("class_head")
            })
    });

    let mt_lower = model_type.to_lowercase();
    let arch_has_joint = architectures.iter().any(|a| {
        let al = a.to_lowercase();
        al.contains("gliclass") || al.contains("laya")
    });

    let is_joint_single_pass = has_mask_scorer
        || rl_cfg_path.exists()
        || gliclass_cfg_path.exists()
        || mt_lower.contains("gliclass")
        || mt_lower.contains("laya")
        || arch_has_joint;

    let is_raw_mlm_only = !architectures.is_empty()
        && architectures
            .iter()
            .all(|a| a.ends_with("ForMaskedLM") || a.ends_with("LMHead"))
        && !st_report.as_ref().is_some_and(|s| {
            s.tensor_shapes
                .keys()
                .any(|k| k == "classifier.weight" || k.ends_with(".classifier.weight"))
        });

    let has_classifier_head = st_report
        .as_ref()
        .is_some_and(|s| s.has_classifier_head && !is_raw_mlm_only);

    let id2label_has_entailment =
        get_val("id2label")
            .and_then(|v| v.as_object())
            .is_some_and(|m| {
                m.values()
                    .filter_map(|x| x.as_str())
                    .any(|lbl| lbl.to_lowercase().contains("entailment"))
            });

    let arch_is_seq_cls = architectures
        .iter()
        .any(|a| a.ends_with("ForSequenceClassification"));

    let detected_paradigm = if is_joint_single_pass {
        HeadParadigm::JointSinglePass
    } else if has_classifier_head || id2label_has_entailment || arch_is_seq_cls {
        HeadParadigm::PairwiseNliCrossEncoder
    } else {
        HeadParadigm::GenerativeMlmVerbalizer
    };

    let total_parameters = if let Some(ref st) = st_report {
        st.total_parameters as usize
    } else {
        get_usize("num_parameters")
            .or_else(|| get_usize("total_parameters"))
            .unwrap_or_else(|| {
                vocab_size.saturating_mul(hidden_size)
                    + num_hidden_layers.saturating_mul(12 * hidden_size * hidden_size)
            })
    };

    let disk_usage_bytes = compute_dir_size_bytes(dir);
    let is_valid = st_report
        .as_ref()
        .is_some_and(|s| s.total_parameters > 0 && s.file_size_bytes > 0);

    let model_id = canonical_model_id(dir)
        .map(String::from)
        .or_else(|| get_str("model_id"))
        .or_else(|| get_str("_name_or_path"))
        .or_else(|| get_str("model_name"))
        .or_else(|| get_str("encoder"))
        .unwrap_or_else(|| {
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("custom-model")
                .to_string()
        });

    Ok(DynamicModelSpec {
        model_id,
        model_type,
        architectures,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        max_position_embeddings,
        vocab_size,
        detected_paradigm,
        total_parameters,
        disk_usage_bytes,
        is_valid,
        local_path: dir.to_path_buf(),
    })
}

/// Scan all distinct registered checkpoint directories under `models/` plus any custom
/// subdirectories containing `config.json`.
pub fn scan_registered_models() -> Vec<DynamicModelSpec> {
    let canonical_dirs = [
        "models/laya",
        "models/laya/multilingual",
        "models/laya/typed-decisions",
        "models/modernbert-zeroshot",
        "models/deberta-v3",
        "models/eurobert",
        "models/neobert",
    ];

    let mut visited = HashSet::new();
    let mut specs = Vec::new();

    for rel in canonical_dirs {
        let p = PathBuf::from(rel);
        let key = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
        if visited.insert(key)
            && p.exists()
            && let Ok(spec) = inspect_model_directory(&p)
        {
            specs.push(spec);
        }
    }

    if let Ok(entries) = std::fs::read_dir("models") {
        let mut extra_paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.join("config.json").exists())
            .collect();
        extra_paths.sort();

        for p in extra_paths {
            let key = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
            if visited.insert(key)
                && let Ok(spec) = inspect_model_directory(&p)
            {
                specs.push(spec);
            }
        }
    }

    specs
}

/// Render a compact table inventory of dynamically inspected local models.
pub fn print_models_inventory(specs: &[DynamicModelSpec]) {
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Model ID").fg(Color::Cyan),
        Cell::new("Type / Arch").fg(Color::Blue),
        Cell::new("Paradigm").fg(Color::Magenta),
        Cell::new("Dims (L×H×A)").fg(Color::White),
        Cell::new("Context").fg(Color::Yellow),
        Cell::new("Params").fg(Color::Green),
        Cell::new("Disk").fg(Color::Green),
        Cell::new("Status").fg(Color::White),
    ]);

    for s in specs {
        let arch = s
            .architectures
            .first()
            .map(|a| a.as_str())
            .unwrap_or("UnknownArch");
        table.add_row(vec![
            Cell::new(&s.model_id),
            Cell::new(format!("{} ({})", s.model_type, arch)),
            Cell::new(s.detected_paradigm.as_str()),
            Cell::new(format!(
                "{}L×{}H×{}A",
                s.num_hidden_layers, s.hidden_size, s.num_attention_heads
            )),
            Cell::new(format!("{} tok", s.max_position_embeddings)),
            Cell::new(format!("{:.1}M", s.total_parameters as f64 / 1_000_000.0)),
            Cell::new(format!("{:.1} MB", s.disk_usage_bytes as f64 / 1_048_576.0)),
            Cell::new(if s.is_valid {
                "✅ Valid"
            } else {
                "⚠️ Partial"
            }),
        ]);
    }

    println!("{table}");
}
