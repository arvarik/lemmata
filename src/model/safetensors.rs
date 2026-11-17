use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

fn read_safetensors_header_bytes(path: &Path) -> Result<(u64, u64, Vec<u8>)> {
    let resolved = if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    };
    let mut file = File::open(&resolved)
        .with_context(|| format!("Cannot open Safetensors file: {}", resolved.display()))?;

    let file_size = file.metadata()?.len();
    if file_size < 8 {
        bail!(
            "File is too small to be a valid Safetensors file (size: {} bytes)",
            file_size
        );
    }

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);

    if header_len > MAX_HEADER_BYTES {
        bail!("Safetensors header abnormally large (>100MB)");
    }

    if 8u64
        .checked_add(header_len)
        .is_none_or(|end| end > file_size)
    {
        bail!(
            "Corrupted Safetensors file: header size ({} bytes) exceeds file size ({} bytes)",
            header_len,
            file_size
        );
    }

    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)?;
    Ok((file_size, header_len, header_buf))
}

/// Fast, portable SHA-256 fingerprint of `(file_size_u64_le || safetensors_header_bytes)`.
///
/// Runs in sub-millisecond time without hashing multi-gigabyte tensor payloads while remaining
/// identical across machines/filesystems (independent of `mtime`).
pub fn safetensors_header_sha256(path: &Path) -> Result<String> {
    let (file_size, _, header_buf) = read_safetensors_header_bytes(path)?;
    let mut hasher = Sha256::new();
    hasher.update(file_size.to_le_bytes());
    hasher.update(&header_buf);
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[derive(Debug, Deserialize)]
pub struct RawTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (u64, u64),
}

#[derive(Debug, Clone)]
pub struct SafetensorsReport {
    pub total_tensors: usize,
    pub total_parameters: u64,
    pub dtypes: BTreeMap<String, usize>,
    pub tensor_shapes: BTreeMap<String, Vec<usize>>,
    pub architecture: String,
    pub is_laya_compatible: bool,
    pub has_classifier_head: bool,
    pub has_act_head: bool,
    pub has_decision_scorer: bool,
    pub file_size_bytes: u64,
    pub header_size_bytes: u64,
}

pub fn inspect_safetensors(path: &Path) -> Result<SafetensorsReport> {
    let (file_size, header_len, header_buf) = read_safetensors_header_bytes(path)?;

    let raw_header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)
        .context("Failed to parse JSON metadata in Safetensors header")?;

    let data_base = 8u64
        .checked_add(header_len)
        .ok_or_else(|| anyhow::anyhow!("Safetensors header offset overflow"))?;

    let mut total_parameters = 0u64;
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut tensor_shapes = BTreeMap::new();
    let mut has_act_head = false;
    let mut has_decision_scorer = false;
    let mut has_nli_classifier = false;
    let mut has_lm_head = false;
    let mut is_deberta = false;
    let mut is_modernbert = false;
    let mut is_eurobert = false;
    let mut is_neobert = false;

    for (name, val) in raw_header {
        if name == "__metadata__" {
            continue;
        }

        let info: RawTensorInfo = serde_json::from_value(val)
            .with_context(|| format!("Invalid Safetensors tensor entry '{}'", name))?;

        let (start, end) = info.data_offsets;
        if start > end {
            bail!(
                "Corrupted Safetensors tensor '{}': inverted data_offsets ({} > {})",
                name,
                start,
                end
            );
        }
        let abs_end = data_base.checked_add(end).ok_or_else(|| {
            anyhow::anyhow!("Safetensors tensor '{}' data_offsets overflow", name)
        })?;
        if abs_end > file_size {
            bail!(
                "Corrupted Safetensors tensor '{}': data_offsets end ({}) exceeds file size ({})",
                name,
                abs_end,
                file_size
            );
        }

        let mut num_elements = 1u64;
        for &dim in &info.shape {
            num_elements = num_elements.checked_mul(dim as u64).ok_or_else(|| {
                anyhow::anyhow!(
                    "Safetensors tensor '{}' shape {:?} overflows u64 element count",
                    name,
                    info.shape
                )
            })?;
        }
        total_parameters = total_parameters
            .checked_add(num_elements)
            .ok_or_else(|| anyhow::anyhow!("Safetensors total parameter count overflow"))?;
        *dtypes.entry(info.dtype).or_default() += 1;

        if name.contains("act_head") {
            has_act_head = true;
        }
        if name.contains("scorer") || name.contains("mask_scorer") {
            has_decision_scorer = true;
        }
        if name == "classifier.weight" || name.ends_with(".classifier.weight") {
            has_nli_classifier = true;
        }
        if name.contains("lm_head") || name.starts_with("decoder.") {
            has_lm_head = true;
        }
        if name.starts_with("deberta.") || name.contains("rel_embeddings") {
            is_deberta = true;
        }
        if name.contains("tok_embeddings") || name.contains("Wqkv") {
            is_modernbert = true;
        }
        if name.contains("self_attn.q_proj") || name.contains("mlp.gate_proj") {
            is_eurobert = true;
        }
        if name.starts_with("transformer_encoder.") || name.contains("qkv.weight") {
            is_neobert = true;
        }

        tensor_shapes.insert(name, info.shape);
    }

    let is_laya = has_act_head || has_decision_scorer || tensor_shapes.contains_key("temperature");
    let has_classifier_head = is_laya || has_nli_classifier || has_lm_head;
    let architecture = if is_laya {
        "Laya System 1 (ModernBERT/mmBERT + Decision/Act Heads)".to_string()
    } else if is_deberta && has_nli_classifier {
        "DeBERTa-v3 Zero-Shot Sequence Classifier (NLI)".to_string()
    } else if is_modernbert && has_nli_classifier {
        "ModernBERT Zero-Shot Sequence Classifier (NLI)".to_string()
    } else if is_eurobert {
        "EuroBERT Bidirectional Encoder (RoPE + SwiGLU + MLM Head)".to_string()
    } else if is_neobert {
        "NeoBERT Representation Encoder (RoPE + SwiGLU + MLM Head)".to_string()
    } else if has_nli_classifier {
        "Encoder Sequence Classifier".to_string()
    } else {
        "Pretrained Transformer Encoder".to_string()
    };

    Ok(SafetensorsReport {
        total_tensors: tensor_shapes.len(),
        total_parameters,
        dtypes,
        tensor_shapes,
        architecture,
        is_laya_compatible: is_laya,
        has_classifier_head,
        has_act_head,
        has_decision_scorer,
        file_size_bytes: file_size,
        header_size_bytes: header_len,
    })
}
