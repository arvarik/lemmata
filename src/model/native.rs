//! Native Laya inference on CPU: the ModernBERT/mmBERT encoder plus the Laya decision head.
//!
//! This is a port of the reference code that ships with each checkpoint (`rl_agent_api.py`
//! and `rl_common.py`). `tests/laya_parity_test.rs` compares it with that code.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use candle_transformers::models::modernbert::{Config as EncoderConfig, ModernBert};
use tokenizers::Tokenizer;

use super::{Checkpoint, QuestionType};

/// Upstream `build_sequence` keeps at most this many tokens per option text.
const MAX_OPTION_TOKENS: usize = 48;

static NATIVE: AtomicBool = AtomicBool::new(false);
static MOCK_NOTICE: AtomicBool = AtomicBool::new(false);
static MODELS: OnceLock<Mutex<HashMap<Checkpoint, Arc<LayaModel>>>> = OnceLock::new();

/// Route all option scoring through the real Laya weights (`true`) or the keyword mock (`false`).
pub fn set_native(on: bool) {
    NATIVE.store(on, Ordering::Relaxed);
}

pub fn native_enabled() -> bool {
    NATIVE.load(Ordering::Relaxed)
}

/// Ask the mock backend to print a one-time stderr warning when it scores options.
pub fn set_mock_notice(on: bool) {
    MOCK_NOTICE.store(on, Ordering::Relaxed);
}

pub(crate) fn mock_notice_enabled() -> bool {
    MOCK_NOTICE.load(Ordering::Relaxed)
}

/// True when the weights for `ckpt` are on disk.
pub fn weights_present(ckpt: Checkpoint) -> bool {
    ckpt.local_dir().join("model.safetensors").exists()
}

/// Checkpoints that commands such as `compare all` can run with the active backend.
pub fn runnable_checkpoints() -> &'static [Checkpoint] {
    if native_enabled() {
        &[
            Checkpoint::English,
            Checkpoint::Multilingual,
            Checkpoint::TypedDecisions,
            Checkpoint::AutoRoute,
        ]
    } else {
        Checkpoint::all()
    }
}

/// Load `ckpt` once per process and share it across threads.
pub fn shared(ckpt: Checkpoint) -> Result<Arc<LayaModel>> {
    if !Checkpoint::laya_checkpoints().contains(&ckpt) {
        bail!(
            "lemma has no native inference for {ckpt}. Native inference supports english, multilingual and typed-decisions. Run with --backend mock to use the keyword simulator."
        );
    }
    // ponytail: one lock covers loading, so a second checkpoint waits while the first loads.
    let mut models = MODELS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| anyhow!("native model cache lock poisoned"))?;
    if let Some(m) = models.get(&ckpt) {
        return Ok(m.clone());
    }
    let dir = ckpt.local_dir();
    let model = Arc::new(LayaModel::load(&dir).with_context(|| {
        format!(
            "Failed to load {ckpt} from {}. Run `lemma download` first.",
            dir.display()
        )
    })?);
    models.insert(ckpt, model.clone());
    Ok(model)
}

/// One pre-norm `nn.TransformerEncoderLayer` (ReLU feed-forward) of the decision head.
struct HeadLayer {
    norm1: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    n_heads: usize,
}

impl HeadLayer {
    fn load(vb: VarBuilder, d: usize) -> candle_core::Result<Self> {
        let attn = vb.pp("self_attn");
        Ok(Self {
            norm1: candle_nn::layer_norm(d, 1e-5, vb.pp("norm1"))?,
            in_proj: Linear::new(
                attn.get((3 * d, d), "in_proj_weight")?,
                Some(attn.get(3 * d, "in_proj_bias")?),
            ),
            out_proj: candle_nn::linear(d, d, attn.pp("out_proj"))?,
            norm2: candle_nn::layer_norm(d, 1e-5, vb.pp("norm2"))?,
            linear1: candle_nn::linear(d, 4 * d, vb.pp("linear1"))?,
            linear2: candle_nn::linear(4 * d, d, vb.pp("linear2"))?,
            n_heads: (d / 64).max(1),
        })
    }

    /// `x` is (1, seq, d). There is no padding mask, because lemma runs one sequence per pass.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let hd = d / self.n_heads;
        let qkv = x
            .apply(&self.norm1)?
            .apply(&self.in_proj)?
            .reshape((b, t, 3, self.n_heads, hd))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let att = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let ctx = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        let x = (x + ctx.apply(&self.out_proj)?)?;
        let ff = x
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        x + ff
    }
}

/// A loaded Laya checkpoint.
pub struct LayaModel {
    tokenizer: Tokenizer,
    encoder: ModernBert,
    head: Vec<HeadLayer>,
    type_emb: Embedding,
    scorer_norm: LayerNorm,
    scorer_hidden: Linear,
    scorer_out: Linear,
    max_len: usize,
    head_max_len: usize,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
    mask_token: String,
    device: Device,
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("Malformed JSON in {}", path.display()))
}

fn json_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| anyhow!("missing integer field '{key}'"))
}

/// Map a transformers ModernBERT config (v4 or v5 rope keys) to candle's config.
fn encoder_config(raw: &serde_json::Value) -> Result<EncoderConfig> {
    let rope = |kind: &str, legacy: &str, default: f64| {
        raw.pointer(&format!("/rope_parameters/{kind}/rope_theta"))
            .or_else(|| raw.get(legacy))
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    };
    let mut cfg = raw.clone();
    cfg["global_rope_theta"] = rope("full_attention", "global_rope_theta", 160_000.0).into();
    cfg["local_rope_theta"] = rope("sliding_attention", "local_rope_theta", 10_000.0).into();
    if let Some(obj) = cfg.as_object_mut() {
        // The decision head replaces the MLM classifier, so drop the classifier fields.
        obj.retain(|k, _| !k.starts_with("classifier_") && k != "id2label" && k != "label2id");
    }
    Ok(serde_json::from_value(cfg)?)
}

impl LayaModel {
    /// Load a checkpoint directory laid out like `convaiinnovations/laya`.
    pub fn load(dir: &Path) -> Result<Self> {
        let agent_cfg = read_json(&dir.join("rl_agent_config.json"))?;
        let enc_raw = read_json(&dir.join("encoder/config.json"))?;
        let tok_cfg = read_json(&dir.join("tokenizer/tokenizer_config.json"))?;
        let enc_cfg = encoder_config(&enc_raw)?;
        let d = enc_cfg.hidden_size;

        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).map_err(|e| anyhow!(e))?;
        let special = |key: &str| -> Result<(String, u32)> {
            let tok = tok_cfg
                .get(key)
                .and_then(|v| v.as_str().or_else(|| v.get("content")?.as_str()))
                .ok_or_else(|| anyhow!("tokenizer_config.json has no {key}"))?;
            let id = tokenizer
                .token_to_id(tok)
                .ok_or_else(|| anyhow!("{key} '{tok}' is not in the vocabulary"))?;
            Ok((tok.to_string(), id))
        };
        let (_, cls_id) = special("cls_token")?;
        let (_, sep_id) = special("sep_token")?;
        let (mask_token, mask_id) = special("mask_token")?;

        let device = Device::Cpu;
        // SAFETY: the file is memory-mapped read-only and is not modified while lemma runs.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        // candle names the encoder weights `model.*`. Laya stores them as `encoder.*`.
        let encoder = ModernBert::load(
            vb.clone()
                .rename_f(|name| name.replacen("model.", "encoder.", 1)),
            &enc_cfg,
        )?;
        let head = (0..json_usize(&agent_cfg, "head_layers")?)
            .map(|i| HeadLayer::load(vb.pp(format!("head.layers.{i}")), d))
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self {
            tokenizer,
            encoder,
            head,
            type_emb: candle_nn::embedding(3, d, vb.pp("type_emb"))?,
            scorer_norm: candle_nn::layer_norm(d, 1e-5, vb.pp("scorer.0"))?,
            scorer_hidden: candle_nn::linear(d, d, vb.pp("scorer.1"))?,
            scorer_out: candle_nn::linear(d, 1, vb.pp("scorer.3"))?,
            max_len: json_usize(&agent_cfg, "max_len")?,
            head_max_len: json_usize(&agent_cfg, "head_max_len")?,
            cls_id,
            sep_id,
            mask_id,
            mask_token,
            device,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec())
    }

    /// `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`.
    ///
    /// Returns the token IDs and the position of each option's `[MASK]` marker.
    pub fn build_sequence(
        &self,
        state: &str,
        qtype: QuestionType,
        instructions: &str,
        options: &[String],
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let mask = self.mask_token.as_str();
        let qname = match qtype {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        };
        let head_ids = self.encode(&format!(
            "{qname} question: {}",
            instructions.replace(mask, " ")
        ))?;
        let mut opt_ids = options
            .iter()
            .map(|o| {
                let text = self.encode(&format!(" {}", o.replace(mask, " ")))?;
                let mut ids = vec![self.mask_id];
                ids.extend(text.into_iter().take(MAX_OPTION_TOKENS));
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;

        let used = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>() as i64;
        let mut opt_budget = self.head_max_len as i64 - used(&opt_ids);
        if opt_budget < 16 {
            // Too many or too long options: shrink every option text evenly.
            let per = (self.head_max_len.saturating_sub(16) / opt_ids.len().max(1)).max(4);
            opt_ids.iter_mut().for_each(|o| o.truncate(per));
            opt_budget = self.head_max_len as i64 - used(&opt_ids);
        }
        let head_len = head_ids.len().min(opt_budget.max(8) as usize);

        let mut ids = vec![self.cls_id];
        ids.extend_from_slice(&head_ids[..head_len]);
        ids.push(self.sep_id);
        let mut markers = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            markers.push(ids.len());
            ids.extend_from_slice(o);
        }
        ids.push(self.sep_id);
        let room = self.max_len.saturating_sub(ids.len() + 1);
        let state_ids = self.encode(&state.replace(mask, " "))?;
        ids.extend_from_slice(&state_ids[..state_ids.len().min(room)]);
        ids.push(self.sep_id);
        ids.truncate(self.max_len);
        markers.retain(|&m| m < self.max_len);
        Ok((ids, markers))
    }

    /// Raw (pre-temperature) logit per option, in `options` order.
    // ponytail: one sequence per forward pass on CPU. Batch questions with a padding mask, or add
    // a Metal/CUDA device, if throughput matters.
    pub fn option_logits(
        &self,
        state: &str,
        qtype: QuestionType,
        instructions: &str,
        options: &[String],
    ) -> Result<Vec<f64>> {
        let (ids, markers) = self.build_sequence(state, qtype, instructions, options)?;
        if markers.len() != options.len() {
            bail!(
                "{} options do not fit in head_max_len={} tokens",
                options.len(),
                self.head_max_len
            );
        }
        let dev = &self.device;
        let input = Tensor::new(ids.as_slice(), dev)?.unsqueeze(0)?;
        let attention = Tensor::ones((1, ids.len()), DType::F32, dev)?;
        // Upstream QTYPES order: choice = 0, score = 1, noul = 2.
        let qtype_idx = Tensor::new(&[qtype as u32], dev)?;

        let mut h = self.encoder.forward(&input, &attention)?;
        h = h.broadcast_add(&self.type_emb.forward(&qtype_idx)?.unsqueeze(1)?)?;
        for layer in &self.head {
            h = layer.forward(&h)?;
        }
        let marker_idx = Tensor::new(markers.iter().map(|&m| m as u32).collect::<Vec<_>>(), dev)?;
        let logits = h
            .squeeze(0)?
            .index_select(&marker_idx, 0)?
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_hidden)?
            .gelu_erf()?
            .apply(&self.scorer_out)?
            .squeeze(1)?;
        Ok(logits
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    }
}
