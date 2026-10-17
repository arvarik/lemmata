//! EuroBERT masked-LM inference on CPU with candle, ported from `modeling_eurobert.py`, the
//! reference code that ships with `EuroBERT/EuroBERT-210m`.
//!
//! EuroBERT is a Llama-style encoder: pre-norm RMSNorm layers, half-split RoPE, a SwiGLU MLP, and
//! bidirectional attention with no causal mask. The masked-LM head is one linear layer.
//!
//! `tests/eurobert_parity_test.rs` compares it with transformers.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};
use candle_nn::{ops::softmax_last_dim, rotary_emb::rope};
use candle_transformers::utils::repeat_kv;
use serde::Deserialize;

/// The `config.json` fields the encoder reads. Optional fields default as in
/// `configuration_eurobert.py`.
#[derive(Deserialize)]
struct Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: Option<serde_json::Value>,
    hidden_act: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
}

/// One `EuroBertDecoderLayer`: RMSNorm, attention, residual, then RMSNorm, SwiGLU MLP, residual.
struct Layer {
    input_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    post_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Layer {
    fn load(vb: VarBuilder, cfg: &Config, n_kv_heads: usize, head_dim: usize) -> Result<Self> {
        let (d, ff) = (cfg.hidden_size, cfg.intermediate_size);
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = n_kv_heads * head_dim;
        let attn = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        let linear = |i, o, bias, vb| candle_nn::linear_b(i, o, bias, vb);
        Ok(Self {
            input_norm: candle_nn::rms_norm(d, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            q_proj: linear(d, q_out, cfg.attention_bias, attn.pp("q_proj"))?,
            k_proj: linear(d, kv_out, cfg.attention_bias, attn.pp("k_proj"))?,
            v_proj: linear(d, kv_out, cfg.attention_bias, attn.pp("v_proj"))?,
            o_proj: linear(q_out, d, cfg.attention_bias, attn.pp("o_proj"))?,
            post_norm: candle_nn::rms_norm(d, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            gate_proj: linear(d, ff, cfg.mlp_bias, mlp.pp("gate_proj"))?,
            up_proj: linear(d, ff, cfg.mlp_bias, mlp.pp("up_proj"))?,
            down_proj: linear(ff, d, cfg.mlp_bias, mlp.pp("down_proj"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// `x` is (1, seq, d). `cos` and `sin` are (seq, head_dim / 2).
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = x.apply(&self.input_norm)?;
        let heads = |proj: &Linear, n: usize| -> candle_core::Result<Tensor> {
            h.apply(proj)?
                .reshape((b, t, n, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = rope(&heads(&self.q_proj, self.n_heads)?, cos, sin)?;
        let k = rope(&heads(&self.k_proj, self.n_kv_heads)?, cos, sin)?;
        let v = heads(&self.v_proj, self.n_kv_heads)?;
        let groups = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, groups)?;
        let v = repeat_kv(v, groups)?;
        // Bidirectional: every token attends to every token, so there is no mask.
        let att = (q.matmul(&k.t()?)? * (self.head_dim as f64).powf(-0.5))?;
        let ctx = softmax_last_dim(&att)?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.n_heads * self.head_dim))?;
        let x = (x + ctx.apply(&self.o_proj)?)?;
        let h = x.apply(&self.post_norm)?;
        let ff = (h.apply(&self.gate_proj)?.silu()? * h.apply(&self.up_proj)?)?;
        x + ff.apply(&self.down_proj)?
    }
}

/// EuroBERT masked-LM (Llama-style bidirectional encoder), ported from modeling_eurobert.py.
pub struct EuroBertMlm {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    head_dim: usize,
    rope_theta: f64,
    device: Device,
}

impl EuroBertMlm {
    /// Load `dir` (config.json + model.safetensors), F32 on CPU.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text)
            .with_context(|| format!("Malformed EuroBERT config in {}", path.display()))?;
        if cfg.hidden_act != "silu" {
            bail!("unsupported hidden_act '{}', expected silu", cfg.hidden_act);
        }
        // ponytail: default RoPE only. Port the rope_scaling variants if a checkpoint sets one.
        if cfg.rope_scaling.is_some() {
            bail!("rope_scaling is not supported");
        }
        let n_kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        if n_kv_heads == 0 || !cfg.num_attention_heads.is_multiple_of(n_kv_heads) {
            bail!(
                "num_attention_heads={} is not a multiple of num_key_value_heads={n_kv_heads}",
                cfg.num_attention_heads
            );
        }

        let device = crate::model::native::device()?;
        // SAFETY: the file is memory-mapped read-only and is not modified while lemmata runs.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let model = vb.pp("model");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(model.pp(format!("layers.{i}")), &cfg, n_kv_heads, head_dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: candle_nn::embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                model.pp("embed_tokens"),
            )?,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model.pp("norm"))?,
            // `nn.Linear(hidden, vocab, config.mlp_bias)`, untied from the embeddings.
            lm_head: candle_nn::linear_b(
                cfg.hidden_size,
                cfg.vocab_size,
                cfg.mlp_bias,
                vb.pp("lm_head"),
            )?,
            head_dim,
            rope_theta: cfg.rope_theta,
            device,
        })
    }

    /// `EuroBertRotaryEmbedding` for positions `0..seq`: (seq, head_dim / 2) cos and sin tables,
    /// computed in f32 like the reference.
    fn rope_tables(&self, seq: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (self.rope_theta as f32).powf((2 * i) as f32 / self.head_dim as f32))
            .collect();
        let angles: Vec<f32> = (0..seq)
            .flat_map(|p| inv_freq.iter().map(move |f| p as f32 * f))
            .collect();
        let angles = Tensor::from_vec(angles, (seq, half), &self.device)?;
        Ok((angles.cos()?, angles.sin()?))
    }

    /// Masked-LM logits for one unpadded sequence of token ids, at `positions` only.
    /// Returns one vocab-sized row per position, in the order given.
    // ponytail: one sequence per forward pass on CPU. Batch with a padding mask if throughput matters.
    pub fn mlm_logits_at(&self, ids: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        if ids.is_empty() {
            bail!("EuroBERT needs at least one token");
        }
        if let Some(&p) = positions.iter().find(|&&p| p >= ids.len()) {
            bail!("position {p} is outside the {}-token sequence", ids.len());
        }
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        let (cos, sin) = self.rope_tables(ids.len())?;
        let mut h = self
            .embed_tokens
            .forward(&Tensor::new(ids, &self.device)?.unsqueeze(0)?)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin)?;
        }
        let idx = Tensor::new(
            positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            &self.device,
        )?;
        // The vocabulary is 128256 tokens, so the head reads only the requested rows.
        let logits = self
            .norm
            .forward(&h.squeeze(0)?)?
            .index_select(&idx, 0)?
            .apply(&self.lm_head)?;
        Ok(logits.to_vec2::<f32>()?)
    }

    /// Mean of the final normalized hidden states over all tokens, for linear probes.
    pub fn mean_hidden(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            bail!("EuroBERT needs at least one token");
        }
        let (cos, sin) = self.rope_tables(ids.len())?;
        let mut h = self
            .embed_tokens
            .forward(&Tensor::new(ids, &self.device)?.unsqueeze(0)?)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin)?;
        }
        Ok(self
            .norm
            .forward(&h.squeeze(0)?)?
            .mean(0)?
            .to_vec1::<f32>()?)
    }
}
