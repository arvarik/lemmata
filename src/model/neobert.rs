//! NeoBERT masked-LM inference on CPU with candle, ported from the reference code that ships
//! with `chandar-lab/NeoBERT` (`model.py` and `rotary.py`).
//!
//! `tests/neobert_parity_test.rs` compares it with the Python reference code.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder};

/// `precompute_freqs_cis` default. config.json has no rope theta.
const ROPE_THETA: f32 = 10_000.0;

/// The `NeoBERTConfig` fields the forward pass reads.
#[derive(serde::Deserialize)]
struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    norm_eps: f64,
    vocab_size: usize,
    max_length: usize,
}

/// One pre-norm `EncoderBlock`: x + attn(attention_norm(x)), then x + ffn(ffn_norm(x)).
struct Block {
    attention_norm: RmsNorm,
    qkv: Linear,
    wo: Linear,
    ffn_norm: RmsNorm,
    /// xformers `SwiGLU` packs w1 (gate) over w2 (up) in one `w12` matrix.
    w12: Linear,
    w3: Linear,
    n_heads: usize,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let d = cfg.hidden_size;
        // EncoderBlock: int(2 * intermediate_size / 3), rounded up to a multiple of 8.
        let ffn = (2 * cfg.intermediate_size / 3).div_ceil(8) * 8;
        Ok(Self {
            attention_norm: candle_nn::rms_norm(d, cfg.norm_eps, vb.pp("attention_norm"))?,
            qkv: candle_nn::linear_no_bias(d, 3 * d, vb.pp("qkv"))?,
            wo: candle_nn::linear_no_bias(d, d, vb.pp("wo"))?,
            ffn_norm: candle_nn::rms_norm(d, cfg.norm_eps, vb.pp("ffn_norm"))?,
            w12: candle_nn::linear_no_bias(d, 2 * ffn, vb.pp("ffn.w12"))?,
            w3: candle_nn::linear_no_bias(ffn, d, vb.pp("ffn.w3"))?,
            n_heads: cfg.num_attention_heads,
        })
    }

    /// `x` is (1, seq, d). There is no attention mask, because lemma runs one unpadded sequence.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let hd = d / self.n_heads;
        // `qkv(x).view(b, t, heads, 3 * hd).chunk(3)`: each head's q, k and v sit side by side,
        // not all queries, then all keys, then all values.
        let qkv = x
            .apply(&self.attention_norm)?
            .apply(&self.qkv)?
            .reshape((b, t, self.n_heads, 3, hd))?
            .permute((3, 0, 2, 1, 4))?;
        // `rotary.py` rotates (x[2i], x[2i+1]) as one complex number, which is `rope_i`.
        let q = candle_nn::rotary_emb::rope_i(&qkv.get(0)?.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&qkv.get(1)?.contiguous()?, cos, sin)?;
        let v = qkv.get(2)?.contiguous()?;
        let att = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let ctx = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        let x = (x + ctx.apply(&self.wo)?)?;
        // SwiGLU: w3(silu(x w1) * (x w2)).
        let ff = candle_nn::ops::swiglu(&x.apply(&self.ffn_norm)?.apply(&self.w12)?)?;
        x + ff.apply(&self.w3)?
    }
}

/// NeoBERT masked-LM (pre-RMSNorm, SwiGLU, RoPE), ported from model.py.
pub struct NeoBertMlm {
    embed: Embedding,
    blocks: Vec<Block>,
    layer_norm: RmsNorm,
    decoder: Linear,
    /// RoPE tables for positions 0..max_length, (max_length, head_dim / 2), like `freqs_cis`.
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl NeoBertMlm {
    /// Load `dir` (config.json + model.safetensors), F32 on CPU.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text)
            .with_context(|| format!("Malformed NeoBERT config in {}", path.display()))?;

        let device = crate::model::native::device()?;
        // SAFETY: the file is memory-mapped read-only and is not modified while lemma runs.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.transformer_encoder.{i}")), &cfg))
            .collect::<candle_core::Result<Vec<_>>>()?;

        // precompute_freqs_cis, in f32 as torch computes it: angle = pos / theta^(2i / head_dim).
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let angles: Vec<f32> = (0..cfg.max_length)
            .flat_map(|p| inv_freq.iter().map(move |f| p as f32 * f))
            .collect();
        let angles = Tensor::from_vec(angles, (cfg.max_length, half), &device)?;

        Ok(Self {
            embed: candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.encoder"))?,
            blocks,
            layer_norm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.norm_eps,
                vb.pp("model.layer_norm"),
            )?,
            decoder: candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("decoder"))?,
            cos: angles.cos()?,
            sin: angles.sin()?,
            device,
        })
    }

    /// Masked-LM logits for one unpadded sequence of token ids, at `positions` only.
    /// Returns one vocab-sized row per position, in the order given.
    // ponytail: one sequence per forward pass on CPU. Batch with a padding mask if throughput matters.
    pub fn mlm_logits_at(&self, ids: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        let max_len = self.cos.dim(0)?;
        if ids.is_empty() || ids.len() > max_len {
            bail!("NeoBERT reads 1 to {max_len} tokens, got {}", ids.len());
        }
        if let Some(p) = positions.iter().find(|&&p| p >= ids.len()) {
            bail!("position {p} is outside the {}-token sequence", ids.len());
        }
        if positions.is_empty() {
            return Ok(Vec::new());
        }
        let t = ids.len();
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;
        let mut x = self
            .embed
            .forward(&Tensor::new(ids, &self.device)?.unsqueeze(0)?)?;
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }
        // The final RMSNorm and the decoder act per token, so only the requested rows need them.
        let rows = positions.iter().map(|&p| p as u32).collect::<Vec<_>>();
        let logits = x
            .squeeze(0)?
            .index_select(&Tensor::new(rows, &self.device)?, 0)?
            .apply(&self.layer_norm)?
            .apply(&self.decoder)?;
        Ok(logits.to_vec2::<f32>()?)
    }

    /// Mean of the final normalized hidden states over all tokens, for linear probes.
    pub fn mean_hidden(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let max_len = self.cos.dim(0)?;
        if ids.is_empty() || ids.len() > max_len {
            bail!("NeoBERT reads 1 to {max_len} tokens, got {}", ids.len());
        }
        let t = ids.len();
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;
        let mut x = self
            .embed
            .forward(&Tensor::new(ids, &self.device)?.unsqueeze(0)?)?;
        for block in &self.blocks {
            x = block.forward(&x, &cos, &sin)?;
        }
        Ok(x.squeeze(0)?
            .apply(&self.layer_norm)?
            .mean(0)?
            .to_vec1::<f32>()?)
    }
}
