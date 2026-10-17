//! Native inference on CPU with candle.
//!
//! - Laya: the ModernBERT/mmBERT encoder plus the Laya decision head, ported from the reference
//!   code that ships with each checkpoint (`rl_agent_api.py` and `rl_common.py`).
//! - NLI zero-shot cross-encoders: ModernBERT and DeBERTa-v3 sequence classifiers.
//!
//! `tests/native_parity_test.rs` compares both with the Python reference code.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use candle_transformers::models::debertav2::{
    Config as DebertaConfig, DebertaV2ContextPooler, DebertaV2Model,
};
use candle_transformers::models::modernbert::{Config as EncoderConfig, ModernBert};
use tokenizers::Tokenizer;

use super::{Checkpoint, ChunkStrategy, Question, QuestionType, pool_window_logits};

/// Upstream `build_sequence` keeps at most this many tokens per option text.
const MAX_OPTION_TOKENS: usize = 48;
/// Both NLI fine-tunes were trained at 512 tokens (`model_max_length` in tokenizer_config.json).
const DEFAULT_NLI_MAX_LEN: usize = 512;

/// Where option logits come from.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Keyword-rule simulator for tests and demos. It runs no model
    Mock = 0,
    /// Real weights for each checkpoint that has them, the labelled mock for the rest
    Auto = 1,
    /// Real weights only. Fails for a checkpoint without weights or native support
    Native = 2,
}

/// Where native models run.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The Metal GPU when one is present (macOS), otherwise the CPU
    Auto,
    /// The CPU
    Cpu,
    /// The Metal GPU. Fails when none is present
    Metal,
}

static BACKEND: AtomicU8 = AtomicU8::new(Backend::Mock as u8);
static DEVICE_CHOICE: OnceLock<DeviceChoice> = OnceLock::new();
static DEVICE: OnceLock<Device> = OnceLock::new();

/// Choose the device before the first model loads. Later calls have no effect.
pub fn set_device(choice: DeviceChoice) {
    let _ = DEVICE_CHOICE.set(choice);
}

/// The device every native model loads onto.
pub fn device() -> Result<Device> {
    if let Some(d) = DEVICE.get() {
        return Ok(d.clone());
    }
    let metal = || Device::new_metal(0).map_err(|e| anyhow!("no Metal GPU: {e}"));
    // The CLI sets the choice. Library callers can set LEMMATA_DEVICE, and otherwise get the CPU.
    let from_env = || {
        let value = std::env::var("LEMMATA_DEVICE").ok()?;
        <DeviceChoice as clap::ValueEnum>::from_str(&value, true).ok()
    };
    let choice = DEVICE_CHOICE.get().copied().or_else(from_env);
    let device = match choice.unwrap_or(DeviceChoice::Cpu) {
        DeviceChoice::Cpu => Device::Cpu,
        DeviceChoice::Metal => metal()?,
        DeviceChoice::Auto => metal().unwrap_or(Device::Cpu),
    };
    Ok(DEVICE.get_or_init(|| device).clone())
}
static MODELS: OnceLock<Mutex<HashMap<PathBuf, Arc<NativeModel>>>> = OnceLock::new();

pub fn set_backend(backend: Backend) {
    BACKEND.store(backend as u8, Ordering::Relaxed);
}

pub fn backend() -> Backend {
    match BACKEND.load(Ordering::Relaxed) {
        1 => Backend::Auto,
        2 => Backend::Native,
        _ => Backend::Mock,
    }
}

/// Checkpoints lemmata can run natively.
pub fn has_native_impl(ckpt: Checkpoint) -> bool {
    matches!(
        ckpt,
        Checkpoint::English
            | Checkpoint::Multilingual
            | Checkpoint::TypedDecisions
            | Checkpoint::ModernBertZeroShot
            | Checkpoint::ModernBertNli
            | Checkpoint::DebertaV3ZeroShot
            | Checkpoint::EuroBert
            | Checkpoint::NeoBert
    )
}

/// True when the weights for `ckpt` are in its default directory.
pub fn weights_present(ckpt: Checkpoint) -> bool {
    ckpt.local_dir().join("model.safetensors").exists()
}

/// True when `ckpt` can run natively from its default directory.
pub fn native_available(ckpt: Checkpoint) -> bool {
    has_native_impl(ckpt) && weights_present(ckpt)
}

/// Whether scoring for `ckpt` (weights in `dir`) runs the real model under the active backend.
pub fn uses_native(ckpt: Checkpoint, dir: &Path) -> bool {
    match backend() {
        Backend::Mock => false,
        Backend::Native => true,
        Backend::Auto => has_native_impl(ckpt) && dir.join("model.safetensors").exists(),
    }
}

/// Under `auto`, say once per checkpoint that its decisions are simulated.
pub(crate) fn warn_simulated(ckpt: Checkpoint, dir: &Path) {
    static WARNED: OnceLock<Mutex<HashSet<Checkpoint>>> = OnceLock::new();
    if backend() != Backend::Auto {
        return;
    }
    let first = WARNED
        .get_or_init(Default::default)
        .lock()
        .is_ok_and(|mut w| w.insert(ckpt));
    if first {
        let why = if has_native_impl(ckpt) {
            format!("no weights in {}", dir.display())
        } else {
            "lemmata has no native implementation for it".to_string()
        };
        eprintln!(
            "⚠️  {ckpt}: {why}. Its decisions are simulated by the keyword mock (model id mock:{ckpt})."
        );
    }
}

/// True when native models run on a GPU, where classify calls from many threads only queue.
pub fn runs_on_gpu() -> bool {
    backend() != Backend::Mock && device().is_ok_and(|d| d.is_metal())
}

/// Map `f` over `items`: in parallel on the CPU, in order on a GPU.
pub fn map_items<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    if runs_on_gpu() {
        items.iter().map(f).collect()
    } else {
        items.par_iter().map(f).collect()
    }
}

/// Checkpoints that commands such as `compare all` run under the active backend.
pub fn runnable_checkpoints() -> Vec<Checkpoint> {
    if backend() == Backend::Mock {
        return Checkpoint::all().to_vec();
    }
    let mut ready: Vec<Checkpoint> = Checkpoint::all()
        .iter()
        .copied()
        .filter(|&c| native_available(c))
        .collect();
    if ready
        .iter()
        .any(|c| Checkpoint::laya_checkpoints().contains(c))
    {
        ready.push(Checkpoint::AutoRoute);
    }
    if ready.is_empty() {
        // Nothing is installed: `auto` simulates everything, `native` reports the missing weights.
        return Checkpoint::all().to_vec();
    }
    ready
}

/// Load the model in `dir` once per process and share it across threads.
pub fn shared(ckpt: Checkpoint, dir: &Path) -> Result<Arc<NativeModel>> {
    if !has_native_impl(ckpt) {
        bail!(
            "lemmata has no native inference for {ckpt}. Native inference supports english, multilingual, typed-decisions, modernbert-zeroshot and deberta-v3. Run with --backend auto or mock to simulate it."
        );
    }
    // ponytail: one lock covers loading, so a second checkpoint waits while the first loads.
    let mut models = MODELS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| anyhow!("native model cache lock poisoned"))?;
    if let Some(m) = models.get(dir) {
        return Ok(m.clone());
    }
    let started = Instant::now();
    let model = match ckpt {
        Checkpoint::English | Checkpoint::Multilingual | Checkpoint::TypedDecisions => {
            LayaModel::load(dir).map(|m| NativeModel::Laya(Box::new(m)))
        }
        Checkpoint::EuroBert | Checkpoint::NeoBert => {
            MlmModel::load(ckpt, dir).map(|m| NativeModel::Mlm(Box::new(m)))
        }
        _ => NliModel::load(dir).map(|m| NativeModel::Nli(Box::new(m))),
    }
    .with_context(|| {
        format!(
            "Failed to load {ckpt} from {}. Run `lemmata download` first.",
            dir.display()
        )
    })?;
    eprintln!(
        "🔧 Loaded {ckpt} from {} in {:.0} ms",
        dir.display(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    let model = Arc::new(model);
    models.insert(dir.to_path_buf(), model.clone());
    Ok(model)
}

/// Drop every cached model. Callers that use one checkpoint at a time call this between
/// checkpoints, so memory holds one model instead of all of them.
pub fn release_models() {
    if let Some(models) = MODELS.get()
        && let Ok(mut models) = models.lock()
    {
        models.clear();
    }
}

/// One scoring call: a state text, a question, and its rendered options.
pub struct ScoreRequest<'a> {
    pub state: &'a str,
    pub question: &'a Question,
    pub options: &'a [String],
    pub strategy: ChunkStrategy,
    pub hypothesis_template: Option<&'a str>,
    pub multi_label: bool,
}

/// Raw (pre-temperature) logit per option, and the real number of tokens the model read.
pub struct Scored {
    pub logits: Vec<f64>,
    pub input_tokens: usize,
}

/// A loaded native model.
pub enum NativeModel {
    Laya(Box<LayaModel>),
    Nli(Box<NliModel>),
    Mlm(Box<MlmModel>),
}

impl NativeModel {
    pub fn score(&self, req: &ScoreRequest) -> Result<Scored> {
        Ok(self.score_batch(std::slice::from_ref(req))?.remove(0))
    }

    /// Mean-pooled final hidden states of the encoder for `text`, for linear probes. Long text
    /// is split with `strategy` like scoring, and the window embeddings are averaged.
    pub fn embed(&self, text: &str, strategy: ChunkStrategy) -> Result<Vec<f32>> {
        let windows: Vec<Vec<f32>> = match self {
            Self::Laya(m) => {
                let ids = m.encode(&text.replace(&m.mask_token, " "))?;
                state_windows(&ids, m.max_len.saturating_sub(2), strategy)
                    .iter()
                    .map(|w| mean_hidden(&m.encoder, &m.device, &framed(m.cls_id, w, m.sep_id)))
                    .collect::<Result<_>>()?
            }
            Self::Nli(m) => {
                let ids = m.encode(text)?;
                state_windows(&ids, m.max_len.saturating_sub(2), strategy)
                    .iter()
                    .map(|w| m.mean_hidden(&framed(m.cls_id, w, m.sep_id)))
                    .collect::<Result<_>>()?
            }
            Self::Mlm(m) => {
                let ids = m.encode(text)?;
                let room = m.max_len.saturating_sub(m.prefix.len() + m.suffix.len());
                state_windows(&ids, room, strategy)
                    .iter()
                    .map(|w| {
                        m.encoder
                            .mean_hidden(&[&m.prefix[..], w, &m.suffix[..]].concat())
                    })
                    .collect::<Result<_>>()?
            }
        };
        let n = windows.len().max(1) as f32;
        let mut mean = vec![0.0; windows.first().map_or(0, Vec::len)];
        for w in &windows {
            for (m, v) in mean.iter_mut().zip(w) {
                *m += v / n;
            }
        }
        Ok(mean)
    }

    /// Score several questions about the same text. Laya runs them all in one forward pass.
    pub fn score_batch(&self, reqs: &[ScoreRequest]) -> Result<Vec<Scored>> {
        match self {
            Self::Laya(m) => m.score_batch(reqs),
            Self::Nli(m) => reqs.iter().map(|r| m.score(r)).collect(),
            Self::Mlm(m) => reqs.iter().map(|r| m.score(r)).collect(),
        }
    }
}

/// Split the state tokens into the windows the model reads, each at most `room` tokens.
///
/// Head+tail keeps the first quarter and the last three quarters of the budget, as in
/// Sun et al. 2019 (128 + 382 of 510). Sliding windows overlap by a third and are pooled.
pub fn state_windows(state: &[u32], room: usize, strategy: ChunkStrategy) -> Vec<Vec<u32>> {
    let n = state.len();
    if n <= room || room == 0 {
        return vec![state[..n.min(room)].to_vec()];
    }
    match strategy {
        ChunkStrategy::TruncatePrefix => vec![state[..room].to_vec()],
        ChunkStrategy::HeadTail => {
            let head = room / 4;
            vec![[&state[..head], &state[n - (room - head)..]].concat()]
        }
        ChunkStrategy::SlidingWindow => {
            let step = (room * 2 / 3).max(1);
            let mut windows = Vec::new();
            let mut start = 0;
            loop {
                let end = (start + room).min(n);
                windows.push(state[start..end].to_vec());
                if end == n {
                    return windows;
                }
                start += step;
            }
        }
    }
}

fn framed(first: u32, body: &[u32], last: u32) -> Vec<u32> {
    let mut ids = Vec::with_capacity(body.len() + 2);
    ids.push(first);
    ids.extend_from_slice(body);
    ids.push(last);
    ids
}

/// Mean over tokens of a ModernBERT encoder's final hidden states for one unpadded sequence.
fn mean_hidden(encoder: &ModernBert, device: &Device, ids: &[u32]) -> Result<Vec<f32>> {
    let input = Tensor::new(ids, device)?.unsqueeze(0)?;
    let mask = Tensor::ones((1, ids.len()), DType::F32, device)?;
    Ok(encoder
        .forward(&input, &mask)?
        .squeeze(0)?
        .mean(0)?
        .to_vec1::<f32>()?)
}

fn pool(per_window: Vec<Vec<f64>>) -> Vec<f64> {
    if per_window.len() == 1 {
        per_window.into_iter().next().unwrap_or_default()
    } else {
        pool_window_logits(&per_window)
    }
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

    /// `x` is (batch, seq, d). `key_bias` is (batch, 1, 1, seq): 0 for real tokens and a large
    /// negative value for padding, so no query attends to padding.
    fn forward(&self, x: &Tensor, key_bias: &Tensor) -> candle_core::Result<Tensor> {
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
        let att = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?.broadcast_add(key_bias)?;
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
    pad_id: u32,
    mask_token: String,
    device: Device,
}

/// Load a tokenizer with its padding and truncation turned off. lemmata builds and cuts every
/// sequence itself, and some tokenizer.json files pad each encoding to a fixed length.
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow!(e))?;
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).map_err(|e| anyhow!(e))?;
    Ok(tokenizer)
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

        let tokenizer = load_tokenizer(&dir.join("tokenizer/tokenizer.json"))?;
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
        let (_, pad_id) = special("pad_token")?;

        let device = device()?;
        // SAFETY: the file is memory-mapped read-only and is not modified while lemmata runs.
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
            pad_id,
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

    /// `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`, the part
    /// before the state. Returns the token IDs and the position of each option's `[MASK]` marker.
    fn build_prefix(
        &self,
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
        Ok((ids, markers))
    }

    /// Tokens left for the state after the prefix and the final `[SEP]`.
    fn state_room(&self, prefix_len: usize) -> usize {
        self.max_len.saturating_sub(prefix_len + 1)
    }

    fn with_state(
        &self,
        prefix: &[u32],
        markers: &[usize],
        state: &[u32],
    ) -> (Vec<u32>, Vec<usize>) {
        let mut ids = prefix.to_vec();
        ids.extend_from_slice(state);
        ids.push(self.sep_id);
        ids.truncate(self.max_len);
        let markers = markers
            .iter()
            .copied()
            .filter(|&m| m < self.max_len)
            .collect();
        (ids, markers)
    }

    /// Upstream `build_sequence`: the prefix, then the state cut to fit, then `[SEP]`.
    pub fn build_sequence(
        &self,
        state: &str,
        qtype: QuestionType,
        instructions: &str,
        options: &[String],
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let (prefix, markers) = self.build_prefix(qtype, instructions, options)?;
        let state_ids = self.encode(&state.replace(&self.mask_token, " "))?;
        let room = self.state_room(prefix.len());
        Ok(self.with_state(&prefix, &markers, &state_ids[..state_ids.len().min(room)]))
    }

    /// Raw (pre-temperature) logit per option, in `options` order, for a truncated state.
    pub fn option_logits(
        &self,
        state: &str,
        qtype: QuestionType,
        instructions: &str,
        options: &[String],
    ) -> Result<Vec<f64>> {
        let (ids, markers) = self.build_sequence(state, qtype, instructions, options)?;
        Ok(self.forward_batch(&[(ids, markers, qtype)])?.remove(0))
    }

    /// Every question's windows go into one padded forward pass, then each question pools its
    /// own windows.
    fn score_batch(&self, reqs: &[ScoreRequest]) -> Result<Vec<Scored>> {
        let mut out: Vec<Option<Scored>> = (0..reqs.len()).map(|_| None).collect();
        let mut fitting = Vec::new();
        for (i, req) in reqs.iter().enumerate() {
            let lens = self.option_lengths(req.options)?;
            if lens.iter().sum::<usize>() + 16 > self.head_max_len {
                out[i] = Some(self.score_tournament(req, &lens)?);
            } else {
                fitting.push(i);
            }
        }
        let fitted = self.score_fitting(&fitting.iter().map(|&i| &reqs[i]).collect::<Vec<_>>())?;
        for (i, scored) in fitting.into_iter().zip(fitted) {
            out[i] = Some(scored);
        }
        Ok(out.into_iter().flatten().collect())
    }

    /// Tokens each option takes in the prefix: its `[MASK]` marker plus up to 48 text tokens.
    fn option_lengths(&self, options: &[String]) -> Result<Vec<usize>> {
        options
            .iter()
            .map(|o| {
                let text = self.encode(&format!(" {}", o.replace(&self.mask_token, " ")))?;
                Ok(1 + text.len().min(MAX_OPTION_TOKENS))
            })
            .collect()
    }

    /// Upstream shrinks every option evenly when the options overflow `head_max_len`, which cuts
    /// Banking77's 77 labels to 2 tokens each. Instead, score groups that fit, then score the
    /// leaders together. Eliminated options keep their first-round order, below every finalist.
    fn score_tournament(&self, req: &ScoreRequest, lens: &[usize]) -> Result<Scored> {
        let budget = self.head_max_len.saturating_sub(16);
        let fits = |idx: &[usize]| idx.iter().map(|&i| lens[i]).sum::<usize>() <= budget;
        let mut groups: Vec<Vec<usize>> = vec![Vec::new()];
        for i in 0..lens.len() {
            let mut candidate = groups.last().cloned().unwrap_or_default();
            candidate.push(i);
            if fits(&candidate) || groups.last().is_some_and(Vec::is_empty) {
                *groups.last_mut().unwrap() = candidate;
            } else {
                groups.push(vec![i]);
            }
        }
        let round = |group: &[usize]| -> Vec<String> {
            group.iter().map(|&i| req.options[i].clone()).collect()
        };

        let group_options: Vec<Vec<String>> = groups.iter().map(|g| round(g)).collect();
        let first: Vec<ScoreRequest> = group_options
            .iter()
            .map(|options| ScoreRequest { options, ..*req })
            .collect();
        let mut r1 = vec![f64::NEG_INFINITY; lens.len()];
        let mut input_tokens = 0;
        for (group, scored) in groups
            .iter()
            .zip(self.score_fitting(&first.iter().collect::<Vec<_>>())?)
        {
            input_tokens += scored.input_tokens;
            for (&i, z) in group.iter().zip(scored.logits) {
                r1[i] = z;
            }
        }

        // Finalists: the best first-round options that fit in one prefix, in their original order.
        let mut ranked: Vec<usize> = (0..lens.len()).collect();
        ranked.sort_by(|&a, &b| r1[b].total_cmp(&r1[a]));
        let mut finalists = Vec::new();
        for &i in &ranked {
            let mut next = finalists.clone();
            next.push(i);
            if !fits(&next) {
                break;
            }
            finalists = next;
        }
        finalists.sort_unstable();
        let final_options = round(&finalists);
        let last = ScoreRequest {
            options: &final_options,
            ..*req
        };
        let scored = self.score_fitting(&[&last])?.remove(0);
        input_tokens += scored.input_tokens;

        let mut logits = r1.clone();
        let lowest_final = scored.logits.iter().copied().fold(f64::INFINITY, f64::min);
        let best_out = (0..lens.len())
            .filter(|i| !finalists.contains(i))
            .map(|i| r1[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let offset = if best_out.is_finite() {
            best_out - lowest_final + 5.0
        } else {
            0.0
        };
        for z in logits.iter_mut() {
            *z -= offset;
        }
        for (&i, z) in finalists.iter().zip(scored.logits) {
            logits[i] = z;
        }
        Ok(Scored {
            logits,
            input_tokens,
        })
    }

    /// Requests whose options fit in the prefix: every window of every request in one pass.
    fn score_fitting(&self, reqs: &[&ScoreRequest]) -> Result<Vec<Scored>> {
        if reqs.is_empty() {
            return Ok(Vec::new());
        }
        let mut seqs = Vec::new();
        let mut owner = Vec::new();
        for (i, req) in reqs.iter().enumerate() {
            let q = req.question;
            let (prefix, markers) = self.build_prefix(q.qtype, &q.instructions, req.options)?;
            let state_ids = self.encode(&req.state.replace(&self.mask_token, " "))?;
            for window in state_windows(&state_ids, self.state_room(prefix.len()), req.strategy) {
                let (ids, m) = self.with_state(&prefix, &markers, &window);
                if m.len() != req.options.len() {
                    bail!(
                        "{} options do not fit in head_max_len={} tokens",
                        req.options.len(),
                        self.head_max_len
                    );
                }
                seqs.push((ids, m, q.qtype));
                owner.push(i);
            }
        }
        let tokens: Vec<usize> = seqs.iter().map(|(ids, _, _)| ids.len()).collect();
        let mut per_req: Vec<(Vec<Vec<f64>>, usize)> = vec![(Vec::new(), 0); reqs.len()];
        // ponytail: one batch for all windows; a very long sliding-window input pads every
        // question's sequences to its longest window. Split the batch if memory matters.
        for ((logits, i), n) in self
            .forward_batch(&seqs)?
            .into_iter()
            .zip(owner)
            .zip(tokens)
        {
            per_req[i].0.push(logits);
            per_req[i].1 += n;
        }
        Ok(per_req
            .into_iter()
            .map(|(windows, input_tokens)| Scored {
                logits: pool(windows),
                input_tokens,
            })
            .collect())
    }

    /// Raw option logits for several `(ids, markers, qtype)` sequences in one padded pass.
    fn forward_batch(
        &self,
        seqs: &[(Vec<u32>, Vec<usize>, QuestionType)],
    ) -> Result<Vec<Vec<f64>>> {
        let dev = &self.device;
        let (b, t) = (
            seqs.len(),
            seqs.iter().map(|s| s.0.len()).max().unwrap_or(0),
        );
        let mut ids = Vec::with_capacity(b * t);
        let mut mask = Vec::with_capacity(b * t);
        for (seq, _, _) in seqs {
            ids.extend_from_slice(seq);
            ids.extend(std::iter::repeat_n(self.pad_id, t - seq.len()));
            mask.extend((0..t).map(|i| if i < seq.len() { 1f32 } else { 0f32 }));
        }
        let input = Tensor::from_vec(ids, (b, t), dev)?;
        let mask = Tensor::from_vec(mask, (b, t), dev)?;
        // Upstream QTYPES order: choice = 0, score = 1, noul = 2.
        let qtypes = Tensor::new(seqs.iter().map(|s| s.2 as u32).collect::<Vec<_>>(), dev)?;
        let key_bias = ((mask.ones_like()? - &mask)? * -1e9)?.reshape((b, 1, 1, t))?;

        let mut h = self.encoder.forward(&input, &mask)?;
        h = h.broadcast_add(&self.type_emb.forward(&qtypes)?.unsqueeze(1)?)?;
        for layer in &self.head {
            h = layer.forward(&h, &key_bias)?;
        }
        // Gather every marker row from the flattened batch, then split per sequence.
        let rows: Vec<u32> = seqs
            .iter()
            .enumerate()
            .flat_map(|(i, s)| s.1.iter().map(move |&m| (i * t + m) as u32))
            .collect();
        let d = h.dim(2)?;
        let flat = h.reshape((b * t, d))?;
        let logits = flat
            .index_select(&Tensor::new(rows, dev)?, 0)?
            .apply(&self.scorer_norm)?
            .apply(&self.scorer_hidden)?
            .gelu_erf()?
            .apply(&self.scorer_out)?
            .squeeze(1)?
            .to_vec1::<f32>()?;
        let mut out = Vec::with_capacity(b);
        let mut at = 0;
        for (_, markers, _) in seqs {
            out.push(
                logits[at..at + markers.len()]
                    .iter()
                    .map(|&z| f64::from(z))
                    .collect(),
            );
            at += markers.len();
        }
        Ok(out)
    }
}

enum NliBackbone {
    /// `ModernBertForSequenceClassification`, rebuilt here: candle's version returns softmax
    /// output and drops its config when `label2id` holds integers.
    ModernBert {
        encoder: ModernBert,
        dense: Linear,
        norm: LayerNorm,
        classifier: Linear,
        mean_pool: bool,
    },
    /// `DebertaV2ForSequenceClassification` in parts, so probes can read the hidden states.
    Deberta {
        encoder: Box<DebertaV2Model>,
        pooler: Box<DebertaV2ContextPooler>,
        classifier: Linear,
    },
}

/// A zero-shot NLI cross-encoder: scores (text, hypothesis) pairs as entailment.
pub struct NliModel {
    tokenizer: Tokenizer,
    backbone: NliBackbone,
    entail: usize,
    not_entail: usize,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    max_len: usize,
    device: Device,
}

impl NliModel {
    /// Load a Hugging Face `*ForSequenceClassification` NLI directory (ModernBERT or DeBERTa-v2/v3).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg = read_json(&dir.join("config.json"))?;
        let tok_cfg = read_json(&dir.join("tokenizer_config.json"))?;
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json"))?;
        let special = |key: &str| -> Result<u32> {
            let tok = tok_cfg
                .get(key)
                .and_then(|v| v.as_str().or_else(|| v.get("content")?.as_str()))
                .ok_or_else(|| anyhow!("tokenizer_config.json has no {key}"))?;
            tokenizer
                .token_to_id(tok)
                .ok_or_else(|| anyhow!("{key} '{tok}' is not in the vocabulary"))
        };

        let labels: Vec<(usize, String)> = cfg
            .get("id2label")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("config.json has no id2label"))?
            .iter()
            .filter_map(|(k, v)| Some((k.parse().ok()?, v.as_str()?.to_lowercase())))
            .collect();
        let find =
            |pred: &dyn Fn(&str) -> bool| labels.iter().find(|(_, l)| pred(l)).map(|(i, _)| *i);
        let entail = find(&|l| l.starts_with("entail"))
            .ok_or_else(|| anyhow!("id2label has no entailment label"))?;
        let not_entail = find(&|l| l.starts_with("contra") || l.starts_with("not"))
            .or_else(|| labels.iter().map(|(i, _)| *i).find(|&i| i != entail))
            .ok_or_else(|| anyhow!("id2label needs a second label"))?;

        let device = device()?;
        // SAFETY: the file is memory-mapped read-only and is not modified while lemmata runs.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let backbone = match cfg.get("model_type").and_then(|v| v.as_str()) {
            Some("modernbert") => {
                let enc_cfg = encoder_config(&cfg)?;
                let d = enc_cfg.hidden_size;
                let eps = cfg.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5);
                NliBackbone::ModernBert {
                    encoder: ModernBert::load(vb.clone(), &enc_cfg)?,
                    dense: candle_nn::linear_no_bias(d, d, vb.pp("head.dense"))?,
                    norm: candle_nn::layer_norm_no_bias(d, eps, vb.pp("head.norm"))?,
                    classifier: candle_nn::linear(d, labels.len(), vb.pp("classifier"))?,
                    mean_pool: cfg.get("classifier_pooling").and_then(|v| v.as_str())
                        == Some("mean"),
                }
            }
            Some("deberta-v2") => {
                let deb_cfg: DebertaConfig = serde_json::from_value(cfg.clone())?;
                let pooler = DebertaV2ContextPooler::load(vb.pp("deberta"), &deb_cfg)?;
                let classifier =
                    candle_nn::linear(pooler.output_dim()?, labels.len(), vb.pp("classifier"))?;
                NliBackbone::Deberta {
                    encoder: Box::new(DebertaV2Model::load(vb.pp("deberta"), &deb_cfg)?),
                    pooler: Box::new(pooler),
                    classifier,
                }
            }
            other => bail!("unsupported NLI model_type {other:?}"),
        };

        Ok(Self {
            backbone,
            entail,
            not_entail,
            cls_id: special("cls_token")?,
            sep_id: special("sep_token")?,
            pad_id: special("pad_token")?,
            max_len: tok_cfg
                .get("model_max_length")
                .and_then(|v| v.as_u64())
                .map_or(DEFAULT_NLI_MAX_LEN, |n| {
                    (n as usize).min(DEFAULT_NLI_MAX_LEN)
                }),
            tokenizer,
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

    /// `[CLS] text [SEP] hypothesis [SEP]`, with only the text cut to fit (`only_first`).
    pub fn pair_ids(&self, text: &str, hypothesis: &str) -> Result<Vec<u32>> {
        let hyp = self.encode(hypothesis)?;
        let text = self.encode(text)?;
        let room = self.max_len.saturating_sub(hyp.len() + 3);
        Ok(self.join(&text[..text.len().min(room)], &hyp))
    }

    fn join(&self, text: &[u32], hyp: &[u32]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() + hyp.len() + 3);
        ids.push(self.cls_id);
        ids.extend_from_slice(text);
        ids.push(self.sep_id);
        ids.extend_from_slice(hyp);
        ids.push(self.sep_id);
        ids
    }

    /// Mean over tokens of the encoder's final hidden states for one unpadded sequence.
    fn mean_hidden(&self, ids: &[u32]) -> Result<Vec<f32>> {
        match &self.backbone {
            NliBackbone::ModernBert { encoder, .. } => mean_hidden(encoder, &self.device, ids),
            NliBackbone::Deberta { encoder, .. } => {
                let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
                Ok(encoder
                    .forward(&input, None, None)?
                    .squeeze(0)?
                    .mean(0)?
                    .to_vec1::<f32>()?)
            }
        }
    }

    /// Raw classifier logits for one encoded pair, in `id2label` order.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f64>> {
        let mut rows = self.forward_batch(&[ids.to_vec()])?;
        Ok(rows.remove(0))
    }

    /// Raw classifier logits for several encoded pairs in one forward pass. Shorter pairs are
    /// padded, and the attention mask and the pooling ignore the padding.
    pub fn forward_batch(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f64>>> {
        let len = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(seqs.len() * len);
        let mut mask = Vec::with_capacity(seqs.len() * len);
        for seq in seqs {
            ids.extend_from_slice(seq);
            ids.extend(std::iter::repeat_n(self.pad_id, len - seq.len()));
            mask.extend((0..len).map(|i| u8::from(i < seq.len())));
        }
        let shape = (seqs.len(), len);
        let input = Tensor::from_vec(ids, shape, &self.device)?;
        let mask = Tensor::from_vec(mask, shape, &self.device)?;
        let logits = match &self.backbone {
            NliBackbone::ModernBert {
                encoder,
                dense,
                norm,
                classifier,
                mean_pool,
            } => {
                let mask = mask.to_dtype(DType::F32)?;
                let h = encoder.forward(&input, &mask)?;
                let pooled = if *mean_pool {
                    let m = mask.unsqueeze(2)?;
                    h.broadcast_mul(&m)?.sum(1)?.broadcast_div(&m.sum(1)?)?
                } else {
                    h.get_on_dim(1, 0)?
                };
                pooled
                    .apply(dense)?
                    .gelu_erf()?
                    .apply(norm)?
                    .apply(classifier)?
            }
            NliBackbone::Deberta {
                encoder,
                pooler,
                classifier,
            } => {
                let h = encoder.forward(&input, None, Some(mask.to_dtype(DType::I64)?))?;
                pooler.forward(&h)?.apply(classifier)?
            }
        };
        Ok(logits
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| row.into_iter().map(f64::from).collect())
            .collect())
    }

    /// Choice and score options get one hypothesis each, scored by the entailment logit, which
    /// the zero-shot pipeline softmaxes across labels. A `noul` question is one hypothesis (its
    /// statement) scored as [not entailment, entailment]. Multi-label choice uses entailment minus
    /// not-entailment, so a sigmoid gives the pipeline's per-label probability.
    fn score(&self, req: &ScoreRequest) -> Result<Scored> {
        let q = req.question;
        // A `noul` question with false/true descriptions scores them as two competing hypotheses,
        // like a choice. Without descriptions, its statement is the one hypothesis.
        let noul_pair = q.qtype == QuestionType::Noul && q.noul_descriptions().is_some();
        let hypotheses: Vec<String> = if q.qtype == QuestionType::Noul && !noul_pair {
            vec![q.instructions.clone()]
        } else if noul_pair {
            req.options
                .iter()
                .map(|o| {
                    let desc = o.split_once(':').map_or(o.as_str(), |(_, d)| d).trim();
                    format!("{}.", desc.trim_end_matches('.'))
                })
                .collect()
        } else {
            req.options
                .iter()
                .map(|o| hypothesis(q, o, req.hypothesis_template))
                .collect()
        };
        let hyp_ids = hypotheses
            .iter()
            .map(|h| self.encode(h))
            .collect::<Result<Vec<_>>>()?;
        let longest = hyp_ids.iter().map(Vec::len).max().unwrap_or(0);
        let text = self.encode(req.state)?;

        let mut input_tokens = 0;
        let mut per_window = Vec::new();
        let room = self.max_len.saturating_sub(longest + 3);
        for window in state_windows(&text, room, req.strategy) {
            let pairs: Vec<Vec<u32>> = hyp_ids.iter().map(|h| self.join(&window, h)).collect();
            input_tokens += pairs.iter().map(Vec::len).sum::<usize>();
            let mut scores = Vec::with_capacity(req.options.len());
            for z in self.forward_batch(&pairs)? {
                let (ent, not) = (z[self.entail], z[self.not_entail]);
                if q.qtype == QuestionType::Noul && !noul_pair {
                    scores.extend([not, ent]);
                } else if req.multi_label {
                    scores.push(ent - not);
                } else {
                    scores.push(ent);
                }
            }
            per_window.push(scores);
        }
        Ok(Scored {
            logits: pool(per_window),
            input_tokens,
        })
    }
}

/// Longest label, in tokens, that the masked-LM scorer fills in.
const MAX_LABEL_TOKENS: usize = 8;
/// Masked-LM context cap. candle builds the full attention matrix, which needs gigabytes at 8k.
const MLM_MAX_LEN: usize = 2048;

enum MlmEncoder {
    EuroBert(super::eurobert::EuroBertMlm),
    NeoBert(super::neobert::NeoBertMlm),
}

impl MlmEncoder {
    fn logits_at(&self, ids: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::EuroBert(m) => m.mlm_logits_at(ids, positions),
            Self::NeoBert(m) => m.mlm_logits_at(ids, positions),
        }
    }

    fn mean_hidden(&self, ids: &[u32]) -> Result<Vec<f32>> {
        match self {
            Self::EuroBert(m) => m.mean_hidden(ids),
            Self::NeoBert(m) => m.mean_hidden(ids),
        }
    }
}

/// A pretrained masked-LM encoder used as a zero-shot classifier through label words (PET cloze
/// scoring, Schick & Schütze 2021): the label's tokens fill masks after the question.
pub struct MlmModel {
    tokenizer: Tokenizer,
    encoder: MlmEncoder,
    mask_id: u32,
    /// Special tokens the tokenizer puts around a single sequence, e.g. [CLS] ... [SEP].
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    max_len: usize,
    /// Content-free label scores per (question, labels), for contextual calibration.
    priors: Mutex<HashMap<String, Vec<f64>>>,
}

impl MlmModel {
    pub fn load(ckpt: Checkpoint, dir: &Path) -> Result<Self> {
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json"))?;
        let cfg = read_json(&dir.join("config.json"))?;
        let mask_token = read_json(&dir.join("tokenizer_config.json"))?
            .get("mask_token")
            .and_then(|v| {
                v.as_str()
                    .map(str::to_string)
                    .or_else(|| v.get("content")?.as_str().map(str::to_string))
            })
            .ok_or_else(|| anyhow!("tokenizer_config.json has no mask_token"))?;
        let mask_id = tokenizer
            .token_to_id(&mask_token)
            .ok_or_else(|| anyhow!("mask token '{mask_token}' is not in the vocabulary"))?;

        // Find the special tokens around one sequence by encoding a probe with and without them.
        let with = tokenizer.encode("probe", true).map_err(|e| anyhow!(e))?;
        let bare = tokenizer.encode("probe", false).map_err(|e| anyhow!(e))?;
        let (with, bare) = (with.get_ids(), bare.get_ids());
        let start = with
            .windows(bare.len())
            .position(|w| w == bare)
            .ok_or_else(|| anyhow!("cannot locate the text inside a special-token encoding"))?;

        let encoder = match ckpt {
            Checkpoint::EuroBert => MlmEncoder::EuroBert(super::eurobert::EuroBertMlm::load(dir)?),
            Checkpoint::NeoBert => MlmEncoder::NeoBert(super::neobert::NeoBertMlm::load(dir)?),
            other => bail!("{other} is not a masked-LM checkpoint"),
        };
        let positions = ["max_position_embeddings", "max_length"]
            .iter()
            .find_map(|k| cfg.get(*k).and_then(|v| v.as_u64()))
            .unwrap_or(512) as usize;
        Ok(Self {
            prefix: with[..start].to_vec(),
            suffix: with[start + bare.len()..].to_vec(),
            max_len: positions.min(MLM_MAX_LEN),
            priors: Mutex::default(),
            tokenizer,
            encoder,
            mask_id,
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

    /// The words that stand for each option: the choice label, the score level's description,
    /// or "no" / "yes" for a `noul` statement.
    fn label_words(q: &Question, options: &[String]) -> Vec<String> {
        use super::render_nli_hypothesis as render;
        if q.qtype == QuestionType::Noul {
            return vec!["no".into(), "yes".into()];
        }
        options
            .iter()
            .map(|o| {
                level_description(q, o)
                    .unwrap_or_else(|| render(q, o, Some("{label}")).replace('_', " "))
            })
            .collect()
    }

    /// Log-probability of each label (the sum over its tokens), with one mask per token after
    /// `question`. Returns the scores and the number of tokens the model read.
    fn label_log_probs(
        &self,
        context: &[u32],
        question: &[u32],
        labels: &[Vec<u32>],
    ) -> Result<(Vec<f64>, usize)> {
        let end = self.encode(".")?;
        let mut scores = vec![0.0; labels.len()];
        let mut tokens = 0;
        // One forward pass per distinct label length, with that many masks.
        let mut lengths: Vec<usize> = labels.iter().map(Vec::len).collect();
        lengths.sort_unstable();
        lengths.dedup();
        for n in lengths {
            let mut ids = self.prefix.clone();
            ids.extend_from_slice(context);
            ids.extend_from_slice(question);
            let first_mask = ids.len();
            ids.extend(std::iter::repeat_n(self.mask_id, n));
            ids.extend_from_slice(&end);
            ids.extend_from_slice(&self.suffix);
            tokens += ids.len();
            let positions: Vec<usize> = (first_mask..first_mask + n).collect();
            let log_probs: Vec<Vec<f64>> = self
                .encoder
                .logits_at(&ids, &positions)?
                .iter()
                .map(|row| log_softmax(row))
                .collect();
            for (score, label) in scores.iter_mut().zip(labels) {
                if label.len() == n {
                    *score = label
                        .iter()
                        .zip(&log_probs)
                        .map(|(&t, row)| row[t as usize])
                        .sum::<f64>();
                }
            }
        }
        Ok((scores, tokens))
    }

    /// Each option's score is the log-probability of its label after the question, minus the same
    /// score for the content-free input "N/A" (contextual calibration, Zhao et al. 2021): the
    /// pointwise mutual information between label and text. The subtraction removes the label
    /// words' prior, which otherwise makes common words such as "other" win. Summing over tokens
    /// (not averaging) keeps multi-token labels comparable: an easy token such as "/" in
    /// "Sci/Tech" scores about the same with and without the text, so it cancels.
    fn score(&self, req: &ScoreRequest) -> Result<Scored> {
        let q = req.question;
        let words = Self::label_words(q, req.options);
        let ins = q.instructions.trim();
        let question = if q.qtype == QuestionType::Noul && ins.ends_with('?') {
            format!("\nQuestion: {ins}\nAnswer:")
        } else if q.qtype == QuestionType::Noul {
            format!("\nStatement: {ins}\nIs the statement true? Answer:")
        } else {
            // No options list: with one, the model copies the only option that fits the number of
            // masks (for example the 3-token "Sci/Tech"), whatever the text says.
            format!("\nQuestion: {ins}\nAnswer:")
        };
        let question = self.encode(&question)?;
        let labels = words
            .iter()
            .map(|w| {
                let mut ids = self.encode(&format!(" {w}"))?;
                ids.truncate(MAX_LABEL_TOKENS);
                if ids.is_empty() {
                    bail!("option label '{w}' has no tokens");
                }
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;

        let key = format!("{question:?}{labels:?}");
        let cached = self.priors.lock().ok().and_then(|p| p.get(&key).cloned());
        let prior = match cached {
            Some(prior) => prior,
            None => {
                let (prior, _) = self.label_log_probs(&self.encode("N/A")?, &question, &labels)?;
                if let Ok(mut priors) = self.priors.lock() {
                    priors.insert(key, prior.clone());
                }
                prior
            }
        };

        let longest = labels.iter().map(Vec::len).max().unwrap_or(1);
        let overhead = self.prefix.len() + question.len() + longest + 1 + self.suffix.len();
        let state = self.encode(req.state)?;
        let mut input_tokens = 0;
        let mut per_window = Vec::new();
        for window in state_windows(&state, self.max_len.saturating_sub(overhead), req.strategy) {
            let (lp, tokens) = self.label_log_probs(&window, &question, &labels)?;
            input_tokens += tokens;
            per_window.push(lp.iter().zip(&prior).map(|(l, p)| l - p).collect());
        }
        Ok(Scored {
            logits: pool(per_window),
            input_tokens,
        })
    }
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max
        + row
            .iter()
            .map(|&z| (z as f64 - max).exp())
            .sum::<f64>()
            .ln();
    row.iter().map(|&z| z as f64 - lse).collect()
}

/// A score level's own description, without the "level N:" and "N:" prefixes that schemas add.
fn level_description(q: &Question, option: &str) -> Option<String> {
    if q.qtype != QuestionType::Score {
        return None;
    }
    let desc = option.split_once(':')?.1.trim();
    let desc = match desc.split_once(':') {
        Some((n, rest)) if n.trim().chars().all(|c| c.is_ascii_digit()) => rest.trim(),
        _ => desc,
    };
    (!desc.is_empty()).then(|| desc.trim_end_matches('.').to_string())
}

/// The question's or the caller's template when one is set. Otherwise "This text is about
/// {label}.", with the option's description in parentheses when it has one. A score level is
/// "This text is about {description}.".
fn hypothesis(q: &Question, option: &str, template: Option<&str>) -> String {
    use super::render_nli_hypothesis as render;
    if template.is_some() || q.hypothesis_template.is_some() {
        return render(q, option, template);
    }
    if let Some(desc) = level_description(q, option) {
        return format!("This text is about {desc}.");
    }
    let raw = render(q, option, Some("{label}"));
    // Underscored labels such as `activate_my_card` read better as words.
    let label = raw.replace('_', " ");
    let desc = render(q, option, Some("{description}"));
    // The renderer falls back to the instructions (or the label) when the option has no description.
    if desc == raw || desc == q.instructions.trim() {
        format!("This text is about {label}.")
    } else {
        format!("This text is about {label} ({desc}).")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_windows_follow_the_strategy() {
        let ids: Vec<u32> = (0..20).collect();
        assert_eq!(
            state_windows(&ids, 30, ChunkStrategy::HeadTail),
            vec![ids.clone()]
        );
        assert_eq!(
            state_windows(&ids, 8, ChunkStrategy::TruncatePrefix),
            vec![(0..8).collect::<Vec<_>>()]
        );
        // Head+tail: the first quarter of the budget, then the end of the text.
        assert_eq!(
            state_windows(&ids, 8, ChunkStrategy::HeadTail),
            vec![vec![0, 1, 14, 15, 16, 17, 18, 19]]
        );
        // Sliding windows: step 5 (two thirds of 8), and the last window reaches the end.
        let windows = state_windows(&ids, 8, ChunkStrategy::SlidingWindow);
        assert_eq!(windows.first().unwrap(), &(0..8).collect::<Vec<_>>());
        assert_eq!(windows.last().unwrap().last(), Some(&19));
        assert!(windows.iter().all(|w| w.len() <= 8));
    }
}
