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
    Config as DebertaConfig, DebertaV2SeqClassificationModel,
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

static BACKEND: AtomicU8 = AtomicU8::new(Backend::Mock as u8);
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

/// Checkpoints lemma can run natively. `Jev` and `ModernBertNli` share another model's
/// directory, so loading that directory would mislabel the result.
pub fn has_native_impl(ckpt: Checkpoint) -> bool {
    matches!(
        ckpt,
        Checkpoint::English
            | Checkpoint::Multilingual
            | Checkpoint::TypedDecisions
            | Checkpoint::ModernBertZeroShot
            | Checkpoint::DebertaV3ZeroShot
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
            "lemma has no native implementation for it".to_string()
        };
        eprintln!(
            "⚠️  {ckpt}: {why}. Its decisions are simulated by the keyword mock (model id mock:{ckpt})."
        );
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
            "lemma has no native inference for {ckpt}. Native inference supports english, multilingual, typed-decisions, modernbert-zeroshot and deberta-v3. Run with --backend auto or mock to simulate it."
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
    let model = if Checkpoint::laya_checkpoints().contains(&ckpt) {
        LayaModel::load(dir).map(|m| NativeModel::Laya(Box::new(m)))
    } else {
        NliModel::load(dir).map(|m| NativeModel::Nli(Box::new(m)))
    }
    .with_context(|| {
        format!(
            "Failed to load {ckpt} from {}. Run `lemma download` first.",
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
}

impl NativeModel {
    pub fn score(&self, req: &ScoreRequest) -> Result<Scored> {
        match self {
            Self::Laya(m) => m.score(req),
            Self::Nli(m) => m.score(req),
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
        self.forward(&ids, &markers, qtype, options.len())
    }

    fn score(&self, req: &ScoreRequest) -> Result<Scored> {
        let q = req.question;
        let (prefix, markers) = self.build_prefix(q.qtype, &q.instructions, req.options)?;
        let state_ids = self.encode(&req.state.replace(&self.mask_token, " "))?;
        let mut input_tokens = 0;
        let mut per_window = Vec::new();
        for window in state_windows(&state_ids, self.state_room(prefix.len()), req.strategy) {
            let (ids, markers) = self.with_state(&prefix, &markers, &window);
            input_tokens += ids.len();
            per_window.push(self.forward(&ids, &markers, q.qtype, req.options.len())?);
        }
        Ok(Scored {
            logits: pool(per_window),
            input_tokens,
        })
    }

    // ponytail: one sequence per forward pass on CPU. Batch questions with a padding mask, or add
    // a Metal/CUDA device, if throughput matters.
    fn forward(
        &self,
        ids: &[u32],
        markers: &[usize],
        qtype: QuestionType,
        n_options: usize,
    ) -> Result<Vec<f64>> {
        if markers.len() != n_options {
            bail!(
                "{n_options} options do not fit in head_max_len={} tokens",
                self.head_max_len
            );
        }
        let dev = &self.device;
        let input = Tensor::new(ids, dev)?.unsqueeze(0)?;
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
    Deberta(Box<DebertaV2SeqClassificationModel>),
}

/// A zero-shot NLI cross-encoder: scores (text, hypothesis) pairs as entailment.
pub struct NliModel {
    tokenizer: Tokenizer,
    backbone: NliBackbone,
    entail: usize,
    not_entail: usize,
    cls_id: u32,
    sep_id: u32,
    max_len: usize,
    device: Device,
}

impl NliModel {
    /// Load a Hugging Face `*ForSequenceClassification` NLI directory (ModernBERT or DeBERTa-v2/v3).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg = read_json(&dir.join("config.json"))?;
        let tok_cfg = read_json(&dir.join("tokenizer_config.json"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!(e))?;
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

        let device = Device::Cpu;
        // SAFETY: the file is memory-mapped read-only and is not modified while lemma runs.
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
                NliBackbone::Deberta(Box::new(DebertaV2SeqClassificationModel::load(
                    vb.pp("deberta"),
                    &deb_cfg,
                    None,
                )?))
            }
            other => bail!("unsupported NLI model_type {other:?}"),
        };

        Ok(Self {
            backbone,
            entail,
            not_entail,
            cls_id: special("cls_token")?,
            sep_id: special("sep_token")?,
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

    /// Raw classifier logits for one encoded pair, in `id2label` order.
    // ponytail: one pair per forward pass. Batch the K hypotheses with a padding mask if throughput matters.
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f64>> {
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let logits = match &self.backbone {
            NliBackbone::ModernBert {
                encoder,
                dense,
                norm,
                classifier,
                mean_pool,
            } => {
                let mask = Tensor::ones((1, ids.len()), DType::F32, &self.device)?;
                let h = encoder.forward(&input, &mask)?;
                // One unpadded sequence, so the masked mean is the plain mean.
                let pooled = if *mean_pool {
                    h.mean(1)?
                } else {
                    h.get_on_dim(1, 0)?
                };
                pooled
                    .apply(dense)?
                    .gelu_erf()?
                    .apply(norm)?
                    .apply(classifier)?
            }
            NliBackbone::Deberta(model) => model.forward(&input, None, None)?,
        };
        Ok(logits
            .squeeze(0)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    }

    /// Choice and score options get one hypothesis each, scored by the entailment logit, which
    /// the zero-shot pipeline softmaxes across labels. A `noul` question is one hypothesis (its
    /// statement) scored as [not entailment, entailment]. Multi-label choice uses entailment minus
    /// not-entailment, so a sigmoid gives the pipeline's per-label probability.
    fn score(&self, req: &ScoreRequest) -> Result<Scored> {
        let q = req.question;
        let hypotheses: Vec<String> = if q.qtype == QuestionType::Noul {
            vec![q.instructions.clone()]
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
            let mut scores = Vec::with_capacity(req.options.len());
            for hyp in &hyp_ids {
                let ids = self.join(&window, hyp);
                input_tokens += ids.len();
                let z = self.forward(&ids)?;
                let (ent, not) = (z[self.entail], z[self.not_entail]);
                if q.qtype == QuestionType::Noul {
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

/// The question's or the caller's template when one is set. Otherwise "This text is about
/// {label}.", with the option's description in parentheses when it has one.
fn hypothesis(q: &Question, option: &str, template: Option<&str>) -> String {
    use super::render_nli_hypothesis as render;
    if template.is_some() || q.hypothesis_template.is_some() {
        return render(q, option, template);
    }
    let label = render(q, option, Some("{label}"));
    let desc = render(q, option, Some("{description}"));
    // The renderer falls back to the instructions when the option has no description.
    if desc == label || desc == q.instructions.trim() {
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
