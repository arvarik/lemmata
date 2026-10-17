//! Supervised linear probes: softmax regression on a native model's mean-pooled embeddings.
//!
//! A probe learns one choice question from labelled examples. Base encoders without a
//! classifier (EuroBERT, NeoBERT) score far better through a probe than through zero-shot label
//! words, and every model gains on tasks with many labels.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{D, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::datasource::DataItem;
use crate::model::{Checkpoint, ChunkStrategy, Question, QuestionType};

/// File format of this build's probes.
pub const PROBE_KIND: &str = "lemma-probe-v1";
/// L2 strengths tried on the held-out fifth of the training data.
const L2_GRID: [f64; 4] = [1e-4, 1e-3, 1e-2, 1e-1];
const ITERATIONS: usize = 1000;

/// A fitted probe for one choice question of one checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub kind: String,
    pub checkpoint: Checkpoint,
    pub question_id: String,
    pub labels: Vec<String>,
    /// Per-dimension mean and standard deviation of the training embeddings.
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// Row-major `labels × dim` weights, and one bias per label.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub l2: f64,
    pub train_samples: usize,
    /// Accuracy on the held-out fifth, with the chosen L2.
    pub validation_accuracy: f64,
    /// Zero-shot accuracy of the same checkpoint on the same held-out fifth.
    #[serde(default)]
    pub zero_shot_validation_accuracy: f64,
}

impl Probe {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read probe {}", path.display()))?;
        let probe: Probe = serde_json::from_str(&text)
            .with_context(|| format!("Malformed probe {}", path.display()))?;
        if probe.kind != PROBE_KIND {
            bail!(
                "{} is a '{}' probe, and this build reads '{PROBE_KIND}'. Refit it with `lemma probe fit`.",
                path.display(),
                probe.kind
            );
        }
        Ok(probe)
    }

    /// One logit per label, in `labels` order.
    pub fn logits(&self, embedding: &[f32]) -> Vec<f64> {
        let d = self.mean.len();
        let x: Vec<f32> = embedding
            .iter()
            .zip(&self.mean)
            .zip(&self.std)
            .map(|((e, m), s)| (e - m) / s)
            .collect();
        self.bias
            .iter()
            .enumerate()
            .map(|(j, b)| {
                let row = &self.weights[j * d..(j + 1) * d];
                f64::from(b + row.iter().zip(&x).map(|(w, v)| w * v).sum::<f32>())
            })
            .collect()
    }
}

/// The choice labels of `q`, in schema order.
pub fn choice_labels(q: &Question) -> Vec<String> {
    match &q.criteria {
        Some(serde_json::Value::Object(map)) => map.keys().cloned().collect(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.split(':').next().unwrap_or(s).trim().to_string())
            .collect(),
        _ => Vec::new(),
    }
}

/// Fit a probe for choice question `qid` on labelled `items`, with `ckpt`'s real weights.
pub fn fit_probe(
    ckpt: Checkpoint,
    items: &[DataItem],
    qid: &str,
    q: &Question,
    strategy: ChunkStrategy,
) -> Result<Probe> {
    if q.qtype != QuestionType::Choice {
        bail!("probes support choice questions, and '{qid}' is not one");
    }
    let labels = choice_labels(q);
    let index: BTreeMap<String, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.to_lowercase(), i))
        .collect();
    let model = crate::model::native::shared(ckpt, &ckpt.local_dir())?;
    let questions = BTreeMap::from([(qid.to_string(), q.clone())]);

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut kept = Vec::new();
    for item in items {
        let truth = crate::engine::metrics::resolve_ground_truth_for_question(
            item.ground_truth.as_ref(),
            qid,
            q,
            questions.len(),
        )
        .and_then(|v| v.as_str().map(str::to_lowercase))
        .and_then(|t| index.get(&t).copied());
        let Some(y) = truth else { continue };
        let text = match &item.state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };
        xs.push(model.embed(&text, strategy)?);
        ys.push(y);
        kept.push(item);
    }
    if xs.len() < 10 {
        bail!(
            "'{qid}' has {} labelled items that match its labels. A probe needs at least 10.",
            xs.len()
        );
    }

    let d = xs[0].len();
    let n = xs.len();
    let mut mean = vec![0f32; d];
    for x in &xs {
        for (m, v) in mean.iter_mut().zip(x) {
            *m += v / n as f32;
        }
    }
    let mut std = vec![0f32; d];
    for x in &xs {
        for ((s, v), m) in std.iter_mut().zip(x).zip(&mean) {
            *s += (v - m).powi(2) / n as f32;
        }
    }
    let std: Vec<f32> = std.into_iter().map(|v| v.sqrt() + 1e-6).collect();
    let rows: Vec<Vec<f32>> = xs
        .iter()
        .map(|x| {
            x.iter()
                .zip(&mean)
                .zip(&std)
                .map(|((v, m), s)| (v - m) / s)
                .collect()
        })
        .collect();

    // Hold out every fifth item to choose the L2 strength, then refit on everything.
    let (train, valid): (Vec<usize>, Vec<usize>) = (0..n).partition(|i| i % 5 != 4);
    let k = labels.len();
    let mut best = (f64::NEG_INFINITY, L2_GRID[0]);
    for l2 in L2_GRID {
        let (w, b) = softmax_regression(&rows, &ys, &train, k, l2)?;
        let correct = valid
            .iter()
            .filter(|&&i| argmax(&linear(&w, &b, &rows[i], d)) == ys[i])
            .count();
        let acc = correct as f64 / valid.len().max(1) as f64;
        // On ties, prefer the stronger penalty.
        if acc >= best.0 {
            best = (acc, l2);
        }
    }
    let all: Vec<usize> = (0..n).collect();
    let (weights, bias) = softmax_regression(&rows, &ys, &all, k, best.1)?;

    // The same held-out items, scored zero-shot, so the caller can see whether the probe helps.
    let zero_shot =
        crate::engine::LayaClassifier::with_checkpoint(ckpt).with_chunk_strategy(strategy);
    let mut zs_correct = 0;
    for &i in &valid {
        let res = zero_shot.classify(&kept[i].state, &questions)?;
        let choice = res.answers[qid].choice.as_deref().map(str::to_lowercase);
        zs_correct += usize::from(choice.as_deref() == Some(labels[ys[i]].to_lowercase().as_str()));
    }
    Ok(Probe {
        kind: PROBE_KIND.to_string(),
        checkpoint: ckpt,
        question_id: qid.to_string(),
        labels,
        mean,
        std,
        weights,
        bias,
        l2: best.1,
        train_samples: n,
        validation_accuracy: best.0,
        zero_shot_validation_accuracy: zs_correct as f64 / valid.len().max(1) as f64,
    })
}

fn linear(w: &[f32], b: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    b.iter()
        .enumerate()
        .map(|(j, bj)| {
            bj + w[j * d..(j + 1) * d]
                .iter()
                .zip(x)
                .map(|(a, v)| a * v)
                .sum::<f32>()
        })
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

/// Full-batch gradient descent on mean cross-entropy plus `l2 · |W|²`. The step is
/// `1 / (L / 2 + l2)`, with `L` the largest eigenvalue of `XᵀX / n` (power iteration), which
/// keeps the convex objective's descent stable for any feature scale.
fn softmax_regression(
    rows: &[Vec<f32>],
    ys: &[usize],
    subset: &[usize],
    k: usize,
    l2: f64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let dev = Device::Cpu;
    let n = subset.len();
    let d = rows[0].len();
    let flat: Vec<f32> = subset
        .iter()
        .flat_map(|&i| rows[i].iter().copied())
        .collect();
    let x = Tensor::from_vec(flat, (n, d), &dev)?;
    let mut onehot = vec![0f32; n * k];
    for (r, &i) in subset.iter().enumerate() {
        onehot[r * k + ys[i]] = 1.0;
    }
    let y = Tensor::from_vec(onehot, (n, k), &dev)?;

    let xt = x.t()?.contiguous()?;
    let mut v = Tensor::ones((d, 1), candle_core::DType::F32, &dev)?;
    let mut eig = 1.0f32;
    for _ in 0..30 {
        let next = (xt.matmul(&x.matmul(&v)?)? / n as f64)?;
        eig = next.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        v = (next / f64::from(eig.max(1e-12)))?;
    }
    let lr = 1.0 / (f64::from(eig) / 2.0 + l2);

    let mut w = Tensor::zeros((d, k), candle_core::DType::F32, &dev)?;
    let mut b = Tensor::zeros(k, candle_core::DType::F32, &dev)?;
    for _ in 0..ITERATIONS {
        let z = x.matmul(&w)?.broadcast_add(&b)?;
        let p = candle_nn::ops::softmax(&z, D::Minus1)?;
        let g = ((p - &y)? / n as f64)?;
        let gw = (xt.matmul(&g)? + (&w * (2.0 * l2))?)?;
        let gb = g.sum(0)?;
        w = (w - (gw * lr)?)?;
        b = (b - (gb * lr)?)?;
    }
    // Store `labels × dim`, so each label's weights are one contiguous row.
    let weights = w.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    Ok((weights, b.to_vec1::<f32>()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_regression_separates_shifted_clusters() {
        // Two labels whose embeddings differ only in the first dimension.
        let rows: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                vec![sign + (i as f32 * 0.01), 0.3, -0.2]
            })
            .collect();
        let ys: Vec<usize> = (0..40).map(|i| i % 2).collect();
        let all: Vec<usize> = (0..40).collect();
        let (w, b) = softmax_regression(&rows, &ys, &all, 2, 1e-3).unwrap();
        let correct = all
            .iter()
            .filter(|&&i| argmax(&linear(&w, &b, &rows[i], 3)) == ys[i])
            .count();
        assert_eq!(correct, 40);
    }
}
