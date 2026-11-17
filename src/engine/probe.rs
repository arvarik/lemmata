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
pub const PROBE_KIND: &str = "lemmata-probe-v1";
/// L2 strengths tried on the held-out fifth of the training data. The large values let a
/// residual probe shrink to the zero-shot logits plus a per-label bias.
const L2_GRID: [f64; 6] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0];
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
    /// The probe's logits are added to the checkpoint's zero-shot logits for the question, so it
    /// learns a correction instead of replacing the zero-shot head. `fit_probe` keeps this when it
    /// beats the plain probe on the held-out fifth.
    #[serde(default)]
    pub residual: bool,
    /// The zero-shot logits are appended to the embedding as extra features, so `mean`, `std`,
    /// and each weight row have one entry per label after the embedding's. With enough data this
    /// learns how far to trust the zero-shot head, which a residual probe cannot rescale.
    #[serde(default)]
    pub stacked: bool,
}

impl Probe {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read probe {}", path.display()))?;
        let probe: Probe = serde_json::from_str(&text)
            .with_context(|| format!("Malformed probe {}", path.display()))?;
        if probe.kind != PROBE_KIND {
            bail!(
                "{} is a '{}' probe, and this build reads '{PROBE_KIND}'. Refit it with `lemmata probe fit`.",
                path.display(),
                probe.kind
            );
        }
        probe.validate_dimensions(path)?;
        Ok(probe)
    }

    fn validate_dimensions(&self, path: &Path) -> Result<()> {
        let p = path.display();
        if self.labels.is_empty() {
            bail!("Probe {p} has no labels");
        }
        if self.mean.is_empty() || self.mean.len() != self.std.len() {
            bail!(
                "Probe {p} has mismatched standardization vectors (mean={}, std={})",
                self.mean.len(),
                self.std.len()
            );
        }
        if self.bias.len() != self.labels.len() {
            bail!(
                "Probe {p} bias length ({}) does not match label count ({})",
                self.bias.len(),
                self.labels.len()
            );
        }
        let expected_weights = self
            .labels
            .len()
            .checked_mul(self.mean.len())
            .ok_or_else(|| anyhow::anyhow!("Probe {p} weight matrix dimensions overflow"))?;
        if self.weights.len() != expected_weights {
            bail!(
                "Probe {p} weight count ({}) does not match labels * dim ({} * {} = {})",
                self.weights.len(),
                self.labels.len(),
                self.mean.len(),
                expected_weights
            );
        }
        if self.residual && self.stacked {
            bail!("Probe {p} cannot be both residual and stacked simultaneously");
        }
        if self.stacked && self.mean.len() <= self.labels.len() {
            bail!(
                "Stacked probe {p} feature dimension ({}) must exceed label count ({})",
                self.mean.len(),
                self.labels.len()
            );
        }
        if self.std.iter().any(|s| !s.is_finite() || *s <= 0.0)
            || self.mean.iter().any(|v| !v.is_finite())
            || self.weights.iter().any(|v| !v.is_finite())
            || self.bias.iter().any(|v| !v.is_finite())
            || !self.l2.is_finite()
        {
            bail!("Probe {p} contains non-finite or non-positive parameters");
        }
        Ok(())
    }

    /// Whether scoring needs the checkpoint's zero-shot logits.
    pub fn uses_zero_shot(&self) -> bool {
        self.residual || self.stacked
    }

    /// Length of the embedding the probe expects, without stacked zero-shot features.
    pub fn embedding_dim(&self) -> usize {
        self.mean.len() - if self.stacked { self.labels.len() } else { 0 }
    }

    pub fn mode(&self) -> &'static str {
        match (self.residual, self.stacked) {
            (true, _) => "residual",
            (_, true) => "stacked",
            _ => "plain",
        }
    }

    /// One logit per label, in `labels` order. `zero_shot` holds the checkpoint's raw logits for
    /// the question, rounded to 4 decimals as answers report them. A residual probe adds them, and
    /// a stacked probe reads them as features.
    pub fn logits(&self, embedding: &[f32], zero_shot: Option<&[f64]>) -> Vec<f64> {
        let base = zero_shot.filter(|_| self.residual);
        let d = self.mean.len();
        let stacked: Vec<f32> = zero_shot
            .filter(|_| self.stacked)
            .map(|z| z.iter().map(|&v| v as f32).collect())
            .unwrap_or_default();
        let x: Vec<f32> = embedding
            .iter()
            .chain(&stacked)
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
                    + base.map_or(0.0, |z| z[j])
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

    let mut matched: Vec<(&DataItem, usize)> = Vec::new();
    for item in items {
        let truth = crate::engine::metrics::resolve_ground_truth_for_question(
            item.ground_truth.as_ref(),
            qid,
            q,
            questions.len(),
        )
        .and_then(|v| v.as_str().map(str::to_lowercase))
        .and_then(|t| index.get(&t).copied());
        if let Some(y) = truth {
            matched.push((item, y));
        }
    }
    if matched.len() < 10 {
        bail!(
            "'{qid}' has {} labelled items that match its labels. A probe needs at least 10.",
            matched.len()
        );
    }

    let ys: Vec<usize> = matched.iter().map(|&(_, y)| y).collect();
    let kept_items: Vec<DataItem> = matched.iter().map(|&(it, _)| it.clone()).collect();
    let xs: Vec<Vec<f32>> = crate::model::native::map_items(&kept_items, |item| {
        let text = item.state_text();
        model.embed(&text, strategy)
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    // Zero-shot logits for every item: the residual probe's base, and the baseline to beat.
    let zero_shot =
        crate::engine::LayaClassifier::with_checkpoint(ckpt).with_chunk_strategy(strategy);
    let zs_batch = zero_shot.classify_batch(&kept_items, &questions)?;
    let mut zs_logits = Vec::with_capacity(kept_items.len());
    let mut zs_hit = Vec::with_capacity(kept_items.len());
    for ((_, mut res), &y) in zs_batch.into_iter().zip(&ys) {
        let ans = res
            .answers
            .remove(qid)
            .ok_or_else(|| anyhow::anyhow!("no zero-shot answer for '{qid}'"))?;
        zs_hit.push(ans.choice.as_deref().map(str::to_lowercase) == Some(labels[y].to_lowercase()));
        zs_logits.push(
            ans.logits
                .unwrap_or_default()
                .iter()
                .map(|&z| z as f32)
                .collect::<Vec<_>>(),
        );
    }

    let k = labels.len();
    if zs_logits.iter().any(|z| z.len() != k) {
        bail!("zero-shot scoring of '{qid}' did not return one logit per label");
    }
    let stacked_xs: Vec<Vec<f32>> = xs
        .iter()
        .zip(&zs_logits)
        .map(|(x, z)| x.iter().chain(z).copied().collect())
        .collect();
    // Keep the best held-out accuracy. On ties, prefer the residual probe, which stays closest
    // to zero-shot, then the stacked one.
    let (fit, residual, stacked) = [
        (fit_features(&xs, &ys, k, Some(&zs_logits))?, true, false),
        (fit_features(&stacked_xs, &ys, k, None)?, false, true),
        (fit_features(&xs, &ys, k, None)?, false, false),
    ]
    .into_iter()
    .reduce(|best, c| {
        if c.0.validation_accuracy > best.0.validation_accuracy {
            c
        } else {
            best
        }
    })
    .expect("three candidates");

    let n = ys.len();
    let (_, valid) = stratified_train_valid_split(&ys, k);
    let zs_correct = valid.iter().filter(|&&i| zs_hit[i]).count();
    Ok(Probe {
        kind: PROBE_KIND.to_string(),
        checkpoint: ckpt,
        question_id: qid.to_string(),
        labels,
        mean: fit.mean,
        std: fit.std,
        weights: fit.weights,
        bias: fit.bias,
        l2: fit.l2,
        train_samples: n,
        validation_accuracy: fit.validation_accuracy,
        zero_shot_validation_accuracy: zs_correct as f64 / valid.len().max(1) as f64,
        residual,
        stacked,
    })
}

struct FeatureFit {
    mean: Vec<f32>,
    std: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    l2: f64,
    validation_accuracy: f64,
}

/// Deterministic per-class stratified 80/20 split so class-sorted or round-robin datasets
/// never starve `train` or `valid` of an entire class.
fn stratified_train_valid_split(ys: &[usize], k: usize) -> (Vec<usize>, Vec<usize>) {
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); k.max(1)];
    for (idx, &y) in ys.iter().enumerate() {
        if let Some(bucket) = by_class.get_mut(y) {
            bucket.push(idx);
        }
    }
    let mut is_valid = vec![false; ys.len()];
    for indices in &by_class {
        let m = indices.len();
        if m >= 5 {
            for (pos, &idx) in indices.iter().enumerate() {
                if pos % 5 == 4 {
                    is_valid[idx] = true;
                }
            }
        } else if m >= 2 {
            is_valid[indices[m - 1]] = true;
        }
    }
    // Fallback if every class was a singleton (e.g. extreme K with 1 example per class).
    if !is_valid.iter().any(|&v| v) && ys.len() >= 5 {
        for (i, flag) in is_valid.iter_mut().enumerate() {
            if i % 5 == 4 {
                *flag = true;
            }
        }
    }
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (i, &v) in is_valid.iter().enumerate() {
        if v {
            valid.push(i);
        } else {
            train.push(i);
        }
    }
    (train, valid)
}

fn compute_standardization(xs: &[Vec<f32>], subset: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let d = xs[0].len();
    let denom = subset.len().max(1) as f32;
    let mut mean = vec![0f32; d];
    for &idx in subset {
        for (m, v) in mean.iter_mut().zip(&xs[idx]) {
            *m += v / denom;
        }
    }
    let mut var = vec![0f32; d];
    for &idx in subset {
        for ((s, v), m) in var.iter_mut().zip(&xs[idx]).zip(&mean) {
            *s += (v - m).powi(2) / denom;
        }
    }
    let std: Vec<f32> = var.into_iter().map(|v| v.sqrt() + 1e-6).collect();
    (mean, std)
}

fn standardize_rows(xs: &[Vec<f32>], mean: &[f32], std: &[f32]) -> Vec<Vec<f32>> {
    xs.iter()
        .map(|x| {
            x.iter()
                .zip(mean)
                .zip(std)
                .map(|((v, m), s)| (v - m) / s)
                .collect()
        })
        .collect()
}

/// Standardize `xs` on `train` only when selecting L2 on the stratified held-out fifth, then
/// recompute standardization and refit on all items with that L2.
/// `base` holds per-item logits that the model's output is added to (a residual probe).
fn fit_features(
    xs: &[Vec<f32>],
    ys: &[usize],
    k: usize,
    base: Option<&[Vec<f32>]>,
) -> Result<FeatureFit> {
    let d = xs[0].len();
    let n = xs.len();
    let (train, valid) = stratified_train_valid_split(ys, k);
    let (cv_mean, cv_std) = compute_standardization(xs, &train);
    let cv_rows = standardize_rows(xs, &cv_mean, &cv_std);

    let mut best = (f64::NEG_INFINITY, L2_GRID[0]);
    let score = |w: &[f32], b: &[f32], i: usize| -> Vec<f32> {
        let mut z = linear(w, b, &cv_rows[i], d);
        if let Some(base) = base {
            for (zj, bj) in z.iter_mut().zip(&base[i]) {
                *zj += bj;
            }
        }
        z
    };
    for l2 in L2_GRID {
        let (w, b) = softmax_regression(&cv_rows, ys, &train, k, l2, base)?;
        let correct = valid
            .iter()
            .filter(|&&i| argmax(&score(&w, &b, i)) == ys[i])
            .count();
        let acc = correct as f64 / valid.len().max(1) as f64;
        // On ties, prefer the stronger penalty.
        if acc >= best.0 {
            best = (acc, l2);
        }
    }
    let all: Vec<usize> = (0..n).collect();
    let (mean, std) = compute_standardization(xs, &all);
    let full_rows = standardize_rows(xs, &mean, &std);
    let (weights, bias) = softmax_regression(&full_rows, ys, &all, k, best.1, base)?;
    Ok(FeatureFit {
        mean,
        std,
        weights,
        bias,
        l2: best.1,
        validation_accuracy: best.0,
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
    base: Option<&[Vec<f32>]>,
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
    let offset = match base {
        Some(base) => {
            let flat: Vec<f32> = subset
                .iter()
                .flat_map(|&i| base[i].iter().copied())
                .collect();
            Tensor::from_vec(flat, (n, k), &dev)?
        }
        None => Tensor::zeros((n, k), candle_core::DType::F32, &dev)?,
    };

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
        let z = (x.matmul(&w)?.broadcast_add(&b)? + &offset)?;
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
        let (w, b) = softmax_regression(&rows, &ys, &all, 2, 1e-3, None).unwrap();
        let correct = all
            .iter()
            .filter(|&&i| argmax(&linear(&w, &b, &rows[i], 3)) == ys[i])
            .count();
        assert_eq!(correct, 40);
    }

    #[test]
    fn stacked_probe_reads_zero_shot_logits_after_the_embedding() {
        // One embedding dimension, then one feature per label. Each label's weight picks its own
        // zero-shot logit, so the probe's logits equal the zero-shot logits.
        let probe = Probe {
            kind: PROBE_KIND.to_string(),
            checkpoint: Checkpoint::English,
            question_id: "q".to_string(),
            labels: vec!["a".to_string(), "b".to_string()],
            mean: vec![0.0; 3],
            std: vec![1.0; 3],
            weights: vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bias: vec![0.0; 2],
            l2: 0.0,
            train_samples: 0,
            validation_accuracy: 0.0,
            zero_shot_validation_accuracy: 0.0,
            residual: false,
            stacked: true,
        };
        assert_eq!(probe.embedding_dim(), 1);
        assert_eq!(probe.logits(&[5.0], Some(&[0.25, -1.0])), [0.25, -1.0]);
    }

    #[test]
    fn residual_fit_keeps_correct_zero_shot_logits() {
        // The zero-shot logits already name the label, and the embedding is noise.
        let ys: Vec<usize> = (0..60).map(|i| i % 3).collect();
        let xs: Vec<Vec<f32>> = (0..60)
            .map(|i| vec![((i * 37) % 11) as f32, ((i * 17) % 7) as f32])
            .collect();
        let base: Vec<Vec<f32>> = ys
            .iter()
            .map(|&y| (0..3).map(|j| if j == y { 2.0 } else { 0.0 }).collect())
            .collect();
        let fit = fit_features(&xs, &ys, 3, Some(&base)).unwrap();
        assert_eq!(fit.validation_accuracy, 1.0);
    }

    #[test]
    fn stratified_split_handles_five_class_round_robin_and_sorted_labels() {
        // Previously `i % 5 == 4` put 100% of class 4 into valid and 0% into train!
        let ys_rr: Vec<usize> = (0..50).map(|i| i % 5).collect();
        let xs_rr: Vec<Vec<f32>> = ys_rr
            .iter()
            .map(|&y| (0..5).map(|j| if j == y { 3.0 } else { 0.0 }).collect())
            .collect();
        let fit_rr = fit_features(&xs_rr, &ys_rr, 5, None).unwrap();
        assert_eq!(fit_rr.validation_accuracy, 1.0);

        let ys_sorted: Vec<usize> = (0..50).map(|i| i / 10).collect();
        let xs_sorted: Vec<Vec<f32>> = ys_sorted
            .iter()
            .map(|&y| (0..5).map(|j| if j == y { 3.0 } else { 0.0 }).collect())
            .collect();
        let fit_sorted = fit_features(&xs_sorted, &ys_sorted, 5, None).unwrap();
        assert_eq!(fit_sorted.validation_accuracy, 1.0);
    }

    #[test]
    fn probe_load_rejects_malformed_dimensions() {
        let path = std::env::temp_dir().join(format!("bad_probe_{}.json", std::process::id()));
        let bad = Probe {
            kind: PROBE_KIND.to_string(),
            checkpoint: Checkpoint::English,
            question_id: "q".to_string(),
            labels: vec!["a".into(), "b".into(), "c".into()],
            mean: vec![0.0; 2],
            std: vec![1.0; 2],
            weights: vec![0.0; 6],
            bias: vec![0.0; 3],
            l2: 0.1,
            train_samples: 10,
            validation_accuracy: 0.5,
            zero_shot_validation_accuracy: 0.5,
            residual: false,
            stacked: true, // invalid: mean.len() == 2 <= labels.len() == 3
        };
        std::fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
        assert!(Probe::load(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
