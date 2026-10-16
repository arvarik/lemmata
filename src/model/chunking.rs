use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use crate::engine::classifier::confidence_from_probs;
use crate::model::estimate_tokens;

/// Context window adaptation strategy for documents exceeding a checkpoint's token limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChunkStrategy {
    #[default]
    TruncatePrefix,
    HeadTail,
    SlidingWindow,
}

impl ChunkStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TruncatePrefix => "truncate-prefix",
            Self::HeadTail => "head-tail",
            Self::SlidingWindow => "sliding-window",
        }
    }
}

impl fmt::Display for ChunkStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for ChunkStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "truncate" | "prefix" | "truncate-prefix" | "truncate_prefix" => {
                Ok(Self::TruncatePrefix)
            }
            "head-tail" | "head_tail" | "headtail" => Ok(Self::HeadTail),
            "sliding" | "sliding-window" | "sliding_window" | "window" => Ok(Self::SlidingWindow),
            other => Err(format!(
                "Invalid chunk strategy '{}'. Options: truncate-prefix, head-tail, sliding-window",
                other
            )),
        }
    }
}

fn word_byte_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, idx));
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

/// Adapt `text` into one or more chunks fitting within `ctx_window` tokens according to `strategy`.
pub fn chunk_text_for_window<'a>(
    text: &'a str,
    ctx_window: usize,
    strategy: ChunkStrategy,
) -> Vec<Cow<'a, str>> {
    if estimate_tokens(text) <= ctx_window {
        return vec![Cow::Borrowed(text)];
    }

    let spans = word_byte_spans(text);
    if spans.len() <= 4 {
        let cut = text
            .char_indices()
            .nth(ctx_window.max(4))
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        return vec![Cow::Borrowed(&text[..cut])];
    }

    let max_words = (ctx_window.saturating_sub(1) * 3 / 4)
        .min(spans.len().saturating_sub(1))
        .max(4);

    match strategy {
        ChunkStrategy::TruncatePrefix => {
            let end_byte = spans[max_words - 1].1;
            vec![Cow::Borrowed(&text[..end_byte])]
        }
        ChunkStrategy::HeadTail => {
            let head_words = (max_words / 4).max(1);
            let tail_words = max_words.saturating_sub(head_words + 1).max(1);
            let head_end = spans[head_words - 1].1;
            let tail_start = spans[spans.len() - tail_words].0;
            vec![Cow::Owned(format!(
                "{} ... {}",
                &text[..head_end],
                &text[tail_start..]
            ))]
        }
        ChunkStrategy::SlidingWindow => {
            let stride = (max_words * 2 / 3).max(1);
            let mut windows = Vec::new();
            let mut start_word = 0usize;
            while start_word < spans.len() {
                let end_word = (start_word + max_words).min(spans.len());
                let start_byte = spans[start_word].0;
                let end_byte = spans[end_word - 1].1;
                windows.push(Cow::Borrowed(&text[start_byte..end_byte]));
                if end_word == spans.len() {
                    break;
                }
                start_word += stride;
            }
            windows
        }
    }
}

/// Pool option logits across multiple sliding windows using softmax negentropy confidence weighting.
pub fn pool_window_logits(window_logits: &[Vec<f64>]) -> Vec<f64> {
    if window_logits.is_empty() {
        return Vec::new();
    }
    if window_logits.len() == 1 {
        return window_logits[0].clone();
    }

    let k = window_logits[0].len();
    if k == 0 {
        return Vec::new();
    }

    let n = window_logits.len();
    let mut normalized_windows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut weights: Vec<f64> = Vec::with_capacity(n);
    let mut best_idx = 0usize;
    let mut best_weight = f64::NEG_INFINITY;

    for (w_idx, logits) in window_logits.iter().enumerate() {
        let max_z = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_z = logits.iter().cloned().fold(f64::INFINITY, f64::min);
        let exp_sum: f64 = logits.iter().map(|&z| (z - max_z).exp()).sum();
        let probs: Vec<f64> = if exp_sum > 0.0 {
            logits
                .iter()
                .map(|&z| (z - max_z).exp() / exp_sum)
                .collect()
        } else {
            vec![1.0 / k as f64; k]
        };

        let conf = confidence_from_probs(&probs);
        let tail_recency = if n > 1 {
            1.0 + 0.5 * (w_idx as f64 / (n - 1) as f64)
        } else {
            1.0
        };
        let weight = conf.powi(2) * tail_recency + 1e-6;
        if weight > best_weight {
            best_weight = weight;
            best_idx = w_idx;
        }
        weights.push(weight);

        // ponytail: bound per-window logit span so a verbose preamble never dwarfs a concise tail incident
        let span = (max_z - min_z).max(1e-6);
        let scale = if span > 4.5 { 4.5 / span } else { 1.0 };
        normalized_windows.push(
            logits
                .iter()
                .map(|&z| min_z + (z - min_z) * scale)
                .collect(),
        );
    }

    let weight_sum: f64 = weights.iter().sum::<f64>().max(1e-9);
    (0..k)
        .map(|i| {
            let weighted_mean: f64 = normalized_windows
                .iter()
                .zip(weights.iter())
                .map(|(w_logits, &w)| w_logits.get(i).copied().unwrap_or(0.0) * w)
                .sum::<f64>()
                / weight_sum;
            let best_val = normalized_windows[best_idx].get(i).copied().unwrap_or(0.0);
            0.85 * best_val + 0.15 * weighted_mean
        })
        .collect()
}
