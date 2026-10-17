//! Hashed word features: a fast, model-free text vector for shortlisting and distance checks.

use crate::engine::metrics::fnv1a64;

/// Compute cosine similarity between two unit-normalized vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    if dot.is_finite() {
        dot.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Convert input text into a normalized 128-dimensional dense feature representation.
/// Uses a combination of word frequency, character n-gram hashing, and token length distribution.
pub fn vectorize_text(text: &str) -> Vec<f64> {
    const DIM: usize = 128;
    let mut vec = vec![0.0f64; DIM];

    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "was",
        "are", "were", "be", "been", "have", "has", "had", "do", "does", "did", "this", "that",
        "it", "i", "we", "you", "they", "my", "our", "your", "their", "of", "from", "by", "as",
        "can", "will", "would", "should", "could", "get", "please",
    ];

    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 2 && !STOPWORDS.contains(&s.as_str()))
        .collect();

    if tokens.is_empty() {
        return vec;
    }

    // Word hashing into feature buckets
    for token in &tokens {
        let hash = fnv1a64(token.as_bytes());
        vec[(hash as usize) % DIM] += 1.5;

        // Subword bigram hashing for stem/morphology generalization without heap allocation
        let chars: Vec<char> = token.chars().collect();
        for window in chars.windows(2) {
            let mut buf = [0u8; 8];
            let len0 = window[0].encode_utf8(&mut buf).len();
            let len1 = window[1].encode_utf8(&mut buf[len0..]).len();
            let bi_hash = fnv1a64(&buf[..len0 + len1]);
            vec[(bi_hash as usize) % DIM] += 0.5;
        }
    }

    // L2 normalize vector onto unit hypersphere
    let norm_sq: f64 = vec.iter().map(|v| v * v).sum();
    if norm_sq > 0.0 {
        let norm = norm_sq.sqrt();
        for val in &mut vec {
            *val /= norm;
        }
    }

    vec
}
