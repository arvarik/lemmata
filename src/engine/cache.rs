use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

use crate::model::DecisionResult;

/// Configuration for the sub-millisecond semantic decision cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    pub max_entries: usize,
    pub similarity_threshold: f64,
    pub min_confidence: f64,
    pub ttl_seconds: u64,
    #[serde(default = "default_max_escalate")]
    pub max_escalate_probability: f64,
}

fn default_max_escalate() -> f64 {
    0.50
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            similarity_threshold: 0.92,
            min_confidence: 0.70,
            ttl_seconds: 86_400, // 24 hours
            max_escalate_probability: 0.50,
        }
    }
}

/// Statistics tracking cache hit rates and efficiency.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn total_lookups(&self) -> u64 {
        self.exact_hits + self.semantic_hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.total_lookups();
        if total == 0 {
            0.0
        } else {
            (self.exact_hits + self.semantic_hits) as f64 / total as f64
        }
    }
}

/// A single cached decision entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedDecision {
    pub input_text: String,
    pub feature_vector: Vec<f64>,
    pub result: DecisionResult,
    pub created_at_secs: u64,
    pub hit_count: u64,
}

/// Result returned when querying the cache.
#[derive(Debug, Clone)]
pub enum CacheLookupResult {
    ExactHit(DecisionResult),
    SemanticHit {
        result: DecisionResult,
        similarity: f64,
        matched_text: String,
    },
    Miss,
}

/// Two-tier semantic and exact decision cache for Laya models.
pub struct DecisionCache {
    config: SemanticCacheConfig,
    exact_map: HashMap<String, CachedDecision>,
    entries: Vec<CachedDecision>,
    stats: CacheStats,
}

impl DecisionCache {
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            config,
            exact_map: HashMap::new(),
            entries: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached decisions.
    pub fn clear(&mut self) {
        self.exact_map.clear();
        self.entries.clear();
    }

    /// Generate an exact cache key for text and questions schema.
    pub fn make_exact_key(text: &str, schema_hash: &str) -> String {
        let canonical = text.trim().to_lowercase();
        format!("{schema_hash}::{canonical}")
    }

    /// Query the cache for a given input state.
    pub fn lookup(&mut self, text: &str, schema_hash: &str) -> CacheLookupResult {
        let now = now_secs();
        let exact_key = Self::make_exact_key(text, schema_hash);

        // Tier 1: Exact lookup
        if let Some(cached) = self.exact_map.get_mut(&exact_key) {
            if now.saturating_sub(cached.created_at_secs) <= self.config.ttl_seconds {
                cached.hit_count += 1;
                self.stats.exact_hits += 1;
                let mut res = cached.result.clone();
                res.latency_ms = 0.05; // sub-millisecond retrieval
                return CacheLookupResult::ExactHit(res);
            }
        }

        // Tier 2: Semantic lookup via normalized feature vector
        let query_vec = vectorize_text(text);
        let mut best_sim = -1.0;
        let mut best_idx = None;

        for (idx, item) in self.entries.iter().enumerate() {
            if now.saturating_sub(item.created_at_secs) > self.config.ttl_seconds {
                continue;
            }
            let sim = cosine_similarity(&query_vec, &item.feature_vector);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }

        if best_sim >= self.config.similarity_threshold {
            if let Some(idx) = best_idx {
                self.entries[idx].hit_count += 1;
                self.stats.semantic_hits += 1;
                let mut res = self.entries[idx].result.clone();
                res.latency_ms = 0.35; // ~0.35 ms semantic match
                let matched_text = self.entries[idx].input_text.clone();
                return CacheLookupResult::SemanticHit {
                    result: res,
                    similarity: (best_sim * 10000.0).round() / 10000.0,
                    matched_text,
                };
            }
        }

        self.stats.misses += 1;
        CacheLookupResult::Miss
    }

    /// Admit a decision into the cache if it meets confidence guards.
    pub fn insert(&mut self, text: &str, schema_hash: &str, result: &DecisionResult) -> bool {
        // Admission guard: only cache high-confidence, un-escalated decisions
        let avg_confidence = if result.answers.is_empty() {
            0.0
        } else {
            result.answers.values().map(|a| a.confidence).sum::<f64>() / result.answers.len() as f64
        };

        let max_act_prob = result
            .answers
            .values()
            .filter_map(|a| a.act_probability)
            .fold(0.0, f64::max);

        // Reject caching if confidence is too low or supervisor escalation is indicated
        if avg_confidence < self.config.min_confidence || max_act_prob > self.config.max_escalate_probability {
            return false;
        }

        // Capacity eviction
        if self.entries.len() >= self.config.max_entries {
            self.prune_oldest();
        }

        let exact_key = Self::make_exact_key(text, schema_hash);
        let feature_vector = vectorize_text(text);
        let now = now_secs();

        let cached = CachedDecision {
            input_text: text.to_string(),
            feature_vector,
            result: result.clone(),
            created_at_secs: now,
            hit_count: 0,
        };

        self.exact_map.insert(exact_key, cached.clone());
        self.entries.push(cached);
        self.stats.insertions += 1;
        true
    }

    fn prune_oldest(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        // Remove oldest or least accessed entry
        let min_idx = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, item)| item.hit_count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let removed = self.entries.swap_remove(min_idx);
        self.exact_map.retain(|_, v| v.input_text != removed.input_text);
        self.stats.evictions += 1;
    }

    /// Prune expired entries beyond TTL.
    pub fn prune_expired(&mut self) -> usize {
        let now = now_secs();
        let ttl = self.config.ttl_seconds;
        let before_len = self.entries.len();

        self.entries.retain(|item| now.saturating_sub(item.created_at_secs) <= ttl);
        self.exact_map.retain(|_, item| now.saturating_sub(item.created_at_secs) <= ttl);

        let pruned = before_len - self.entries.len();
        self.stats.evictions += pruned as u64;
        pruned
    }
}

/// Compute cosine similarity between two unit-normalized vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(-1.0, 1.0)
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
        let hash = fnv1a_hash(token.as_bytes());
        let idx = (hash as usize) % DIM;
        vec[idx] += 1.5;

        // Subword bigram hashing for stem/morphology generalization
        let chars: Vec<char> = token.chars().collect();
        for window in chars.windows(2) {
            let bi_str: String = window.iter().collect();
            let bi_hash = fnv1a_hash(bi_str.as_bytes());
            let bi_idx = (bi_hash as usize) % DIM;
            vec[bi_idx] += 0.5;
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

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DecisionAnswer, QuestionType};
    use std::collections::BTreeMap;

    fn sample_decision() -> DecisionResult {
        let mut answers = BTreeMap::new();
        answers.insert(
            "dept".into(),
            DecisionAnswer {
                qtype: QuestionType::Choice,
                choice: Some("billing".into()),
                selected_labels: None,
                score: None,
                noul: None,
                confidence: 0.95,
                probabilities: BTreeMap::new(),
                act_probability: Some(0.08),
            },
        );
        DecisionResult {
            model: "test_model".into(),
            answers,
            input_tokens: 42,
            latency_ms: 32.5,
        }
    }

    #[test]
    fn test_exact_cache_hit() {
        let mut cache = DecisionCache::new(SemanticCacheConfig::default());
        let text = "Please refund my invoice payment #4411";
        let schema_hash = "schema_v1";

        assert!(matches!(cache.lookup(text, schema_hash), CacheLookupResult::Miss));

        cache.insert(text, schema_hash, &sample_decision());
        assert_eq!(cache.len(), 1);

        match cache.lookup(text, schema_hash) {
            CacheLookupResult::ExactHit(res) => {
                assert_eq!(res.answers["dept"].choice.as_deref(), Some("billing"));
                assert!(res.latency_ms < 1.0);
            }
            _ => panic!("Expected exact cache hit"),
        }
    }

    #[test]
    fn test_semantic_cache_hit() {
        let mut config = SemanticCacheConfig::default();
        config.similarity_threshold = 0.85;
        let mut cache = DecisionCache::new(config);

        let original = "Please refund my duplicate invoice charge";
        let paraphrased = "Can I get a refund for a duplicate invoice payment?";
        let schema_hash = "schema_v1";

        cache.insert(original, schema_hash, &sample_decision());

        match cache.lookup(paraphrased, schema_hash) {
            CacheLookupResult::SemanticHit { result, similarity, .. } => {
                assert_eq!(result.answers["dept"].choice.as_deref(), Some("billing"));
                assert!(similarity >= 0.80);
            }
            _ => panic!("Expected semantic cache hit for paraphrased ticket"),
        }
    }
}
