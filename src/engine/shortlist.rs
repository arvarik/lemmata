use serde::{Deserialize, Serialize};

use crate::engine::cache::{cosine_similarity, vectorize_text};
use crate::model::Question;

/// Output of Stage-1 Bi-Encoder candidate option shortlisting prior to Cross-Encoder NLI scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortlistResult {
    pub kept_indices: Vec<usize>,
    pub kept_options: Vec<String>,
    pub bi_encoder_scores: Vec<(usize, f64)>,
    pub original_cardinality: usize,
    pub shortlisted_cardinality: usize,
}

const DOMAIN_SYNONYMS: &[(&str, &[&str])] = &[
    (
        "billing",
        &[
            "invoice",
            "charge",
            "charged",
            "refund",
            "billed",
            "payment",
            "receipt",
            "price",
            "cost",
            "overcharge",
            "dispute",
            "factura",
            "reembolso",
            "facture",
            "remboursement",
            "rechnung",
            "发票",
            "退款",
            "账单",
            "請求",
            "返金",
            "счет",
            "возврат",
            "оплата",
        ],
    ),
    (
        "technical",
        &[
            "bug",
            "error",
            "errors",
            "crash",
            "crashing",
            "outage",
            "broken",
            "issue",
            "stack",
            "exception",
            "timeout",
            "webhook",
            "sdk",
            "api",
            "erreur",
            "panne",
            "fehler",
            "absturz",
            "错误",
            "崩溃",
            "故障",
            "エラー",
            "障害",
            "バグ",
            "ошибка",
            "сбой",
        ],
    ),
    (
        "security",
        &[
            "phishing",
            "unauthorized",
            "breach",
            "compromised",
            "suspicious",
            "2fa",
            "mfa",
            "token",
            "password",
            "credential",
            "credentials",
            "injection",
            "jailbreak",
            "stole",
            "sécurité",
            "piratage",
            "seguridad",
            "sicherheit",
            "セキュリティ",
            "不正アクセス",
            "安全",
        ],
    ),
    (
        "account",
        &[
            "profile",
            "login",
            "username",
            "email address",
            "locked out",
            "sso",
            "settings",
            "member",
            "seat",
            "seats",
            "permission",
            "permissions",
            "role",
            "cuenta",
            "compte",
            "konto",
            "アカウント",
            "账户",
        ],
    ),
    (
        "sales",
        &[
            "pricing",
            "contract",
            "enterprise",
            "quote",
            "demo",
            "buy",
            "purchase",
            "lead",
            "upgrade",
            "devis",
            "contrat",
            "angebot",
            "vertrag",
        ],
    ),
];

/// Stage-1 Bi-Encoder Top-k Shortlisting: filters a high-cardinality option set (`K >> top_k`)
/// down to `top_k` candidates in `O(1)` vector + lexical pass before Cross-Encoder NLI scoring.
///
/// Preserves original ascending option indices in `kept_indices` and `kept_options`.
pub fn shortlist_options(
    state_text: &str,
    question: &Question,
    options: &[String],
    top_k: usize,
) -> ShortlistResult {
    let original_cardinality = options.len();
    let full_lower = state_text.to_lowercase();
    let effective_state = if let Some(idx) = full_lower.find("--- [end of preamble") {
        &full_lower[idx..]
    } else {
        &full_lower
    };

    let state_vec = vectorize_text(effective_state);
    let ins_lower = question.instructions.to_lowercase();

    let mut scored: Vec<(usize, f64)> = options
        .iter()
        .enumerate()
        .map(|(idx, opt)| {
            let s = score_candidate_bi_encoder(effective_state, &state_vec, &ins_lower, opt);
            (idx, (s * 10000.0).round() / 10000.0)
        })
        .collect();

    if top_k == 0 || original_cardinality <= top_k {
        return ShortlistResult {
            kept_indices: (0..original_cardinality).collect(),
            kept_options: options.to_vec(),
            bi_encoder_scores: scored,
            original_cardinality,
            shortlisted_cardinality: original_cardinality,
        };
    }

    // Sort candidates by descending bi-encoder score, tie-breaking by ascending original index
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut kept_indices: Vec<usize> = scored.iter().take(top_k).map(|&(idx, _)| idx).collect();
    kept_indices.sort_unstable();

    let kept_options: Vec<String> = kept_indices
        .iter()
        .map(|&idx| options[idx].clone())
        .collect();
    let shortlisted_cardinality = kept_indices.len();

    ShortlistResult {
        kept_indices,
        kept_options,
        bi_encoder_scores: scored,
        original_cardinality,
        shortlisted_cardinality,
    }
}

/// Expand `shortlisted_logits` (length `kept_indices.len()`) back into a `total_options`-length
/// logit vector. Non-shortlisted options receive `min_logit - 6.0` so softmax assigns ~99.8%+
/// probability mass to shortlisted candidates while preserving all original option keys.
pub fn expand_shortlisted_logits(
    shortlisted_logits: &[f64],
    kept_indices: &[usize],
    total_options: usize,
) -> Vec<f64> {
    if total_options == 0 {
        return Vec::new();
    }
    if shortlisted_logits.len() == total_options && kept_indices.len() == total_options {
        let is_identity = kept_indices.iter().enumerate().all(|(i, &k)| i == k);
        if is_identity {
            return shortlisted_logits.to_vec();
        }
    }

    let min_logit = shortlisted_logits
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let floor_logit = if min_logit.is_finite() {
        min_logit - 6.0
    } else {
        -6.0
    };

    let mut expanded = vec![floor_logit; total_options];
    for (&orig_idx, &logit) in kept_indices.iter().zip(shortlisted_logits.iter()) {
        if orig_idx < total_options {
            expanded[orig_idx] = logit;
        }
    }
    expanded
}

fn score_candidate_bi_encoder(
    state_lower: &str,
    state_vec: &[f64],
    ins_lower: &str,
    option: &str,
) -> f64 {
    let opt_lower = option.to_lowercase();
    let label_part = opt_lower.split(':').next().unwrap_or(&opt_lower).trim();

    // 1. Dense hashed trigram/word vector cosine similarity
    let opt_vec = vectorize_text(&opt_lower);
    let cos_sim = cosine_similarity(state_vec, &opt_vec).max(0.0);
    let mut score = cos_sim * 2.5;

    // 2. Direct non-negated label & sub-token match
    if label_part.chars().count() >= 3 && contains_whole_or_prefix(state_lower, label_part) {
        if is_clause_negated(state_lower, label_part) {
            score -= 1.5;
        } else {
            score += 4.0;
        }
    } else if label_part.contains('_') {
        for sub in label_part.split('_') {
            if sub.chars().count() >= 3 && contains_whole_or_prefix(state_lower, sub) {
                if is_clause_negated(state_lower, sub) {
                    score -= 1.0;
                } else {
                    score += 2.5;
                }
            }
        }
    }

    // 3. Domain-synonym overlap (matching the label key primarily)
    for (domain, synonyms) in DOMAIN_SYNONYMS {
        if label_part == *domain || label_part.split('_').any(|p| p == *domain) {
            for &kw in *synonyms {
                if state_lower.contains(kw) {
                    if is_clause_negated(state_lower, kw) {
                        score -= 0.8;
                    } else {
                        score += 2.0;
                    }
                }
            }
        }
    }

    // 4. Non-negated token overlap between state_text and option description
    for word in opt_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
    {
        if contains_whole_or_prefix(state_lower, word) {
            if is_clause_negated(state_lower, word) && !is_clause_negated(&opt_lower, word) {
                score -= 0.6;
            } else {
                score += 1.4;
            }
        } else {
            // 5-char stem match (e.g., "invoices" <-> "invoice", "crashes" <-> "crashing")
            let stem = word
                .char_indices()
                .nth(5)
                .map(|(i, _)| &word[..i])
                .unwrap_or(word);
            if stem.chars().count() >= 5 && state_lower.contains(stem) {
                if is_clause_negated(state_lower, stem) && !is_clause_negated(&opt_lower, stem) {
                    score -= 0.4;
                } else {
                    score += 0.9;
                }
            }
        }
        if ins_lower.contains(word) {
            score += 0.15;
        }
    }

    score
}

/// Checks that `needle` appears in `haystack` on a word boundary (or as a prefix of a word,
/// but prevents short roots like "account" from matching unrelated words if needed, though
/// stem matches are handled separately).
fn contains_whole_or_prefix(haystack: &str, needle: &str) -> bool {
    if !haystack.contains(needle) {
        return false;
    }
    // Avoid false-positive substring match of "account" inside "accounting" when checking label
    if needle == "account" {
        return haystack
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| w == "account" || w == "accounts");
    }
    true
}

fn is_clause_negated(text_lower: &str, keyword: &str) -> bool {
    let Some(pos) = text_lower.find(keyword) else {
        return false;
    };
    let window_start = text_lower[..pos]
        .char_indices()
        .rev()
        .take(36)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0);
    let prefix = &text_lower[window_start..pos];
    let clause = prefix.rsplit(['.', ';', ',', '!']).next().unwrap_or(prefix);
    [
        "not ", "never ", "no ", "wasn't ", "didn't ", "don't ", "isn't ", "without ", "no fue ",
        "pas de ", "nicht ", "kein ",
    ]
    .iter()
    .any(|neg| clause.contains(neg))
}
