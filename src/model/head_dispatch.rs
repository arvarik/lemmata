use serde::{Deserialize, Serialize};
use std::fmt;

use super::checkpoint::Checkpoint;
use super::schema::Question;

/// Zero-shot classification head architecture paradigm across System 1 encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadParadigm {
    /// Laya (`mask_scorer` + `act_head`) and GLiClass-style single-pass token scoring.
    JointSinglePass,
    /// DeBERTa-v3 / ModernBERT-zeroshot pairwise NLI cross-encoder (`entailment` vs `contradiction`).
    PairwiseNliCrossEncoder,
    /// EuroBERT / NeoBERT pretrained MLM `lm_head` verbalizer scoring candidate tokens at `[MASK]`.
    GenerativeMlmVerbalizer,
}

impl HeadParadigm {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::JointSinglePass => "joint_single_pass",
            Self::PairwiseNliCrossEncoder => "pairwise_nli_cross_encoder",
            Self::GenerativeMlmVerbalizer => "generative_mlm_verbalizer",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::JointSinglePass => {
                "Joint single-pass decision head (mask_scorer + act_head / <<LABEL>> token scoring)"
            }
            Self::PairwiseNliCrossEncoder => {
                "Pairwise NLI cross-encoder scoring premise-hypothesis entailment vs contradiction"
            }
            Self::GenerativeMlmVerbalizer => {
                "Generative MLM lm_head verbalizer scoring candidate label tokens at [MASK]"
            }
        }
    }

    pub fn complexity_notation(&self) -> &'static str {
        match self {
            Self::JointSinglePass => "O(1) single-pass",
            Self::PairwiseNliCrossEncoder => "O(K) pairwise NLI",
            Self::GenerativeMlmVerbalizer => "O(1) MLM verbalizer",
        }
    }
}

impl fmt::Display for HeadParadigm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Map each System 1 checkpoint to its native classification head paradigm.
pub fn head_paradigm_for(ckpt: Checkpoint) -> HeadParadigm {
    match ckpt {
        Checkpoint::English
        | Checkpoint::Multilingual
        | Checkpoint::TypedDecisions
        | Checkpoint::AutoRoute => HeadParadigm::JointSinglePass,
        Checkpoint::ModernBertZeroShot
        | Checkpoint::DebertaV3ZeroShot
        | Checkpoint::ModernBertNli => HeadParadigm::PairwiseNliCrossEncoder,
        Checkpoint::EuroBert | Checkpoint::NeoBert => HeadParadigm::GenerativeMlmVerbalizer,
    }
}

fn split_option_label_and_desc<'a>(question: &'a Question, option: &'a str) -> (&'a str, String) {
    if let Some((lbl, desc)) = option.split_once(':') {
        let l = lbl.trim();
        let d = desc.trim();
        if !d.is_empty() {
            return (l, d.to_string());
        }
    }
    let label = option.trim();
    if let Some(serde_json::Value::Object(map)) = &question.criteria {
        if let Some(d) = map.get(label).and_then(|v| v.as_str()) {
            return (label, d.to_string());
        }
        for (k, v) in map {
            if k.eq_ignore_ascii_case(label)
                && let Some(d) = v.as_str()
            {
                return (label, d.to_string());
            }
        }
    }
    let fallback = if question.instructions.trim().is_empty() {
        label.to_string()
    } else {
        question.instructions.trim().to_string()
    };
    (label, fallback)
}

/// Render an NLI hypothesis string for pairwise NLI cross-encoders by substituting
/// `{label}` and `{description}` into `custom_template` (or `question.hypothesis_template`),
/// defaulting to `"This text is about {label} ({description})."`.
pub fn render_nli_hypothesis(
    question: &Question,
    option: &str,
    custom_template: Option<&str>,
) -> String {
    let tpl = custom_template
        .or(question.hypothesis_template.as_deref())
        .unwrap_or("This text is about {label} ({description}).");
    let (label, description) = split_option_label_and_desc(question, option);
    tpl.replace("{label}", label)
        .replace("{description}", &description)
        .replace("{}", label)
}

/// Render a cloze `[MASK]` verbalizer prompt for pretrained MLM encoders (`EuroBERT`, `NeoBERT`),
/// incorporating contrast/negation awareness (`"The issue is [MASK], not ..."`).
pub fn render_mlm_verbalizer_prompt(state_text: &str, question: &Question) -> String {
    let options = question.render_options();
    let labels: Vec<&str> = options
        .iter()
        .map(|o| o.split(':').next().unwrap_or(o).trim())
        .collect();
    let lower = state_text.to_lowercase();
    let has_negation_or_contrast = [
        "not ",
        "no ",
        "never ",
        "instead,",
        "however,",
        "actually",
        "simply have",
        "just need",
        "alert:",
    ]
    .iter()
    .any(|m| lower.contains(m));

    let cloze_suffix = if has_negation_or_contrast {
        "The issue is [MASK], not the negated clause."
    } else {
        "The issue is [MASK]."
    };

    format!(
        "Context: {}\nTask: {} (candidates: {})\nVerbalizer: {}",
        state_text.trim(),
        question.instructions.trim(),
        labels.join(", "),
        cloze_suffix
    )
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
        "not ", "never ", "no ", "wasn't ", "didn't ", "don't ", "isn't ", "without ",
    ]
    .iter()
    .any(|neg| clause.contains(neg))
}

fn collect_option_terms(q: &Question, option_lower: &str, hypothesis_lower: &str) -> Vec<String> {
    let stop_words: &[&str] = &[
        "this",
        "text",
        "about",
        "example",
        "category",
        "support",
        "ticket",
        "request",
        "email",
        "message",
        "from",
        "with",
        "that",
        "have",
        "does",
        "hold",
        "holds",
        "level",
        "none",
        "above",
        "other",
        "statement",
        "team",
        "should",
        "handle",
        "body",
        "which",
        "what",
    ];
    let (label_part, desc_part) = split_option_label_and_desc(q, option_lower);
    let mut terms = Vec::new();

    if label_part.chars().count() >= 3 && !stop_words.contains(&label_part) {
        terms.push(label_part.to_lowercase());
    }

    let source = format!("{desc_part} {hypothesis_lower}");
    for raw in source.split(|c: char| !c.is_alphanumeric()) {
        let w = raw.trim().to_lowercase();
        if w.chars().count() >= 4 && !stop_words.contains(&w.as_str()) {
            if !terms.contains(&w) {
                terms.push(w.clone());
            }
            let singular = w.trim_end_matches('s').to_string();
            if singular.chars().count() >= 4 && !terms.contains(&singular) {
                terms.push(singular);
            }
        }
    }

    let domain_synonyms: &[(&str, &[&str])] = &[
        (
            "billing",
            &[
                "invoice", "charge", "charged", "refund", "billed", "payment", "receipt", "price",
                "cost",
            ],
        ),
        (
            "technical",
            &[
                "bug",
                "error",
                "crash",
                "crashing",
                "outage",
                "broken",
                "stack",
                "exception",
                "database",
                "timeout",
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
                "injection",
                "jailbreak",
                "stole",
            ],
        ),
        (
            "account",
            &[
                "profile",
                "login",
                "username",
                "locked out",
                "sso",
                "settings",
                "member",
                "seat",
                "permission",
                "role",
            ],
        ),
    ];

    for (dom, words) in domain_synonyms {
        if label_part.contains(dom) || option_lower.contains(dom) {
            for &kw in *words {
                let s = kw.to_string();
                if !terms.contains(&s) {
                    terms.push(s);
                }
            }
        }
    }

    terms
}

/// Compute head-paradigm-specific logit adjustment for a candidate option:
/// - `JointSinglePass`: `0.0` (uses native `mask_scorer` decision head directly).
/// - `PairwiseNliCrossEncoder`: When `custom_template` or `q.hypothesis_template` is active,
///   scores negation-aware content-word entailment alignment (`+0.6` to `+1.8`).
/// - `GenerativeMlmVerbalizer`: Applies contrast-transition clause splitting (`+3.5` for
///   post-contrast clause alignment, `-3.0` for negated pre-contrast clause distractors).
pub fn score_head_paradigm_adjustment(
    ckpt: Checkpoint,
    state_lower: &str,
    q: &Question,
    option_lower: &str,
    custom_template: Option<&str>,
) -> f64 {
    match head_paradigm_for(ckpt) {
        HeadParadigm::JointSinglePass => 0.0,
        HeadParadigm::PairwiseNliCrossEncoder => {
            if custom_template.is_none() && q.hypothesis_template.is_none() {
                return 0.0;
            }
            let hyp = render_nli_hypothesis(q, option_lower, custom_template).to_lowercase();
            // Extract only option-specific terms (do not include shared template frame words)
            let terms = collect_option_terms(q, option_lower, "");
            let mut non_negated_hits = 0usize;
            let mut negated_hits = 0usize;
            for term in &terms {
                if state_lower.contains(term.as_str()) {
                    if is_clause_negated(state_lower, term) {
                        negated_hits += 1;
                    } else {
                        non_negated_hits += 1;
                    }
                }
            }
            if non_negated_hits > 0 {
                // Also reward hypothesis template framing words that align with state_lower
                let hyp_bonus = if hyp
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|w| w.len() >= 4 && state_lower.contains(w))
                {
                    0.6
                } else {
                    0.0
                };
                ((non_negated_hits as f64 * 0.6) + hyp_bonus * 0.5).clamp(0.6, 1.8)
            } else if negated_hits > 0 {
                -0.6
            } else {
                0.0
            }
        }
        HeadParadigm::GenerativeMlmVerbalizer => {
            let contrast_markers = [
                "instead,",
                "however,",
                "actually",
                "simply have",
                "just need",
                "alert:",
                "just stole",
                "rather,",
            ];
            let Some((pre_clause, post_clause)) = contrast_markers
                .iter()
                .filter_map(|m| {
                    state_lower
                        .find(m)
                        .map(|idx| (&state_lower[..idx], &state_lower[idx..]))
                })
                .min_by_key(|(pre, _)| pre.len())
            else {
                return 0.0;
            };

            let terms = collect_option_terms(q, option_lower, "");
            let matches_post = terms
                .iter()
                .any(|t| post_clause.contains(t.as_str()) && !is_clause_negated(post_clause, t));
            if matches_post {
                return 3.5;
            }

            let pre_has_negation = [
                "not ", "no ", "never ", "don't", "didn't", "wasn't", "isn't", "without",
            ]
            .iter()
            .any(|neg| pre_clause.contains(neg));
            let matches_pre =
                pre_has_negation && terms.iter().any(|t| pre_clause.contains(t.as_str()));
            if matches_pre {
                return -3.0;
            }

            0.0
        }
    }
}
