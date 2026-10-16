use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Primitive question types supported natively by Laya System 1 models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    /// Pick one label from a set of discrete candidates.
    Choice,
    /// Return an ordinal rubric value (e.g. 0 to 4).
    Score,
    /// Return calibrated probability for a boolean assertion (True/False).
    Noul,
}

/// A typed question schema definition conforming to Laya / Jev request specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    pub instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<serde_json::Value>,
}

impl Question {
    pub fn new_choice(instructions: impl Into<String>, labels: Vec<String>) -> Self {
        let mut map = serde_json::Map::new();
        for label in labels {
            map.insert(label, serde_json::Value::Null);
        }
        Self {
            qtype: QuestionType::Choice,
            instructions: instructions.into(),
            criteria: Some(serde_json::Value::Object(map)),
        }
    }

    pub fn new_score(instructions: impl Into<String>, levels: Vec<String>) -> Self {
        let arr = levels.into_iter().map(serde_json::Value::String).collect();
        Self {
            qtype: QuestionType::Score,
            instructions: instructions.into(),
            criteria: Some(serde_json::Value::Array(arr)),
        }
    }

    pub fn new_noul(instructions: impl Into<String>) -> Self {
        Self {
            qtype: QuestionType::Noul,
            instructions: instructions.into(),
            criteria: None,
        }
    }

    /// Extract rendered option texts for this question.
    pub fn render_options(&self) -> Vec<String> {
        match self.qtype {
            QuestionType::Choice => {
                if let Some(serde_json::Value::Object(map)) = &self.criteria {
                    map.iter()
                        .map(|(k, v)| {
                            if let Some(desc) = v.as_str() {
                                format!("{k}: {desc}")
                            } else {
                                k.clone()
                            }
                        })
                        .collect()
                } else if let Some(serde_json::Value::Array(arr)) = &self.criteria {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                } else {
                    vec![]
                }
            }
            QuestionType::Score => {
                if let Some(serde_json::Value::Array(arr)) = &self.criteria {
                    arr.iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let text = v.as_str().unwrap_or("");
                            format!("level {i}: {text}")
                        })
                        .collect()
                } else {
                    vec!["level 0".into(), "level 1".into(), "level 2".into()]
                }
            }
            QuestionType::Noul => {
                if let Some(serde_json::Value::Object(map)) = &self.criteria {
                    let f_desc = map
                        .get("false")
                        .and_then(|v| v.as_str())
                        .unwrap_or("no, the statement does not hold");
                    let t_desc = map
                        .get("true")
                        .and_then(|v| v.as_str())
                        .unwrap_or("yes, the statement holds");
                    vec![format!("false: {f_desc}"), format!("true: {t_desc}")]
                } else {
                    vec![
                        "false: no, the statement does not hold".into(),
                        "true: yes, the statement holds".into(),
                    ]
                }
            }
        }
    }

    /// Validate that the question has sufficient options and fits within the Laya head token budget.
    pub fn validate(&self, qid: &str, head_budget: usize) -> Result<Vec<String>, String> {
        let mut warnings = Vec::new();
        let options = self.render_options();

        match self.qtype {
            QuestionType::Choice => {
                if options.len() < 2 {
                    return Err(format!(
                        "Choice question '{}' must specify at least 2 candidate options",
                        qid
                    ));
                }
            }
            QuestionType::Score => {
                if options.len() < 2 {
                    return Err(format!(
                        "Score question '{}' requires at least 2 rubric levels",
                        qid
                    ));
                }
            }
            QuestionType::Noul => {}
        }

        let est_tokens: usize = options.iter().map(|o| estimate_tokens(o)).sum::<usize>()
            + estimate_tokens(&self.instructions).saturating_sub(4);

        if est_tokens > head_budget {
            warnings.push(format!(
                "Question '{}' options (~{} tokens) exceed head budget ({} tokens). Truncation will degrade accuracy.",
                qid, est_tokens, head_budget
            ));
        }

        Ok(warnings)
    }
}

/// Estimate subword token count for both whitespace-delimited and non-Latin/CJK scripts.
pub fn estimate_tokens(text: &str) -> usize {
    let word_tokens = text.split_whitespace().count() * 4 / 3;
    let non_latin = text
        .chars()
        .filter(|&c| c as u32 > 0x024F && !c.is_whitespace())
        .count();
    word_tokens + non_latin + 1
}

/// The answer returned by Laya for a single question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAnswer {
    #[serde(rename = "type")]
    pub qtype: QuestionType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_labels: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,

    pub confidence: f64,

    pub probabilities: BTreeMap<String, f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub act_probability: Option<f64>,
}

/// The overall response for a single state input across all questions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    pub model: String,
    pub answers: BTreeMap<String, DecisionAnswer>,
    pub input_tokens: usize,
    pub latency_ms: f64,
}

/// A node in a hierarchical decision tree: either a recursive subtree or a terminal leaf question.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HierarchicalNode {
    Subtree(Box<HierarchicalSchema>),
    Leaf(Question),
}

impl HierarchicalNode {
    pub fn question(&self) -> &Question {
        match self {
            HierarchicalNode::Subtree(sub) => &sub.coarse,
            HierarchicalNode::Leaf(q) => q,
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            HierarchicalNode::Subtree(sub) => sub.branches.is_empty(),
            HierarchicalNode::Leaf(_) => true,
        }
    }
}

/// Validation report for a hierarchical schema against head token budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalValidationReport {
    pub root_tokens: usize,
    pub max_branch_tokens: usize,
    pub total_leaf_classes: usize,
    pub branch_count: usize,
    pub warnings: Vec<String>,
    pub valid: bool,
}

/// A hierarchical decision schema structuring high-cardinality classification into coarse-to-fine stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSchema {
    /// The coarse / root question (supports "coarse" or "root" in JSON)
    #[serde(alias = "root")]
    pub coarse: Question,

    /// Sub-branches mapping candidate option keys to child questions or subtrees
    #[serde(default, alias = "subtrees")]
    pub branches: BTreeMap<String, HierarchicalNode>,
}

impl HierarchicalSchema {
    /// Load a hierarchical schema from a JSON file.
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json_str(&content)
    }

    /// Parse a hierarchical schema from a JSON string (supports single schema or map wrapper).
    pub fn from_json_str(content: &str) -> anyhow::Result<Self> {
        if let Ok(schema) = serde_json::from_str::<HierarchicalSchema>(content) {
            return Ok(schema);
        }
        if let Ok(map) = serde_json::from_str::<BTreeMap<String, HierarchicalSchema>>(content)
            && let Some((_name, schema)) = map.into_iter().next()
        {
            return Ok(schema);
        }
        anyhow::bail!(
            "Failed to parse JSON as HierarchicalSchema (requires 'coarse'/'root' question and optional 'branches'/'subtrees')"
        )
    }

    /// Calculate total leaf classes in this hierarchy
    pub fn total_leaf_count(&self) -> usize {
        let mut count = 0;
        let coarse_options = self.coarse.render_options();
        for opt in &coarse_options {
            let key = opt.split(':').next().unwrap_or(opt).trim();
            if let Some(node) = self.branches.get(key) {
                match node {
                    HierarchicalNode::Leaf(q) => count += q.render_options().len(),
                    HierarchicalNode::Subtree(sub) => count += sub.total_leaf_count(),
                }
            } else {
                count += 1;
            }
        }
        count
    }

    /// Calculate maximum tree depth (1 for single coarse, 2 for coarse + branch, etc.)
    pub fn max_depth(&self) -> usize {
        if self.branches.is_empty() {
            return 1;
        }
        let branch_depth = self
            .branches
            .values()
            .map(|node| match node {
                HierarchicalNode::Leaf(_) => 1,
                HierarchicalNode::Subtree(sub) => sub.max_depth(),
            })
            .max()
            .unwrap_or(0);
        1 + branch_depth
    }

    /// Validate the entire hierarchy against a model checkpoint head token budget.
    pub fn validate(&self, head_budget: usize) -> Result<HierarchicalValidationReport, String> {
        let mut warnings = Vec::new();

        // 1. Validate coarse question
        let coarse_warnings = self.coarse.validate("coarse_root", head_budget)?;
        warnings.extend(coarse_warnings);

        let root_tokens: usize = self
            .coarse
            .render_options()
            .iter()
            .map(|o| estimate_tokens(o))
            .sum();

        let mut max_branch_tokens = 0;

        // 2. Validate all branch questions
        for (branch_key, node) in &self.branches {
            match node {
                HierarchicalNode::Leaf(q) => {
                    let branch_warnings =
                        q.validate(&format!("branch:{}", branch_key), head_budget)?;
                    warnings.extend(branch_warnings);
                    let b_tokens: usize =
                        q.render_options().iter().map(|o| estimate_tokens(o)).sum();
                    if b_tokens > max_branch_tokens {
                        max_branch_tokens = b_tokens;
                    }
                }
                HierarchicalNode::Subtree(sub) => {
                    let sub_report = sub.validate(head_budget)?;
                    warnings.extend(sub_report.warnings);
                    if sub_report.root_tokens > max_branch_tokens {
                        max_branch_tokens = sub_report.root_tokens;
                    }
                    if sub_report.max_branch_tokens > max_branch_tokens {
                        max_branch_tokens = sub_report.max_branch_tokens;
                    }
                }
            }
        }

        let total_leaves = self.total_leaf_count();
        let branch_count = self.branches.len();

        Ok(HierarchicalValidationReport {
            root_tokens,
            max_branch_tokens,
            total_leaf_classes: total_leaves,
            branch_count,
            warnings,
            valid: true,
        })
    }

    /// Render visual ASCII tree diagram of the hierarchy
    pub fn render_ascii_tree(&self, head_budget: usize) -> String {
        let mut out = String::new();
        let coarse_options = self.coarse.render_options();
        let root_tokens: usize = coarse_options.iter().map(|o| estimate_tokens(o)).sum();

        out.push_str(&format!(
            "Root: {} ({} options, ~{} tokens / budget {})\n",
            self.coarse.instructions,
            coarse_options.len(),
            root_tokens,
            head_budget
        ));

        let total_coarse = coarse_options.len();
        for (i, opt) in coarse_options.iter().enumerate() {
            let is_last_coarse = i + 1 == total_coarse;
            let prefix = if is_last_coarse {
                "└── "
            } else {
                "├── "
            };
            let child_indent = if is_last_coarse { "    " } else { "│   " };

            let key = opt.split(':').next().unwrap_or(opt).trim();
            if let Some(node) = self.branches.get(key) {
                match node {
                    HierarchicalNode::Leaf(q) => {
                        let sub_opts = q.render_options();
                        let sub_tokens: usize = sub_opts.iter().map(|o| estimate_tokens(o)).sum();
                        out.push_str(&format!(
                            "{}{}: {} ({} sub-options, ~{} tokens)\n",
                            prefix,
                            key,
                            q.instructions,
                            sub_opts.len(),
                            sub_tokens
                        ));
                        let sub_total = sub_opts.len();
                        for (si, sopt) in sub_opts.iter().enumerate() {
                            let is_last_sub = si + 1 == sub_total;
                            let sub_prefix = if is_last_sub {
                                "└── "
                            } else {
                                "├── "
                            };
                            out.push_str(&format!("{}{}{}\n", child_indent, sub_prefix, sopt));
                        }
                    }
                    HierarchicalNode::Subtree(sub) => {
                        out.push_str(&format!(
                            "{}{}: [Subtree: {}]\n",
                            prefix, key, sub.coarse.instructions
                        ));
                    }
                }
            } else {
                out.push_str(&format!("{}{}: (Leaf at root level)\n", prefix, opt));
            }
        }

        let leaves = self.total_leaf_count();
        let max_b = self
            .branches
            .values()
            .map(|n| match n {
                HierarchicalNode::Leaf(q) => {
                    q.render_options().iter().map(|o| estimate_tokens(o)).sum()
                }
                HierarchicalNode::Subtree(s) => s.total_leaf_count() * 10,
            })
            .max()
            .unwrap_or(0);
        let max_stage = root_tokens.max(max_b);
        out.push_str("\nSummary:\n");
        out.push_str(&format!("- Total Leaf Classes: {}\n", leaves));
        out.push_str(&format!("- Coarse Branches: {}\n", self.branches.len()));
        out.push_str(&format!(
            "- Max Tokens per Stage: {} / {}\n",
            max_stage, head_budget
        ));
        out.push_str(&format!("- Tree Depth: {} levels\n", self.max_depth()));

        out
    }
}

fn is_quote_header(line: &str) -> bool {
    let t = line.trim();
    let lower = t.to_lowercase();
    (lower.starts_with("on ") && lower.ends_with("wrote:") && t.len() <= 310)
        || (t.starts_with("--")
            && (lower.contains("original message") || lower.contains("forwarded message")))
        || (t.len() >= 8 && t.chars().all(|c| c == '_'))
        || lower.starts_with("from: ")
}

fn is_signature_marker(line: &str) -> bool {
    let t = line.trim();
    if t == "--" {
        return true;
    }
    let lower = t.to_lowercase();
    if lower.starts_with("sent from my iphone")
        || lower.starts_with("sent from my android")
        || lower.starts_with("sent from my mobile")
        || lower.starts_with("sent from my ipad")
    {
        return true;
    }
    for prefix in [
        "best regards",
        "kind regards",
        "warm regards",
        "many thanks",
        "thank you",
        "thanks",
        "regards",
        "cheers",
        "sincerely",
        "best",
    ] {
        if let Some(rest) = lower.strip_prefix(prefix)
            && rest
                .chars()
                .all(|c| c.is_alphanumeric() || " ,!.-".contains(c))
        {
            return true;
        }
    }
    false
}

fn is_disclaimer(paragraph: &str) -> bool {
    let lower = paragraph.to_lowercase();
    lower.contains("confidential")
        || (lower.contains("intended")
            && (lower.contains("addressee") || lower.contains("recipient")))
        || (lower.contains("received this") && lower.contains("in error"))
}

/// Clean raw email body by stripping quoted reply headers, `>` quote lines, signatures, and legal disclaimers.
pub fn clean_email_body(body: &str, max_chars: usize) -> String {
    let norm = body
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace("\\n", "\n");
    let mut lines = Vec::new();
    for line in norm.split('\n') {
        if is_quote_header(line) && !lines.is_empty() {
            break;
        }
        if line.trim_start().starts_with('>') {
            continue;
        }
        lines.push(line.trim_end());
    }
    let start_idx = ((lines.len() * 3 / 5).min(lines.len().saturating_sub(8))).max(1);
    let mut cut = lines.len();
    for (i, line) in lines.iter().enumerate().skip(start_idx) {
        if line.trim().len() <= 40 && is_signature_marker(line) {
            cut = i;
            break;
        }
    }
    let mut paragraphs = Vec::new();
    let mut cur_para = Vec::new();
    for line in &lines[..cut] {
        if line.trim().is_empty() {
            if !cur_para.is_empty() {
                let p = cur_para.join("\n");
                if !is_disclaimer(&p) {
                    paragraphs.push(p);
                }
                cur_para.clear();
            }
        } else {
            let collapsed = line
                .split([' ', '\t'])
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            cur_para.push(collapsed);
        }
    }
    if !cur_para.is_empty() {
        let p = cur_para.join("\n");
        if !is_disclaimer(&p) {
            paragraphs.push(p);
        }
    }
    let joined = paragraphs.join("\n\n");
    joined
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| joined[..i].to_string())
        .unwrap_or(joined)
}

/// Default fan-out of Laya email triage questions (category, spam, phishing, urgency, reply, sentiment).
pub fn email_questions() -> BTreeMap<String, Question> {
    BTreeMap::from([
        (
            "category".into(),
            Question {
                qtype: QuestionType::Choice,
                instructions: "Which team should handle the email in `body`?".into(),
                criteria: Some(serde_json::json!({
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, integrations",
                    "sales": "pricing, demos, new purchases",
                    "account": "login, access, profile changes",
                    "hr": "hiring, leave, payroll",
                    "other": "none of the above"
                })),
            },
        ),
        (
            "is_spam".into(),
            Question::new_noul("Is this email unsolicited spam or bulk marketing?"),
        ),
        (
            "is_phishing".into(),
            Question {
                qtype: QuestionType::Noul,
                instructions: "Is this email a phishing or scam attempt to steal money, credentials, or personal data?".into(),
                criteria: Some(serde_json::json!({
                    "true": "phishing, scam, or fraud",
                    "false": "a legitimate email"
                })),
            },
        ),
        (
            "urgency".into(),
            Question::new_score(
                "How urgent is the issue described in `body`?",
                vec![
                    "no time pressure".into(),
                    "needs attention soon".into(),
                    "blocking issue or hard deadline".into(),
                ],
            ),
        ),
        (
            "needs_reply".into(),
            Question::new_noul("Does the sender expect a reply?"),
        ),
        (
            "sentiment".into(),
            Question::new_score(
                "What is the sender's tone in `body`?",
                vec![
                    "angry or very negative".into(),
                    "negative".into(),
                    "neutral".into(),
                    "positive".into(),
                ],
            ),
        ),
    ])
}
