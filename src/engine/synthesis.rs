use anyhow::{Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::model::{Checkpoint, Question, QuestionType};

/// Inferred schema report with token budget analysis.
#[derive(Debug, Clone)]
pub struct InferredSchemaResult {
    pub schema: BTreeMap<String, Question>,
    pub total_questions: usize,
    pub total_samples_analyzed: usize,
    pub token_budgets: BTreeMap<String, (usize, usize)>, // qid -> (est_tokens, max_budget)
}

pub struct SchemaSynthesizer {
    checkpoint: Checkpoint,
    text_col: Option<String>,
    target_cols: Vec<String>,
}

impl SchemaSynthesizer {
    pub fn new(checkpoint: Checkpoint) -> Self {
        Self {
            checkpoint,
            text_col: None,
            target_cols: Vec::new(),
        }
    }

    pub fn with_text_col(mut self, col: impl Into<String>) -> Self {
        self.text_col = Some(col.into());
        self
    }

    pub fn with_target_cols(mut self, cols: Vec<String>) -> Self {
        self.target_cols = cols;
        self
    }

    /// Infer schema from a CSV, JSON, or JSONL file.
    pub fn infer_from_file(&self, path: &Path) -> Result<InferredSchemaResult> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let rows: Vec<BTreeMap<String, String>> = match ext.as_str() {
            "csv" => self.read_csv_rows(path)?,
            "jsonl" | "json" => self.read_json_rows(path)?,
            _ => bail!(
                "Unsupported data format for schema inference: {}. Use CSV or JSONL.",
                path.display()
            ),
        };

        if rows.is_empty() {
            bail!("Input file contains no rows: {}", path.display());
        }

        // Identify text column
        let text_column = if let Some(tc) = &self.text_col {
            tc.clone()
        } else {
            self.detect_text_column(&rows)?
        };

        // Identify target columns
        let targets = if !self.target_cols.is_empty() {
            self.target_cols.clone()
        } else {
            self.detect_target_columns(&rows, &text_column)
        };

        if targets.is_empty() {
            bail!("Could not identify any categorical or rubric columns to infer questions from");
        }

        let head_budget = self.checkpoint.head_budget();
        let mut questions = BTreeMap::new();
        let mut token_budgets = BTreeMap::new();

        for target in &targets {
            let (q, est_tokens) =
                self.synthesize_question(target, &text_column, &rows, head_budget);
            token_budgets.insert(target.clone(), (est_tokens, head_budget));
            questions.insert(target.clone(), q);
        }

        Ok(InferredSchemaResult {
            total_questions: questions.len(),
            total_samples_analyzed: rows.len(),
            token_budgets,
            schema: questions,
        })
    }

    fn synthesize_question(
        &self,
        target_col: &str,
        text_col: &str,
        rows: &[BTreeMap<String, String>],
        head_budget: usize,
    ) -> (Question, usize) {
        // Collect unique values and group texts by label
        let mut label_texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for r in rows {
            if let (Some(target_val), Some(text_val)) = (r.get(target_col), r.get(text_col)) {
                let clean_target = target_val.trim().to_lowercase();
                if !clean_target.is_empty() {
                    label_texts
                        .entry(clean_target)
                        .or_default()
                        .push(text_val.clone());
                }
            }
        }

        let unique_labels: Vec<String> = label_texts.keys().cloned().collect();

        // 1. Check for boolean assertion -> Noul
        if unique_labels.len() == 2 {
            let is_bool = unique_labels.iter().all(|l| {
                matches!(
                    l.as_str(),
                    "true" | "false" | "yes" | "no" | "0" | "1" | "t" | "f"
                )
            });
            if is_bool {
                let instructions = format!(
                    "Evaluate whether {} holds true for this input",
                    target_col.replace('_', " ")
                );
                let q = Question::new_noul(instructions);
                let est_tokens = q
                    .render_options()
                    .iter()
                    .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
                    .sum();
                return (q, est_tokens);
            }
        }

        // 2. Check for ordinal rubric -> Score
        let all_integers =
            !unique_labels.is_empty() && unique_labels.iter().all(|l| l.parse::<i64>().is_ok());
        if all_integers && unique_labels.len() <= 6 {
            let mut int_vals: Vec<i64> = unique_labels
                .iter()
                .filter_map(|l| l.parse::<i64>().ok())
                .collect();
            int_vals.sort();
            let levels: Vec<String> = int_vals.into_iter().map(|v| v.to_string()).collect();
            let instructions = format!("Rate {} on ordinal rubric", target_col.replace('_', " "));
            let q = Question::new_score(instructions, levels);
            let est_tokens = q
                .render_options()
                .iter()
                .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
                .sum();
            return (q, est_tokens);
        }

        // 3. Multi-class Categorical -> Choice with synthesized criteria
        let mut criteria_map = serde_json::Map::new();
        let total_docs = rows.len() as f64;

        // Compute global document frequency
        let mut doc_freq: BTreeMap<String, usize> = BTreeMap::new();
        for r in rows {
            if let Some(txt) = r.get(text_col) {
                let words: BTreeSet<String> = extract_keywords(txt);
                for w in words {
                    *doc_freq.entry(w).or_default() += 1;
                }
            }
        }

        for (label, texts) in &label_texts {
            let mut label_word_counts: BTreeMap<String, usize> = BTreeMap::new();
            for t in texts {
                for w in extract_keywords(t) {
                    *label_word_counts.entry(w).or_default() += 1;
                }
            }

            // Rank words by TF-IDF relative to corpus
            let mut scored_words: Vec<(String, f64)> = label_word_counts
                .into_iter()
                .map(|(w, count)| {
                    let df = *doc_freq.get(&w).unwrap_or(&1) as f64;
                    let idf = (total_docs / (df + 1.0)).ln().max(0.1);
                    let tf = count as f64 / texts.len() as f64;
                    let score = tf * idf;
                    (w, score)
                })
                .collect();

            scored_words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top 4-5 distinctive keywords
            let top_keywords: Vec<String> =
                scored_words.into_iter().take(5).map(|(w, _)| w).collect();

            let desc = if top_keywords.is_empty() {
                label.clone()
            } else {
                top_keywords.join(", ")
            };

            criteria_map.insert(label.clone(), serde_json::Value::String(desc));
        }

        // Apply knapsack token budget reduction if criteria exceed head budget
        let mut question = Question {
            qtype: QuestionType::Choice,
            instructions: format!(
                "Which {} best categorizes this request?",
                target_col.replace('_', " ")
            ),
            criteria: Some(serde_json::Value::Object(criteria_map)),
        };

        // Iterative pruning to guarantee fitting into head_budget
        loop {
            let est_tokens: usize = question
                .render_options()
                .iter()
                .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
                .sum();

            if est_tokens <= head_budget {
                return (question, est_tokens);
            }

            // Prune longest criteria description
            if let Some(serde_json::Value::Object(map)) = &mut question.criteria {
                let mut longest_key = None;
                let mut max_words = 0;
                for (k, v) in map.iter() {
                    if let Some(s) = v.as_str() {
                        let cnt = s.split_whitespace().count();
                        if cnt > max_words && cnt > 1 {
                            max_words = cnt;
                            longest_key = Some(k.clone());
                        }
                    }
                }

                if let Some(lk) = longest_key {
                    if let Some(serde_json::Value::String(s)) = map.get_mut(&lk) {
                        let trimmed: Vec<&str> = s.split(',').map(|w| w.trim()).collect();
                        if trimmed.len() > 1 {
                            *s = trimmed[..trimmed.len() - 1].join(", ");
                        } else {
                            break;
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let final_tokens = question
            .render_options()
            .iter()
            .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
            .sum();

        (question, final_tokens)
    }

    fn read_csv_rows(&self, path: &Path) -> Result<Vec<BTreeMap<String, String>>> {
        let file = File::open(path)?;
        let mut rdr = csv::Reader::from_reader(file);
        let headers = rdr.headers()?.clone();

        let mut rows = Vec::new();
        for res in rdr.records() {
            let record = res?;
            let mut map = BTreeMap::new();
            for (h, val) in headers.iter().zip(record.iter()) {
                map.insert(h.trim().to_string(), val.trim().to_string());
            }
            rows.push(map);
        }
        Ok(rows)
    }

    fn read_json_rows(&self, path: &Path) -> Result<Vec<BTreeMap<String, String>>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut rows = Vec::new();

        for line_res in reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(serde_json::Value::Object(map)) = serde_json::from_str(trimmed) {
                let mut row = BTreeMap::new();
                for (k, v) in map {
                    let s = match v {
                        serde_json::Value::String(text) => text,
                        other => other.to_string(),
                    };
                    row.insert(k, s);
                }
                rows.push(row);
            }
        }
        Ok(rows)
    }

    fn detect_text_column(&self, rows: &[BTreeMap<String, String>]) -> Result<String> {
        let candidate_names = [
            "text",
            "body",
            "input",
            "message",
            "description",
            "content",
            "feedback",
            "snippet",
        ];
        for name in candidate_names {
            if rows.iter().any(|r| r.contains_key(name)) {
                return Ok(name.to_string());
            }
        }

        // Fallback: choose the column with longest average length
        let mut col_lens: BTreeMap<String, usize> = BTreeMap::new();
        for r in rows.iter().take(50) {
            for (col, val) in r {
                *col_lens.entry(col.clone()).or_default() += val.len();
            }
        }

        let best_col = col_lens
            .into_iter()
            .max_by_key(|(_, l)| *l)
            .map(|(c, _)| c)
            .ok_or_else(|| anyhow::anyhow!("No columns found in data"))?;

        Ok(best_col)
    }

    fn detect_target_columns(
        &self,
        rows: &[BTreeMap<String, String>],
        text_col: &str,
    ) -> Vec<String> {
        let mut targets = Vec::new();
        let total = rows.len().max(1);

        let mut unique_vals_per_col: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for r in rows {
            for (k, v) in r {
                if k != text_col && !v.is_empty() {
                    unique_vals_per_col
                        .entry(k.clone())
                        .or_default()
                        .insert(v.clone());
                }
            }
        }

        for (col, vals) in unique_vals_per_col {
            // Target columns typically have between 2 and 20 discrete categories and are not identifier columns
            if col != "id"
                && !col.ends_with("_id")
                && vals.len() >= 2
                && vals.len() <= 20
                && (total <= 10 || vals.len() <= total / 2)
            {
                targets.push(col);
            }
        }

        targets
    }
}

/// Tokenize and clean text into informative keywords (ignoring standard English stopwords).
fn extract_keywords(text: &str) -> BTreeSet<String> {
    let stopwords = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "was",
        "are", "were", "be", "been", "have", "has", "had", "do", "does", "did", "this", "that",
        "it", "i", "we", "you", "they", "my", "our", "your", "their", "of", "from", "by", "as",
        "can", "will", "would", "should", "could", "please", "hello", "hi", "thanks", "thank",
    ];

    let mut words = BTreeSet::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        let lower = raw.to_lowercase();
        if lower.len() >= 3 && !stopwords.contains(&lower.as_str()) {
            words.insert(lower);
        }
    }
    words
}

/// Report detailing schema criteria knapsack compression.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaOptimizationResult {
    pub original_tokens: usize,
    pub optimized_tokens: usize,
    pub target_budget: usize,
    pub pruned_clauses: usize,
    pub per_question_tokens: BTreeMap<String, (usize, usize)>, // qid -> (before, after)
    pub optimized_schema: BTreeMap<String, Question>,
}

/// Knapsack dynamic programming optimizer to compact verbose option criteria
/// into ModernBERT (192) or mmBERT (256) head token budgets.
pub struct SchemaKnapsackOptimizer;

impl SchemaKnapsackOptimizer {
    pub fn optimize(
        questions: &BTreeMap<String, Question>,
        head_budget: usize,
    ) -> SchemaOptimizationResult {
        let mut optimized_schema = questions.clone();
        let mut per_question_tokens = BTreeMap::new();
        let mut total_original = 0;
        let mut total_optimized = 0;
        let mut total_pruned = 0;

        for (qid, q) in &mut optimized_schema {
            let options = q.render_options();
            let est_tokens: usize = options
                .iter()
                .map(|o| o.split_whitespace().count() * 4 / 3 + 1)
                .sum();
            total_original += est_tokens;

            if est_tokens <= head_budget {
                per_question_tokens.insert(qid.clone(), (est_tokens, est_tokens));
                total_optimized += est_tokens;
                continue;
            }

            // Options exceed head budget: optimize via 0/1 knapsack across criteria
            let (new_criteria, pruned_count, after_tokens) = match q.qtype {
                QuestionType::Choice => {
                    if let Some(serde_json::Value::Object(map)) = &q.criteria {
                        Self::optimize_choice_criteria(map, head_budget)
                    } else {
                        (q.criteria.clone(), 0, est_tokens)
                    }
                }
                _ => (q.criteria.clone(), 0, est_tokens),
            };

            q.criteria = new_criteria;
            total_pruned += pruned_count;
            total_optimized += after_tokens;
            per_question_tokens.insert(qid.clone(), (est_tokens, after_tokens));
        }

        SchemaOptimizationResult {
            original_tokens: total_original,
            optimized_tokens: total_optimized,
            target_budget: head_budget,
            pruned_clauses: total_pruned,
            per_question_tokens,
            optimized_schema,
        }
    }

    fn optimize_choice_criteria(
        map: &serde_json::Map<String, serde_json::Value>,
        budget: usize,
    ) -> (Option<serde_json::Value>, usize, usize) {
        let k = map.len().max(1);
        // Reserve token budget for label names and delimiters
        let label_overhead: usize = map
            .keys()
            .map(|k| k.split_whitespace().count() * 4 / 3 + 2)
            .sum();
        let avail_budget = budget.saturating_sub(label_overhead + 8).max(k * 2);

        // Target token budget per option
        let target_per_option = (avail_budget / k).max(3);

        let mut new_map = serde_json::Map::new();
        let mut pruned_clauses = 0;

        for (label, val) in map {
            if let Some(desc) = val.as_str() {
                // Split description into clauses by punctuation
                let clauses: Vec<&str> = desc
                    .split([',', ';', '.', '\n'])
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();

                if clauses.is_empty() {
                    new_map.insert(label.clone(), serde_json::Value::String(desc.to_string()));
                    continue;
                }

                // Score clauses by information density (count of non-stopwords)
                let mut scored_clauses: Vec<(&str, usize, usize)> = Vec::new();
                for clause in &clauses {
                    let words = extract_keywords(clause);
                    let weight = clause.split_whitespace().count() * 4 / 3 + 1;
                    let score = words.len();
                    scored_clauses.push((clause, weight, score));
                }

                // 0/1 Knapsack DP on clauses to maximize information within target_per_option tokens
                let cap = target_per_option;
                let n = scored_clauses.len();
                let mut dp = vec![vec![0usize; cap + 1]; n + 1];

                for i in 1..=n {
                    let (_, w, v) = scored_clauses[i - 1];
                    for j in 0..=cap {
                        dp[i][j] = dp[i - 1][j];
                        if j >= w && dp[i - 1][j - w] + v > dp[i][j] {
                            dp[i][j] = dp[i - 1][j - w] + v;
                        }
                    }
                }

                // Backtrack to select chosen clauses
                let mut chosen = Vec::new();
                let mut curr_cap = cap;
                for i in (1..=n).rev() {
                    let (clause, w, v) = scored_clauses[i - 1];
                    if curr_cap >= w && dp[i][curr_cap] == dp[i - 1][curr_cap - w] + v && v > 0 {
                        chosen.push(clause);
                        curr_cap -= w;
                    }
                }

                chosen.reverse();

                // If knapsack picked nothing, retain the single most informative clause
                if chosen.is_empty() {
                    let best = scored_clauses.iter().max_by_key(|(_, _, v)| *v);
                    if let Some((clause, _, _)) = best {
                        chosen.push(clause);
                    }
                }

                let pruned = clauses.len().saturating_sub(chosen.len());
                pruned_clauses += pruned;

                let compact_desc = chosen.join(", ");
                new_map.insert(label.clone(), serde_json::Value::String(compact_desc));
            } else {
                new_map.insert(label.clone(), val.clone());
            }
        }

        let total_tokens: usize = new_map
            .iter()
            .map(|(k, v)| {
                let d = v.as_str().unwrap_or("");
                format!("{k}: {d}").split_whitespace().count() * 4 / 3 + 1
            })
            .sum();

        (
            Some(serde_json::Value::Object(new_map)),
            pruned_clauses,
            total_tokens,
        )
    }
}
