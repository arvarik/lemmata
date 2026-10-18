use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::datasource::DataItem;
use crate::engine::cache::{DecisionCache, compute_schema_hash};
use crate::engine::classifier::LayaClassifier;
use crate::model::{
    DecisionAnswer, DecisionResult, HierarchicalNode, HierarchicalSchema, Question,
};

/// Nested taxonomies deeper than this stop expanding.
const MAX_DEPTH: usize = 10;

/// A single step in a hierarchical routing trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalStep {
    pub level: usize,
    pub stage_name: String,
    pub question_instructions: String,
    pub chosen_option: String,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub act_probability: Option<f64>,
    pub probabilities: BTreeMap<String, f64>,
    pub latency_ms: f64,
}

/// The end-to-end outcome of hierarchical coarse-to-fine classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalDecisionResult {
    pub item_id: Option<String>,
    pub input_text: String,
    pub path: Vec<String>,
    pub formatted_path: String,
    pub final_choice: String,
    pub compound_confidence: f64,
    pub coarse_confidence: f64,
    pub fine_confidence: Option<f64>,
    pub steps: Vec<HierarchicalStep>,
    pub total_latency_ms: f64,
    pub escalated: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub escalation_reason: Option<String>,
    pub cache_hit: bool,
}

/// Benchmark statistics for hierarchical classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalBenchResult {
    pub total_samples: usize,
    pub iterations: usize,
    pub avg_coarse_latency_ms: f64,
    pub avg_fine_latency_ms: f64,
    pub avg_total_latency_ms: f64,
    pub escalation_rate: f64,
    pub completion_rate: f64,
    pub avg_compound_confidence: f64,
    pub throughput_qps: f64,
}

/// One scored path through the taxonomy.
struct Candidate {
    probability: f64,
    steps: Vec<HierarchicalStep>,
}

/// Classifies text against a label taxonomy. At each level it follows the top `beam` options,
/// scores each path as the product of its probabilities, and returns the best path.
pub struct HierarchicalRouter {
    schema: HierarchicalSchema,
    classifier: LayaClassifier,
    beam: usize,
    path_threshold: f64,
    cache: Option<Arc<Mutex<DecisionCache>>>,
}

impl HierarchicalRouter {
    pub fn new(schema: HierarchicalSchema, classifier: LayaClassifier) -> Self {
        Self {
            schema,
            classifier,
            beam: 2,
            path_threshold: 0.50,
            cache: None,
        }
    }

    /// Follow the top `beam` options at each level. 1 is the greedy top-down router.
    pub fn with_beam(mut self, beam: usize) -> Self {
        self.beam = beam.max(1);
        self
    }

    /// Escalate an item when its best path probability is below `threshold`.
    pub fn with_path_threshold(mut self, threshold: f64) -> Self {
        self.path_threshold = threshold;
        self
    }

    pub fn with_cache(mut self, cache: Arc<Mutex<DecisionCache>>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn schema(&self) -> &HierarchicalSchema {
        &self.schema
    }

    /// Route one text through the taxonomy.
    pub fn route_text(
        &self,
        text: &str,
        item_id: Option<String>,
    ) -> Result<HierarchicalDecisionResult> {
        let clean_text = text.trim();
        let start = Instant::now();
        let mut all_hits = self.cache.is_some();
        let mut candidates = Vec::new();
        self.expand(
            clean_text,
            &self.schema,
            0,
            1.0,
            Vec::new(),
            &mut candidates,
            &mut all_hits,
        )?;
        let best = candidates
            .into_iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .ok_or_else(|| anyhow!("The taxonomy produced no path for this input"))?;

        let path: Vec<String> = best.steps.iter().map(|s| s.chosen_option.clone()).collect();
        let escalated = best.probability < self.path_threshold;
        Ok(HierarchicalDecisionResult {
            item_id,
            input_text: clean_text.to_string(),
            formatted_path: path.join(" -> "),
            final_choice: path.last().cloned().unwrap_or_default(),
            compound_confidence: (best.probability * 10000.0).round() / 10000.0,
            coarse_confidence: best.steps.first().map_or(0.0, |s| s.confidence),
            fine_confidence: (best.steps.len() > 1)
                .then(|| best.steps.last().map_or(0.0, |s| s.confidence)),
            path,
            steps: best.steps,
            total_latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            escalated,
            status: if escalated { "ESCALATED" } else { "COMPLETED" }.to_string(),
            escalation_reason: escalated.then(|| {
                format!(
                    "Path probability {:.4} is below the threshold {:.4}",
                    best.probability, self.path_threshold
                )
            }),
            cache_hit: all_hits,
        })
    }

    /// Score the top `beam` options of `node.coarse`, then expand each one into its branch.
    #[allow(clippy::too_many_arguments)]
    fn expand(
        &self,
        text: &str,
        node: &HierarchicalSchema,
        level: usize,
        probability: f64,
        steps: Vec<HierarchicalStep>,
        out: &mut Vec<Candidate>,
        all_hits: &mut bool,
    ) -> Result<()> {
        if level >= MAX_DEPTH {
            out.push(Candidate { probability, steps });
            return Ok(());
        }
        let (answer, latency_ms) = self.classify(text, &node.coarse, all_hits)?;
        for (label, p) in top_options(&answer, self.beam) {
            let mut path_steps = steps.clone();
            path_steps.push(step(level, &node.coarse, &label, p, &answer, latency_ms));
            let path_probability = probability * p;
            match self.find_branch(&label, &node.branches) {
                None => out.push(Candidate {
                    probability: path_probability,
                    steps: path_steps,
                }),
                // A leaf question ends the path, so only its best option can win.
                Some(HierarchicalNode::Leaf(question)) => {
                    let (leaf, leaf_ms) = self.classify(text, question, all_hits)?;
                    for (leaf_label, lp) in top_options(&leaf, 1) {
                        let mut leaf_steps = path_steps.clone();
                        leaf_steps.push(step(level + 1, question, &leaf_label, lp, &leaf, leaf_ms));
                        out.push(Candidate {
                            probability: path_probability * lp,
                            steps: leaf_steps,
                        });
                    }
                }
                Some(HierarchicalNode::Subtree(sub)) => self.expand(
                    text,
                    sub,
                    level + 1,
                    path_probability,
                    path_steps,
                    out,
                    all_hits,
                )?,
            }
        }
        Ok(())
    }

    /// Classify `text` against one question, through the cache when one is set. The cache key
    /// is the question itself, so different taxonomies and levels never share an entry.
    fn classify(
        &self,
        text: &str,
        question: &Question,
        all_hits: &mut bool,
    ) -> Result<(DecisionAnswer, f64)> {
        let questions = BTreeMap::from([("q".to_string(), question.clone())]);
        let schema_hash = compute_schema_hash(&questions);
        if let Some(cache) = &self.cache
            && let Some(hit) = cache
                .lock()
                .ok()
                .and_then(|mut c| c.lookup(text, &schema_hash))
            && let Some(answer) = hit.answers.get("q")
        {
            return Ok((answer.clone(), 0.0));
        }
        *all_hits = false;
        let start = Instant::now();
        let answer = self.classifier.classify_single(text, question)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(cache) = &self.cache
            && let Ok(mut c) = cache.lock()
        {
            let result = DecisionResult {
                model: self.classifier.config().checkpoint.repo_id().to_string(),
                answers: BTreeMap::from([("q".to_string(), answer.clone())]),
                input_tokens: 0,
                latency_ms,
            };
            c.insert(text, &schema_hash, &result);
        }
        Ok((answer, latency_ms))
    }

    /// Match coarse choice against branch map keys (exact or normalized).
    fn find_branch<'a>(
        &self,
        key: &str,
        branches: &'a BTreeMap<String, HierarchicalNode>,
    ) -> Option<&'a HierarchicalNode> {
        if let Some(node) = branches.get(key) {
            return Some(node);
        }
        let clean = key.trim().to_lowercase();
        let clean_prefix = clean.split(':').next().unwrap_or(&clean).trim();

        for (k, node) in branches {
            let k_clean = k.trim().to_lowercase();
            let k_prefix = k_clean.split(':').next().unwrap_or(&k_clean).trim();
            if k_clean == clean
                || k_prefix == clean_prefix
                || clean.starts_with(k_prefix)
                || k_clean.starts_with(clean_prefix)
            {
                return Some(node);
            }
        }
        None
    }

    /// Classify a slice of DataItems in parallel using Rayon.
    pub fn classify_items(&self, items: &[DataItem]) -> Result<Vec<HierarchicalDecisionResult>> {
        crate::model::native::map_items(items, |item| {
            let text = match &item.state {
                serde_json::Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            self.route_text(&text, Some(item.id.clone()))
        })
        .into_iter()
        .collect()
    }

    /// Benchmark hierarchical routing over a test dataset.
    pub fn benchmark(
        &self,
        items: &[DataItem],
        iterations: usize,
    ) -> Result<HierarchicalBenchResult> {
        if items.is_empty() || iterations == 0 {
            anyhow::bail!("Cannot benchmark with 0 items or 0 iterations");
        }

        let total_samples = items.len() * iterations;
        let mut total_coarse_lat = 0.0;
        let mut total_fine_lat = 0.0;
        let mut total_lat = 0.0;
        let mut escalations = 0;
        let mut completions = 0;
        let mut fine_stage_count = 0;
        let mut compound_conf_sum = 0.0;

        let bench_start = Instant::now();

        for _ in 0..iterations {
            let results = self.classify_items(items)?;
            for res in results {
                total_lat += res.total_latency_ms;
                compound_conf_sum += res.compound_confidence;

                if let Some(s0) = res.steps.first() {
                    total_coarse_lat += s0.latency_ms;
                }
                if let Some(s1) = res.steps.get(1) {
                    total_fine_lat += s1.latency_ms;
                    fine_stage_count += 1;
                }

                if res.escalated {
                    escalations += 1;
                } else {
                    completions += 1;
                }
            }
        }

        let wall_time_sec = bench_start.elapsed().as_secs_f64();
        let throughput = if wall_time_sec > 0.0 {
            total_samples as f64 / wall_time_sec
        } else {
            0.0
        };

        Ok(HierarchicalBenchResult {
            total_samples,
            iterations,
            avg_coarse_latency_ms: total_coarse_lat / total_samples as f64,
            avg_fine_latency_ms: if fine_stage_count > 0 {
                total_fine_lat / fine_stage_count as f64
            } else {
                0.0
            },
            avg_total_latency_ms: total_lat / total_samples as f64,
            escalation_rate: escalations as f64 / total_samples as f64,
            completion_rate: completions as f64 / total_samples as f64,
            avg_compound_confidence: compound_conf_sum / total_samples as f64,
            throughput_qps: throughput,
        })
    }

    /// Format results as an ASCII table.
    pub fn format_table(results: &[HierarchicalDecisionResult]) -> String {
        let mut table = comfy_table::Table::new();
        table.set_header(vec![
            "#",
            "Input Snippet",
            "Coarse Intent",
            "Branch Intent",
            "Comp.Conf",
            "Status",
            "Latency",
        ]);

        for (i, r) in results.iter().enumerate() {
            let snippet: String = r.input_text.chars().take(26).collect();
            let snippet = if r.input_text.chars().count() > 26 {
                format!("{snippet}..")
            } else {
                snippet
            };
            let coarse = r
                .steps
                .first()
                .map(|s| format!("{}:{:.2}", s.chosen_option, s.confidence))
                .or_else(|| {
                    r.path
                        .first()
                        .map(|p| format!("{p}:{:.2}", r.coarse_confidence))
                })
                .unwrap_or_else(|| "-".into());
            let fine = r
                .steps
                .get(1)
                .map(|s| format!("{}:{:.2}", s.chosen_option, s.confidence))
                .or_else(|| {
                    r.path.get(1).map(|p| {
                        format!(
                            "{p}:{:.2}",
                            r.fine_confidence.unwrap_or(r.compound_confidence)
                        )
                    })
                })
                .unwrap_or_else(|| "-".into());

            table.add_row(vec![
                (i + 1).to_string(),
                snippet,
                coarse,
                fine,
                format!("{:.3}", r.compound_confidence),
                r.status.clone(),
                format!("{:.1}ms", r.total_latency_ms),
            ]);
        }

        table.to_string()
    }

    /// Format results as CSV string.
    pub fn format_csv(results: &[HierarchicalDecisionResult]) -> String {
        let mut out = String::new();
        out.push_str("id,input_snippet,coarse_intent,coarse_conf,fine_intent,fine_conf,path,compound_conf,status,escalated,latency_ms\n");
        for (i, r) in results.iter().enumerate() {
            let id = r.item_id.clone().unwrap_or_else(|| format!("item_{}", i));
            let clean_snip = r.input_text.replace('\"', "\"\"").replace('\n', " ");
            let coarse_opt = r
                .steps
                .first()
                .map(|s| s.chosen_option.as_str())
                .or_else(|| r.path.first().map(|s| s.as_str()))
                .unwrap_or("");
            let coarse_conf = r.coarse_confidence;
            let fine_opt = r
                .steps
                .get(1)
                .map(|s| s.chosen_option.as_str())
                .or_else(|| r.path.get(1).map(|s| s.as_str()))
                .unwrap_or("");
            let fine_conf = r.fine_confidence.unwrap_or(0.0);

            out.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",{:.4},\"{}\",{:.4},\"{}\",{:.4},\"{}\",{},{:.2}\n",
                id,
                clean_snip,
                coarse_opt,
                coarse_conf,
                fine_opt,
                fine_conf,
                r.formatted_path,
                r.compound_confidence,
                r.status,
                r.escalated,
                r.total_latency_ms
            ));
        }
        out
    }

    /// Format results as JSONL string.
    pub fn format_jsonl(results: &[HierarchicalDecisionResult]) -> Result<String> {
        let mut out = String::new();
        for r in results {
            out.push_str(&serde_json::to_string(r)?);
            out.push('\n');
        }
        Ok(out)
    }
}

/// The `k` most probable options of an answer. An answer without option probabilities (a score
/// or yes/no question) gives its single choice.
fn top_options(answer: &DecisionAnswer, k: usize) -> Vec<(String, f64)> {
    let mut options: Vec<(String, f64)> = answer
        .probabilities
        .iter()
        .map(|(label, &p)| (label.clone(), p))
        .collect();
    if options.is_empty() {
        options.push((
            answer
                .choice
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            answer.confidence,
        ));
    }
    options.sort_by(|a, b| b.1.total_cmp(&a.1));
    options.truncate(k.max(1));
    options
}

fn step(
    level: usize,
    question: &Question,
    label: &str,
    probability: f64,
    answer: &DecisionAnswer,
    latency_ms: f64,
) -> HierarchicalStep {
    HierarchicalStep {
        level,
        stage_name: if level == 0 {
            "coarse".to_string()
        } else {
            format!("level_{level}")
        },
        question_instructions: question.instructions.clone(),
        chosen_option: label.to_string(),
        confidence: probability,
        act_probability: answer.act_probability,
        probabilities: answer.probabilities.clone(),
        latency_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Checkpoint, HierarchicalNode, Question};

    #[test]
    fn wider_beam_never_lowers_the_best_path() {
        let schema: HierarchicalSchema =
            serde_json::from_str(include_str!("../../schema/banking_hierarchy.json")).unwrap();
        let greedy = HierarchicalRouter::new(
            schema.clone(),
            LayaClassifier::with_checkpoint(Checkpoint::English),
        )
        .with_beam(1);
        let wide =
            HierarchicalRouter::new(schema, LayaClassifier::with_checkpoint(Checkpoint::English))
                .with_beam(3);
        for text in [
            "my debit card was stolen in London",
            "the wire transfer has not arrived after 48 hours",
            "someone logged into my account from another country",
            "how do I change my ATM PIN",
        ] {
            let g = greedy.route_text(text, None).unwrap();
            let w = wide.route_text(text, None).unwrap();
            // The wide beam also scores the greedy path, so its best path is at least as likely.
            assert!(w.compound_confidence >= g.compound_confidence, "{text}");
            assert_eq!(w.path.len(), w.steps.len());
            assert_eq!(w.escalated, w.compound_confidence < 0.5);
        }
    }

    #[test]
    fn top_options_ranks_by_probability() {
        let answer = DecisionAnswer {
            qtype: crate::model::QuestionType::Choice,
            choice: Some("b".into()),
            selected_labels: None,
            score: None,
            noul: None,
            confidence: 0.6,
            probabilities: BTreeMap::from([
                ("a".to_string(), 0.1),
                ("b".to_string(), 0.6),
                ("c".to_string(), 0.3),
            ]),
            act_probability: None,
            logits: None,
        };
        let top = top_options(&answer, 2);
        assert_eq!(top.len(), 2);
        assert_eq!((top[0].0.as_str(), top[1].0.as_str()), ("b", "c"));
        let no_probs = DecisionAnswer {
            probabilities: BTreeMap::new(),
            ..answer
        };
        assert_eq!(top_options(&no_probs, 3), vec![("b".to_string(), 0.6)]);
    }

    #[test]
    fn test_hierarchical_benchmark_zero_iterations() {
        let schema = HierarchicalSchema {
            coarse: Question::new_choice("Dept", vec!["billing".into()]),
            branches: BTreeMap::new(),
        };
        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let router = HierarchicalRouter::new(schema, classifier);
        let items = vec![DataItem::new(
            "1",
            serde_json::json!({"text": "test"}),
            "test",
        )];
        assert!(router.benchmark(&items, 0).is_err());
    }

    #[test]
    fn test_hierarchical_formatters_cache_hit_fallback() {
        let res = HierarchicalDecisionResult {
            item_id: Some("cached_1".into()),
            input_text: "Need payment assistance".into(),
            path: vec!["billing".into(), "invoices".into()],
            formatted_path: "billing -> invoices".into(),
            final_choice: "invoices".into(),
            compound_confidence: 0.95,
            coarse_confidence: 0.95,
            fine_confidence: Some(0.95),
            steps: vec![],
            total_latency_ms: 0.5,
            escalated: false,
            status: "COMPLETED (CACHED)".into(),
            escalation_reason: None,
            cache_hit: true,
        };

        let table = HierarchicalRouter::format_table(std::slice::from_ref(&res));
        assert!(table.contains("billing:0.95"));
        assert!(table.contains("invoices:0.95"));

        let csv = HierarchicalRouter::format_csv(&[res]);
        assert!(csv.contains("\"billing\""));
        assert!(csv.contains("\"invoices\""));
    }

    #[test]
    fn test_hierarchical_find_branch_prefix_match() {
        let mut branches = BTreeMap::new();
        branches.insert(
            "cards: card issues".into(),
            HierarchicalNode::Leaf(Question::new_choice("Card Inquiry", vec!["lost".into()])),
        );
        let schema = HierarchicalSchema {
            coarse: Question::new_choice("Dept", vec!["cards".into()]),
            branches,
        };
        let classifier = LayaClassifier::with_checkpoint(Checkpoint::English);
        let router = HierarchicalRouter::new(schema, classifier);

        assert!(
            router
                .find_branch("cards", &router.schema.branches)
                .is_some()
        );
        assert!(
            router
                .find_branch("cards: card issues", &router.schema.branches)
                .is_some()
        );
    }
}
