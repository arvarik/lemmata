use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::datasource::DataItem;
use crate::engine::cache::{CacheLookupResult, DecisionCache};
use crate::engine::classifier::LayaClassifier;
use crate::engine::ood::OodDetector;
use crate::model::{
    DecisionAnswer, DecisionResult, HierarchicalNode, HierarchicalSchema, QuestionType,
};

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
    pub coarse_escalation_rate: f64,
    pub completion_rate: f64,
    pub avg_compound_confidence: f64,
    pub throughput_qps: f64,
}

/// The hierarchical decision router orchestrating coarse-to-fine inference.
pub struct HierarchicalRouter {
    schema: HierarchicalSchema,
    classifier: LayaClassifier,
    coarse_threshold: f64,
    escalate_threshold: f64,
    cache: Option<Arc<Mutex<DecisionCache>>>,
    ood_detector: Option<Arc<OodDetector>>,
}

impl HierarchicalRouter {
    pub fn new(schema: HierarchicalSchema, classifier: LayaClassifier) -> Self {
        Self {
            schema,
            classifier,
            coarse_threshold: 0.50,
            escalate_threshold: 0.65,
            cache: None,
            ood_detector: None,
        }
    }

    pub fn with_coarse_threshold(mut self, threshold: f64) -> Self {
        self.coarse_threshold = threshold;
        self
    }

    pub fn with_escalate_threshold(mut self, threshold: f64) -> Self {
        self.escalate_threshold = threshold;
        self
    }

    pub fn with_cache(mut self, cache: Arc<Mutex<DecisionCache>>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn with_ood_detector(mut self, detector: Arc<OodDetector>) -> Self {
        self.ood_detector = Some(detector);
        self
    }

    pub fn schema(&self) -> &HierarchicalSchema {
        &self.schema
    }

    /// Route a single text input through the coarse-to-fine hierarchy.
    pub fn route_text(
        &self,
        text: &str,
        item_id: Option<String>,
    ) -> Result<HierarchicalDecisionResult> {
        let clean_text = text.trim();

        // 1. Check semantic / exact cache
        if let Some(cache_mutex) = &self.cache
            && let Ok(mut c) = cache_mutex.lock()
        {
            match c.lookup(clean_text, "hierarchical") {
                CacheLookupResult::ExactHit(res)
                | CacheLookupResult::SemanticHit { result: res, .. } => {
                    if let Some(ans) = res.answers.get("final") {
                        let compound_conf = ans.confidence;
                        let final_choice = ans.choice.clone().unwrap_or_default();
                        let formatted_path = ans
                            .selected_labels
                            .as_ref()
                            .map(|v| v.join(" -> "))
                            .unwrap_or_else(|| final_choice.clone());
                        let path = ans
                            .selected_labels
                            .clone()
                            .unwrap_or_else(|| vec![final_choice.clone()]);
                        return Ok(HierarchicalDecisionResult {
                            item_id,
                            input_text: clean_text.to_string(),
                            path,
                            formatted_path,
                            final_choice,
                            compound_confidence: compound_conf,
                            coarse_confidence: compound_conf,
                            fine_confidence: None,
                            steps: vec![],
                            total_latency_ms: res.latency_ms,
                            escalated: false,
                            status: "COMPLETED (CACHED)".to_string(),
                            escalation_reason: None,
                            cache_hit: true,
                        });
                    }
                }
                CacheLookupResult::Miss => {}
            }
        }

        let total_start = Instant::now();
        let mut steps = Vec::new();

        // 2. Stage 1: Coarse Intent Router
        let coarse_start = Instant::now();

        // Check OOD gate at root if detector is active
        if let Some(ood) = &self.ood_detector {
            let mut qmap = BTreeMap::new();
            qmap.insert("coarse".to_string(), self.schema.coarse.clone());
            if let Ok(eval) = ood.evaluate("coarse", clean_text, &self.classifier, &qmap)
                && eval.is_ood
            {
                let total_latency = total_start.elapsed().as_secs_f64() * 1000.0;
                return Ok(HierarchicalDecisionResult {
                    item_id,
                    input_text: clean_text.to_string(),
                    path: vec!["REJECTED_OOD".to_string()],
                    formatted_path: "REJECTED_OOD".to_string(),
                    final_choice: "REJECTED_OOD".to_string(),
                    compound_confidence: 0.0,
                    coarse_confidence: 0.0,
                    fine_confidence: None,
                    steps: vec![],
                    total_latency_ms: total_latency,
                    escalated: true,
                    status: "REJECTED_OOD".to_string(),
                    escalation_reason: Some(format!(
                        "Out-of-Distribution detected at coarse stage (anomaly score: {:.4})",
                        eval.anomaly_score
                    )),
                    cache_hit: false,
                });
            }
        }

        let coarse_ans = self
            .classifier
            .classify_single(clean_text, &self.schema.coarse)?;
        let coarse_latency = coarse_start.elapsed().as_secs_f64() * 1000.0;
        let coarse_choice = coarse_ans
            .choice
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let coarse_conf = coarse_ans.confidence;
        let coarse_act = coarse_ans.act_probability.unwrap_or(0.0);

        steps.push(HierarchicalStep {
            level: 0,
            stage_name: "coarse".to_string(),
            question_instructions: self.schema.coarse.instructions.clone(),
            chosen_option: coarse_choice.clone(),
            confidence: coarse_conf,
            act_probability: coarse_ans.act_probability,
            probabilities: coarse_ans.probabilities.clone(),
            latency_ms: coarse_latency,
        });

        // Coarse Gate: Escalation check
        if coarse_conf < self.coarse_threshold {
            let total_latency = total_start.elapsed().as_secs_f64() * 1000.0;
            return Ok(HierarchicalDecisionResult {
                item_id,
                input_text: clean_text.to_string(),
                path: vec![coarse_choice.clone()],
                formatted_path: coarse_choice.clone(),
                final_choice: coarse_choice,
                compound_confidence: coarse_conf,
                coarse_confidence: coarse_conf,
                fine_confidence: None,
                steps,
                total_latency_ms: total_latency,
                escalated: true,
                status: "ESCALATED_AT_COARSE".to_string(),
                escalation_reason: Some(format!(
                    "Coarse confidence {:.4} below threshold {:.4}",
                    coarse_conf, self.coarse_threshold
                )),
                cache_hit: false,
            });
        }

        if coarse_act > self.escalate_threshold {
            let total_latency = total_start.elapsed().as_secs_f64() * 1000.0;
            return Ok(HierarchicalDecisionResult {
                item_id,
                input_text: clean_text.to_string(),
                path: vec![coarse_choice.clone()],
                formatted_path: coarse_choice.clone(),
                final_choice: coarse_choice,
                compound_confidence: coarse_conf,
                coarse_confidence: coarse_conf,
                fine_confidence: None,
                steps,
                total_latency_ms: total_latency,
                escalated: true,
                status: "ESCALATED_AT_COARSE".to_string(),
                escalation_reason: Some(format!(
                    "Coarse act probability {:.4} exceeds threshold {:.4}",
                    coarse_act, self.escalate_threshold
                )),
                cache_hit: false,
            });
        }

        // 3. Stage 2: Branch Sub-Classification
        let branch_node = self.find_branch(&coarse_choice, &self.schema.branches);

        let result = match branch_node {
            None => {
                // Coarse choice is a terminal leaf
                let total_latency = total_start.elapsed().as_secs_f64() * 1000.0;
                HierarchicalDecisionResult {
                    item_id,
                    input_text: clean_text.to_string(),
                    path: vec![coarse_choice.clone()],
                    formatted_path: coarse_choice.clone(),
                    final_choice: coarse_choice,
                    compound_confidence: coarse_conf,
                    coarse_confidence: coarse_conf,
                    fine_confidence: None,
                    steps,
                    total_latency_ms: total_latency,
                    escalated: false,
                    status: "COMPLETED".to_string(),
                    escalation_reason: None,
                    cache_hit: false,
                }
            }
            Some(HierarchicalNode::Leaf(sub_q)) => {
                let fine_start = Instant::now();
                let fine_ans = self.classifier.classify_single(clean_text, sub_q)?;
                let fine_latency = fine_start.elapsed().as_secs_f64() * 1000.0;
                let fine_choice = fine_ans
                    .choice
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                let fine_conf = fine_ans.confidence;
                let fine_act = fine_ans.act_probability.unwrap_or(0.0);

                steps.push(HierarchicalStep {
                    level: 1,
                    stage_name: format!("branch:{}", coarse_choice),
                    question_instructions: sub_q.instructions.clone(),
                    chosen_option: fine_choice.clone(),
                    confidence: fine_conf,
                    act_probability: fine_ans.act_probability,
                    probabilities: fine_ans.probabilities.clone(),
                    latency_ms: fine_latency,
                });

                let compound_conf = ((coarse_conf * fine_conf) * 10000.0).round() / 10000.0;
                let total_latency = total_start.elapsed().as_secs_f64() * 1000.0;
                let path = vec![coarse_choice.clone(), fine_choice.clone()];
                let formatted_path = format!("{} -> {}", coarse_choice, fine_choice);

                let (escalated, status, reason) = if fine_act > self.escalate_threshold {
                    (
                        true,
                        "ESCALATED_AT_BRANCH".to_string(),
                        Some(format!(
                            "Fine branch act probability {:.4} exceeds threshold {:.4}",
                            fine_act, self.escalate_threshold
                        )),
                    )
                } else {
                    (false, "COMPLETED".to_string(), None)
                };

                HierarchicalDecisionResult {
                    item_id,
                    input_text: clean_text.to_string(),
                    path,
                    formatted_path,
                    final_choice: fine_choice,
                    compound_confidence: compound_conf,
                    coarse_confidence: coarse_conf,
                    fine_confidence: Some(fine_conf),
                    steps,
                    total_latency_ms: total_latency,
                    escalated,
                    status,
                    escalation_reason: reason,
                    cache_hit: false,
                }
            }
            Some(HierarchicalNode::Subtree(sub_schema)) => {
                let mut path = vec![coarse_choice.clone()];
                let mut current_compound = coarse_conf;
                let mut fine_conf_val = None;
                let mut fine_choice_val = coarse_choice.clone();
                let mut escalated = false;
                let mut status = "COMPLETED".to_string();
                let mut reason = None;

                self.traverse_subtree(
                    clean_text,
                    sub_schema,
                    1,
                    &mut path,
                    &mut steps,
                    &mut current_compound,
                    &mut fine_conf_val,
                    &mut fine_choice_val,
                    &mut escalated,
                    &mut status,
                    &mut reason,
                )?;

                let total_latency = total_start.elapsed().as_secs_f64() * 1000.0;
                let formatted_path = path.join(" -> ");

                HierarchicalDecisionResult {
                    item_id,
                    input_text: clean_text.to_string(),
                    path,
                    formatted_path,
                    final_choice: fine_choice_val,
                    compound_confidence: current_compound,
                    coarse_confidence: coarse_conf,
                    fine_confidence: fine_conf_val,
                    steps,
                    total_latency_ms: total_latency,
                    escalated,
                    status,
                    escalation_reason: reason,
                    cache_hit: false,
                }
            }
        };

        // Cache admission check
        if let Some(cache_mutex) = &self.cache
            && !result.escalated
            && result.compound_confidence >= 0.70
        {
            let mut answers = BTreeMap::new();
            answers.insert(
                "final".to_string(),
                DecisionAnswer {
                    qtype: QuestionType::Choice,
                    choice: Some(result.final_choice.clone()),
                    selected_labels: Some(result.path.clone()),
                    score: None,
                    noul: None,
                    confidence: result.compound_confidence,
                    probabilities: BTreeMap::new(),
                    act_probability: Some(0.1),
                },
            );
            let dec_res = DecisionResult {
                model: "laya-hierarchical".to_string(),
                answers,
                input_tokens: clean_text.split_whitespace().count(),
                latency_ms: result.total_latency_ms,
            };
            if let Ok(mut c) = cache_mutex.lock() {
                c.insert(clean_text, "hierarchical", &dec_res);
            }
        }

        Ok(result)
    }

    /// Recursively traverse nested subtrees up to depth limit.
    #[allow(clippy::too_many_arguments)]
    fn traverse_subtree(
        &self,
        text: &str,
        schema: &HierarchicalSchema,
        level: usize,
        path: &mut Vec<String>,
        steps: &mut Vec<HierarchicalStep>,
        compound_confidence: &mut f64,
        last_fine_conf: &mut Option<f64>,
        final_choice: &mut String,
        escalated: &mut bool,
        status: &mut String,
        reason: &mut Option<String>,
    ) -> Result<()> {
        if level > 10 {
            *escalated = true;
            *status = "ESCALATED_DEPTH_LIMIT".to_string();
            *reason = Some("Hierarchical traversal exceeded maximum depth limit of 10".to_string());
            return Ok(());
        }

        let step_start = Instant::now();
        let ans = self.classifier.classify_single(text, &schema.coarse)?;
        let latency = step_start.elapsed().as_secs_f64() * 1000.0;
        let choice = ans.choice.clone().unwrap_or_else(|| "unknown".to_string());
        let conf = ans.confidence;
        let act = ans.act_probability.unwrap_or(0.0);

        steps.push(HierarchicalStep {
            level,
            stage_name: format!("level_{}:{}", level, choice),
            question_instructions: schema.coarse.instructions.clone(),
            chosen_option: choice.clone(),
            confidence: conf,
            act_probability: ans.act_probability,
            probabilities: ans.probabilities.clone(),
            latency_ms: latency,
        });

        path.push(choice.clone());
        *compound_confidence = ((*compound_confidence * conf) * 10000.0).round() / 10000.0;
        *last_fine_conf = Some(conf);
        *final_choice = choice.clone();

        if act > self.escalate_threshold {
            *escalated = true;
            *status = format!("ESCALATED_AT_LEVEL_{}", level);
            *reason = Some(format!(
                "Subtree act probability {:.4} exceeds threshold {:.4}",
                act, self.escalate_threshold
            ));
            return Ok(());
        }

        if let Some(node) = self.find_branch(&choice, &schema.branches) {
            match node {
                HierarchicalNode::Leaf(sub_q) => {
                    let leaf_start = Instant::now();
                    let leaf_ans = self.classifier.classify_single(text, sub_q)?;
                    let leaf_latency = leaf_start.elapsed().as_secs_f64() * 1000.0;
                    let leaf_choice = leaf_ans
                        .choice
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string());
                    let leaf_conf = leaf_ans.confidence;
                    let leaf_act = leaf_ans.act_probability.unwrap_or(0.0);

                    steps.push(HierarchicalStep {
                        level: level + 1,
                        stage_name: format!("leaf:{}", leaf_choice),
                        question_instructions: sub_q.instructions.clone(),
                        chosen_option: leaf_choice.clone(),
                        confidence: leaf_conf,
                        act_probability: leaf_ans.act_probability,
                        probabilities: leaf_ans.probabilities.clone(),
                        latency_ms: leaf_latency,
                    });

                    path.push(leaf_choice.clone());
                    *compound_confidence =
                        ((*compound_confidence * leaf_conf) * 10000.0).round() / 10000.0;
                    *last_fine_conf = Some(leaf_conf);
                    *final_choice = leaf_choice;

                    if leaf_act > self.escalate_threshold {
                        *escalated = true;
                        *status = "ESCALATED_AT_LEAF".to_string();
                        *reason = Some(format!(
                            "Leaf act probability {:.4} exceeds threshold {:.4}",
                            leaf_act, self.escalate_threshold
                        ));
                    }
                }
                HierarchicalNode::Subtree(sub) => {
                    self.traverse_subtree(
                        text,
                        sub,
                        level + 1,
                        path,
                        steps,
                        compound_confidence,
                        last_fine_conf,
                        final_choice,
                        escalated,
                        status,
                        reason,
                    )?;
                }
            }
        }

        Ok(())
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
        items
            .par_iter()
            .map(|item| {
                let text = match &item.state {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                self.route_text(&text, Some(item.id.clone()))
            })
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
        let mut coarse_escalations = 0;
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

                if res.status == "ESCALATED_AT_COARSE" {
                    coarse_escalations += 1;
                } else if res.status == "COMPLETED" {
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
            coarse_escalation_rate: coarse_escalations as f64 / total_samples as f64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Checkpoint, HierarchicalNode, Question};

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
