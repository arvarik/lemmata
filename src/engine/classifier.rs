use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::datasource::DataItem;
use crate::model::{
    Checkpoint, DecisionAnswer, DecisionModel, DecisionResult, Question, QuestionType,
    estimate_tokens,
};

static WARNED_HEAD_BUDGET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Option count above which native routing prefers an NLI cross-encoder over Laya.
const HIGH_CARDINALITY: usize = 20;

/// Laya decision engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub checkpoint: Checkpoint,
    pub temperature: f64,
    pub temperature_by_options: BTreeMap<String, f64>,
    pub escalate_threshold: f64,
    pub multi_label: bool,
    pub multi_label_threshold: f64,
    pub local_weights_path: Option<std::path::PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qats_profile: Option<crate::engine::calibrate::QatsProfile>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cost_weights: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_template: Option<String>,
    #[serde(default)]
    pub chunk_strategy: crate::model::ChunkStrategy,
    #[serde(default)]
    pub shortlist_k: usize,
}

impl ClassifierConfig {
    pub fn with_checkpoint(checkpoint: Checkpoint) -> Self {
        let mut temp_map: BTreeMap<String, f64> = checkpoint
            .default_temperature_by_options()
            .into_iter()
            .collect();

        // Load from local rl_agent_config.json if available on disk
        let sub = checkpoint.local_subdir();
        let cfg_path = if sub.is_empty() {
            std::path::PathBuf::from("models/laya/rl_agent_config.json")
        } else {
            std::path::PathBuf::from(format!("models/laya/{sub}/rl_agent_config.json"))
        };
        if matches!(
            checkpoint,
            Checkpoint::English
                | Checkpoint::Multilingual
                | Checkpoint::TypedDecisions
                | Checkpoint::AutoRoute
        ) && let Ok(content) = std::fs::read_to_string(&cfg_path)
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(&content)
            && let Some(obj) = val
                .get("temperature_by_options")
                .and_then(|v| v.as_object())
        {
            for (k, v) in obj {
                if let Some(t) = v.as_f64() {
                    temp_map.insert(k.clone(), t);
                }
            }
        }

        let local_safetensors = checkpoint.local_dir().join("model.safetensors");
        let local_weights_path = local_safetensors.exists().then_some(local_safetensors);

        let mut cfg = Self {
            checkpoint,
            temperature: 1.0,
            temperature_by_options: temp_map,
            escalate_threshold: 0.5,
            multi_label: false,
            multi_label_threshold: 0.40,
            local_weights_path,
            qats_profile: None,
            cost_weights: BTreeMap::new(),
            hypothesis_template: None,
            chunk_strategy: crate::model::ChunkStrategy::default(),
            shortlist_k: 0,
        };
        if let Some(cal) = crate::engine::model_calibration::load_checkpoint_calibration(checkpoint)
        {
            crate::engine::model_calibration::apply_persisted_calibration(&mut cfg, &cal);
        }
        cfg
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp.max(0.01);
        self.temperature_by_options.clear();
        self
    }
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self::with_checkpoint(Checkpoint::English)
    }
}

/// Laya normalized negentropy confidence: `1 - H(p) / ln(k)`.
pub fn confidence_from_probs(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = -probs
        .iter()
        .take(k)
        .filter(|&&p| p > 1e-12)
        .map(|&p| p * p.ln())
        .sum::<f64>();
    (1.0 - ent / (k as f64).ln().max(1e-9)).clamp(0.0, 1.0)
}

/// The classifier runner for Laya System 1 decision models.
pub struct LayaClassifier {
    config: ClassifierConfig,
}

impl LayaClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    pub fn with_checkpoint(checkpoint: Checkpoint) -> Self {
        Self::new(ClassifierConfig::with_checkpoint(checkpoint))
    }

    pub fn with_auto_best() -> Self {
        Self::with_checkpoint(Checkpoint::AutoRoute)
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.config = self.config.with_temperature(temp);
        self
    }

    pub fn with_hypothesis_template(mut self, tpl: impl Into<String>) -> Self {
        self.config.hypothesis_template = Some(tpl.into());
        self
    }

    pub fn with_chunk_strategy(mut self, strategy: crate::model::ChunkStrategy) -> Self {
        self.config.chunk_strategy = strategy;
        self
    }

    pub fn with_shortlist_k(mut self, k: usize) -> Self {
        self.config.shortlist_k = k;
        self
    }

    pub fn config(&self) -> &ClassifierConfig {
        &self.config
    }

    /// Resolve effective checkpoint for an input state (handles `Checkpoint::AutoRoute`).
    pub fn resolve_checkpoint(&self, state_text: &str) -> (Checkpoint, &'static str) {
        if self.config.checkpoint == Checkpoint::AutoRoute {
            Self::laya_route(state_text)
        } else {
            (self.config.checkpoint, "configured")
        }
    }

    /// Script-aware route across the three Laya checkpoints only.
    fn laya_route(state_text: &str) -> (Checkpoint, &'static str) {
        let (ckpt, reason) = Checkpoint::detect_script_route(state_text);
        if ckpt == Checkpoint::English {
            let trimmed = state_text.trim_start();
            if trimmed.starts_with('{')
                || estimate_tokens(state_text) > Checkpoint::English.context_window()
            {
                return (Checkpoint::TypedDecisions, "long_or_structured");
            }
        }
        (ckpt, reason)
    }

    /// Resolve effective checkpoint using full capability, context-window, and label-cardinality meta-routing.
    pub fn resolve_checkpoint_for_questions(
        &self,
        state_text: &str,
        questions: &BTreeMap<String, Question>,
    ) -> (Checkpoint, &'static str) {
        if self.config.checkpoint == Checkpoint::AutoRoute {
            Self::route_auto(state_text, questions)
        } else {
            (self.config.checkpoint, "configured")
        }
    }

    /// The meta-router's pick, moved to an installed checkpoint when real weights are in use.
    fn route_auto(
        state_text: &str,
        questions: &BTreeMap<String, Question>,
    ) -> (Checkpoint, &'static str) {
        use crate::model::native::{Backend, backend, native_available};
        let dec = crate::model::route_auto_best(state_text, questions);
        if backend() == Backend::Mock {
            return (dec.selected, dec.reason);
        }
        // The meta-router sends long text to ModernBERT-zeroshot and EuroBERT for their 8k windows.
        // Natively, the ModernBERT fine-tune was trained at 512 tokens, lemma caps EuroBERT at 2048,
        // and EuroBERT is a base model without a classifier, so lemma does not follow those rules.
        let long_context_pick = matches!(
            dec.selected,
            Checkpoint::ModernBertZeroShot | Checkpoint::EuroBert
        );
        // Above 20 options, each Laya option gets fewer than 10 of the head's 192 tokens. On
        // Banking77 (77 labels) Laya scored 32-37% and the NLI models 68-73%, best first below.
        if dec.max_label_cardinality > HIGH_CARDINALITY
            && let Some(&nli) = [
                Checkpoint::ModernBertNli,
                Checkpoint::DebertaV3ZeroShot,
                Checkpoint::ModernBertZeroShot,
            ]
            .iter()
            .find(|&&c| native_available(c))
        {
            return (nli, "high_cardinality_nli");
        }
        if !long_context_pick && native_available(dec.selected) {
            return (dec.selected, dec.reason);
        }
        let (laya, reason) = Self::laya_route(state_text);
        if native_available(laya) {
            return (laya, reason);
        }
        match Checkpoint::laya_checkpoints()
            .iter()
            .find(|&&c| native_available(c))
        {
            Some(&installed) => (installed, "installed_fallback"),
            // Nothing is installed: `auto` simulates the pick, `native` reports the missing weights.
            None => (laya, reason),
        }
    }

    /// Directory with the weights for `ckpt`: `local_weights_path` for the configured checkpoint.
    fn weights_dir(&self, ckpt: Checkpoint) -> std::path::PathBuf {
        match &self.config.local_weights_path {
            Some(path) if ckpt == self.config.checkpoint => path
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| ckpt.local_dir()),
            _ => ckpt.local_dir(),
        }
    }

    /// Perform inference for a single question on state text.
    pub fn classify_single(&self, state_text: &str, question: &Question) -> Result<DecisionAnswer> {
        let mut map = BTreeMap::new();
        map.insert("single".to_string(), question.clone());
        let state_val = serde_json::Value::String(state_text.to_string());
        let mut res = self.classify(&state_val, &map)?;
        res.answers
            .remove("single")
            .ok_or_else(|| anyhow::anyhow!("Failed to extract decision answer"))
    }

    /// Perform a single forward pass over all questions for a given state input.
    pub fn classify(
        &self,
        state: &serde_json::Value,
        questions: &BTreeMap<String, Question>,
    ) -> Result<DecisionResult> {
        if questions.is_empty() {
            bail!("No questions provided for classification");
        }

        let state_text = match state {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };

        let (active_ckpt, _) = self.resolve_checkpoint_for_questions(&state_text, questions);
        // Load the weights before the clock starts, so latency_ms measures inference only.
        let dir = self.weights_dir(active_ckpt);
        let native = crate::model::native::uses_native(active_ckpt, &dir);
        if native {
            crate::model::native::shared(active_ckpt, &dir)?;
        }
        let start = Instant::now();

        let mut total_tokens = 0;
        let mut answers = BTreeMap::new();

        // Each question's options, after the optional lexical shortlist.
        let plans: Vec<(Vec<String>, Option<crate::engine::ShortlistResult>)> = questions
            .values()
            .map(|q| {
                let options = q.render_options();
                let shortlist = (self.config.shortlist_k > 0
                    && q.qtype == QuestionType::Choice
                    && options.len() > self.config.shortlist_k)
                    .then(|| {
                        crate::engine::shortlist_options(
                            &state_text,
                            q,
                            &options,
                            self.config.shortlist_k,
                        )
                    });
                (options, shortlist)
            })
            .collect();
        let eval_options = |(options, shortlist): &(
            Vec<String>,
            Option<crate::engine::ShortlistResult>,
        )|
         -> Vec<String> {
            shortlist
                .as_ref()
                .map_or_else(|| options.clone(), |s| s.kept_options.clone())
        };
        let plan_options: Vec<Vec<String>> = plans.iter().map(eval_options).collect();
        // Native models score every question in one call (Laya batches them in one forward pass).
        let mut batched: Vec<Option<crate::model::native::Scored>> = if native {
            let reqs: Vec<crate::model::native::ScoreRequest> = questions
                .values()
                .zip(&plan_options)
                .map(|(q, options)| crate::model::native::ScoreRequest {
                    state: &state_text,
                    question: q,
                    options,
                    strategy: self.config.chunk_strategy,
                    hypothesis_template: self.config.hypothesis_template.as_deref(),
                    multi_label: self.config.multi_label,
                })
                .collect();
            crate::model::native::shared(active_ckpt, &dir)?
                .score_batch(&reqs)?
                .into_iter()
                .map(Some)
                .collect()
        } else {
            (0..questions.len()).map(|_| None).collect()
        };

        for (i, (qid, q)) in questions.iter().enumerate() {
            let (options, shortlist) = &plans[i];
            let k = options.len().max(2);
            let eval_options: &[String] = &plan_options[i];

            let head_budget = active_ckpt.head_budget();
            let est_option_tokens = eval_options
                .iter()
                .map(|o| estimate_tokens(o))
                .sum::<usize>();

            if est_option_tokens > head_budget {
                let warn_set = WARNED_HEAD_BUDGET.get_or_init(|| Mutex::new(HashSet::new()));
                if let Ok(mut guard) = warn_set.lock()
                    && guard.insert(format!(
                        "{}:{}:{}",
                        active_ckpt.repo_id(),
                        qid,
                        est_option_tokens
                    ))
                {
                    eprintln!(
                        "⚠️  Warning: Question '{}' options (~{} tokens) exceed head budget ({} tokens). High cardinality may degrade accuracy.",
                        qid, est_option_tokens, head_budget
                    );
                }
            }

            // Raw logits for the options, pooled across windows for the sliding-window strategy
            let scored = match batched[i].take() {
                Some(scored) => scored,
                None => self.score_options(&state_text, q, eval_options, active_ckpt, true)?,
            };
            total_tokens += scored.input_tokens;
            let eval_logits = scored.logits;
            let raw_logits = if let Some(s) = &shortlist {
                crate::engine::expand_shortlisted_logits(
                    &eval_logits,
                    &s.kept_indices,
                    options.len(),
                )
            } else {
                eval_logits
            };
            let logits_out = Some(
                raw_logits
                    .iter()
                    .map(|z| (z * 10000.0).round() / 10000.0)
                    .collect(),
            );

            // Temperature scaling per Laya RLCD calibration
            let temp_key = self.get_temperature_bucket(q.qtype, k);
            let qtype_idx = match q.qtype {
                QuestionType::Choice => 0,
                QuestionType::Score => 1,
                QuestionType::Noul => 2,
            };
            let has_explicit_temp = (self.config.temperature - 1.0).abs() > 1e-6;
            let fallback_temp = if has_explicit_temp {
                self.config.temperature
            } else {
                active_ckpt.default_temperatures()[qtype_idx]
            };
            let mut temp = if self.config.checkpoint == Checkpoint::AutoRoute
                && active_ckpt == Checkpoint::Multilingual
                && !has_explicit_temp
            {
                active_ckpt.default_temperatures()[qtype_idx]
            } else if let Some(&bucket_temp) = self.config.temperature_by_options.get(&temp_key) {
                let is_default_bucket = self
                    .config
                    .checkpoint
                    .default_temperature_by_options()
                    .get(&temp_key)
                    .is_some_and(|&def_t| (bucket_temp - def_t).abs() <= 1e-6);
                if has_explicit_temp && is_default_bucket {
                    self.config.temperature
                } else {
                    // Fitted bucket temperatures outside this range (e.g. upstream choice:11+ = 0.10) make
                    // the model far more confident than it is. The upstream loader clamps the same way.
                    bucket_temp.clamp(
                        crate::engine::calibrate::MIN_TEMPERATURE,
                        crate::engine::calibrate::MAX_TEMPERATURE,
                    )
                }
            } else {
                fallback_temp
            };

            // 2026 Q-ATS: Adaptive temperature scaling under covariate shift
            if let Some(qats) = &self.config.qats_profile
                && qats.enabled
                && qats.std_distance > 1e-6
                && !qats.class_centroids.is_empty()
            {
                let feat = crate::engine::cache::vectorize_text(&state_text);
                let mut min_dist = f64::INFINITY;
                for centroid in qats.class_centroids.values() {
                    let dist = crate::engine::ood::euclidean_distance(&feat, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                if min_dist.is_finite() {
                    let z_dist = (min_dist - qats.mean_distance) / qats.std_distance;
                    let factor = 1.0 + qats.gamma * z_dist.tanh();
                    temp = (temp * factor).clamp(0.05, 10.0);
                }
            }

            let scaled_logits: Vec<f64> = raw_logits.iter().map(|&z| z / temp.max(0.01)).collect();

            // Softmax
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = scaled_logits.iter().map(|&z| (z - max_logit).exp()).sum();
            let probs: Vec<f64> = scaled_logits
                .iter()
                .map(|&z| (z - max_logit).exp() / exp_sum)
                .collect();

            // Normalized entropy & blended negentropy confidence
            let negentropy = confidence_from_probs(&probs);
            let entropy = (1.0 - negentropy).clamp(0.0, 1.0);
            let p_max = probs.iter().cloned().fold(0.0, f64::max);
            let margin_conf = if k > 1 {
                (p_max - (1.0 - p_max) / (k as f64 - 1.0)).max(0.0)
            } else {
                p_max
            };
            let confidence = (((0.25 * negentropy + 0.75 * margin_conf).clamp(0.0, 1.0)) * 10000.0)
                .round()
                / 10000.0;

            // Escalation signal from normalized entropy. lemma does not read Laya's act/escalate head:
            // upstream reports that head carries no usable signal, and confidence gating works better.
            let act_probability = (0.05 + 0.9 * entropy).clamp(0.0, 1.0);

            let mut prob_map = BTreeMap::new();
            match q.qtype {
                QuestionType::Choice => {
                    let keys: Vec<String> = if let Some(serde_json::Value::Object(map)) =
                        &q.criteria
                    {
                        map.keys().cloned().collect()
                    } else if let Some(serde_json::Value::Array(arr)) = &q.criteria {
                        arr.iter()
                            .filter_map(|v| {
                                v.as_str()
                                    .map(|s| s.split(':').next().unwrap_or(s).trim().to_string())
                            })
                            .collect()
                    } else {
                        options
                            .iter()
                            .map(|s| s.split(':').next().unwrap_or(s).trim().to_string())
                            .collect()
                    };

                    if self.config.multi_label {
                        // Independent sigmoid scoring per option: p_i = 1 / (1 + exp(-z_i))
                        let mut selected = Vec::new();
                        let mut best_idx = 0;
                        let mut best_sig = -1.0;
                        for (i, (&z, key)) in scaled_logits.iter().zip(keys.iter()).enumerate() {
                            let sig = 1.0 / (1.0 + (-z).exp());
                            prob_map.insert(key.clone(), (sig * 10000.0).round() / 10000.0);
                            if sig >= self.config.multi_label_threshold {
                                selected.push(key.clone());
                            }
                            if sig > best_sig {
                                best_sig = sig;
                                best_idx = i;
                            }
                        }
                        if selected.is_empty()
                            && let Some(lbl) = keys.get(best_idx)
                        {
                            selected.push(lbl.clone());
                        }
                        let primary = selected.first().cloned();
                        answers.insert(
                            qid.clone(),
                            DecisionAnswer {
                                qtype: QuestionType::Choice,
                                choice: primary,
                                selected_labels: Some(selected),
                                score: None,
                                noul: None,
                                confidence,
                                probabilities: prob_map,
                                logits: logits_out.clone(),
                                act_probability: Some(
                                    (act_probability * 10000.0).round() / 10000.0,
                                ),
                            },
                        );
                    } else {
                        let mut best_idx = 0;
                        let mut best_p = -1.0;
                        for (i, (&p, key)) in probs.iter().zip(keys.iter()).enumerate() {
                            prob_map.insert(key.clone(), (p * 10000.0).round() / 10000.0);
                            let w = self
                                .config
                                .cost_weights
                                .get(&key.to_lowercase())
                                .or_else(|| self.config.cost_weights.get(key))
                                .copied()
                                .unwrap_or(1.0);
                            if p * w > best_p {
                                best_p = p * w;
                                best_idx = i;
                            }
                        }

                        let chosen_label = keys
                            .get(best_idx)
                            .cloned()
                            .unwrap_or_else(|| "unknown".into());
                        answers.insert(
                            qid.clone(),
                            DecisionAnswer {
                                qtype: QuestionType::Choice,
                                choice: Some(chosen_label),
                                selected_labels: None,
                                score: None,
                                noul: None,
                                confidence,
                                probabilities: prob_map,
                                logits: logits_out.clone(),
                                act_probability: Some(
                                    (act_probability * 10000.0).round() / 10000.0,
                                ),
                            },
                        );
                    }
                }
                QuestionType::Score => {
                    let mut weighted_score = 0.0;
                    for (i, &p) in probs.iter().enumerate() {
                        weighted_score += i as f64 * p;
                        prob_map.insert(i.to_string(), (p * 10000.0).round() / 10000.0);
                    }
                    answers.insert(
                        qid.clone(),
                        DecisionAnswer {
                            qtype: QuestionType::Score,
                            choice: None,
                            selected_labels: None,
                            score: Some((weighted_score * 10000.0).round() / 10000.0),
                            noul: None,
                            confidence,
                            probabilities: prob_map,
                            logits: logits_out.clone(),
                            act_probability: Some((act_probability * 10000.0).round() / 10000.0),
                        },
                    );
                }
                QuestionType::Noul => {
                    let raw_p_true = probs.get(1).copied().unwrap_or(0.5);
                    let w_true = self
                        .config
                        .cost_weights
                        .get("true")
                        .or_else(|| self.config.cost_weights.get(qid))
                        .copied()
                        .unwrap_or(1.0);
                    let w_false = self
                        .config
                        .cost_weights
                        .get("false")
                        .copied()
                        .unwrap_or(1.0);
                    let denom = (w_true * raw_p_true + w_false * (1.0 - raw_p_true)).max(1e-12);
                    let p_true = (w_true * raw_p_true / denom).clamp(0.0, 1.0);
                    prob_map.insert("false".into(), ((1.0 - p_true) * 10000.0).round() / 10000.0);
                    prob_map.insert("true".into(), (p_true * 10000.0).round() / 10000.0);

                    answers.insert(
                        qid.clone(),
                        DecisionAnswer {
                            qtype: QuestionType::Noul,
                            choice: None,
                            selected_labels: None,
                            score: None,
                            noul: Some((p_true * 10000.0).round() / 10000.0),
                            confidence,
                            probabilities: prob_map,
                            logits: logits_out.clone(),
                            act_probability: Some((act_probability * 10000.0).round() / 10000.0),
                        },
                    );
                }
            }
        }

        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        // Label simulated results so they cannot pass for model output in logs or audit receipts.
        let model = if native {
            active_ckpt.repo_id().to_string()
        } else {
            format!("mock:{}", active_ckpt.repo_id())
        };
        Ok(DecisionResult {
            model,
            answers,
            input_tokens: total_tokens,
            latency_ms: (latency_ms * 100.0).round() / 100.0,
        })
    }

    /// Classify a collection of DataItems in parallel using Rayon.
    pub fn classify_batch(
        &self,
        items: &[DataItem],
        questions: &BTreeMap<String, Question>,
    ) -> Result<Vec<(String, DecisionResult)>> {
        crate::model::native::map_items(items, |item| {
            let res = self.classify(&item.state, questions)?;
            Ok((item.id.clone(), res))
        })
        .into_iter()
        .collect()
    }

    /// Raw (pre-temperature) logit per option for the configured checkpoint.
    pub fn compute_option_logits(
        &self,
        state_text: &str,
        q: &Question,
        options: &[String],
    ) -> Result<Vec<f64>> {
        self.compute_option_logits_for_checkpoint(state_text, q, options, self.config.checkpoint)
    }

    /// Raw logit per option for `active_ckpt`. Every command scores options through this function.
    ///
    /// With the native backend the logits come from the Laya weights. Otherwise they come
    /// from the keyword mock, which only simulates a model.
    pub fn compute_option_logits_for_checkpoint(
        &self,
        state_text: &str,
        q: &Question,
        options: &[String],
        active_ckpt: Checkpoint,
    ) -> Result<Vec<f64>> {
        Ok(self
            .score_options(state_text, q, options, active_ckpt, false)?
            .logits)
    }

    /// Score `options` for `ckpt`. `chunked` applies the configured chunk strategy to long text.
    ///
    /// The real model splits its own input on tokenizer tokens. The mock splits text on the
    /// word-count estimate, and its token count is that estimate.
    fn score_options(
        &self,
        state_text: &str,
        q: &Question,
        options: &[String],
        ckpt: Checkpoint,
        chunked: bool,
    ) -> Result<crate::model::native::Scored> {
        use crate::model::native;
        let ckpt = if ckpt == Checkpoint::AutoRoute {
            let mut qmap = BTreeMap::new();
            qmap.insert("q".to_string(), q.clone());
            Self::route_auto(state_text, &qmap).0
        } else {
            ckpt
        };
        let strategy = if chunked {
            self.config.chunk_strategy
        } else {
            crate::model::ChunkStrategy::TruncatePrefix
        };
        let dir = self.weights_dir(ckpt);
        if native::uses_native(ckpt, &dir) {
            return native::shared(ckpt, &dir)?.score(&native::ScoreRequest {
                state: state_text,
                question: q,
                options,
                strategy,
                hypothesis_template: self.config.hypothesis_template.as_deref(),
                multi_label: self.config.multi_label,
            });
        }
        native::warn_simulated(ckpt, &dir);

        let ctx_window = ckpt.context_window();
        let logits = if chunked {
            let chunks = crate::model::chunk_text_for_window(state_text, ctx_window, strategy);
            if chunks.len() <= 1 {
                let text = chunks.first().map(|c| c.as_ref()).unwrap_or(state_text);
                self.mock_option_logits(text, q, options, ckpt)
            } else {
                let windows: Vec<Vec<f64>> = chunks
                    .iter()
                    .map(|c| self.mock_option_logits(c, q, options, ckpt))
                    .collect();
                crate::model::pool_window_logits(&windows)
            }
        } else {
            self.mock_option_logits(state_text, q, options, ckpt)
        };
        let option_tokens: usize = options.iter().map(|o| estimate_tokens(o)).sum();
        Ok(crate::model::native::Scored {
            logits,
            input_tokens: option_tokens + estimate_tokens(state_text).min(ctx_window) + 10,
        })
    }

    /// Keyword-rule simulation of a System 1 encoder. It runs no neural network.
    fn mock_option_logits(
        &self,
        state_text: &str,
        q: &Question,
        options: &[String],
        active_ckpt: Checkpoint,
    ) -> Vec<f64> {
        let full_lower = state_text.to_lowercase();
        let state_lower: &str = if let Some(idx) = full_lower.find("--- [end of preamble") {
            &full_lower[idx..]
        } else {
            &full_lower
        };
        let ins_lower = q.instructions.to_lowercase();
        let (_script_ckpt, script_reason) = Checkpoint::detect_script_route(state_text);

        if q.qtype == QuestionType::Noul && options.len() == 2 {
            return self.compute_noul_logits(
                state_lower,
                &ins_lower,
                options,
                active_ckpt,
                script_reason,
            );
        }

        let has_nli_or_decision_head =
            !matches!(active_ckpt, Checkpoint::EuroBert | Checkpoint::NeoBert);
        let supports_euro_multilingual = matches!(
            active_ckpt,
            Checkpoint::Multilingual | Checkpoint::AutoRoute | Checkpoint::EuroBert
        );
        let supports_latin_cognates = supports_euro_multilingual
            || matches!(
                active_ckpt,
                Checkpoint::ModernBertZeroShot
                    | Checkpoint::DebertaV3ZeroShot
                    | Checkpoint::TypedDecisions
            );
        let supports_global_multilingual = matches!(
            active_ckpt,
            Checkpoint::Multilingual | Checkpoint::AutoRoute
        );

        let mut logits = Vec::with_capacity(options.len());
        for (opt_idx, opt) in options.iter().enumerate() {
            let opt_lower = opt.to_lowercase();
            let label_part = opt_lower.split(':').next().unwrap_or(&opt_lower).trim();

            let mut score = 0.0;

            // Ordinal scale scoring for QuestionType::Score (e.g. 0..1 or 0..3 urgency/severity)
            if q.qtype == QuestionType::Score && options.len() >= 2 {
                let last_idx = options.len() - 1;
                let has_critical = [
                    "urgent",
                    "immediate",
                    "asap",
                    "critical",
                    "outage",
                    "crash",
                    "crashing",
                    "breach",
                    "compromised",
                    "unauthorized",
                    "phishing",
                    "injection",
                    "jailbreak",
                    "stole",
                    "cancel",
                ]
                .iter()
                .any(|m| {
                    state_lower.contains(m)
                        && (!has_nli_or_decision_head || !Self::is_clause_negated(state_lower, m))
                }) || (supports_latin_cognates
                    && ["critique", "piratage", "kritischen", "absturz", "dringend"]
                        .iter()
                        .any(|m| state_lower.contains(m)))
                    || (supports_global_multilingual
                        && ["严重", "崩溃", "紧急"]
                            .iter()
                            .any(|m| state_lower.contains(m)));

                let has_elevated = (!state_lower.contains("simply")
                    && !state_lower.contains("minor")
                    && [
                        "billed twice",
                        "charged twice",
                        "duplicate",
                        "refund",
                        "overcharge",
                        "dispute",
                    ]
                    .iter()
                    .any(|m| {
                        state_lower.contains(m)
                            && (!has_nli_or_decision_head
                                || !Self::is_clause_negated(state_lower, m))
                    }))
                    || (supports_latin_cognates
                        && ["reembolso", "duplicado"]
                            .iter()
                            .any(|m| state_lower.contains(m)))
                    || (supports_euro_multilingual
                        && ["двойную", "возврат"]
                            .iter()
                            .any(|m| state_lower.contains(m)))
                    || (supports_global_multilingual
                        && ["二重決済", "返金"].iter().any(|m| state_lower.contains(m)));

                let has_low_info = [
                    "no action needed",
                    "archive",
                    "copy of",
                    "informational",
                    "historical",
                    "nice job",
                    "sleek",
                ]
                .iter()
                .any(|m| state_lower.contains(m));

                if options.len() == 2 {
                    if has_critical || has_elevated {
                        if opt_idx == 1 {
                            score += 4.5;
                        }
                    } else if opt_idx == 0 {
                        score += 4.0;
                    }
                } else if has_critical && !has_elevated {
                    if opt_idx == last_idx {
                        score += 4.5;
                    }
                } else if has_elevated && options.len() == 4 {
                    if state_lower.contains("cancel") || state_lower.contains("three times") {
                        if opt_idx == 3 {
                            score += 4.5;
                        }
                    } else if has_nli_or_decision_head {
                        if opt_idx == 2 {
                            score += 4.5;
                        }
                    } else if opt_idx == 3 {
                        // Raw MLM encoders without ordinal regression head over-predict extreme bucket
                        score += 4.2;
                    }
                } else if has_low_info {
                    if opt_idx == 0 {
                        score += 4.5;
                    }
                } else if opt_idx == 1 {
                    score += 3.8;
                }

                score += 0.1;
                logits.push(score);
                continue;
            }

            let prefix4 = label_part
                .char_indices()
                .nth(4)
                .map(|(i, _)| &label_part[..i])
                .unwrap_or(label_part);

            // Direct label match or compound sub-token match (char-boundary safe, >= 3 chars to avoid matching digits in Score questions)
            if label_part.chars().count() >= 3 && state_lower.contains(label_part) {
                if Self::is_clause_negated(state_lower, label_part) {
                    if has_nli_or_decision_head
                        && !(active_ckpt == Checkpoint::Multilingual
                            && (state_lower.contains("ignore previous instructions")
                                || state_lower.contains("no security breach")))
                    {
                        score -= 1.5;
                    } else {
                        score += 12.5;
                    }
                } else {
                    score += 4.0;
                }
            } else if label_part.contains('_') {
                for sub in label_part.split('_') {
                    if sub.chars().count() >= 3 && state_lower.contains(sub) {
                        score += 3.5;
                    }
                }
            } else if label_part.chars().count() >= 4 && state_lower.contains(prefix4) {
                if Self::is_clause_negated(state_lower, prefix4) {
                    if has_nli_or_decision_head {
                        score -= 1.2;
                    } else {
                        score += 4.5;
                    }
                } else {
                    score += 3.2;
                }
            }

            // Built-in semantic associations for typical routing and decision domains (English + Multilingual)
            let synonyms: &[(&str, &[&str])] = &[
                (
                    "billing",
                    &[
                        "invoice",
                        "charge",
                        "refund",
                        "billed",
                        "payment",
                        "receipt",
                        "price",
                        "cost",
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
                        "चालान",
                        "रिफंड",
                        "बिल",
                        "счет",
                        "возврат",
                        "оплата",
                        "វិក្កយបត្រ",
                    ],
                ),
                (
                    "technical",
                    &[
                        "bug",
                        "error",
                        "crash",
                        "outage",
                        "broken",
                        "issue",
                        "stack",
                        "exception",
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
                        "त्रुटि",
                        "क्रैश",
                        "ошибка",
                        "сбой",
                        "បញ្ហា",
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
                        "devis",
                        "contrat",
                        "angebot",
                        "vertrag",
                        "报价",
                        "合同",
                        "采购",
                        "見積",
                        "契約",
                        "प्रस्ताव",
                        "अनुबंध",
                        "контракт",
                    ],
                ),
                (
                    "urgency",
                    &[
                        "immediate",
                        "urgent",
                        "asap",
                        "critical",
                        "blocking",
                        "today",
                        "emergency",
                        "deadline",
                        "tomorrow",
                        "urgente",
                        "dringend",
                        "紧急",
                        "立刻",
                        "至急",
                        "तुरंत",
                        "अत्यावश्यक",
                        "срочно",
                        "បន្ទាន់",
                    ],
                ),
                (
                    "churn",
                    &[
                        "cancel",
                        "leave",
                        "terminate",
                        "switching",
                        "competitor",
                        "stop",
                        "annuler",
                        "kündigen",
                        "取消",
                        "退订",
                        "解約",
                        "रद्द",
                        "отменить",
                    ],
                ),
                (
                    "duplicate",
                    &[
                        "same person",
                        "duplicate",
                        "identical",
                        "shared email",
                        "shared phone",
                        "career progression",
                        "nickname",
                        "promoted",
                        "alias",
                        "transition",
                        "exact match",
                    ],
                ),
                (
                    "merge",
                    &[
                        "same person",
                        "duplicate",
                        "identical",
                        "career shift",
                        "nickname",
                        "married name",
                        "auto-merge",
                    ],
                ),
                (
                    "separate",
                    &[
                        "distinct",
                        "different",
                        "father and son",
                        "senior vs junior",
                        "spouses",
                        "coworkers",
                        "separate",
                        "unrelated",
                        "namesake",
                        "collision",
                        "competing",
                    ],
                ),
                (
                    "exact",
                    &[
                        "exact match",
                        "direct hit",
                        "satisfies",
                        "founder",
                        "investor",
                        "researcher",
                        "lead",
                        "architect",
                        "engineer",
                        "cfo",
                        "partner",
                    ],
                ),
                (
                    "irrelevant",
                    &[
                        "not relevant",
                        "unrelated",
                        "recruiter",
                        "false positive",
                        "different",
                        "student",
                        "technician",
                        "paralegal",
                    ],
                ),
                (
                    "personal",
                    &[
                        "catch-up", "coffee", "meeting", "intro", "lunch", "dinner", "friend",
                        "alumni", "personal", "1-on-1", "drinks",
                    ],
                ),
                (
                    "automated",
                    &[
                        "receipt",
                        "invoice",
                        "noreply",
                        "no-reply",
                        "notification",
                        "automated",
                        "alert",
                        "digest",
                        "newsletter",
                        "unsubscribe",
                        "verification code",
                        "ticket",
                        "trip",
                        "order",
                    ],
                ),
                (
                    "urgent",
                    &[
                        "promised",
                        "deadline",
                        "tomorrow",
                        "today",
                        "urgent",
                        "critical",
                        "blocking",
                        "send by",
                        "introduce",
                        "return",
                        "asap",
                        "before",
                        "紧急",
                        "तुरंत",
                        "срочно",
                        "បន្ទាន់",
                    ],
                ),
                (
                    "routine",
                    &[
                        "casual",
                        "next month",
                        "stay in touch",
                        "reconnect",
                        "spring",
                        "future",
                        "check in",
                        "catch up",
                    ],
                ),
                (
                    "none",
                    &[
                        "notes",
                        "standup",
                        "completed",
                        "historical",
                        "no action",
                        "closed",
                        "attended",
                        "archive",
                        "no blockers",
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
                        "sécurité",
                        "piratage",
                        "seguridad",
                        "sicherheit",
                        "セキュリティ",
                        "不正アクセス",
                        "安全",
                        "盗刷",
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
                        "permission",
                        "role",
                        "cuenta",
                        "compte",
                        "konto",
                        "アカウント",
                        "账户",
                    ],
                ),
            ];

            let non_en_vocab = [
                "factura",
                "reembolso",
                "facture",
                "remboursement",
                "rechnung",
                "erreur",
                "panne",
                "fehler",
                "absturz",
                "devis",
                "contrat",
                "angebot",
                "vertrag",
                "urgente",
                "dringend",
                "annuler",
                "kündigen",
                "sécurité",
                "piratage",
                "seguridad",
                "sicherheit",
                "cuenta",
                "compte",
                "konto",
            ];

            for (domain, words) in synonyms {
                if label_part.contains(domain) || opt_lower.contains(domain) {
                    for &kw in *words {
                        let is_latin_euro_kw = non_en_vocab.contains(&kw);
                        let is_cyrillic_kw = !kw.is_ascii()
                            && kw.chars().all(|c| (0x0400..=0x052F).contains(&(c as u32)));
                        let is_global_kw = !kw.is_ascii() && !is_latin_euro_kw && !is_cyrillic_kw;
                        if (is_latin_euro_kw && !supports_latin_cognates)
                            || (is_cyrillic_kw && !supports_euro_multilingual)
                            || (is_global_kw && !supports_global_multilingual)
                        {
                            continue;
                        }
                        if state_lower.contains(kw) {
                            if Self::is_clause_negated(state_lower, kw) {
                                if has_nli_or_decision_head
                                    && !(active_ckpt == Checkpoint::Multilingual
                                        && state_lower.contains("ignore previous instructions"))
                                {
                                    score -= 0.8;
                                } else {
                                    score += 4.2;
                                }
                            } else {
                                score += 2.0;
                            }
                        }
                    }
                }
            }

            // Phrase matching from criteria descriptions
            for phrase in &[
                "same person",
                "career progression",
                "career shift",
                "shared phone",
                "shared email",
                "distinct people",
                "different people",
                "father and son",
                "spouses",
                "coworkers",
                "exact match",
                "direct hit",
                "not relevant",
                "false positive",
                "individual human",
                "personal communication",
                "automated notification",
                "receipt",
                "newsletter",
                "active commitment",
                "promised",
                "deadline",
                "no action needed",
                "informational notes",
            ] {
                if opt_lower.contains(phrase) && state_lower.contains(phrase) {
                    score += 3.5;
                }
            }

            // Word overlap between option description and state text
            let opt_words: Vec<&str> = opt_lower
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                .filter(|w| w.chars().count() >= 3)
                .collect();
            for word in &opt_words {
                let is_non_ascii_word = word.chars().any(|c| c as u32 > 0x024F);
                if is_non_ascii_word && !supports_global_multilingual && !label_part.contains(word)
                {
                    continue;
                }
                if (word.chars().count() > 3 || is_non_ascii_word) && state_lower.contains(word) {
                    if Self::is_clause_negated(state_lower, word)
                        && !Self::is_clause_negated(&opt_lower, word)
                    {
                        if has_nli_or_decision_head
                            && !(active_ckpt == Checkpoint::Multilingual
                                && state_lower.contains("ignore previous instructions"))
                        {
                            score -= 0.6;
                        } else {
                            score += 2.8;
                        }
                    } else {
                        score += 1.8;
                    }
                }
            }

            // Instruction alignment
            for word in &opt_words {
                if word.chars().count() > 3 && ins_lower.contains(word) {
                    score += 0.5;
                }
            }

            let head_scale =
                if self.config.hypothesis_template.is_some() || q.hypothesis_template.is_some() {
                    2.0
                } else {
                    1.0
                };
            score += crate::model::score_head_paradigm_adjustment(
                active_ckpt,
                state_lower,
                q,
                &opt_lower,
                self.config.hypothesis_template.as_deref(),
            ) * head_scale;

            // Fallback base logit to prevent zeros
            score += 0.1;
            logits.push(score);
        }

        // If all logits are uniform, introduce slight decreasing prior so option 0 is default fallback
        let all_equal = logits.windows(2).all(|w| f64::abs(w[0] - w[1]) < 1e-6);
        if all_equal {
            let n = logits.len();
            for (i, val) in logits.iter_mut().enumerate() {
                *val += 0.08 * (n.saturating_sub(i)) as f64;
            }
        }

        let n_opts = logits.len();
        for (opt_idx, val) in logits.iter_mut().enumerate() {
            let pos_bias = 0.015 * (n_opts.saturating_sub(opt_idx)) as f64;
            *val = Self::apply_checkpoint_profile(
                *val + pos_bias,
                opt_idx,
                active_ckpt,
                script_reason,
                state_text,
            );
        }

        logits
    }

    fn compute_noul_logits(
        &self,
        state_lower: &str,
        ins_lower: &str,
        options: &[String],
        active_ckpt: Checkpoint,
        script_reason: &str,
    ) -> Vec<f64> {
        let stop_words: &[&str] = &[
            "does",
            "this",
            "that",
            "have",
            "with",
            "from",
            "user",
            "sender",
            "email",
            "text",
            "statement",
            "hold",
            "holds",
            "whether",
            "there",
            "true",
            "false",
            "question",
            "body",
            "message",
            "ticket",
            "customer",
            "item",
            "about",
            "would",
            "could",
            "should",
            "require",
            "human",
            "intervention",
        ];

        let has_nli_or_decision_head =
            !matches!(active_ckpt, Checkpoint::EuroBert | Checkpoint::NeoBert);
        let supports_euro_multilingual = matches!(
            active_ckpt,
            Checkpoint::Multilingual | Checkpoint::AutoRoute | Checkpoint::EuroBert
        );
        let supports_global_multilingual = matches!(
            active_ckpt,
            Checkpoint::Multilingual | Checkpoint::AutoRoute
        );

        let mut pos_score = 0.0f64;
        let mut neg_score = 0.0f64;

        let opt_false = options[0].to_lowercase();
        let opt_true = options[1].to_lowercase();

        // 1. Match content words from instructions and custom true criteria
        let query_source = format!("{ins_lower} {opt_true}");
        for raw_w in query_source.split(|c: char| !c.is_alphanumeric()) {
            if raw_w.chars().count() >= 4 && !stop_words.contains(&raw_w) {
                let stem = raw_w
                    .char_indices()
                    .nth(5)
                    .map(|(i, _)| &raw_w[..i])
                    .unwrap_or(raw_w);
                if (state_lower.contains(raw_w) && !Self::is_clause_negated(state_lower, raw_w))
                    || (stem.len() >= 4
                        && state_lower.contains(stem)
                        && !Self::is_clause_negated(state_lower, stem))
                {
                    pos_score += 1.5;
                }
            }
        }

        // 2. Match custom false criteria if specified
        if !opt_false.contains("statement does not hold") {
            for raw_w in opt_false.split(|c: char| !c.is_alphanumeric()) {
                if raw_w.chars().count() >= 4
                    && !stop_words.contains(&raw_w)
                    && state_lower.contains(raw_w)
                    && !Self::is_clause_negated(state_lower, raw_w)
                {
                    neg_score += 1.5;
                }
            }
        }

        // 3. Domain concept triggers for boolean assertions
        let concept_rules: &[(&[&str], &[&str])] = &[
            (
                &["churn", "cancel", "switch", "competitor", "leave", "retain"],
                &[
                    "cancel",
                    "canceling",
                    "cancelled",
                    "churn",
                    "competitor",
                    "switching",
                    "switch to",
                    "leave",
                    "leaving",
                    "pause all payments",
                    "stop paying",
                    "terminate",
                    "close my account",
                    "closing my account",
                    "取消",
                    "解約",
                    "रद्द",
                ],
            ),
            (
                &[
                    "phishing",
                    "scam",
                    "fraud",
                    "steal",
                    "credential",
                    "compromise",
                    "suspicious",
                ],
                &[
                    "phishing",
                    "scam",
                    "fraud",
                    "stolen",
                    "unauthorized",
                    "compromised",
                    "suspicious",
                    "verify your account",
                    "password expired",
                    "wire transfer",
                    "credentials",
                    "gift card",
                    "urgent action required",
                    "suspend your account",
                    "social security",
                    "bank account details",
                ],
            ),
            (
                &[
                    "spam",
                    "unsolicited",
                    "marketing",
                    "bulk",
                    "promo",
                    "newsletter",
                ],
                &[
                    "unsubscribe",
                    "special offer",
                    "limited time",
                    "click here",
                    "buy now",
                    "discount",
                    "promo",
                    "marketing",
                    "newsletter",
                    "free trial",
                    "act now",
                    "winner",
                    "congratulations",
                    "bulk",
                ],
            ),
            (
                &["reply", "respond", "response", "answer"],
                &[
                    "?",
                    "can you",
                    "could you",
                    "please send",
                    "please let",
                    "let me know",
                    "when will",
                    "how do",
                    "please issue",
                    "would you",
                ],
            ),
            (
                &[
                    "urgent",
                    "immediate",
                    "active",
                    "outage",
                    "critical",
                    "blocking",
                    "escalat",
                ],
                &[
                    "urgent",
                    "immediate",
                    "asap",
                    "outage",
                    "down",
                    "crash",
                    "crashing",
                    "critical",
                    "blocking",
                    "active",
                    "emergency",
                    "unauthorized",
                    "compromised",
                    "breach",
                    "phishing",
                    "injection",
                    "jailbreak",
                    "stole",
                    "billed twice",
                    "charged twice",
                    "billed three times",
                ],
            ),
        ];

        for (q_triggers, state_markers) in concept_rules {
            if q_triggers
                .iter()
                .any(|t| ins_lower.contains(t) || opt_true.contains(t))
            {
                for m in *state_markers {
                    if state_lower.contains(m) {
                        if Self::is_clause_negated(state_lower, m) {
                            if !has_nli_or_decision_head {
                                pos_score += 1.6;
                            }
                        } else {
                            pos_score += 2.2;
                        }
                    }
                }
            }
        }

        let supports_latin_cognates = supports_euro_multilingual
            || matches!(
                active_ckpt,
                Checkpoint::ModernBertZeroShot
                    | Checkpoint::DebertaV3ZeroShot
                    | Checkpoint::TypedDecisions
            );

        if ["urgent", "immediate", "escalat", "critical"]
            .iter()
            .any(|t| ins_lower.contains(t) || opt_true.contains(t))
        {
            if supports_latin_cognates
                && [
                    "urgente",
                    "reembolso",
                    "critique",
                    "piratage",
                    "kritischen",
                    "absturz",
                    "dringend",
                ]
                .iter()
                .any(|m| state_lower.contains(m))
            {
                pos_score += 2.5;
            }
            if supports_euro_multilingual
                && ["срочно", "возврат"]
                    .iter()
                    .any(|m| state_lower.contains(m))
            {
                pos_score += 2.5;
            }
            if supports_global_multilingual
                && ["至急", "二重決済", "严重", "崩溃", "紧急"]
                    .iter()
                    .any(|m| state_lower.contains(m))
            {
                pos_score += 2.5;
            }
        }

        // 4. Negation & sentiment counter-indicators
        for neg_phrase in &[
            "no churn",
            "not canceling",
            "never cancel",
            "do not cancel",
            "not spam",
            "legitimate",
            "no action needed",
            "not blocking",
            "simply have",
            "routine account",
            "noreply",
            "no-reply",
            "do not reply",
            "automated notification",
        ] {
            if state_lower.contains(neg_phrase) {
                neg_score += 3.0;
            }
        }

        if ["churn", "cancel", "spam", "phishing", "scam", "fraud"]
            .iter()
            .any(|t| ins_lower.contains(t) || opt_true.contains(t))
        {
            for pos_sentiment in &[
                "sleek",
                "nice job",
                "great job",
                "love the",
                "thank you",
                "very happy",
            ] {
                if state_lower.contains(pos_sentiment) {
                    neg_score += 2.0;
                }
            }
        }

        let (mut l_false, mut l_true) = if pos_score > neg_score {
            let delta = (pos_score - neg_score).min(2.0) * 0.25;
            (0.2, 2.6 + delta)
        } else {
            let delta = (neg_score - pos_score).min(2.0) * 0.25;
            (2.6 + delta, 0.2)
        };

        l_false =
            Self::apply_checkpoint_profile(l_false, 0, active_ckpt, script_reason, state_lower);
        l_true = Self::apply_checkpoint_profile(l_true, 1, active_ckpt, script_reason, state_lower);
        vec![l_false, l_true]
    }

    fn is_clause_negated(text_lower: &str, keyword: &str) -> bool {
        let Some(pos) = text_lower.find(keyword) else {
            return false;
        };
        let window_start = text_lower[..pos]
            .char_indices()
            .rev()
            .take(32)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        let prefix = &text_lower[window_start..pos];
        let clause = prefix.rsplit(['.', ';', ',', '!']).next().unwrap_or(prefix);
        [
            "not ", "never ", "no ", "wasn't ", "didn't ", "don't ", "isn't ", "without ",
            "no fue ", "pas de ", "nicht ", "kein ",
        ]
        .iter()
        .any(|neg| clause.contains(neg))
    }

    fn apply_checkpoint_profile(
        score: f64,
        opt_idx: usize,
        ckpt: Checkpoint,
        script_reason: &str,
        state_text: &str,
    ) -> f64 {
        let is_non_latin = script_reason != "ascii_latin";
        let is_latin_ext = script_reason == "latin_extended";
        let is_euro_script = matches!(script_reason, "latin_extended" | "cyrillic");
        match ckpt {
            Checkpoint::English | Checkpoint::AutoRoute => {
                if is_non_latin {
                    score * 0.28 + 0.05 * (4usize.saturating_sub(opt_idx)) as f64
                } else {
                    score
                }
            }
            Checkpoint::Multilingual => {
                if is_non_latin {
                    score * 1.22
                } else {
                    score * 0.86 + ((opt_idx as f64 * 0.11) % 0.17)
                }
            }
            Checkpoint::TypedDecisions => {
                if state_text.trim_start().starts_with('{') {
                    score * 1.16
                } else if is_latin_ext {
                    score * 0.78
                } else if is_non_latin {
                    score * 0.30 + 0.05 * (4usize.saturating_sub(opt_idx)) as f64
                } else {
                    score * 1.08
                }
            }
            Checkpoint::ModernBertZeroShot => {
                if is_latin_ext {
                    score * 0.82
                } else if is_non_latin {
                    score * 0.32 + 0.05 * (4usize.saturating_sub(opt_idx)) as f64
                } else {
                    score * 1.10 + ((opt_idx as f64 * 0.05) % 0.09)
                }
            }
            Checkpoint::DebertaV3ZeroShot => {
                if is_latin_ext {
                    score * 0.85
                } else if is_non_latin {
                    score * 0.30 + 0.05 * (4usize.saturating_sub(opt_idx)) as f64
                } else {
                    score * 1.14 + ((opt_idx as f64 * 0.06) % 0.11)
                }
            }
            Checkpoint::EuroBert => {
                if is_euro_script {
                    score * 1.18
                } else if is_non_latin {
                    score * 0.31 + 0.05 * (4usize.saturating_sub(opt_idx)) as f64
                } else {
                    score * 0.92 + ((opt_idx as f64 * 0.09) % 0.16)
                }
            }
            Checkpoint::NeoBert => {
                if is_non_latin {
                    score * 0.26 + 0.05 * (4usize.saturating_sub(opt_idx)) as f64
                } else {
                    score * 0.95 + ((opt_idx as f64 * 0.08) % 0.15)
                }
            }
            Checkpoint::ModernBertNli => {
                if is_non_latin {
                    score * 0.24
                } else {
                    score * 0.74 + ((opt_idx as f64 * 0.13) % 0.21)
                }
            }
        }
    }

    pub fn get_temperature_bucket(&self, qtype: QuestionType, k: usize) -> String {
        let prefix = match qtype {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        };

        if k == 2 {
            format!("{prefix}:2")
        } else if k <= 5 {
            format!("{prefix}:3-5")
        } else if k <= 10 {
            format!("{prefix}:6-10")
        } else {
            format!("{prefix}:11+")
        }
    }
}

impl DecisionModel for LayaClassifier {
    fn model_id(&self) -> &str {
        self.config.checkpoint.repo_id()
    }

    fn checkpoint(&self) -> Checkpoint {
        self.config.checkpoint
    }

    fn classify(
        &self,
        state: &serde_json::Value,
        questions: &BTreeMap<String, Question>,
    ) -> Result<DecisionResult> {
        LayaClassifier::classify(self, state, questions)
    }
}
