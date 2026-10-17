use anyhow::Result;
use comfy_table::{Cell, Color};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, BufRead, Write};
use std::str::FromStr;

use crate::datasource::DataItem;
use crate::engine::classifier::{LayaClassifier, confidence_from_probs};
use crate::engine::conformal::ConformalEngine;
use crate::engine::consensus::ConsensusEngine;
use crate::engine::learn_visuals::{
    c_bold, c_cyan, c_green, c_magenta, c_yellow, colorize_ascii_diagram, colorize_math_block,
    render_callout_card, render_phase_breadcrumb, render_phase_visual_demonstration, styled_table,
};
use crate::engine::shortlist::shortlist_options;
use crate::model::{
    Checkpoint, ChunkStrategy, Question, chunk_text_for_window, estimate_tokens, head_paradigm_for,
    render_mlm_verbalizer_prompt, render_nli_hypothesis, route_auto_best,
};

/// Five-phase Spiral & CRA (Concrete-Representational-Abstract) learning curriculum
/// for mastering System 1 encoder architectures, zero-shot heads, calibration, and production ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurriculumPhase {
    Phase1Foundations,
    Phase2HeadParadigms,
    Phase3ScalingContextAndTaxonomies,
    Phase4CalibrationAndUncertainty,
    Phase5ProductionCascadingAndOps,
}

impl CurriculumPhase {
    pub fn all() -> [CurriculumPhase; 5] {
        [
            Self::Phase1Foundations,
            Self::Phase2HeadParadigms,
            Self::Phase3ScalingContextAndTaxonomies,
            Self::Phase4CalibrationAndUncertainty,
            Self::Phase5ProductionCascadingAndOps,
        ]
    }

    pub fn number(&self) -> u8 {
        match self {
            Self::Phase1Foundations => 1,
            Self::Phase2HeadParadigms => 2,
            Self::Phase3ScalingContextAndTaxonomies => 3,
            Self::Phase4CalibrationAndUncertainty => 4,
            Self::Phase5ProductionCascadingAndOps => 5,
        }
    }

    pub fn slug(&self) -> &'static str {
        match self {
            Self::Phase1Foundations => "foundations",
            Self::Phase2HeadParadigms => "paradigms",
            Self::Phase3ScalingContextAndTaxonomies => "scaling",
            Self::Phase4CalibrationAndUncertainty => "calibration",
            Self::Phase5ProductionCascadingAndOps => "production",
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Self::Phase1Foundations => {
                "Foundations — System 1 vs. System 2 & Bidirectional Encoders"
            }
            Self::Phase2HeadParadigms => "The 3 Zero-Shot Head Paradigms & Typed Decisions",
            Self::Phase3ScalingContextAndTaxonomies => {
                "Scaling Context Windows, Languages & High-Cardinality Taxonomies"
            }
            Self::Phase4CalibrationAndUncertainty => {
                "Trustworthy Uncertainty, Temperature Calibration & Conformal Guarantees"
            }
            Self::Phase5ProductionCascadingAndOps => {
                "Production Architecture — Cascading, Consensus, Slicing & Active Loops"
            }
        }
    }

    pub fn subtitle(&self) -> &'static str {
        match self {
            Self::Phase1Foundations => {
                "Why non-autoregressive bidirectional attention beats causal LLMs for sub-40ms routing & triage"
            }
            Self::Phase2HeadParadigms => {
                "How JointSinglePass O(1), Pairwise NLI Cross-Encoders O(K), and MLM Verbalizers extract decisions"
            }
            Self::Phase3ScalingContextAndTaxonomies => {
                "Overcoming the 512-token context wall, 50-label cardinality bottleneck, and multilingual script drift"
            }
            Self::Phase4CalibrationAndUncertainty => {
                "Turning raw uncalibrated logits into true probabilities with Q-ATS, Free-Energy OOD, and Conformal Sets"
            }
            Self::Phase5ProductionCascadingAndOps => {
                "Combining System 1 speed with System 2 reasoning via Pareto cascades, JSD consensus, and active distillation"
            }
        }
    }

    pub fn key_takeaways(&self) -> &'static [&'static str] {
        match self {
            Self::Phase1Foundations => &[
                "Bidirectional encoders see full left+right context simultaneously in a single forward pass (<40ms).",
                "2025/2026 architectures (ModernBERT, mmBERT, DeBERTa-v3, EuroBERT, NeoBERT) add RoPE, SwiGLU, FlashAttention, and 8k context.",
                "System 1 handles high-throughput structured classification at ~250x lower cost than autoregressive System 2 LLMs.",
            ],
            Self::Phase2HeadParadigms => &[
                "JointSinglePass (Laya/GLiClass) scores all K labels in O(1) passes and supports Choice, ordinal Score E[y], and binary Noul.",
                "PairwiseNliCrossEncoder (DeBERTa-v3-ZS, ModernBERT-ZS) runs O(K) premise-hypothesis passes and depends heavily on --hypothesis-template.",
                "GenerativeMlmVerbalizer (EuroBERT, NeoBERT) predicts label tokens at [MASK] and requires contrast-aware framing for negation.",
            ],
            Self::Phase3ScalingContextAndTaxonomies => &[
                "Naive prefix truncation drops late-document incidents ('Lost in the Tail'); --chunk-strategy head-tail or sliding-window preserves tail signals.",
                "--shortlist-k prunes high-cardinality taxonomies (K > 10) in O(1) via bi-encoder cosine similarity before cross-encoder scoring.",
                "--model auto-best routes each item in <0.5ms using Unicode script, token length, label count K, and typed-head requirements.",
            ],
            Self::Phase4CalibrationAndUncertainty => &[
                "Raw softmax probabilities are often overconfident; normalized negentropy 1 - H(p)/ln(K) + top-2 margin yields a richer confidence signal.",
                "Temperature scaling (z_i / T) and Quantile-Adaptive Temperature Scaling (Q-ATS) align confidence with empirical accuracy (minimizing ECE).",
                "Split Conformal Prediction (lemma conformal --alpha 0.05) guarantees >=95% true-label coverage via adaptive prediction sets C(x).",
            ],
            Self::Phase5ProductionCascadingAndOps => &[
                "Pareto Cascading (lemma cascade) resolves 75-85% of traffic in System 1 (<40ms) and escalates only high-risk items to System 2.",
                "Multi-Model Consensus (lemma consensus) combines diverse paradigms and quantifies epistemic disagreement via Jensen-Shannon Divergence.",
                "Diagnostic slicing (--slice-by), drift monitoring (lemma drift), active learning (lemma active), and attestation (lemma attest) close the production loop.",
            ],
        }
    }
}

impl fmt::Display for CurriculumPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Phase {} ({}): {}",
            self.number(),
            self.slug(),
            self.title()
        )
    }
}

impl FromStr for CurriculumPhase {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let norm = s.trim().to_ascii_lowercase();
        match norm.as_str() {
            "1" | "phase1" | "phase-1" | "phase_1" | "foundations" | "foundation" | "encoders"
            | "encoder" | "system1" | "bidirectional" | "modernbert" => Ok(Self::Phase1Foundations),
            "2" | "phase2" | "phase-2" | "phase_2" | "paradigms" | "paradigm" | "heads"
            | "head" | "nli" | "mlm" | "typed" | "verbalizer" => Ok(Self::Phase2HeadParadigms),
            "3" | "phase3" | "phase-3" | "phase_3" | "routing" | "chunking" | "shortlist"
            | "scaling" | "context" | "taxonomy" | "taxonomies" | "auto-best" => {
                Ok(Self::Phase3ScalingContextAndTaxonomies)
            }
            "4" | "phase4" | "phase-4" | "phase_4" | "calibration" | "calibrate" | "conformal"
            | "uncertainty" | "ood" | "qats" | "temperature" | "ece" => {
                Ok(Self::Phase4CalibrationAndUncertainty)
            }
            "5" | "phase5" | "phase-5" | "phase_5" | "production" | "cascade" | "cascading"
            | "consensus" | "drift" | "active" | "ops" | "slicing" => {
                Ok(Self::Phase5ProductionCascadingAndOps)
            }
            other => Err(format!(
                "Unknown curriculum phase '{}'. Valid phases: 1 (foundations), 2 (paradigms), 3 (scaling), 4 (calibration), 5 (production)",
                other
            )),
        }
    }
}

/// Concrete worked-example output computed live on real examples during lesson generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveLabOutput {
    pub title: String,
    pub explanation: String,
    pub metrics_summary: Vec<(String, String)>,
    pub try_command: String,
}

/// Diagnostic multiple-choice question targeting a specific conceptual mental model or misconception.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub id: String,
    pub phase: u8,
    pub prompt: String,
    pub options: Vec<String>,
    pub correct_index: usize,
    pub explanation: String,
}

/// Curated paper, technical blog post, or practitioner guide mapped to a curriculum phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadingResource {
    pub phase: u8,
    pub category: String,
    pub title: String,
    pub author_venue: String,
    pub takeaway: String,
    pub url: String,
}

/// Complete Concrete-Representational-Abstract (CRA) lesson for a single curriculum phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseLesson {
    pub phase: u8,
    pub slug: String,
    pub title: String,
    pub subtitle: String,
    pub mental_model: String,
    pub ascii_diagram: String,
    pub core_concepts: Vec<(String, String)>,
    pub deep_intuitions: Vec<(String, String)>,
    pub under_the_hood_math: String,
    pub live_lab: LiveLabOutput,
    pub quiz: Vec<QuizQuestion>,
    pub further_reading: Vec<ReadingResource>,
    pub next_step_commands: Vec<String>,
}

/// Searchable glossary entry linking theoretical ML concepts to concrete `lemma` CLI flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlossaryEntry {
    pub term: String,
    pub category: String,
    pub definition: String,
    pub cli_flag_or_cmd: String,
}

/// End-to-end operational workflow recipe for common `lemma` tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRecipe {
    pub slug: String,
    pub title: String,
    pub goal: String,
    pub steps: Vec<(String, String)>,
    pub pro_tip: String,
}

/// Build a complete `PhaseLesson` including live runtime experiments computed via `LayaClassifier`.
pub fn build_phase_lesson(phase: CurriculumPhase) -> PhaseLesson {
    match phase {
        CurriculumPhase::Phase1Foundations => build_phase_1(),
        CurriculumPhase::Phase2HeadParadigms => build_phase_2(),
        CurriculumPhase::Phase3ScalingContextAndTaxonomies => build_phase_3(),
        CurriculumPhase::Phase4CalibrationAndUncertainty => build_phase_4(),
        CurriculumPhase::Phase5ProductionCascadingAndOps => build_phase_5(),
    }
}

fn format_prediction(ans: &crate::model::DecisionAnswer) -> String {
    if let Some(ref c) = ans.choice {
        c.clone()
    } else if let Some(s) = ans.score {
        format!("{s:.2}")
    } else if let Some(n) = ans.noul {
        format!("{} ({:.2})", if n >= 0.5 { "true" } else { "false" }, n)
    } else {
        "none".to_string()
    }
}

fn build_phase_1() -> PhaseLesson {
    let phase = CurriculumPhase::Phase1Foundations;

    // Live Lab: Run English, Multilingual, and TypedDecisions checkpoints on concrete prompts
    let clf_en = LayaClassifier::with_checkpoint(Checkpoint::English);
    let clf_multi = LayaClassifier::with_checkpoint(Checkpoint::Multilingual);
    let clf_typed = LayaClassifier::with_checkpoint(Checkpoint::TypedDecisions);

    let q_route = Question::new_choice(
        "Route incoming customer ticket to the owning department",
        vec![
            "billing".to_string(),
            "technical".to_string(),
            "security".to_string(),
            "sales".to_string(),
        ],
    );
    let en_text =
        "We did not have a technical outage; our April enterprise invoice was charged twice";
    let ans_en = clf_en
        .classify_single(en_text, &q_route)
        .expect("Phase 1 English classification");

    let explain_rep = crate::engine::explain::AttributionEngine::new(&clf_en)
        .explain(en_text, "route", &q_route, "billing", "technical")
        .expect("Phase 1 Contrastive Attribution");
    let top_pos = explain_rep
        .attributions
        .iter()
        .max_by(|a, b| a.delta_margin.total_cmp(&b.delta_margin))
        .map(|a| format!("'{}' ({:+.3})", a.token, a.delta_margin))
        .unwrap_or_else(|| "none".to_string());
    let suppressed_neg = explain_rep
        .attributions
        .iter()
        .min_by(|a, b| a.delta_margin.total_cmp(&b.delta_margin))
        .map(|a| format!("'{}' ({:+.3})", a.token, a.delta_margin))
        .unwrap_or_else(|| "none".to_string());

    let ans_multi = clf_multi
        .classify_single(
            "緊急：本番環境のAPIゲートウェイでタイムアウトエラーが多発しています",
            &q_route,
        )
        .expect("Phase 1 Multilingual classification");

    let q_score = Question::new_score(
        "Rate production incident severity from 1 (minor) to 4 (critical outage)",
        vec![
            "1: Minor cosmetic issue".to_string(),
            "2: Degraded non-critical feature".to_string(),
            "3: Major service disruption".to_string(),
            "4: Critical multi-region outage".to_string(),
        ],
    );
    let q_noul = Question::new_noul("Does this incident require immediate on-call paging?");
    let mut typed_map = BTreeMap::new();
    typed_map.insert("severity".to_string(), q_score);
    typed_map.insert("page_oncall".to_string(), q_noul);
    let typed_res = clf_typed
        .classify(
            &serde_json::Value::String(
                "CRITICAL: Primary database cluster is unreachable across all availability zones; SLA breach imminent.".to_string(),
            ),
            &typed_map,
        )
        .expect("Phase 1 TypedDecisions classification");

    let sev_ans = &typed_res.answers["severity"];
    let page_ans = &typed_res.answers["page_oncall"];

    let live_lab = LiveLabOutput {
        title: "Live Lab 1: Sub-Millisecond Bidirectional Triage Across English, CJK & Typed Heads"
            .to_string(),
        explanation: "We just ran live forward passes across Laya's English (ModernBERT-large), Multilingual (mmBERT), and TypedDecisions checkpoints, plus leave-one-out contrastive token attribution. Notice how bidirectional encoding evaluates all candidate labels and typed questions in a single non-autoregressive pass.".to_string(),
        metrics_summary: vec![
            (
                "English (convaiinnovations/laya)".to_string(),
                format!(
                    "pred={} | conf={:.1}% | window={}t",
                    format_prediction(&ans_en),
                    ans_en.confidence * 100.0,
                    Checkpoint::English.context_window()
                ),
            ),
            (
                "Contrastive Attribution Top Driver".to_string(),
                format!(
                    "top_billing={} | suppressed_negation={} (margin={:+.2})",
                    top_pos, suppressed_neg, explain_rep.baseline_margin
                ),
            ),
            (
                "Multilingual CJK (laya-multilingual)".to_string(),
                format!(
                    "pred={} | conf={:.1}% | window={}t",
                    format_prediction(&ans_multi),
                    ans_multi.confidence * 100.0,
                    Checkpoint::Multilingual.context_window()
                ),
            ),
            (
                "Typed Score E[y] (laya-typed-decisions)".to_string(),
                format!(
                    "pred={} | conf={:.1}% | latency={:.2}ms",
                    format_prediction(sev_ans),
                    sev_ans.confidence * 100.0,
                    typed_res.latency_ms
                ),
            ),
            (
                "Typed Noul Assertion (page_oncall)".to_string(),
                format!(
                    "pred={} | conf={:.1}%",
                    format_prediction(page_ans),
                    page_ans.confidence * 100.0
                ),
            ),
        ],
        try_command: "lemma explain \"Our April invoice was charged twice\" --labels \"billing,technical,security\"".to_string(),
    };

    PhaseLesson {
        phase: phase.number(),
        slug: phase.slug().to_string(),
        title: phase.title().to_string(),
        subtitle: phase.subtitle().to_string(),
        mental_model: concat!(
            "Daniel Kahneman's dual-process theory divides cognition into System 1 (fast, automatic, pattern-recognizing) ",
            "and System 2 (slow, deliberate, step-by-step reasoning).\n\n",
            "In modern AI infrastructure, causal autoregressive LLMs (GPT-4o, Claude 3.7, Llama-3-70B) are System 2 engines: ",
            "they generate tokens one-by-one from left to right (500-3000ms, ~$0.005/call). Using a 70B causal LLM just to ",
            "classify an email into 4 folders or check a guardrail is like hiring a Supreme Court justice to sort postal mail.\n\n",
            "System 1 Decision Models (Laya, ModernBERT, DeBERTa-v3, mmBERT, EuroBERT, NeoBERT) are compact (210M-435M parameter) ",
            "non-autoregressive BIDIRECTIONAL encoders. Because every token attends to both past AND future tokens simultaneously ",
            "in a single forward pass, they resolve classification, ordinal scoring, and policy routing in <40ms at ~250x lower cost."
        )
        .to_string(),
        ascii_diagram: concat!(
            "  CAUSAL DECODER (System 2 LLM: O(N) sequential token generation)\n",
            "  [Token 1] <-- [Token 2] <-- [Token 3] <-- [Token 4] --> Generates: \"b\", \"i\", \"l\", \"l\", \"i\", \"n\", \"g\"\n",
            "  (Token 1 is BLIND to Token 4! Requires KV-cache loop & 500-2000ms latency)\n\n",
            "  BIDIRECTIONAL ENCODER (System 1 Laya / ModernBERT: 1 single parallel pass)\n",
            "  [Token 1: \"Not\"] <===> [Token 2: \"outage\"] <===> [Token 3: \"invoice\"] <===> [Token 4: \"charged\"]\n",
            "         \\                      |                         |                      /\n",
            "          +====================> [ Joint Decision Head ] <======================+\n",
            "                                        |\n",
            "                         P(billing)=0.94, P(technical)=0.03  (<40ms total!)"
        )
        .to_string(),
        core_concepts: vec![
            (
                "Bidirectional Full Attention".to_string(),
                "Unlike causal masks (M_ij = -inf for j > i), encoder self-attention lets early negation or qualifiers ('not a bug, refund needed') directly contextualize every token in layer 1..L.".to_string(),
            ),
            (
                "ModernBERT (2025/2026 Breakthrough)".to_string(),
                "Replaces legacy 2018 BERT's absolute 512-token embeddings with Rotary Position Embeddings (RoPE), GeGLU/SwiGLU layers, alternating local/global FlashAttention, and sequence unpadding up to 8,192 tokens.".to_string(),
            ),
            (
                "DeBERTa-v3 Disentangled Attention".to_string(),
                "Separates token content vectors (C) and relative position vectors (P), computing Content-to-Content, Content-to-Position, and Position-to-Content attention scores—ideal for compositional syntax & negation.".to_string(),
            ),
            (
                "mmBERT & EuroBERT Multilingual Encoders".to_string(),
                "mmBERT-base (322M) and EuroBERT-210m bring non-English scripts (CJK, Cyrillic, Arabic, Devanagari, European languages) into native System 1 latent space without English translation latency.".to_string(),
            ),
        ],
        deep_intuitions: vec![
            (
                "Representation Geometry & Anisotropy (Ethayarajh, 2019)".to_string(),
                "Why don't we just cosine-compare raw [CLS] embeddings? Ethayarajh (2019) proved that raw transformer hidden states occupy a narrow 'anisotropic cone' in R^d where even completely unrelated sentences have cosine similarity > 0.85. Laya's token-level `mask_scorer` and `act_head` linear projections overcome anisotropy by projecting each `<<OPTION>>` token out of the cone onto orthogonal decision hyperplanes.".to_string(),
            ),
            (
                "ModernBERT's Alternating Local/Global Attention & Unpadding".to_string(),
                "Full O(T^2) self-attention across 8,192 tokens would blow up latency. ModernBERT (Answer.AI & LightOn) alternates 128-token local sliding-window attention in 2/3 of layers with full global attention every 3rd layer, and 'unpads' batches into a single contiguous FlashAttention stream—eliminating wasted FLOPs on [PAD] tokens and achieving 2–3x higher throughput.".to_string(),
            ),
        ],
        under_the_hood_math: concat!(
            "1. Bidirectional Self-Attention (No Causal Mask):\n",
            "   Attn(Q, K, V) = softmax( (Q * K^T + B_rel) / sqrt(d_k) ) * V\n",
            "   where all positions i, j in {1..T} attend to each other simultaneously.\n\n",
            "2. DeBERTa-v3 Disentangled Attention Score for token i attending to token j:\n",
            "   A_{i,j} = C_i * C_j^T  +  C_i * P_{delta(i,j)}^T  +  P_{delta(j,i)} * C_j^T\n",
            "   (Explicitly binds relative distance delta(i,j) to semantic content!)"
        )
        .to_string(),
        live_lab,
        quiz: build_phase_1_quiz(),
        further_reading: build_reading_list(Some(1)),
        next_step_commands: vec![
            "lemma info --inventory".to_string(),
            "lemma explain \"We did not have a technical outage; our invoice was double charged\" --labels \"billing,technical,security\"".to_string(),
            "lemma learn --phase 2".to_string(),
        ],
    }
}

fn build_phase_2() -> PhaseLesson {
    let phase = CurriculumPhase::Phase2HeadParadigms;

    let sample_text = "We did not have a technical outage; instead, our invoice was charged twice.";
    let question = Question::new_choice(
        "Identify the customer's actual support issue",
        vec![
            "billing: Invoice, charge, refund, or payment dispute".to_string(),
            "technical: API bug, outage, crash, or integration failure".to_string(),
            "security: Unauthorized access, MFA, or phishing alert".to_string(),
            "account: Profile settings, SSO login, or seat management".to_string(),
        ],
    );

    let clf_joint = LayaClassifier::with_checkpoint(Checkpoint::English);
    let clf_nli = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot);
    let clf_mlm = LayaClassifier::with_checkpoint(Checkpoint::EuroBert);

    let ans_joint = clf_joint
        .classify_single(sample_text, &question)
        .expect("Phase 2 JointSinglePass");
    let ans_nli = clf_nli
        .classify_single(sample_text, &question)
        .expect("Phase 2 PairwiseNliCrossEncoder");
    let ans_mlm = clf_mlm
        .classify_single(sample_text, &question)
        .expect("Phase 2 GenerativeMlmVerbalizer");

    // Minimal-pair counterfactual flip (Distill.pub / Gardner et al. Contrast Sets)
    let cf_tech = clf_joint
        .classify_single("API outage and crash; not a billing charge", &question)
        .expect("Phase 2 Counterfactual A");
    let cf_bill = clf_joint
        .classify_single("API is fine; billing invoice charged twice", &question)
        .expect("Phase 2 Counterfactual B");

    let nli_hyp = render_nli_hypothesis(&question, "billing: Invoice, charge, refund", None);
    let mlm_prompt = render_mlm_verbalizer_prompt(sample_text, &question);

    let live_lab = LiveLabOutput {
        title: "Live Lab 2: Comparing the 3 Zero-Shot Head Paradigms on Adversarial Negation"
            .to_string(),
        explanation: format!(
            "Input: \"{}\"\nWe ran this exact contrastive negation prompt through all 3 head paradigms: JointSinglePass (laya-en), PairwiseNliCrossEncoder (deberta-v3-zs), and GenerativeMlmVerbalizer (eurobert-210m), plus a minimal-pair counterfactual flip.",
            sample_text
        ),
        metrics_summary: vec![
            (
                format!(
                    "1. {} ({})",
                    Checkpoint::English.short_name(),
                    head_paradigm_for(Checkpoint::English).complexity_notation()
                ),
                format!(
                    "pred={} | P(billing)={:.3} | conf={:.1}%",
                    format_prediction(&ans_joint),
                    ans_joint.probabilities.get("billing").copied().unwrap_or(0.0),
                    ans_joint.confidence * 100.0
                ),
            ),
            (
                format!(
                    "2. {} ({})",
                    Checkpoint::DebertaV3ZeroShot.short_name(),
                    head_paradigm_for(Checkpoint::DebertaV3ZeroShot).complexity_notation()
                ),
                format!(
                    "pred={} | P(billing)={:.3} | conf={:.1}%",
                    format_prediction(&ans_nli),
                    ans_nli.probabilities.get("billing").copied().unwrap_or(0.0),
                    ans_nli.confidence * 100.0
                ),
            ),
            (
                format!(
                    "3. {} ({})",
                    Checkpoint::EuroBert.short_name(),
                    head_paradigm_for(Checkpoint::EuroBert).complexity_notation()
                ),
                format!(
                    "pred={} | P(billing)={:.3} | conf={:.1}%",
                    format_prediction(&ans_mlm),
                    ans_mlm.probabilities.get("billing").copied().unwrap_or(0.0),
                    ans_mlm.confidence * 100.0
                ),
            ),
            (
                "Minimal-Pair Counterfactual Flip".to_string(),
                format!(
                    "\"API timed out/outage; not billing\" -> {} ({:.0}%) | \"API is fine; billing invoice charged twice\" -> {} ({:.0}%)",
                    format_prediction(&cf_tech),
                    cf_tech.confidence * 100.0,
                    format_prediction(&cf_bill),
                    cf_bill.confidence * 100.0
                ),
            ),
            (
                "Rendered NLI Hypothesis".to_string(),
                format!("\"{}\"", nli_hyp),
            ),
            (
                "Rendered MLM Cloze Prompt".to_string(),
                format!(
                    "\"{}\"",
                    mlm_prompt.chars().take(72).collect::<String>()
                ),
            ),
        ],
        try_command: "lemma compare --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --slice-by auto".to_string(),
    };

    PhaseLesson {
        phase: phase.number(),
        slug: phase.slug().to_string(),
        title: phase.title().to_string(),
        subtitle: phase.subtitle().to_string(),
        mental_model: concat!(
            "An encoder backbone turns text into contextual token embeddings H in R^{T x d}. ",
            "But how do we extract a zero-shot classification decision across K candidate labels without fine-tuning a new linear layer for every task?\n\n",
            "Modern System 1 models use THREE distinct zero-shot head paradigms—each with radically different scaling laws and failure modes:\n",
            "1. JointSinglePass (Laya / Jev / GLiClass): Packs the state AND all K option tokens into one prompt and scores all K options simultaneously in O(1) forward passes using dedicated mask_scorer + act_head projections.\n",
            "2. PairwiseNliCrossEncoder (DeBERTa-v3-ZS / ModernBERT-ZS): Formats each option k as a Natural Language Inference (NLI) hypothesis ('This text is about {label}') and runs K separate forward passes (O(K) complexity) to score entailment vs contradiction.\n",
            "3. GenerativeMlmVerbalizer (EuroBERT / NeoBERT): Appends a cloze template ('The primary issue is [MASK].') and reads the pretrained language modeling head (lm_head) logits for verbalizer tokens at the [MASK] position."
        )
        .to_string(),
        ascii_diagram: concat!(
            "  PARADIGM 1: JointSinglePass (Laya / GLiClass) --> O(1) Pass for all K labels\n",
            "  [State Text] + [<<OPT_1>> billing] + [<<OPT_2>> technical] + [<<OPT_3>> security]\n",
            "       |---> Single Encoder Pass ---> mask_scorer (Choice/Score) + act_head (Noul)\n\n",
            "  PARADIGM 2: PairwiseNliCrossEncoder (DeBERTa-v3 / ModernBERT-ZS) --> O(K) Passes\n",
            "  Pass 1: [CLS] State [SEP] \"This text is about billing.\"   [SEP] -> [entailment vs contradiction]\n",
            "  Pass 2: [CLS] State [SEP] \"This text is about technical.\" [SEP] -> [entailment vs contradiction]\n\n",
            "  PARADIGM 3: GenerativeMlmVerbalizer (EuroBERT / NeoBERT) --> O(1) Cloze Pass\n",
            "  [State Text] + \"The primary category is [MASK].\" -> lm_head vocab logits at [MASK]"
        )
        .to_string(),
        core_concepts: vec![
            (
                "Typed Decisions (Choice, Score, Noul)".to_string(),
                "Laya's JointSinglePass head supports 3 mathematical output types: Choice (categorical softmax/sigmoid), Score (ordinal expected value E[y] = sum_i i * p_i for 1..K rubrics), and Noul (binary sufficiency/policy gate via act_head).".to_string(),
            ),
            (
                "Hypothesis Template Sensitivity (NLI)".to_string(),
                "Pairwise NLI models judge whether the input ENTAILS a hypothesis sentence. Changing --hypothesis-template from 'This text is about {label}' to 'The customer needs help with {label}: {description}' can swing zero-shot accuracy by 8-15%.".to_string(),
            ),
            (
                "Pre-Contrast Negation Traps in MLM Verbalizers".to_string(),
                "Raw MLM encoders (EuroBERT, NeoBERT) predict which word fits at [MASK] based on lexical co-occurrence. Without contrast-aware verbalization ('The issue is [MASK], not the negated clause'), pre-contrast words like 'outage' in 'not an outage, billing error' can hijack [MASK].".to_string(),
            ),
        ],
        deep_intuitions: vec![
            (
                "Minimal-Pair Counterfactuals (Distill.pub Pattern)".to_string(),
                "Bag-of-words heuristics fail when two sentences contain the exact same domain nouns ('API', 'billing') with flipped discourse pivots. Testing minimal pairs ('API timed out; not a billing issue' vs. 'API is fine; billing invoice charged twice') verifies that bidirectional attention heads bind negation to the correct clause rather than memorizing surface keywords.".to_string(),
            ),
            (
                "Head Paradigm Failure-Mode Matrix".to_string(),
                "Use `JointSinglePass` (Laya/GLiClass) when K > 5, latency budgets are strict (<15ms O(1)), or you need typed ordinal `Score` E[y] and `Noul` gates. Use `PairwiseNliCrossEncoder` (DeBERTa-v3-ZS / ModernBERT-ZS) when K <= 5 and options require rich multi-clause logical hypotheses. Use `GenerativeMlmVerbalizer` (EuroBERT/NeoBERT) for domain/multilingual pretraining alignment paired with contrast-aware verbalizers.".to_string(),
            ),
        ],
        under_the_hood_math: concat!(
            "1. Ordinal Score Expected Value (QuestionType::Score):\n",
            "   E[y] = sum_{i=1}^{K} i * P(y = i | x),   where P(y = i | x) = softmax(z_i / T)\n",
            "   Unlike argmax classification, E[y] preserves continuous severity (e.g., 3.42 / 4.00).\n\n",
            "2. Pairwise NLI Entailment Margin (for candidate option k in 1..K):\n",
            "   s_k = logit_entailment(x, H_k) - logit_contradiction(x, H_k)\n",
            "   P(y = k | x) = exp(s_k / T) / sum_{j=1}^{K} exp(s_j / T)"
        )
        .to_string(),
        live_lab,
        quiz: build_phase_2_quiz(),
        further_reading: build_reading_list(Some(2)),
        next_step_commands: vec![
            "lemma classify --model deberta-v3 --hypothesis-template \"The customer is reporting a {label} issue ({description}).\" --text \"Invoice charged twice\" --labels \"billing,technical\"".to_string(),
            "lemma experiment --mode adversarial".to_string(),
            "lemma learn --phase 3".to_string(),
        ],
    }
}

fn build_phase_3() -> PhaseLesson {
    let phase = CurriculumPhase::Phase3ScalingContextAndTaxonomies;

    // 1. Long-context Lost-in-the-Tail demonstration (>600 tokens on a 512-token model)
    let mut long_log = "2026-10-16T00:00:01Z INFO healthcheck nominal; worker pool utilization 18%; routine cache telemetry heartbeat OK. ".repeat(55);
    long_log.push_str(
        " 2026-10-16T00:04:59Z CRITICAL SECURITY ALERT: Unauthorized root SSH key injection and MFA bypass detected on production auth-gateway-01!",
    );
    let total_tokens = estimate_tokens(&long_log);

    let q_sec = Question::new_choice(
        "Classify the log status",
        vec![
            "security: Unauthorized access, breach, or SSH/MFA compromise".to_string(),
            "technical: Routine telemetry, healthcheck, or performance log".to_string(),
            "billing: Invoice or subscription event".to_string(),
        ],
    );

    let clf_prefix = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot)
        .with_chunk_strategy(ChunkStrategy::TruncatePrefix);
    let clf_head_tail = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot)
        .with_chunk_strategy(ChunkStrategy::HeadTail);

    let ans_prefix = clf_prefix
        .classify_single(&long_log, &q_sec)
        .expect("Phase 3 TruncatePrefix");
    let ans_head_tail = clf_head_tail
        .classify_single(&long_log, &q_sec)
        .expect("Phase 3 HeadTail");

    let ht_chunks = chunk_text_for_window(&long_log, 512, ChunkStrategy::HeadTail);
    let ht_chunk_tokens = estimate_tokens(&ht_chunks[0]);

    // 2. High-cardinality shortlisting (K=10 -> shortlist_k=3)
    let high_card_q = Question::new_choice(
        "Route ticket across 10 enterprise queues",
        vec![
            "billing_refunds".to_string(),
            "billing_tax_vat".to_string(),
            "tech_api_gateway".to_string(),
            "tech_sdk_python".to_string(),
            "security_iam_sso".to_string(),
            "security_phishing".to_string(),
            "sales_enterprise".to_string(),
            "legal_gdpr_dpa".to_string(),
            "hr_payroll".to_string(),
            "facilities_office".to_string(),
        ],
    );
    let high_card_opts = high_card_q.render_options();
    let sl_res = shortlist_options(
        "We need a refund for an overcharge and duplicate VAT on our invoice",
        &high_card_q,
        &high_card_opts,
        3,
    );

    // 3. Meta-router (`route_auto_best`) across heterogeneous inputs
    let mut q_map = BTreeMap::new();
    q_map.insert("dept".to_string(), q_sec.clone());
    let route_short = route_auto_best("Simple English billing refund request", &q_map);
    let route_cjk = route_auto_best("請求書の二重課金について返金をお願いします", &q_map);
    let route_long = route_auto_best(&long_log, &q_map);

    // 4. 0/1 Knapsack Schema Compression (`SchemaKnapsackOptimizer::optimize`)
    let mut verbose_map = BTreeMap::new();
    verbose_map.insert(
        "dept".to_string(),
        Question {
            qtype: crate::model::QuestionType::Choice,
            instructions: "Classify the log status".to_string(),
            criteria: Some(serde_json::json!({
                "security": "Unauthorized root access, SSH key compromise, MFA bypass, or credential phishing alert",
                "technical": "Routine telemetry heartbeat, API gateway timeout, SDK crash, or worker pool health log",
                "billing": "Duplicate invoice charge, VAT tax receipt, credit card refund, or subscription renewal"
            })),
            hypothesis_template: None,
        },
    );
    let knapsack_res =
        crate::engine::synthesis::SchemaKnapsackOptimizer::optimize(&verbose_map, 24);

    let live_lab = LiveLabOutput {
        title: "Live Lab 3: Tail-Incident Preservation, O(1) Label Shortlisting & Meta-Routing"
            .to_string(),
        explanation: format!(
            "We generated a {}-token server log where the first 90% is routine telemetry and a CRITICAL security breach occurs in the final 25 words, then tested 512-token DeBERTa-v3 with TruncatePrefix vs HeadTail, plus 0/1 Knapsack schema compression.",
            total_tokens
        ),
        metrics_summary: vec![
            (
                "Long Log (TruncatePrefix @ 512t)".to_string(),
                format!(
                    "pred={} | P(security)={:.3} (Tail incident truncated!)",
                    format_prediction(&ans_prefix),
                    ans_prefix.probabilities.get("security").copied().unwrap_or(0.0)
                ),
            ),
            (
                format!("Long Log (HeadTail @ {}t)", ht_chunk_tokens),
                format!(
                    "pred={} | P(security)={:.3} (25% head + 75% tail preserved!)",
                    format_prediction(&ans_head_tail),
                    ans_head_tail.probabilities.get("security").copied().unwrap_or(0.0)
                ),
            ),
            (
                "Bi-Encoder Shortlist (K=10 -> k=3)".to_string(),
                format!(
                    "kept={:?} (70% fewer NLI passes)",
                    sl_res.kept_options
                ),
            ),
            (
                "Knapsack Schema Compression".to_string(),
                format!(
                    "{}t -> {}t (budget={}t, pruned={} low-MI clauses)",
                    knapsack_res.original_tokens,
                    knapsack_res.optimized_tokens,
                    knapsack_res.target_budget,
                    knapsack_res.pruned_clauses
                ),
            ),
            (
                "auto-best Route (Short English)".to_string(),
                format!("{} ({})", route_short.selected.short_name(), route_short.reason),
            ),
            (
                "auto-best Route (Japanese CJK / Long 600t+)".to_string(),
                format!(
                    "{} ({}) / {} ({})",
                    route_cjk.selected.short_name(),
                    route_cjk.reason,
                    route_long.selected.short_name(),
                    route_long.reason
                ),
            ),
        ],
        try_command: "lemma classify --model auto-best --chunk-strategy head-tail --shortlist-k 4 --file data/customer_support.jsonl".to_string(),
    };

    PhaseLesson {
        phase: phase.number(),
        slug: phase.slug().to_string(),
        title: phase.title().to_string(),
        subtitle: phase.subtitle().to_string(),
        mental_model: concat!(
            "When moving System 1 encoders from toy benchmarks to real production streams, engineers hit THREE structural walls:\n\n",
            "1. The Context Window Wall (512 vs 8,192 tokens): Checkpoints like laya-en and deberta-v3-zs have a 512-token limit. ",
            "In customer email threads and stack traces, the actual failure or escalation request is almost always at the BOTTOM (or in the first sentence + final reply). ",
            "Naive prefix truncation ('take first 512 tokens') slices off the tail and causes catastrophic silent misses.\n\n",
            "2. The Label Cardinality Wall (K >> 10 options): Pairwise NLI cross-encoders run O(K) forward passes per item, while JointSinglePass models have a finite head token budget (192-512 tokens). ",
            "When K = 40 categories, scoring all 40 options wastes latency and overflows the head budget.\n\n",
            "3. The Heterogeneous Stream Wall: Real traffic mixes short English queries, 3,000-token logs, JSON payloads, and Japanese/Spanish/Arabic tickets. ",
            "No single static checkpoint is optimal for every slice—which is why capability-aware meta-routing (--model auto-best) is essential."
        )
        .to_string(),
        ascii_diagram: concat!(
            "  1. HEAD-TAIL CHUNKING (--chunk-strategy head-tail on 512t model)\n",
            "  [Head 25%: Subject & Opening] ... [Middle Routine Noise Dropped] ... [Tail 75%: Latest Reply / Stack Trace]\n\n",
            "  2. TWO-STAGE SHORTLISTING (--shortlist-k 5)\n",
            "  [K = 50 Labels] ===(Stage 1: O(1) Bi-Encoder Cosine + Synonym Filter)===> [Top k = 5 Labels]\n",
            "                                                                                   |\n",
            "                                              [Stage 2: Cross-Encoder / Joint Head Scores Top 5]\n\n",
            "  3. META-ROUTER (--model auto-best, <0.5ms dispatch)\n",
            "  Input + Schema ---> Script != Latin?      ---> laya-multilingual / eurobert-210m\n",
            "                 ---> Tokens > 512?         ---> modernbert-zs (8k) / laya-typed (1k)\n",
            "                 ---> K > 8 or Typed Head?  ---> laya-en (O(1) single-pass)\n",
            "                 ---> Short & K <= 5 Choice ---> deberta-v3-zs (disentangled NLI)"
        )
        .to_string(),
        core_concepts: vec![
            (
                "Context Chunking Strategies (--chunk-strategy)".to_string(),
                "Choose between truncate-prefix (default fast cut), head-tail (25% head + 75% tail in 1 pass—best for email threads & logs), and sliding-window (overlapping windows pooled via negentropy confidence weighting).".to_string(),
            ),
            (
                "Bi-Encoder Option Shortlisting (--shortlist-k)".to_string(),
                "Computes fast character/word n-gram & multilingual domain-synonym cosine similarity in O(1) to prune K options down to top-k before running heavy cross-encoder or joint scoring.".to_string(),
            ),
            (
                "0/1 Knapsack Schema Compression (lemma schema optimize)".to_string(),
                "When your JSON schema exceeds a checkpoint's head token budget (e.g., 192 tokens), lemma's dynamic programming optimizer prunes low-Mutual-Information filler words while preserving discriminative keywords.".to_string(),
            ),
            (
                "Coarse-to-Fine Hierarchical Trees (lemma route)".to_string(),
                "Splits a flat 50-way taxonomy into a root router (e.g., 5 departments) followed by specialized leaf schemas (10 sub-queues each), bounding complexity to O(log K).".to_string(),
            ),
        ],
        deep_intuitions: vec![
            (
                "U-Shaped Positional Attention & \"Lost in the Middle\" (Liu et al., 2024)".to_string(),
                "Liu et al. (TACL 2024) demonstrated that transformer attention exhibits a U-shaped recall curve: models attend strongly to primacy (opening context/subject line) and recency (final reply/stack trace), while middle tokens suffer severe attention dilution. `--chunk-strategy head-tail` exploits this U-curve by allocating 25% of the window to the document head and 75% to the tail.".to_string(),
            ),
            (
                "Information-Theoretic Knapsack Schema Compression".to_string(),
                "Verbose option descriptions overflow JointSinglePass head token budgets (192–256 tokens). `lemma schema optimize` scores each description clause by Mutual Information density I(w; Y) (non-stopword discriminative triggers) and solves a 0/1 Knapsack dynamic program to pack maximum semantic separability into tight token budgets.".to_string(),
            ),
        ],
        under_the_hood_math: concat!(
            "1. Confidence-Weighted Sliding-Window Logit Pooling:\n",
            "   For windows w in {1..W} with probability distribution p^{(w)}:\n",
            "   weight_w = max(0.05, 1 - H(p^{(w)}) / ln(K))\n",
            "   P_pooled(y = k) = (sum_w weight_w * p_k^{(w)}) / sum_w weight_w\n",
            "   (High-entropy noise windows automatically receive near-zero weight!)\n\n",
            "2. 0/1 Knapsack Schema Token Optimization:\n",
            "   max sum_{j} MI(token_j; Y)   subject to   sum_{j} length(token_j) <= B_head"
        )
        .to_string(),
        live_lab,
        quiz: build_phase_3_quiz(),
        further_reading: build_reading_list(Some(3)),
        next_step_commands: vec![
            "lemma schema optimize --schema schema/questions.json --budget 128".to_string(),
            "lemma compare --models laya,deberta-v3,modernbert-zeroshot,auto-best --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --slice-by auto".to_string(),
            "lemma learn --phase 4".to_string(),
        ],
    }
}

fn build_phase_4() -> PhaseLesson {
    let phase = CurriculumPhase::Phase4CalibrationAndUncertainty;

    // 1. Temperature scaling demonstration on an ambiguous multi-intent input
    let raw_logits = [2.10_f64, 1.45, 0.30, -0.40]; // e.g., billing vs account borderline
    let scale_probs = |temp: f64| -> Vec<f64> {
        let max_l = raw_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = raw_logits
            .iter()
            .map(|&z| ((z - max_l) / temp).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    };

    let p_sharp = scale_probs(0.5);
    let p_raw = scale_probs(1.0);
    let p_cal = scale_probs(1.8);

    let conf_sharp = confidence_from_probs(&p_sharp);
    let conf_raw = confidence_from_probs(&p_raw);
    let conf_cal = confidence_from_probs(&p_cal);

    // 2. Helmholtz Free-Energy Translation Invariance Proof (ID tie vs shifted OOD logits)
    let logits_id = [8.2_f64, 8.1, -2.0, -3.0];
    let logits_ood = [-9.8_f64, -9.9, -18.0, -19.0]; // shifted by c = -18 on top classes!
    let raw_helmholtz = |z: &[f64], t: f64| -> (f64, f64) {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = z.iter().map(|&v| ((v - m) / t).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let top_p = exps[0] / sum;
        let energy = -t * (m / t + sum.ln());
        (top_p, energy)
    };
    let (id_top_p, id_helmholtz) = raw_helmholtz(&logits_id, 1.0);
    let (ood_top_p, ood_helmholtz) = raw_helmholtz(&logits_ood, 1.0);
    // The production energy function on the same logits: it matches the inline computation.
    let fe_id = crate::engine::ood::compute_free_energy(&logits_id);
    let fe_ood = crate::engine::ood::compute_free_energy(&logits_ood);

    // 3. Live Split Conformal Prediction on a 6-item calibration set
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let q_conf = Question::new_choice(
        "Classify support department",
        vec![
            "billing".to_string(),
            "technical".to_string(),
            "security".to_string(),
            "account".to_string(),
        ],
    );
    let cal_samples = vec![
        (
            "cal_1",
            "Duplicate charge on our credit card invoice this month",
            "billing",
        ),
        (
            "cal_2",
            "Production API throws 502 Bad Gateway timeout on POST requests",
            "technical",
        ),
        (
            "cal_3",
            "Suspicious login from unknown IP bypassed MFA token",
            "security",
        ),
        (
            "cal_4",
            "Need to update SSO SAML certificate and user seat permissions",
            "account",
        ),
        (
            "cal_5",
            "Requesting a refund for overbilled enterprise seats on invoice",
            "billing",
        ),
        (
            "cal_6",
            "SDK crashes with segmentation fault when parsing webhook payload",
            "technical",
        ),
    ];
    let cal_items: Vec<DataItem> = cal_samples
        .into_iter()
        .map(|(id, text, gt)| {
            let mut gt_map = BTreeMap::new();
            gt_map.insert(
                "dept".to_string(),
                serde_json::Value::String(gt.to_string()),
            );
            DataItem::new(id, serde_json::Value::String(text.to_string()), "live_lab")
                .with_ground_truth(gt_map)
        })
        .collect();

    let (conf_report, _) = ConformalEngine::evaluate(&clf, &cal_items, "dept", &q_conf, 0.10)
        .expect("Phase 4 Conformal evaluation");

    let live_lab = LiveLabOutput {
        title: "Live Lab 4: Temperature Scaling (T=0.5 vs 1.0 vs 1.8) & Split Conformal Coverage"
            .to_string(),
        explanation: "First, we inspect how scaling borderline logits [2.10, 1.45, 0.30, -0.40] by Temperature T tames overconfidence. Second, we prove softmax translation invariance vs Helmholtz Free Energy on ID [8.2, 8.1, -2, -3] vs OOD [-9.8, -9.9, -18, -19]. Third, we run ConformalEngine::evaluate at alpha=0.10.".to_string(),
        metrics_summary: vec![
            (
                "Overconfident (T = 0.50)".to_string(),
                format!(
                    "top_prob={:.1}% | negentropy_conf={:.1}% (Too sharp!)",
                    p_sharp[0] * 100.0,
                    conf_sharp * 100.0
                ),
            ),
            (
                "Uncalibrated Raw (T = 1.00)".to_string(),
                format!(
                    "top_prob={:.1}% | negentropy_conf={:.1}%",
                    p_raw[0] * 100.0,
                    conf_raw * 100.0
                ),
            ),
            (
                "Calibrated (T = 1.80)".to_string(),
                format!(
                    "top_prob={:.1}% | negentropy_conf={:.1}% (Reflects true ambiguity)",
                    p_cal[0] * 100.0,
                    conf_cal * 100.0
                ),
            ),
            (
                "Free-Energy Translation Proof (ID vs OOD)".to_string(),
                format!(
                    "softmax_top1={:.1}% vs {:.1}% (identical!) | E(z)={:+.2} vs {:+.2} (lemma ood: {:+.2} vs {:+.2})",
                    id_top_p * 100.0,
                    ood_top_p * 100.0,
                    id_helmholtz,
                    ood_helmholtz,
                    fe_id,
                    fe_ood
                ),
            ),
            (
                "Conformal Quantile (alpha=0.10)".to_string(),
                format!(
                    "q_hat={:.4} | target_cov={:.0}% | empirical_cov={:.1}%",
                    conf_report.quantile_threshold,
                    conf_report.target_coverage_pct,
                    conf_report.empirical_coverage_pct
                ),
            ),
            (
                "Conformal Prediction Set Size".to_string(),
                format!(
                    "avg_set_size={:.2} labels | singleton_rate={:.1}%",
                    conf_report.average_set_size, conf_report.singleton_rate_pct
                ),
            ),
        ],
        try_command: "lemma conformal --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --alpha 0.05".to_string(),
    };

    PhaseLesson {
        phase: phase.number(),
        slug: phase.slug().to_string(),
        title: phase.title().to_string(),
        subtitle: phase.subtitle().to_string(),
        mental_model: concat!(
            "A raw neural network logit is NOT a calibrated probability! If a model predicts 'billing' with 0.95 softmax score, ",
            "is it actually right 95% of the time? In uncalibrated zero-shot NLI models, empirical accuracy at 0.95 softmax is often only 70-78%.\n\n",
            "To make System 1 decisions trustworthy enough for automated execution, lemma implements FOUR mathematical layers of uncertainty quantification:\n",
            "1. Blended Negentropy + Margin Confidence: Measures the entire probability distribution shape (Shannon entropy) AND top-2 separation, not just the top class probability.\n",
            "2. Temperature Scaling & Q-ATS (Quantile-Adaptive Temperature Scaling): Learns per-bucket temperatures T_k to minimize Expected Calibration Error (ECE), and dynamically inflates T(x) when an input lies far from in-distribution class centroids.\n",
            "3. Helmholtz Free-Energy OOD Detection: Distinguishes in-distribution ambiguity (two known classes tie) from Out-of-Distribution inputs (none of the classes match).\n",
            "4. Split Conformal Prediction: Outputs a variable-size prediction SET C(x) mathematically guaranteed to contain the true label with probability >= 1 - alpha."
        )
        .to_string(),
        ascii_diagram: concat!(
            "  RAW LOGITS z = [z_1, ..., z_K]\n",
            "       |\n",
            "       +---> [Free Energy E(x) = -T ln sum exp(z_i/T)] ---> E(x) > tau_ood? ===> 🚨 OOD REJECT\n",
            "       |\n",
            "       +---> [Q-ATS Temperature Scaling: p_i = softmax(z_i / T(x))] ---> Minimizes ECE & Brier\n",
            "       |\n",
            "       +---> [Split Conformal Set C(x) = {y : sum_{j<=y} p_(j) <= q_hat_{1-alpha}}]\n",
            "                |-- |C(x)| == 1 (Singleton, e.g. {billing})          ===> ✅ AUTO-EXECUTE\n",
            "                |-- |C(x)| >= 2 (Ambiguous, e.g. {billing, account}) ===> ⚠️ ESCALATE TO SYSTEM 2"
        )
        .to_string(),
        core_concepts: vec![
            (
                "Expected Calibration Error (ECE) & Brier Score".to_string(),
                "ECE partitions predictions into M confidence bins and measures the weighted gap |acc(B_m) - conf(B_m)|. Brier Score measures mean squared probability error 1/N sum (p_i - y_i)^2.".to_string(),
            ),
            (
                "Persisted Model Calibration (lemma calibrate --persist-to-model)".to_string(),
                "Tunes optimal temperatures and Q-ATS profiles on a validation dataset and writes them to <model_dir>/lemma_calibration.json, which LayaClassifier auto-loads on every future run.".to_string(),
            ),
            (
                "Free-Energy OOD vs Softmax Entropy".to_string(),
                "Softmax is translation-invariant: logits [-10, -10.1] and [+10, +9.9] produce the exact same softmax probabilities! Helmholtz free energy E(x) preserves logit magnitude to catch alien OOD inputs.".to_string(),
            ),
            (
                "Split Conformal Coverage Guarantee".to_string(),
                "For any exchangeable distribution, sorting non-conformity scores on N calibration items and setting q_hat at rank ceil((N+1)(1-alpha))/N guarantees P(Y_test in C(X_test)) >= 1 - alpha in finite samples.".to_string(),
            ),
        ],
        deep_intuitions: vec![
            (
                "Why Softmax Fails on OOD (Translation Invariance Proof)".to_string(),
                "Adding any constant shift c = -18 to all logits leaves softmax(z + c) = exp(z_i + c)/sum exp(z_j + c) = exp(c)exp(z_i)/[exp(c)sum exp(z_j)] = softmax(z) 100% unchanged! Thus an in-distribution tie [8.2, 8.1] and an alien OOD input [-9.8, -9.9] produce the exact same softmax probabilities. Helmholtz Free Energy E(x) = -T ln sum exp(z_i/T) retains the shift (-c) to expose OOD inputs.".to_string(),
            ),
            (
                "Adaptive Prediction Set Expansion (|C(x)| = 1 -> 2 -> K)".to_string(),
                "Instead of forcing a brittle top-1 guess on every item, Split Conformal Prediction (Angelopoulos & Bates, 2023) adapts set cardinality |C(x)| to instance difficulty: easy inputs yield a singleton |C(x)| = 1 (auto-execute), borderline inputs expand to |C(x)| = 2 (route to disambiguation), and OOD inputs expand to K—all while guaranteeing P(Y in C(X)) >= 1 - alpha.".to_string(),
            ),
        ],
        under_the_hood_math: concat!(
            "1. Normalized Negentropy + Margin Confidence:\n",
            "   H(p) = -sum_{i=1}^{K} p_i ln(p_i),    C_neg(p) = 1 - H(p) / ln(K)\n",
            "   C_blend(p) = 0.65 * C_neg(p) + 0.35 * (p_{(1)} - p_{(2)})\n\n",
            "2. Expected Calibration Error (ECE):\n",
            "   ECE = sum_{m=1}^{M} (|B_m| / N) * | acc(B_m) - conf(B_m) |\n\n",
            "3. Split Conformal Quantile (Adaptive Prediction Sets):\n",
            "   q_hat = Quantile( { s_1, ..., s_N },  ceil((N + 1)(1 - alpha)) / N )"
        )
        .to_string(),
        live_lab,
        quiz: build_phase_4_quiz(),
        further_reading: build_reading_list(Some(4)),
        next_step_commands: vec![
            "lemma calibrate --all-checkpoints --persist-to-model --adaptive --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json".to_string(),
            "lemma conformal --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --alpha 0.05".to_string(),
            "lemma ood fit --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
            "lemma learn --phase 5".to_string(),
        ],
    }
}

fn build_phase_5() -> PhaseLesson {
    let phase = CurriculumPhase::Phase5ProductionCascadingAndOps;

    // Live Lab: Run a 3-paradigm ConsensusEngine across clear vs ambiguous tickets
    let engine = ConsensusEngine::new(vec![
        Checkpoint::English,
        Checkpoint::DebertaV3ZeroShot,
        Checkpoint::ModernBertZeroShot,
    ])
    .with_threshold(0.12);

    let q_triage = Question::new_choice(
        "Route ticket to department",
        vec![
            "billing: Invoice, charge, or refund".to_string(),
            "technical: API outage, bug, or SDK crash".to_string(),
            "security: Unauthorized login, breach, or MFA".to_string(),
            "account: SSO settings or user permissions".to_string(),
        ],
    );
    let mut q_map = BTreeMap::new();
    q_map.insert("dept".to_string(), q_triage);

    let items = vec![
        DataItem::new(
            "clear_ticket",
            serde_json::Value::String(
                "Our credit card was billed twice for the April invoice; please issue a refund."
                    .to_string(),
            ),
            "live_lab",
        ),
        DataItem::new(
            "ambiguous_ticket",
            serde_json::Value::String(
                "After our SSO security rotation, the API billing webhook returned an auth error."
                    .to_string(),
            ),
            "live_lab",
        ),
    ];

    let results = engine
        .evaluate_batch(&items, &q_map)
        .expect("Phase 5 Consensus evaluation");
    let clear_q = &results[0].questions["dept"];
    let ambig_q = &results[1].questions["dept"];

    let live_lab = LiveLabOutput {
        title: "Live Lab 5: Multi-Paradigm Mixture-of-Decisions Consensus & Epistemic JSD"
            .to_string(),
        explanation: "We ran a 3-checkpoint ensemble (laya-en + deberta-v3-zs + modernbert-zs) on a clear billing ticket vs a cross-cutting SSO/API/billing ticket, decomposing uncertainty into Aleatoric (data noise) and Epistemic (inter-model Jensen-Shannon Divergence), alongside production cascade unit economics.".to_string(),
        metrics_summary: vec![
            (
                "Clear Ticket Consensus".to_string(),
                format!(
                    "pred={} | conf={:.1}% | status={}",
                    clear_q.consensus_prediction,
                    clear_q.consensus_confidence * 100.0,
                    clear_q.status.badge()
                ),
            ),
            (
                "Clear Ticket Uncertainty".to_string(),
                format!(
                    "epistemic_JSD={:.4} | aleatoric={:.4}",
                    clear_q.epistemic_uncertainty, clear_q.aleatoric_uncertainty
                ),
            ),
            (
                "Cross-Cutting Ticket Consensus".to_string(),
                format!(
                    "pred={} | conf={:.1}% | status={}",
                    ambig_q.consensus_prediction,
                    ambig_q.consensus_confidence * 100.0,
                    ambig_q.status.badge()
                ),
            ),
            (
                "Cross-Cutting Uncertainty".to_string(),
                format!(
                    "epistemic_JSD={:.4} | aleatoric={:.4} | votes={:?}",
                    ambig_q.epistemic_uncertainty,
                    ambig_q.aleatoric_uncertainty,
                    ambig_q.checkpoint_votes
                ),
            ),
            (
                "Cascade Unit Economics (80% Sys1 Retention)".to_string(),
                "$1,016 / 1M reqs vs $5,000 / 1M pure System 2 (79.7% cost savings)".to_string(),
            ),
        ],
        try_command: "lemma consensus --checkpoints english,deberta-v3,modernbert-zeroshot --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
    };

    PhaseLesson {
        phase: phase.number(),
        slug: phase.slug().to_string(),
        title: phase.title().to_string(),
        subtitle: phase.subtitle().to_string(),
        mental_model: concat!(
            "In production, System 1 and System 2 are not competitors—they form a symbiotic cognitive mesh:\n\n",
            "1. Pareto Cost/Quality Cascading (lemma cascade): System 1 processes 100% of incoming traffic in <40ms. ",
            "For the 75-85% of items where calibrated confidence is high and conformal sets are singletons, System 1 auto-executes immediately. ",
            "Only the 15-25% high-risk tail is escalated to a System 2 LLM—achieving System 2 accuracy at 5x lower latency and 80% lower cloud spend.\n\n",
            "2. Multi-Paradigm Consensus (lemma consensus): Combining models with different inductive biases (e.g. Laya JointSinglePass + DeBERTa-v3 NLI + ModernBERT-ZS) ",
            "lets us separate Aleatoric Uncertainty (the text itself is vague, so all models have flat probabilities) from Epistemic Uncertainty (models strongly disagree with each other, measured via Jensen-Shannon Divergence).\n\n",
            "3. Closed-Loop Flywheel (lemma slice-by -> drift -> active -> attest): Disaggregated slice matrices reveal where models fail; ",
            "PSI/JS drift alerts catch distribution shift; active learning distills hard System 2 teacher labels back into System 1 (RLCD); and SHA-256 hash-chained, HMAC-signed receipts provide tamper-evident compliance."
        )
        .to_string(),
        ascii_diagram: concat!(
            "  INCOMING STREAM (Kafka / HTTP / Lakehouse)\n",
            "        |\n",
            "        v\n",
            "  [Semantic Cache (<0.2ms)] --(Hit)--> Return Cached Decision\n",
            "        |\n",
            "        v (Miss)\n",
            "  [System 1 Router / Consensus (<40ms, $0.00002)]\n",
            "        |\n",
            "        +--- Risk <= tau* (75-85% of traffic) ---> [Guardrail & Merkle Attest] ---> ✅ AUTO-EXECUTE\n",
            "        |\n",
            "        +--- Risk >  tau* (15-25% hard tail)  ---> [System 2 LLM Escalation]   ---> 🔄 ACTIVE RLCD DISTILL"
        )
        .to_string(),
        core_concepts: vec![
            (
                "Aleatoric vs. Epistemic Uncertainty (JSD)".to_string(),
                "Aleatoric uncertainty is average within-model entropy E_m[H(p_m)] (inherently ambiguous input). Epistemic uncertainty is Jensen-Shannon Divergence H(sum w_m p_m) - sum w_m H(p_m) (models disagree and need teacher data).".to_string(),
            ),
            (
                "Pareto Cascade Risk Score (lemma cascade sweep)".to_string(),
                "Combines 1 - confidence, conformal set size |C(x)|, and Free-Energy OOD score into a composite risk metric, sweeping thresholds tau to find the Pareto frontier of accuracy vs $/1k items.".to_string(),
            ),
            (
                "Disaggregated Diagnostic Slicing (--slice-by)".to_string(),
                "Aggregate accuracy hides failure modes! Slicing by auto prefix (neg, multi, long, struct), Unicode script, or token length bucket exposes exactly which sub-population needs routing or calibration.".to_string(),
            ),
            (
                "Active Learning & RLCD Distillation (lemma active)".to_string(),
                "Selects high-uncertainty / OOD items via diversity-aware farthest-first traversal and exports Contrastive Teacher-Student distillation pairs so System 1 continuously absorbs System 2 intelligence.".to_string(),
            ),
        ],
        deep_intuitions: vec![
            (
                "Production Unit Economics (Eugene Yan / FrugalGPT)".to_string(),
                "Routing 100% of 1M requests to a System 2 LLM ($0.005/call) costs $5,000 and ~1,200ms p95 latency. A System 1 -> System 2 cascade resolving 80% of traffic locally in <40ms ($0.00002/call = $16) and escalating only the 20% hard tail to System 2 ($1,000) slashes total spend to ~$1,016 per 1M requests (a 79.7% / ~5x cost reduction) while matching System 2 accuracy.".to_string(),
            ),
            (
                "When to Escalate to User vs. Teacher LLM".to_string(),
                "Not all uncertainty should be routed the same way! High Aleatoric uncertainty (every model in `lemma consensus` outputs flat probabilities) means the user's prompt is genuinely underspecified—ask the user a clarifying question. High Epistemic uncertainty (high JSD between Laya, DeBERTa-v3, and ModernBERT) means System 1 lacks domain coverage—escalate to the System 2 teacher LLM and distill via `lemma active`.".to_string(),
            ),
        ],
        under_the_hood_math: concat!(
            "1. Jensen-Shannon Divergence (Epistemic Disagreement across M models):\n",
            "   JSD(p_1, ..., p_M) = H( sum_{m=1}^{M} w_m p_m ) - sum_{m=1}^{M} w_m H(p_m)\n\n",
            "2. Expected Cascade Cost & Accuracy at Escalation Rate r in [0, 1]:\n",
            "   Cost(r) = C_{sys1} + r * C_{sys2}\n",
            "   Acc(r)  = (1 - r) * Acc_{sys1}(retained) + r * Acc_{sys2}(escalated)"
        )
        .to_string(),
        live_lab,
        quiz: build_phase_5_quiz(),
        further_reading: build_reading_list(Some(5)),
        next_step_commands: vec![
            "lemma cascade sweep --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
            "lemma compare --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --slice-by auto".to_string(),
            "lemma active sample --file data/customer_support.jsonl --schema schema/questions.json --strategy composite --budget 5".to_string(),
            "lemma guide".to_string(),
        ],
    }
}

fn build_phase_1_quiz() -> Vec<QuizQuestion> {
    vec![
        QuizQuestion {
            id: "P1-Q1".to_string(),
            phase: 1,
            prompt: "Why can a bidirectional encoder (like ModernBERT or DeBERTa-v3) resolve a contrastive sentence like 'Not a technical bug, please refund our invoice' more efficiently than a causal LLM?".to_string(),
            options: vec![
                "Because causal LLMs cannot process negative words like 'Not' at all.".to_string(),
                "Because every token attends to both past and future tokens simultaneously in a single non-autoregressive pass (<40ms), binding 'Not' to 'bug' and 'refund' to 'invoice' all at once.".to_string(),
                "Because bidirectional encoders generate text from right to left instead of left to right.".to_string(),
                "Because bidirectional encoders do not use self-attention layers.".to_string(),
            ],
            correct_index: 1,
            explanation: "In a bidirectional encoder, the attention mask is unmasked in both directions, so early tokens ('Not', 'bug') and late tokens ('refund', 'invoice') mutually contextualize each other in one parallel pass without autoregressive token-by-token generation.".to_string(),
        },
        QuizQuestion {
            id: "P1-Q2".to_string(),
            phase: 1,
            prompt: "Which key architectural upgrades distinguish 2025/2026 ModernBERT from legacy 2018 BERT?".to_string(),
            options: vec![
                "Replacing self-attention with recurrent LSTM cells and 128-token limits.".to_string(),
                "Rotary Position Embeddings (RoPE), GeGLU/SwiGLU layers, alternating local/global FlashAttention, and native 8,192-token context windows.".to_string(),
                "Removing all positional information so token order is completely ignored.".to_string(),
                "Requiring 70 billion parameters and multi-GPU tensor parallelism.".to_string(),
            ],
            correct_index: 1,
            explanation: "ModernBERT modernizes the encoder stack with RoPE (enabling 8,192-token extrapolation), FlashAttention unpadding (2-4x higher throughput), and GeGLU feedforward layers while staying under 425M parameters.".to_string(),
        },
        QuizQuestion {
            id: "P1-Q3".to_string(),
            phase: 1,
            prompt: "According to Ethayarajh (2019), why do raw transformer `[CLS]` embeddings suffer from 'anisotropy', and how does Laya's `mask_scorer` / `act_head` architecture overcome this?".to_string(),
            options: vec![
                "Anisotropy means embeddings are all zero vectors; Laya fixes it by adding random Gaussian noise.".to_string(),
                "Raw transformer hidden states occupy a narrow high-dimensional cone where even unrelated sentences have high cosine similarity; Laya's token-level `mask_scorer` and `act_head` projections carve orthogonal decision hyperplanes at each `<<OPTION>>` token.".to_string(),
                "Anisotropy only affects recurrent neural networks, not transformers.".to_string(),
                "Laya avoids anisotropy by discarding all token embeddings except the first character.".to_string(),
            ],
            correct_index: 1,
            explanation: "Ethayarajh (2019) showed that contextualized representations concentrate in a narrow cone (anisotropy). Projecting individual `<<OPTION>>` tokens through dedicated `mask_scorer` and `act_head` heads extracts sharp, orthogonal class margins instead of relying on raw `[CLS]` cosine similarity.".to_string(),
        },
    ]
}

fn build_phase_2_quiz() -> Vec<QuizQuestion> {
    vec![
        QuizQuestion {
            id: "P2-Q1".to_string(),
            phase: 2,
            prompt: "Suppose you have K = 25 candidate labels in a choice question. How many encoder forward passes are required per item by JointSinglePass (Laya) vs. PairwiseNliCrossEncoder (DeBERTa-v3-ZS)?".to_string(),
            options: vec![
                "JointSinglePass runs 25 passes; PairwiseNliCrossEncoder runs 1 pass.".to_string(),
                "JointSinglePass runs 1 pass O(1); PairwiseNliCrossEncoder runs 25 passes O(K) (unless --shortlist-k is enabled).".to_string(),
                "Both paradigms always run 25 passes.".to_string(),
                "Both paradigms require 2^25 passes.".to_string(),
            ],
            correct_index: 1,
            explanation: "JointSinglePass packs all K options into a single prompt and scores their <<LABEL>> tokens simultaneously in O(1) passes. Pairwise NLI cross-encoders evaluate each (premise, hypothesis_k) pair separately in O(K) passes.".to_string(),
        },
        QuizQuestion {
            id: "P2-Q2".to_string(),
            phase: 2,
            prompt: "Why do raw GenerativeMlmVerbalizer models (like EuroBERT or NeoBERT) struggle on 'We did NOT have a security breach; we need a billing refund' if a naive cloze prompt is used?".to_string(),
            options: vec![
                "Because EuroBERT only supports French and German inputs.".to_string(),
                "Because the pretrained lm_head predicts tokens at [MASK] using lexical co-occurrence, so pre-contrast negated terms ('security breach') can strongly activate the 'security' verbalizer unless contrast-aware verbalization isolates the post-pivot clause.".to_string(),
                "Because [MASK] tokens are stripped out by the tokenizer.".to_string(),
                "Because MLM verbalizers use O(K^2) complexity.".to_string(),
            ],
            correct_index: 1,
            explanation: "Pretrained MLM heads are trained to predict co-occurring topic words at [MASK], not logical entailment. Contrast-aware verbalization ('The actual issue is [MASK], not the negated clause') anchors [MASK] to the post-contrast statement.".to_string(),
        },
        QuizQuestion {
            id: "P2-Q3".to_string(),
            phase: 2,
            prompt: "Why does customizing `--hypothesis-template` (e.g. 'The customer is reporting a {label} issue: {description}') significantly improve zero-shot accuracy on Pairwise NLI Cross-Encoders (Yin et al., 2019)?".to_string(),
            options: vec![
                "Because NLI models ignore the input text and only read the hypothesis template.".to_string(),
                "Because NLI cross-encoders judge logical entailment between a premise and a full natural-language proposition, so domain-aligned templates with `{description}` criteria give attention heads explicit semantic anchors to verify.".to_string(),
                "Because longer hypothesis templates reduce the model's parameter count.".to_string(),
                "Because `--hypothesis-template` converts an NLI model into an autoregressive decoder.".to_string(),
            ],
            correct_index: 1,
            explanation: "Yin et al. (2019) showed that zero-shot NLI classification depends on how faithfully the hypothesis sentence represents the target class semantics; injecting `{label}` and `{description}` aligns the task with NLI pretraining.".to_string(),
        },
    ]
}

fn build_phase_3_quiz() -> Vec<QuizQuestion> {
    vec![
        QuizQuestion {
            id: "P3-Q1".to_string(),
            phase: 3,
            prompt: "You are classifying 1,500-token customer support email threads using a 512-token checkpoint, and the customer's actual escalation request is in the final reply at the bottom. Which --chunk-strategy prevents the 'Lost in the Tail' failure mode in a single pass?".to_string(),
            options: vec![
                "--chunk-strategy truncate-prefix".to_string(),
                "--chunk-strategy head-tail (retains 25% opening context + 75% tail context)".to_string(),
                "Increasing temperature to 5.0".to_string(),
                "Disabling the tokenizer".to_string(),
            ],
            correct_index: 1,
            explanation: "head-tail chunking preserves the first 25% of the window (email subject & initial context) and the final 75% of the window (most recent reply/incident), preventing naive prefix truncation from dropping tail signals.".to_string(),
        },
        QuizQuestion {
            id: "P3-Q2".to_string(),
            phase: 3,
            prompt: "How does `--model auto-best` decide which checkpoint to run for each incoming item?".to_string(),
            options: vec![
                "It runs all 10 checkpoints in parallel on every item and picks the slowest one.".to_string(),
                "It inspects Unicode script, subword token count, label cardinality K, and typed head requirements (Score/Noul) in <0.5ms to route each item to the optimal specialist encoder.".to_string(),
                "It picks a checkpoint at random using a coin flip.".to_string(),
                "It requires an external OpenAI API call to choose the model.".to_string(),
            ],
            correct_index: 1,
            explanation: "route_auto_best is a deterministic <0.5ms capability router: non-Latin scripts route to laya-multilingual/eurobert, >512t inputs route to 8k modernbert-zs or 1k laya-typed, high-cardinality K>8 or typed heads route to O(1) laya-en, and short K<=5 choice tasks route to deberta-v3-zs.".to_string(),
        },
        QuizQuestion {
            id: "P3-Q3".to_string(),
            phase: 3,
            prompt: "When a verbose multi-question JSON schema exceeds a checkpoint's head token budget (e.g., 192 tokens), how does `lemma schema optimize` compress the option criteria?".to_string(),
            options: vec![
                "It deletes half of the candidate class labels at random.".to_string(),
                "It scores candidate clauses by information density / Mutual Information I(w; Y) and solves a 0/1 Knapsack dynamic programming problem to retain high-MI discriminative keywords within the token budget.".to_string(),
                "It replaces all option descriptions with empty strings.".to_string(),
                "It Base64-encodes the JSON file.".to_string(),
            ],
            correct_index: 1,
            explanation: "SchemaKnapsackOptimizer treats each clause's token count as weight and its non-stopword keyword density as value, solving a 0/1 Knapsack DP per option so discriminative triggers fit inside the head budget.".to_string(),
        },
    ]
}

fn build_phase_4_quiz() -> Vec<QuizQuestion> {
    vec![
        QuizQuestion {
            id: "P4-Q1".to_string(),
            phase: 4,
            prompt: "A zero-shot NLI classifier is severely overconfident: when it predicts 98% probability, its empirical accuracy is only 75%. How does Temperature Scaling (z_i / T) fix this, and does it change top-1 argmax accuracy?".to_string(),
            options: vec![
                "Setting T > 1.0 softens the probability distribution to match empirical accuracy (lowering ECE) while preserving the exact rank order of logits—so top-1 accuracy is unchanged!".to_string(),
                "Setting T > 1.0 randomly shuffles the predicted class labels.".to_string(),
                "Setting T < 0.1 makes probabilities even closer to 100%, fixing overconfidence.".to_string(),
                "Temperature scaling deletes all classes with probability below 0.5.".to_string(),
            ],
            correct_index: 0,
            explanation: "Dividing logits by T > 1.0 is a strictly monotonic transformation on a single head: it pulls overconfident probabilities toward the uniform prior (reducing ECE and Brier score) without changing which class has the highest logit.".to_string(),
        },
        QuizQuestion {
            id: "P4-Q2".to_string(),
            phase: 4,
            prompt: "Why does `lemma ood` use Helmholtz Free Energy E(x) = -T * ln(sum exp(z_i / T)) instead of just checking softmax entropy?".to_string(),
            options: vec![
                "Because softmax is translation-invariant (logits [-10, -10] and [+10, +10] yield identical [0.5, 0.5] probabilities), whereas Free Energy preserves raw logit magnitude to distinguish an alien OOD input (all logits negative) from an in-distribution tie (two logits strongly positive).".to_string(),
                "Because Free Energy does not require model logits.".to_string(),
                "Because softmax entropy can only be computed for 2 classes.".to_string(),
                "Because Free Energy is always equal to 0.".to_string(),
            ],
            correct_index: 0,
            explanation: "When an input is completely out-of-distribution (e.g., a cooking recipe sent to a DevOps router), all class logits z_i are low/negative. Free energy captures the log-partition sum of raw logits that softmax normalizes away.".to_string(),
        },
        QuizQuestion {
            id: "P4-Q3".to_string(),
            phase: 4,
            prompt: "In Split Conformal Prediction (`lemma conformal --alpha 0.05`), what does it signify when the prediction set C(x) expands from a singleton |C(x)| = 1 on item A to |C(x)| = 3 on item B?".to_string(),
            options: vec![
                "Item A is unambiguous enough to auto-execute under the >=95% coverage guarantee, whereas item B requires 3 candidate labels to reach the calibrated quantile q_hat—signaling boundary ambiguity that should escalate to System 2.".to_string(),
                "It means the model crashed on item B.".to_string(),
                "It means item A has 0% accuracy and item B has 100% accuracy.".to_string(),
                "Conformal prediction sets always have the exact same size for every input.".to_string(),
            ],
            correct_index: 0,
            explanation: "Adaptive Prediction Sets (Angelopoulos & Bates, 2023) dynamically scale set size |C(x)| with instance difficulty: singletons (|C(x)|=1) enable safe automated execution, while multi-label sets (|C(x)|>=2) flag ambiguous inputs for escalation.".to_string(),
        },
    ]
}

fn build_phase_5_quiz() -> Vec<QuizQuestion> {
    vec![
        QuizQuestion {
            id: "P5-Q1".to_string(),
            phase: 5,
            prompt: "In `lemma consensus`, three diverse models each output a low-entropy (90% confident) prediction, but Model A predicts 'billing', Model B predicts 'security', and Model C predicts 'account'. Which uncertainty metric will spike?".to_string(),
            options: vec![
                "Aleatoric uncertainty (within-model entropy) will be maximal, while Epistemic uncertainty (JSD) will be zero.".to_string(),
                "Epistemic uncertainty (Jensen-Shannon Divergence across models) will spike because confident models disagree with each other, triggering a DISPUTED escalation.".to_string(),
                "Both uncertainties will be exactly 0.0.".to_string(),
                "Consensus will silently pick the alphabetical first label with 100% confidence.".to_string(),
            ],
            correct_index: 1,
            explanation: "Each individual model has low entropy (low aleatoric uncertainty), but their mixture distribution has high entropy. Jensen-Shannon Divergence H(mean(p)) - mean(H(p)) isolates this inter-model epistemic disagreement.".to_string(),
        },
        QuizQuestion {
            id: "P5-Q2".to_string(),
            phase: 5,
            prompt: "You run `lemma test` and see 82% overall accuracy, which looks acceptable. Why should you always run `--slice-by auto` or `--slice-by script` before deploying?".to_string(),
            options: vec![
                "Because `--slice-by` automatically rewrites the model weights.".to_string(),
                "Because aggregate accuracy masks Simpson's Paradox / slice failures—a model might score 96% on standard English items but 25% on negated (`neg_*`) or long-context (`long_*`) slices.".to_string(),
                "Because `--slice-by` reduces the file size on disk.".to_string(),
                "Because `--slice-by` skips ground-truth evaluation.".to_string(),
            ],
            correct_index: 1,
            explanation: "Disaggregated evaluation (--slice-by) breaks out Accuracy, ECE, and Confidence per data slice (e.g. negation, multilingual, long-context, structured JSON), immediately showing where routing or chunking fixes are needed.".to_string(),
        },
        QuizQuestion {
            id: "P5-Q3".to_string(),
            phase: 5,
            prompt: "Suppose pure System 2 LLM triage costs $5,000 per 1M requests ($0.005/call). If a `lemma cascade` policy at optimal threshold tau* resolves 80% of traffic in System 1 ($0.00002/call) and escalates 20% to System 2, what is the total cost per 1M requests?".to_string(),
            options: vec![
                "$5,000 per 1M requests (no savings).".to_string(),
                "~$1,016 per 1M requests ($16 for System 1 + $1,000 for 20% System 2 escalations)—a 79.7% (~5x) cost reduction while matching System 2 SLA accuracy.".to_string(),
                "$50,000 per 1M requests.".to_string(),
                "$0 per 1M requests because System 2 is never called.".to_string(),
            ],
            correct_index: 1,
            explanation: "By resolving the 80% high-confidence majority in System 1 ($16) and invoking System 2 only on the 20% high-risk tail ($1,000), a FrugalGPT/Eugene Yan cascade cuts spend from $5,000 to $1,016 per 1M calls (79.7% savings).".to_string(),
        },
    ]
}

/// Build all diagnostic quiz questions across all 5 phases.
pub fn build_all_quiz_questions() -> Vec<QuizQuestion> {
    let mut questions = Vec::new();
    questions.extend(build_phase_1_quiz());
    questions.extend(build_phase_2_quiz());
    questions.extend(build_phase_3_quiz());
    questions.extend(build_phase_4_quiz());
    questions.extend(build_phase_5_quiz());
    questions
}

/// Build the searchable concept & flag glossary.
pub fn build_glossary() -> Vec<GlossaryEntry> {
    vec![
        GlossaryEntry {
            term: "System 1 vs. System 2".to_string(),
            category: "Phase 1: Foundations".to_string(),
            definition: "System 1 models are fast (<40ms), non-autoregressive bidirectional encoders for structured decisions; System 2 models are slow (500-3000ms) autoregressive LLMs for open-ended generation.".to_string(),
            cli_flag_or_cmd: "lemma classify / lemma cascade".to_string(),
        },
        GlossaryEntry {
            term: "Bidirectional Self-Attention".to_string(),
            category: "Phase 1: Foundations".to_string(),
            definition: "Attention mechanism with no causal left-to-right mask, allowing every token to attend to both preceding and succeeding context simultaneously.".to_string(),
            cli_flag_or_cmd: "lemma info --inventory".to_string(),
        },
        GlossaryEntry {
            term: "RoPE (Rotary Position Embeddings)".to_string(),
            category: "Phase 1: Foundations".to_string(),
            definition: "Encodes token positions by rotating query/key vectors in complex space, enabling ModernBERT, EuroBERT, and NeoBERT to scale to 4k-8k context windows.".to_string(),
            cli_flag_or_cmd: "--model modernbert-zeroshot".to_string(),
        },
        GlossaryEntry {
            term: "Disentangled Attention (DeBERTa-v3)".to_string(),
            category: "Phase 1: Foundations".to_string(),
            definition: "Separates content vectors and relative position vectors so attention scores explicitly model content-to-position bindings (crucial for negation).".to_string(),
            cli_flag_or_cmd: "--model deberta-v3".to_string(),
        },
        GlossaryEntry {
            term: "JointSinglePass Head".to_string(),
            category: "Phase 2: Paradigms".to_string(),
            definition: "O(1) single-pass architecture (Laya / GLiClass) that packs all K candidate options into one prompt and scores <<LABEL>> tokens simultaneously.".to_string(),
            cli_flag_or_cmd: "--model english / typed-decisions".to_string(),
        },
        GlossaryEntry {
            term: "PairwiseNliCrossEncoder Head".to_string(),
            category: "Phase 2: Paradigms".to_string(),
            definition: "O(K) architecture (DeBERTa-v3-ZS, ModernBERT-ZS) that pairs the input premise with each label hypothesis and scores entailment vs contradiction.".to_string(),
            cli_flag_or_cmd: "--hypothesis-template \"...\"".to_string(),
        },
        GlossaryEntry {
            term: "GenerativeMlmVerbalizer Head".to_string(),
            category: "Phase 2: Paradigms".to_string(),
            definition: "O(1) cloze architecture (EuroBERT, NeoBERT) that scores candidate label tokens at a [MASK] position using the pretrained lm_head.".to_string(),
            cli_flag_or_cmd: "--model eurobert / neobert".to_string(),
        },
        GlossaryEntry {
            term: "Typed Decisions (Choice, Score, Noul)".to_string(),
            category: "Phase 2: Paradigms".to_string(),
            definition: "Laya's three decision primitives: Choice (categorical softmax/sigmoid), Score (ordinal expected value E[y] = sum i*p_i), and Noul (binary sufficiency assertion).".to_string(),
            cli_flag_or_cmd: "--qtype choice|score|noul".to_string(),
        },
        GlossaryEntry {
            term: "Head-Tail Chunking".to_string(),
            category: "Phase 3: Scaling".to_string(),
            definition: "Context adaptation strategy that concatenates the first 25% and final 75% of a long document to prevent dropping tail incidents on 512-token models.".to_string(),
            cli_flag_or_cmd: "--chunk-strategy head-tail".to_string(),
        },
        GlossaryEntry {
            term: "Bi-Encoder Shortlisting".to_string(),
            category: "Phase 3: Scaling".to_string(),
            definition: "Two-stage filtering that prunes K > 10 candidate options down to top-k in O(1) via cosine similarity before cross-encoder scoring.".to_string(),
            cli_flag_or_cmd: "--shortlist-k <K>".to_string(),
        },
        GlossaryEntry {
            term: "Meta-Router (auto-best)".to_string(),
            category: "Phase 3: Scaling".to_string(),
            definition: "Sub-millisecond (<0.5ms) capability router that selects the optimal encoder per item based on Unicode script, token length, and label count K.".to_string(),
            cli_flag_or_cmd: "--model auto-best".to_string(),
        },
        GlossaryEntry {
            term: "Normalized Negentropy Confidence".to_string(),
            category: "Phase 4: Calibration".to_string(),
            definition: "Information-theoretic confidence 1 - H(p)/ln(K) blended with top-2 margin, measuring distribution certainty across all K classes.".to_string(),
            cli_flag_or_cmd: "lemma test / lemma stats".to_string(),
        },
        GlossaryEntry {
            term: "ECE (Expected Calibration Error)".to_string(),
            category: "Phase 4: Calibration".to_string(),
            definition: "Weighted average gap between predicted confidence and empirical accuracy across M probability bins (0.0 = perfectly calibrated).".to_string(),
            cli_flag_or_cmd: "lemma calibrate --persist-to-model".to_string(),
        },
        GlossaryEntry {
            term: "Q-ATS (Quantile-Adaptive Temperature Scaling)".to_string(),
            category: "Phase 4: Calibration".to_string(),
            definition: "2026 calibration method that dynamically inflates temperature T(x) when an input's energy/distance exceeds in-distribution quantiles.".to_string(),
            cli_flag_or_cmd: "lemma calibrate --all-checkpoints --persist-to-model --adaptive".to_string(),
        },
        GlossaryEntry {
            term: "Split Conformal Prediction".to_string(),
            category: "Phase 4: Calibration".to_string(),
            definition: "Distribution-free uncertainty wrapper that outputs a prediction set C(x) guaranteed to contain the true label with probability >= 1 - alpha.".to_string(),
            cli_flag_or_cmd: "lemma conformal --alpha 0.05".to_string(),
        },
        GlossaryEntry {
            term: "Helmholtz Free-Energy OOD".to_string(),
            category: "Phase 4: Calibration".to_string(),
            definition: "Out-of-distribution score E(x) = -T ln sum exp(z_i/T) that catches alien inputs where all raw class logits are simultaneously low.".to_string(),
            cli_flag_or_cmd: "lemma ood fit / eval".to_string(),
        },
        GlossaryEntry {
            term: "Epistemic vs. Aleatoric Uncertainty".to_string(),
            category: "Phase 5: Production".to_string(),
            definition: "Aleatoric = inherent data ambiguity (high entropy within each model); Epistemic = inter-model disagreement measured via Jensen-Shannon Divergence.".to_string(),
            cli_flag_or_cmd: "lemma consensus".to_string(),
        },
        GlossaryEntry {
            term: "Diagnostic Slicing".to_string(),
            category: "Phase 5: Production".to_string(),
            definition: "Disaggregating accuracy and ECE by slice prefix (en, neg, multi, long, struct), Unicode script, or token length bucket to expose hidden failure modes.".to_string(),
            cli_flag_or_cmd: "--slice-by auto|script|length".to_string(),
        },
    ]
}

/// Build practical step-by-step workflow recipes for `lemma guide`.
pub fn build_workflow_recipes() -> Vec<WorkflowRecipe> {
    vec![
        WorkflowRecipe {
            slug: "quickstart".to_string(),
            title: "1. Zero-to-One Classification & Contrastive Explainability".to_string(),
            goal: "Classify inline text or a JSONL file and see exactly which tokens drove the decision.".to_string(),
            steps: vec![
                (
                    "Classify & explain inline text (auto-detects top-1 vs top-2 labels)".to_string(),
                    "lemma explain \"We did not have an outage, our invoice was double charged\" --labels \"billing,technical,security\"".to_string(),
                ),
                (
                    "Run a multi-question schema (Choice + Score + Noul) on a file".to_string(),
                    "lemma classify --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
                ),
                (
                    "Auto-route heterogeneous multilingual & long-context items".to_string(),
                    "lemma classify --model auto-best --chunk-strategy head-tail --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
                ),
            ],
            pro_tip: "Add --hypothesis-template \"The user needs help with {label}: {description}\" when using NLI models (deberta-v3, modernbert-zeroshot) to boost zero-shot accuracy.".to_string(),
        },
        WorkflowRecipe {
            slug: "arena".to_string(),
            title: "2. Model Arena Benchmarking & Diagnostic Slicing".to_string(),
            goal: "Compare all System 1 checkpoints head-to-head and uncover per-slice strengths (negation, multilingual, long-context).".to_string(),
            steps: vec![
                (
                    "Check local checkpoint inventory and head paradigms".to_string(),
                    "lemma info --inventory".to_string(),
                ),
                (
                    "Run the full System 1 Arena with disaggregated slice matrix".to_string(),
                    "lemma compare --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --slice-by auto".to_string(),
                ),
                (
                    "Evaluate a single model disaggregated by token length or script".to_string(),
                    "lemma test --model deberta-v3 --chunk-strategy head-tail --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --slice-by length".to_string(),
                ),
            ],
            pro_tip: "Never rely on aggregate accuracy alone! Always pass --slice-by auto to see the Checkpoint x Slice matrix before choosing a production model.".to_string(),
        },
        WorkflowRecipe {
            slug: "calibration".to_string(),
            title: "3. Temperature Calibration, Conformal Sets & OOD Guarding".to_string(),
            goal: "Eliminate zero-shot overconfidence, persist per-checkpoint calibration, and guarantee >=95% coverage.".to_string(),
            steps: vec![
                (
                    "Tune & persist temperatures + Q-ATS for all checkpoints".to_string(),
                    "lemma calibrate --all-checkpoints --persist-to-model --adaptive --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json".to_string(),
                ),
                (
                    "Generate Split Conformal prediction sets at 95% coverage (alpha=0.05)".to_string(),
                    "lemma conformal --file data/system1_golden_benchmark.jsonl --schema schema/system1_benchmark_questions.json --alpha 0.05".to_string(),
                ),
                (
                    "Fit Out-of-Distribution (OOD) detector via Helmholtz Free Energy".to_string(),
                    "lemma ood fit --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
                ),
            ],
            pro_tip: "Once `lemma calibrate --persist-to-model` writes `lemma_calibration.json` to the model folder, every subsequent `classify`, `test`, and `compare` run automatically uses the calibrated temperatures!".to_string(),
        },
        WorkflowRecipe {
            slug: "scaling".to_string(),
            title: "4. Scaling to 50+ Labels & 4,000-Token Documents".to_string(),
            goal: "Prevent head-budget overflow on large schemas and preserve tail signals on long documents.".to_string(),
            steps: vec![
                (
                    "Lint & compress a large schema to fit a 128-token head budget".to_string(),
                    "lemma schema optimize --schema schema/questions.json --budget 128".to_string(),
                ),
                (
                    "Enable O(1) Bi-Encoder candidate shortlisting for K > 10 labels".to_string(),
                    "lemma classify --model deberta-v3 --shortlist-k 4 --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
                ),
                (
                    "Use coarse-to-fine hierarchical routing trees".to_string(),
                    "lemma route bench --file data/customer_support.jsonl --tree schema/support_hierarchy.json".to_string(),
                ),
            ],
            pro_tip: "Combining `--shortlist-k 5` with `--chunk-strategy head-tail` lets 512-token NLI cross-encoders handle 2,000-token documents and 30-label schemas at 5x higher speed.".to_string(),
        },
        WorkflowRecipe {
            slug: "production".to_string(),
            title: "5. Production Cascading, Multi-Model Consensus & Active Loops".to_string(),
            goal: "Deploy a fault-tolerant System 1 -> System 2 cascade with drift alerts and cryptographic receipts.".to_string(),
            steps: vec![
                (
                    "Sweep the System 1 / System 2 Pareto cost-accuracy frontier".to_string(),
                    "lemma cascade sweep --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
                ),
                (
                    "Run 3-model Mixture-of-Decisions consensus with JSD dispute detection".to_string(),
                    "lemma consensus --checkpoints english,deberta-v3,modernbert-zeroshot --file data/customer_support.jsonl --schema schema/questions.json".to_string(),
                ),
                (
                    "Curate an active learning review queue & RLCD distillation dataset".to_string(),
                    "lemma active sample --file data/customer_support.jsonl --schema schema/questions.json --strategy composite --budget 10".to_string(),
                ),
                (
                    "Monitor PSI / Jensen-Shannon distribution drift".to_string(),
                    "lemma drift --baseline data/customer_support.jsonl --current data/system1_golden_benchmark.jsonl --schema schema/questions.json".to_string(),
                ),
            ],
            pro_tip: "Use `lemma consensus` on high-stakes decisions: when `epistemic_JSD` is high, models from different paradigms disagree—escalate those items to `lemma active sample` for teacher labeling!".to_string(),
        },
    ]
}

/// Build the curated phase-by-phase reading list of foundational papers, visual essays, and practitioner blogs.
pub fn build_reading_list(phase_filter: Option<u8>) -> Vec<ReadingResource> {
    let all = vec![
        // Phase 1: Foundations — Encoders, Attention & Representation Geometry
        ReadingResource {
            phase: 1,
            category: "Visual Guide".to_string(),
            title: "The Illustrated BERT, ELMo, and Co. (How NLP Cracked Transfer Learning)".to_string(),
            author_venue: "Jay Alammar (jalammar.github.io, 2018)".to_string(),
            takeaway: "Visualizes how bidirectional self-attention lets every token attend to left and right context simultaneously in a single pass.".to_string(),
            url: "https://jalammar.github.io/illustrated-bert/".to_string(),
        },
        ReadingResource {
            phase: 1,
            category: "Foundational Paper".to_string(),
            title: "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT)".to_string(),
            author_venue: "Warner et al., Answer.AI & LightOn (arXiv:2412.13663, 2024/2025)".to_string(),
            takeaway: "Replaces legacy 512-token BERT with RoPE, GeGLU, alternating 128-token local / global FlashAttention, and unpadding up to 8,192 tokens.".to_string(),
            url: "https://arxiv.org/abs/2412.13663".to_string(),
        },
        ReadingResource {
            phase: 1,
            category: "Foundational Paper".to_string(),
            title: "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing".to_string(),
            author_venue: "He, Gao & Chen, Microsoft Research (ICLR 2023, arXiv:2111.09543)".to_string(),
            takeaway: "Disentangles content vectors (C) and relative position vectors (P) so attention explicitly binds negation and syntax across clauses.".to_string(),
            url: "https://arxiv.org/abs/2111.09543".to_string(),
        },
        ReadingResource {
            phase: 1,
            category: "Geometry Paper".to_string(),
            title: "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2".to_string(),
            author_venue: "Kawin Ethayarajh, Stanford (EMNLP 2019, arXiv:1909.00512)".to_string(),
            takeaway: "Proves raw transformer representations occupy a narrow anisotropic cone, explaining why Laya's token-level mask_scorer beats naive [CLS] cosine pooling.".to_string(),
            url: "https://arxiv.org/abs/1909.00512".to_string(),
        },
        // Phase 2: The 3 Zero-Shot Head Paradigms & Typed Decisions
        ReadingResource {
            phase: 2,
            category: "Foundational Paper".to_string(),
            title: "Benchmarking Zero-Shot Text Classification: Datasets, Evaluation and Entailment Approach".to_string(),
            author_venue: "Yin, Hay & Roth, UPenn (EMNLP 2019, arXiv:1909.00161)".to_string(),
            takeaway: "Establishes the Pairwise NLI Cross-Encoder paradigm by reframing zero-shot classification as textual entailment over `--hypothesis-template` propositions.".to_string(),
            url: "https://arxiv.org/abs/1909.00161".to_string(),
        },
        ReadingResource {
            phase: 2,
            category: "Architecture Blog".to_string(),
            title: "GLiClass: Generalist and Lightweight Model for Sequence Classification".to_string(),
            author_venue: "Knowledgator Engineering (Hugging Face Blog, 2024/2025)".to_string(),
            takeaway: "Demonstrates how JointSinglePass architectures pack all K candidate labels into a single bidirectional pass for O(1) zero-shot latency.".to_string(),
            url: "https://huggingface.co/blog/knowledgator/gliclass".to_string(),
        },
        ReadingResource {
            phase: 2,
            category: "Foundational Paper".to_string(),
            title: "Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference (PET)".to_string(),
            author_venue: "Schick & Schütze (EACL 2021, arXiv:2001.07676)".to_string(),
            takeaway: "Formalizes Generative MLM Verbalizers ([MASK] cloze prompts + verbalizer token scoring) used by EuroBERT and NeoBERT adapters.".to_string(),
            url: "https://arxiv.org/abs/2001.07676".to_string(),
        },
        ReadingResource {
            phase: 2,
            category: "Visual Essay".to_string(),
            title: "Evaluating Models’ Local Decision Boundaries via Contrast Sets & Minimal Pairs".to_string(),
            author_venue: "Gardner et al., AI2 (EMNLP Findings 2020) & Distill.pub".to_string(),
            takeaway: "Shows how minimal-pair counterfactuals (flipping a single discourse pivot) test true compositional binding vs. bag-of-words shortcuts.".to_string(),
            url: "https://arxiv.org/abs/2004.02709".to_string(),
        },
        // Phase 3: Scaling Context Windows, Languages & High-Cardinality Taxonomies
        ReadingResource {
            phase: 3,
            category: "Long-Context Paper".to_string(),
            title: "Lost in the Middle: How Language Models Use Long Contexts".to_string(),
            author_venue: "Nelson F. Liu et al., Stanford (TACL 2024, arXiv:2307.03172)".to_string(),
            takeaway: "Demonstrates U-shaped positional attention: models extract highest-MI signals from the opening (head) and final (tail) segments of long documents.".to_string(),
            url: "https://arxiv.org/abs/2307.03172".to_string(),
        },
        ReadingResource {
            phase: 3,
            category: "Foundational Paper".to_string(),
            title: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks".to_string(),
            author_venue: "Reimers & Gurevych, UKP Lab (EMNLP 2019, arXiv:1908.10084)".to_string(),
            takeaway: "Establishes two-stage retrieval + cross-encoding: fast O(1) cosine shortlisting (`--shortlist-k`) before expensive cross-encoder scoring.".to_string(),
            url: "https://arxiv.org/abs/1908.10084".to_string(),
        },
        ReadingResource {
            phase: 3,
            category: "Multilingual Paper".to_string(),
            title: "EuroBERT: Scaling Multilingual Encoders for European Languages".to_string(),
            author_venue: "Boizard et al., LightOn / Inria / Artefact (arXiv:2503.05500, 2025)".to_string(),
            takeaway: "Scales RoPE + SwiGLU bidirectional encoders across multilingual scripts up to 8,192 tokens, motivating `--model auto-best` routing.".to_string(),
            url: "https://arxiv.org/abs/2503.05500".to_string(),
        },
        ReadingResource {
            phase: 3,
            category: "Compression Paper".to_string(),
            title: "LLMLingua: Compressing Prompts for Accelerated Inference via Mutual Information Budgets".to_string(),
            author_venue: "Jiang et al., Microsoft Research (EMNLP 2023, arXiv:2310.05736)".to_string(),
            takeaway: "Motivates `lemma schema optimize`: pruning low-MI filler words under a 0/1 Knapsack token budget while preserving discriminative triggers.".to_string(),
            url: "https://arxiv.org/abs/2310.05736".to_string(),
        },
        // Phase 4: Trustworthy Uncertainty, Temperature Calibration & Conformal Guarantees
        ReadingResource {
            phase: 4,
            category: "Calibration Paper".to_string(),
            title: "On Calibration of Modern Neural Networks".to_string(),
            author_venue: "Guo, Pleiss, Sun & Weinberger, Cornell (ICML 2017, arXiv:1706.04599)".to_string(),
            takeaway: "Proves modern neural networks are systematically overconfident and introduces Temperature Scaling (z_i / T) and Expected Calibration Error (ECE).".to_string(),
            url: "https://arxiv.org/abs/1706.04599".to_string(),
        },
        ReadingResource {
            phase: 4,
            category: "Conformal Tutorial".to_string(),
            title: "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification".to_string(),
            author_venue: "Angelopoulos & Bates, UC Berkeley (Foundations & Trends in ML 2023, arXiv:2107.07511)".to_string(),
            takeaway: "Derives Split Conformal Prediction and Adaptive Prediction Sets C(x), guaranteeing finite-sample >= 1 - alpha coverage (`lemma conformal`).".to_string(),
            url: "https://arxiv.org/abs/2107.07511".to_string(),
        },
        ReadingResource {
            phase: 4,
            category: "OOD Paper".to_string(),
            title: "Energy-based Out-of-distribution Detection".to_string(),
            author_venue: "Weitang Liu et al., UC San Diego (NeurIPS 2020, arXiv:2010.03759)".to_string(),
            takeaway: "Proves softmax is translation-invariant to constant logit shifts, whereas Helmholtz Free Energy E(x) = -T ln sum exp(z_i/T) cleanly detects OOD inputs.".to_string(),
            url: "https://arxiv.org/abs/2010.03759".to_string(),
        },
        ReadingResource {
            phase: 4,
            category: "Adaptive Calibration".to_string(),
            title: "Sample-Dependent Adaptive Temperature Scaling for Improved Calibration (Q-ATS)".to_string(),
            author_venue: "Joy et al., Oxford (AAAI 2023, arXiv:2207.06211)".to_string(),
            takeaway: "Dynamically inflates temperature T(x) when an input's energy or distance exceeds in-distribution quantiles (`lemma calibrate --adaptive`).".to_string(),
            url: "https://arxiv.org/abs/2207.06211".to_string(),
        },
        // Phase 5: Production Architecture — Cascading, Consensus, Slicing & Active Loops
        ReadingResource {
            phase: 5,
            category: "Systems Paper".to_string(),
            title: "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance".to_string(),
            author_venue: "Chen, Zaharia & Zou, Stanford (TMLR 2024, arXiv:2305.05176)".to_string(),
            takeaway: "Formalizes System 1 -> System 2 cascades that resolve 80%+ of traffic locally and cut LLM inference spend by 5x–10x (`lemma cascade`).".to_string(),
            url: "https://arxiv.org/abs/2305.05176".to_string(),
        },
        ReadingResource {
            phase: 5,
            category: "Practitioner Guide".to_string(),
            title: "Patterns for Building LLM-based Systems & Products (Cascades, Guardrails & Caching)".to_string(),
            author_venue: "Eugene Yan (eugeneyan.com, 2023/2025)".to_string(),
            takeaway: "Blueprints production latency SLAs, semantic caching, fast classifier guardrails, and unit-economic trade-offs in real-world AI systems.".to_string(),
            url: "https://eugeneyan.com/writing/llm-patterns/".to_string(),
        },
        ReadingResource {
            phase: 5,
            category: "MLOps Guide".to_string(),
            title: "Designing Machine Learning Systems: Distribution Shift, Monitoring & Continual Learning".to_string(),
            author_venue: "Chip Huyen (O'Reilly / huyenchip.com)".to_string(),
            takeaway: "Covers PSI/Wasserstein drift monitoring (`lemma drift`), uncertainty sampling, and closing the active teacher-student distillation loop (`lemma active`).".to_string(),
            url: "https://huyenchip.com/mlops/".to_string(),
        },
        ReadingResource {
            phase: 5,
            category: "Practitioner Blog".to_string(),
            title: "Your AI Product Needs Evals (Disaggregated Slicing & Error Analysis)".to_string(),
            author_venue: "Hamel Husain (hamel.dev, 2024)".to_string(),
            takeaway: "Explains why aggregate accuracy hides failure modes and how disaggregated slice evaluation (`--slice-by auto`) drives rapid iteration.".to_string(),
            url: "https://hamel.dev/blog/posts/evals/".to_string(),
        },
    ];

    all.into_iter()
        .filter(|r| phase_filter.is_none_or(|p| r.phase == p))
        .collect()
}

/// Print the curated reading trail of foundational papers, visual blogs, and practitioner guides.
pub fn print_reading_list(phase_filter: Option<u8>) {
    let list = build_reading_list(phase_filter);
    println!(
        "{}",
        c_bold(c_cyan(format!(
            "📚 LEMMA CURATED READING TRAIL — Papers, Visual Blogs & Practitioner Guides ({} resources)",
            list.len()
        )))
    );
    println!("══════════════════════════════════════════════════════════════════════════");
    println!(
        "A self-contained learning path from visual transformer intuition to 2025/2026 encoder papers and production MLOps:\n"
    );

    let mut table = styled_table();
    table.set_header(vec![
        Cell::new("Phase").fg(Color::Cyan),
        Cell::new("Type").fg(Color::Yellow),
        Cell::new("Title & Author / Venue").fg(Color::Green),
        Cell::new("Core Engineering Takeaway & URL").fg(Color::White),
    ]);

    for r in &list {
        table.add_row(vec![
            Cell::new(format!("P{}", r.phase)).fg(Color::Cyan),
            Cell::new(&r.category).fg(Color::Yellow),
            Cell::new(format!("{}\n({})", r.title, r.author_venue)).fg(Color::Green),
            Cell::new(format!("{}\n🔗 {}", r.takeaway, r.url)),
        ]);
    }
    println!("{table}");
}

/// Print the 5-Phase System 1 Mastery Curriculum overview and pedagogical guide.
pub fn print_curriculum_overview() {
    println!(
        "{}",
        c_bold(c_cyan(
            "🎓 LEMMA ACADEMY — Interactive System 1 Decision Models & Encoder Mastery"
        ))
    );
    println!("══════════════════════════════════════════════════════════════════════════");
    println!(
        "{}\n",
        render_callout_card(
            "🧭 Pedagogical Design (Concrete-Representational-Abstract & Spiral Curriculum)",
            "Every phase pairs intuitive mental models + semantic color-coded architecture diagrams + co-referenced math\nwith a LIVE executable lab (inline token heatmaps, probability bars, reliability curves) and a Socratic quiz.",
            "cyan"
        )
    );

    let mut table = styled_table();
    table.set_header(vec![
        Cell::new("Phase").fg(Color::Cyan),
        Cell::new("Slug").fg(Color::Yellow),
        Cell::new("Core Topic").fg(Color::Green),
        Cell::new("Command").fg(Color::Magenta),
    ]);

    for p in CurriculumPhase::all() {
        table.add_row(vec![
            Cell::new(format!("Phase {}", p.number())).fg(Color::Cyan),
            Cell::new(p.slug()).fg(Color::Yellow),
            Cell::new(p.title()),
            Cell::new(format!("lemma learn --phase {}", p.number())).fg(Color::Green),
        ]);
    }

    println!("🗺️  The 5-Phase Layered Learning Roadmap:\n{table}\n");

    let mut cmd_table = styled_table();
    cmd_table.set_header(vec![
        Cell::new("Learning Mode").fg(Color::Cyan),
        Cell::new("CLI Command").fg(Color::Green),
        Cell::new("What It Does").fg(Color::White),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Interactive Tutor"),
        Cell::new("lemma learn --interactive").fg(Color::Green),
        Cell::new("Step-by-step guided walkthrough with live labs & graded active-recall quizzes"),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Deep-Dive a Phase"),
        Cell::new("lemma learn --phase 1..5").fg(Color::Green),
        Cell::new("Render a specific phase (e.g. `lemma learn --phase paradigms` or `--phase 4`)"),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Complete Textbook"),
        Cell::new("lemma learn --all").fg(Color::Green),
        Cell::new("Print all 5 phases with live lab benchmarks in one comprehensive reference"),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Diagnostic Quiz"),
        Cell::new("lemma learn --quiz").fg(Color::Green),
        Cell::new("Test your intuition across all 15 Socratic diagnostic checkpoints"),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Concept Glossary"),
        Cell::new("lemma learn --glossary [term]").fg(Color::Green),
        Cell::new("Searchable dictionary linking ML theory (RoPE, ECE, JSD, Q-ATS) to CLI flags"),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Reading Trail"),
        Cell::new("lemma learn --reading").fg(Color::Green),
        Cell::new("Curated reading list of 20 foundational papers, visual essays & MLOps blogs"),
    ]);
    cmd_table.add_row(vec![
        Cell::new("Practical Recipes"),
        Cell::new("lemma guide [topic]").fg(Color::Green),
        Cell::new("Copy-pasteable how-to workflows (quickstart, arena, calibration, scaling, ops)"),
    ]);

    println!("🚀 How to Use Lemma Academy:\n{cmd_table}\n");
    println!(
        "💡 Recommended Start: Run {} or {} to begin!",
        c_bold(c_green("`lemma learn --phase 1`")),
        c_bold(c_green("`lemma learn --interactive`"))
    );
}

/// Print a single `CurriculumPhase` lesson with its mental model, ASCII diagram, math, live lab, and quiz preview.
pub fn print_phase_lesson(phase: CurriculumPhase) {
    let lesson = build_phase_lesson(phase);
    render_phase_lesson(&lesson, true);
}

/// Print all 5 curriculum phases sequentially.
pub fn print_all_phases() {
    for phase in CurriculumPhase::all() {
        let lesson = build_phase_lesson(phase);
        render_phase_lesson(&lesson, true);
        println!("\n──────────────────────────────────────────────────────────────────────────\n");
    }
}

fn render_phase_lesson(lesson: &PhaseLesson, include_quiz_preview: bool) {
    println!("{}\n", render_phase_breadcrumb(lesson.phase));
    println!(
        "{}",
        c_bold(c_cyan(format!(
            "📘 PHASE {} ({}): {}",
            lesson.phase,
            lesson.slug.to_uppercase(),
            lesson.title
        )))
    );
    println!("   {}", c_yellow(&lesson.subtitle));
    println!("══════════════════════════════════════════════════════════════════════════\n");

    println!(
        "{}\n",
        render_callout_card(
            "🧠 1. INTUITIVE MENTAL MODEL (Why This Matters)",
            &lesson.mental_model,
            "cyan"
        )
    );

    println!(
        "{}",
        c_bold(c_magenta(
            "📐 2. VISUAL ARCHITECTURE MECHANICS (Semantic Color-Coded Flow)"
        ))
    );
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "{}\n",
        colorize_ascii_diagram(lesson.phase, &lesson.ascii_diagram)
    );

    println!("{}", c_bold(c_cyan("🔑 3. CORE CONCEPTS & PILLARS")));
    println!("─────────────────────────────");
    let mut c_table = styled_table();
    c_table.set_header(vec![
        Cell::new("Concept").fg(Color::Cyan),
        Cell::new("How It Works in Lemma").fg(Color::White),
    ]);
    for (name, desc) in &lesson.core_concepts {
        c_table.add_row(vec![Cell::new(name).fg(Color::Yellow), Cell::new(desc)]);
    }
    println!("{c_table}\n");

    if !lesson.deep_intuitions.is_empty() {
        let mut body_lines = Vec::new();
        for (title, body) in &lesson.deep_intuitions {
            body_lines.push(format!("▸ {}: {}", c_bold(c_yellow(title)), body));
        }
        println!(
            "{}\n",
            render_callout_card(
                "💡 3b. DEEP PEDAGOGICAL INTUITIONS (\"Aha!\" Mental Models)",
                &body_lines.join("\n\n"),
                "magenta"
            )
        );
    }

    println!(
        "{}",
        c_bold(c_yellow(
            "🧮 4. UNDER THE HOOD: MATHEMATICAL FORMULATION (Co-Referenced Colors)"
        ))
    );
    println!("───────────────────────────────────────────────────────────────────────");
    println!(
        "{}\n",
        colorize_math_block(lesson.phase, &lesson.under_the_hood_math)
    );

    println!(
        "{}",
        c_bold(c_green(format!(
            "🧪 5. {} (Concrete Worked Example)",
            lesson.live_lab.title
        )))
    );
    println!("──────────────────────────────────────────────────────────────────────────");
    println!("{}\n", lesson.live_lab.explanation);

    let mut lab_table = styled_table();
    lab_table.set_header(vec![
        Cell::new("Live Experiment Metric").fg(Color::Cyan),
        Cell::new("Measured Runtime Output").fg(Color::Green),
    ]);
    for (k, v) in &lesson.live_lab.metrics_summary {
        lab_table.add_row(vec![Cell::new(k).fg(Color::Yellow), Cell::new(v)]);
    }
    println!("{lab_table}\n");

    // Render the phase-specific color visual demonstration (heatmaps, probability bars, reliability curves, window bars)
    println!("{}\n", render_phase_visual_demonstration(lesson.phase));

    println!(
        "👉 Try it yourself: {}\n",
        c_bold(c_green(format!("`{}`", lesson.live_lab.try_command)))
    );

    if include_quiz_preview && !lesson.quiz.is_empty() {
        println!(
            "🎯 6. ACTIVE RECALL CHECKPOINT ({} diagnostic questions — run `lemma learn --interactive --phase {}` to take interactively)",
            lesson.quiz.len(),
            lesson.phase
        );
        for q in &lesson.quiz {
            println!("   • [{}] {}", c_cyan(&q.id), q.prompt);
        }
        println!();
    }

    if !lesson.further_reading.is_empty() {
        println!(
            "📚 7. CURATED READING TRAIL (Phase {} Papers & Blogs — `lemma learn --reading --phase {}`)",
            lesson.phase, lesson.phase
        );
        for r in &lesson.further_reading {
            println!(
                "   • [{}] {} — {} ({})",
                c_yellow(&r.category),
                c_bold(c_green(&r.title)),
                r.author_venue,
                c_cyan(&r.url)
            );
            println!("     Takeaway: {}", r.takeaway);
        }
        println!();
    }

    println!("⏭️  Next Steps & Hands-On Commands:");
    for cmd in &lesson.next_step_commands {
        println!("   $ {}", c_green(cmd));
    }
}

/// Print the concept glossary, optionally filtered by a case-insensitive substring.
pub fn print_glossary(filter: Option<&str>) {
    let entries = build_glossary();
    let q = filter.map(|s| s.trim().to_ascii_lowercase());

    let filtered: Vec<&GlossaryEntry> = entries
        .iter()
        .filter(|e| {
            if let Some(ref needle) = q {
                if needle.is_empty() {
                    return true;
                }
                e.term.to_ascii_lowercase().contains(needle)
                    || e.category.to_ascii_lowercase().contains(needle)
                    || e.definition.to_ascii_lowercase().contains(needle)
                    || e.cli_flag_or_cmd.to_ascii_lowercase().contains(needle)
            } else {
                true
            }
        })
        .collect();

    println!(
        "📖 LEMMA CONCEPT & CLI GLOSSARY ({} entries matched)",
        filtered.len()
    );
    println!("══════════════════════════════════════════════════════════════════════════");

    if filtered.is_empty() {
        println!(
            "No glossary entries matched '{}'. Try `lemma learn --glossary` to view all terms.",
            filter.unwrap_or_default()
        );
        return;
    }

    let mut table = styled_table();
    table.set_header(vec![
        Cell::new("Term").fg(Color::Cyan),
        Cell::new("Phase").fg(Color::Yellow),
        Cell::new("Definition & Intuition").fg(Color::White),
        Cell::new("CLI Flag / Command").fg(Color::Green),
    ]);

    for e in filtered {
        table.add_row(vec![
            Cell::new(&e.term).fg(Color::Cyan),
            Cell::new(&e.category).fg(Color::Yellow),
            Cell::new(&e.definition),
            Cell::new(&e.cli_flag_or_cmd).fg(Color::Green),
        ]);
    }
    println!("{table}");
}

/// Print practical workflow recipes (`lemma guide`), optionally filtered by topic slug or keyword.
pub fn print_workflow_guide(topic: Option<&str>) {
    let recipes = build_workflow_recipes();
    let q = topic.map(|s| s.trim().to_ascii_lowercase());

    let matched: Vec<&WorkflowRecipe> = recipes
        .iter()
        .filter(|r| {
            if let Some(ref needle) = q {
                if needle.is_empty() || needle == "all" {
                    return true;
                }
                r.slug.to_ascii_lowercase().contains(needle)
                    || r.title.to_ascii_lowercase().contains(needle)
                    || r.goal.to_ascii_lowercase().contains(needle)
            } else {
                true
            }
        })
        .collect();

    println!(
        "{}",
        c_bold(c_cyan(
            "🛠️  LEMMA PRACTICAL WORKFLOW GUIDE — Copy-Pasteable Recipes"
        ))
    );
    println!("══════════════════════════════════════════════════════════════════════════");
    println!("Tip: Want to learn the deep ML theory behind these commands? Run `lemma learn`!\n");

    if matched.is_empty() {
        println!(
            "No workflow recipe matched '{}'. Available topics: quickstart, arena, calibration, scaling, production",
            topic.unwrap_or_default()
        );
        return;
    }

    for r in matched {
        println!(
            "📌 {}  [topic: {}]",
            c_bold(c_cyan(&r.title)),
            c_yellow(&r.slug)
        );
        println!("   Goal: {}", r.goal);
        let mut table = styled_table();
        table.set_header(vec![
            Cell::new("Step").fg(Color::Cyan),
            Cell::new("Action").fg(Color::Yellow),
            Cell::new("Command to Run").fg(Color::Green),
        ]);
        for (idx, (desc, cmd)) in r.steps.iter().enumerate() {
            table.add_row(vec![
                Cell::new(format!("#{}", idx + 1)).fg(Color::Cyan),
                Cell::new(desc),
                Cell::new(cmd).fg(Color::Green),
            ]);
        }
        println!("{table}");
        println!(
            "{}\n",
            render_callout_card("💡 Pro-Tip", &r.pro_tip, "yellow")
        );
    }
}

/// Print all diagnostic quiz questions and their explanations in non-interactive mode.
pub fn run_quiz_non_interactive(phase_filter: Option<u8>) {
    let all_q = build_all_quiz_questions();
    let questions: Vec<&QuizQuestion> = all_q
        .iter()
        .filter(|q| phase_filter.is_none_or(|p| q.phase == p))
        .collect();

    println!(
        "🧠 LEMMA SOCRATIC DIAGNOSTIC QUIZ ({} questions)",
        questions.len()
    );
    println!("══════════════════════════════════════════════════════════════════════════\n");

    for q in questions {
        println!(
            "[{}] (Phase {}) {}",
            c_cyan(&q.id),
            q.phase,
            c_bold(&q.prompt)
        );
        for (idx, opt) in q.options.iter().enumerate() {
            let marker = if idx == q.correct_index { "✅" } else { "  " };
            println!("  {} ({}) {}", marker, idx + 1, opt);
        }
        println!("  🔎 Why: {}\n", q.explanation);
    }
}

/// Launch an interactive terminal learning or quiz session reading from standard input.
pub fn run_interactive_session(
    start_phase: Option<CurriculumPhase>,
    quiz_only: bool,
) -> Result<()> {
    let stdin = io::stdin();
    let mut lock = stdin.lock();
    let _ = run_interactive_session_with_reader(&mut lock, start_phase, quiz_only)?;
    Ok(())
}

/// Testable interactive session runner accepting any `BufRead` input stream.
/// Returns `(correct_answers, total_attempted)`. Gracefully exits on EOF.
pub fn run_interactive_session_with_reader<R: BufRead>(
    reader: &mut R,
    start_phase: Option<CurriculumPhase>,
    quiz_only: bool,
) -> Result<(usize, usize)> {
    let phases: Vec<CurriculumPhase> = match start_phase {
        Some(p) => CurriculumPhase::all()
            .into_iter()
            .filter(|cand| cand.number() >= p.number())
            .collect(),
        None => CurriculumPhase::all().to_vec(),
    };

    println!("🎓 Welcome to Lemma Interactive Academy!");
    println!(
        "   Controls: [Enter] continue | [1-4] answer quiz | [s] skip phase | [q] quit session\n"
    );

    let mut correct = 0usize;
    let mut attempted = 0usize;

    for phase in phases {
        let lesson = build_phase_lesson(phase);

        if !quiz_only {
            println!("{}", render_phase_breadcrumb(lesson.phase));
            println!(
                "📘 PHASE {} of 5: {}",
                lesson.phase,
                c_bold(c_cyan(&lesson.title))
            );
            println!("   {}\n", c_yellow(&lesson.subtitle));

            println!(
                "{}\n",
                render_callout_card(
                    "🧠 [Step 1/3 — Mental Model & Visual Architecture]",
                    &lesson.mental_model,
                    "cyan"
                )
            );
            println!(
                "{}\n",
                colorize_ascii_diagram(lesson.phase, &lesson.ascii_diagram)
            );
            for (title, body) in &lesson.deep_intuitions {
                println!("   💡 {}: {body}\n", c_bold(c_yellow(title)));
            }

            match prompt_continue(
                reader,
                "Press [Enter] for Math & Live Executable Lab (or [s] skip phase, [q] quit): ",
            )? {
                PromptAction::Continue => {}
                PromptAction::SkipPhase => continue,
                PromptAction::Quit => return Ok((correct, attempted)),
            }

            println!("\n🧮 [Step 2/3 — Mathematical Formulation & Live Lab]");
            println!(
                "{}\n",
                colorize_math_block(lesson.phase, &lesson.under_the_hood_math)
            );
            println!("🧪 {}", c_bold(c_green(&lesson.live_lab.title)));
            println!("{}", lesson.live_lab.explanation);
            for (k, v) in &lesson.live_lab.metrics_summary {
                println!("   • {}: {v}", c_yellow(k));
            }
            println!("\n{}\n", render_phase_visual_demonstration(lesson.phase));
            println!(
                "   👉 Try command: `{}`\n",
                c_green(&lesson.live_lab.try_command)
            );

            match prompt_continue(
                reader,
                "Press [Enter] for the Phase Active-Recall Quiz (or [s] skip phase, [q] quit): ",
            )? {
                PromptAction::Continue => {}
                PromptAction::SkipPhase => continue,
                PromptAction::Quit => return Ok((correct, attempted)),
            }
        }

        println!(
            "\n🎯 [Phase {} Active-Recall Diagnostic Checkpoints]",
            lesson.phase
        );
        for q in &lesson.quiz {
            println!("\n[{}] {}", c_cyan(&q.id), c_bold(&q.prompt));
            for (idx, opt) in q.options.iter().enumerate() {
                println!("  ({}) {}", idx + 1, opt);
            }
            print!("Your answer [1-{} or q to quit]: ", q.options.len());
            let _ = io::stdout().flush();

            let mut line = String::new();
            let bytes = reader.read_line(&mut line)?;
            if bytes == 0 {
                // Graceful EOF in non-TTY / automated test environments
                println!(
                    "\n(Reached end of input stream — concluding interactive session gracefully.)"
                );
                return Ok((correct, attempted));
            }

            let trimmed = line.trim().to_ascii_lowercase();
            if trimmed == "q" || trimmed == "quit" {
                println!("\n👋 Exiting Lemma Interactive Academy. Keep building!");
                return Ok((correct, attempted));
            }

            attempted += 1;
            let parsed_choice = trimmed.parse::<usize>().ok();
            if parsed_choice == Some(q.correct_index + 1) {
                correct += 1;
                println!("✅ Correct! {}", q.explanation);
            } else {
                println!(
                    "💡 Diagnostic Note: The strongest answer is ({}) — {}",
                    q.correct_index + 1,
                    q.options[q.correct_index]
                );
                println!("   Why: {}", q.explanation);
            }
        }

        println!("\n📌 Key Takeaways for Phase {}:", lesson.phase);
        for takeaway in phase.key_takeaways() {
            println!("   ✓ {takeaway}");
        }
        println!();
    }

    if attempted > 0 {
        let pct = (correct as f64 / attempted as f64) * 100.0;
        println!("🏆 Session Complete! Mastery Score: {correct}/{attempted} ({pct:.0}%)");
    } else {
        println!("🏆 Session Complete!");
    }

    Ok((correct, attempted))
}

enum PromptAction {
    Continue,
    SkipPhase,
    Quit,
}

fn prompt_continue<R: BufRead>(reader: &mut R, prompt: &str) -> Result<PromptAction> {
    print!("{prompt}");
    let _ = io::stdout().flush();
    let mut buf = String::new();
    let n = reader.read_line(&mut buf)?;
    if n == 0 {
        println!("\n(End of input stream — finishing session.)");
        return Ok(PromptAction::Quit);
    }
    let cmd = buf.trim().to_ascii_lowercase();
    if cmd == "q" || cmd == "quit" {
        println!("\n👋 Exiting Lemma Interactive Academy.");
        return Ok(PromptAction::Quit);
    }
    if cmd == "s" || cmd == "skip" {
        println!("⏭️  Skipping to next phase...\n");
        return Ok(PromptAction::SkipPhase);
    }
    Ok(PromptAction::Continue)
}
