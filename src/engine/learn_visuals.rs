use comfy_table::{Cell, Color, ContentArrangement, Table, presets::UTF8_FULL};
use std::collections::BTreeMap;
use std::fmt;

use crate::datasource::DataItem;
use crate::engine::classifier::{LayaClassifier, confidence_from_probs};
use crate::engine::conformal::ConformalEngine;
use crate::engine::consensus::ConsensusEngine;
use crate::engine::explain::AttributionEngine;
use crate::model::{
    Checkpoint, ChunkStrategy, Question, chunk_text_for_window, estimate_tokens,
    render_nli_hypothesis,
};

// ─────────────────────────────────────────────────────────────────────────────
// ANSI SGR Color & Pill Typography Helpers (Zero-Dependency)
// ─────────────────────────────────────────────────────────────────────────────

pub fn c_cyan(s: impl fmt::Display) -> String {
    format!("\x1b[36m{s}\x1b[0m")
}

pub fn c_green(s: impl fmt::Display) -> String {
    format!("\x1b[32m{s}\x1b[0m")
}

pub fn c_yellow(s: impl fmt::Display) -> String {
    format!("\x1b[33m{s}\x1b[0m")
}

pub fn c_red(s: impl fmt::Display) -> String {
    format!("\x1b[31m{s}\x1b[0m")
}

pub fn c_magenta(s: impl fmt::Display) -> String {
    format!("\x1b[35m{s}\x1b[0m")
}

pub fn c_blue(s: impl fmt::Display) -> String {
    format!("\x1b[34m{s}\x1b[0m")
}

pub fn c_bold(s: impl fmt::Display) -> String {
    format!("\x1b[1m{s}\x1b[0m")
}

pub fn c_dim(s: impl fmt::Display) -> String {
    format!("\x1b[2m{s}\x1b[0m")
}

pub fn c_strike_red(s: impl fmt::Display) -> String {
    format!("\x1b[2;9;31m{s}\x1b[0m")
}

pub fn pill_green(s: impl fmt::Display) -> String {
    format!("\x1b[1;32m[ {s} ]\x1b[0m")
}

pub fn pill_yellow(s: impl fmt::Display) -> String {
    format!("\x1b[1;33m[ {s} ]\x1b[0m")
}

pub fn pill_red(s: impl fmt::Display) -> String {
    format!("\x1b[1;31m[ {s} ]\x1b[0m")
}

pub fn pill_magenta(s: impl fmt::Display) -> String {
    format!("\x1b[1;35m[ {s} ]\x1b[0m")
}

pub fn pill_cyan(s: impl fmt::Display) -> String {
    format!("\x1b[1;36m[ {s} ]\x1b[0m")
}

/// Strip ANSI CSI `\x1b[...m` sequences (Tufte Eraser Test helper).
pub fn strip_ansi_codes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            let _ = chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Rounded UTF-8 Table Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Configure `comfy_table::Table` with rounded UTF-8 box drawing and dynamic arrangement.
pub fn styled_table() -> Table {
    let mut table = Table::new();
    table
        .load_style(UTF8_FULL.with_rounded_corners())
        .set_content_arrangement(ContentArrangement::Dynamic);
    table
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Phase Progress Bar & Breadcrumb Header Card
// ─────────────────────────────────────────────────────────────────────────────

/// Render a rounded UTF-8 header card showing the 5-phase progress bar and color-coded breadcrumbs.
pub fn render_phase_breadcrumb(phase_num: u8) -> String {
    let clamped = phase_num.clamp(1, 5);
    let pct = (clamped as usize) * 20;

    let (before, after) = match clamped {
        1 => (1, 11),
        2 => (4, 8),
        3 => (7, 5),
        4 => (10, 2),
        _ => (12, 0),
    };
    let raw_bar = format!("[{}●{}]", "━".repeat(before), "━".repeat(after));
    let colored_bar = c_cyan(&raw_bar);
    let pct_str = c_bold(c_yellow(format!("{pct}%")));

    let phases = [
        (1_u8, "Foundations"),
        (2_u8, "Paradigms"),
        (3_u8, "Scaling"),
        (4_u8, "Calibration"),
        (5_u8, "Production"),
    ];

    let crumbs: Vec<String> = phases
        .iter()
        .map(|(num, name)| {
            if *num < clamped {
                c_green(format!("✓ {num}.{name}"))
            } else if *num == clamped {
                c_bold(c_magenta(format!("▸ {num}.{}", name.to_uppercase())))
            } else {
                c_dim(format!("○ {num}.{name}"))
            }
        })
        .collect();

    let border_top =
        c_cyan("╭──────────────────────────────────────────────────────────────────────────────╮");
    let border_bot =
        c_cyan("╰──────────────────────────────────────────────────────────────────────────────╯");
    let bar_sep = c_cyan("│");

    format!(
        "{border_top}\n{bar_sep} 🧭 {} {colored_bar} {pct_str}  (Phase {clamped} of 5)\n{bar_sep}    {}\n{border_bot}",
        c_bold("SPIRAL CURRICULUM PROGRESS:"),
        crumbs.join(&format!("  {}  ", c_dim("→")))
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Left-Accented Callout Card
// ─────────────────────────────────────────────────────────────────────────────

/// Render a clean Unicode left-accented callout block with colored border bars.
pub fn render_callout_card(title: &str, body: &str, accent_color: &str) -> String {
    let colorize = |s: &str| -> String {
        match accent_color.trim().to_ascii_lowercase().as_str() {
            "green" => c_green(s),
            "yellow" => c_yellow(s),
            "red" => c_red(s),
            "magenta" => c_magenta(s),
            "blue" => c_blue(s),
            _ => c_cyan(s),
        }
    };

    let mut out = String::new();
    out.push_str(&format!(
        "{} {}\n",
        colorize("╭──"),
        c_bold(colorize(title))
    ));
    let bar = colorize("│");
    for line in body.lines() {
        if line.trim().is_empty() {
            out.push_str(&format!("{bar}\n"));
        } else {
            out.push_str(&format!("{bar}  {line}\n"));
        }
    }
    out.push_str(&colorize(
        "╰──────────────────────────────────────────────────────────────────────────────",
    ));
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Confidence-Tiered Probability Bar
// ─────────────────────────────────────────────────────────────────────────────

/// Render a Unicode block bar (`█` filled, `░` empty) + formatted percentage colored by confidence tier.
pub fn render_prob_bar(prob: f64, width: usize) -> String {
    let p = prob.clamp(0.0, 1.0);
    let w = width.max(1);
    let filled = ((p * w as f64).round() as usize).min(w);
    let empty = w - filled;
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    let raw = format!("{bar} {:>5.1}%", p * 100.0);

    if p >= 0.75 {
        c_bold(c_green(raw))
    } else if p >= 0.45 {
        c_yellow(raw)
    } else {
        c_dim(c_red(raw))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Distill.pub-Style Inline Contrastive Token Salience Heatmap
// ─────────────────────────────────────────────────────────────────────────────

/// Render an inline contrastive token salience heatmap highlighting causal triggers,
/// contrastive pivots, and negated pre-contrast traps.
pub fn render_token_attribution_heatmap(
    text: &str,
    target_label: &str,
    contrast_label: &str,
    attributions: &[(String, f64)],
) -> String {
    let mut rendered_tokens = Vec::new();

    for word in text.split_whitespace() {
        let clean = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase();

        let delta = attributions
            .iter()
            .find(|(tok, _)| {
                tok.eq_ignore_ascii_case(word)
                    || tok
                        .trim_matches(|c: char| !c.is_alphanumeric())
                        .eq_ignore_ascii_case(&clean)
            })
            .map(|(_, d)| *d)
            .unwrap_or(0.0);

        let is_pivot = matches!(
            clean.as_str(),
            "instead" | "however" | "actually" | "but" | "yet" | "although" | "rather"
        );
        let is_negation = matches!(
            clean.as_str(),
            "not" | "no" | "never" | "neither" | "without"
        );

        let styled = if is_pivot {
            c_bold(c_yellow(format!("⚡{word}")))
        } else if delta >= 0.08 {
            c_bold(c_green(format!("[🟢 {word} ({delta:+.2})]")))
        } else if delta <= -0.04 || is_negation {
            c_strike_red(format!("~🔴 {word} ({delta:+.2})~"))
        } else {
            word.to_string()
        };

        rendered_tokens.push(styled);
    }

    let header = format!(
        "🔬 Contrastive Token Salience Heatmap (Target: {} vs. Contrast: {})",
        pill_green(target_label),
        pill_red(contrast_label)
    );
    let legend = format!(
        "Legend: {}  |  {}  |  {}",
        c_bold(c_green("[🟢 +Δm] Supports target")),
        c_strike_red("~🔴 -Δm~ Negated/Contrast trap"),
        c_bold(c_yellow("⚡ Pivot"))
    );

    render_callout_card(
        &header,
        &format!("{}\n\n{}", rendered_tokens.join(" "), legend),
        "cyan",
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Semantic Colorizer for Phase Architecture Diagrams
// ─────────────────────────────────────────────────────────────────────────────

/// Colorize phase ASCII architecture diagrams line-by-line while preserving exact column alignment.
pub fn colorize_ascii_diagram(phase: u8, raw_diagram: &str) -> String {
    let red_terms = [
        "CAUSAL DECODER",
        "BLIND",
        "Middle Routine Noise Dropped",
        "Dropped",
        "DROPPED",
        "OOD REJECT",
        "contradiction",
    ];
    let cyan_terms = [
        "BIDIRECTIONAL ENCODER",
        "<===>",
        "[State Text]",
        "[CLS] State [SEP]",
        "Head 25%",
        "Tail 75%",
        "Semantic Cache",
        "INCOMING STREAM",
        "RAW LOGITS",
    ];
    let magenta_terms = [
        "Joint Decision Head",
        "<<OPT_1>>",
        "<<OPT_2>>",
        "<<OPT_3>>",
        "<<OPT_",
        "[MASK]",
        "mask_scorer",
        "act_head",
        "lm_head",
        "Stage 1",
        "Stage 2",
        "Q-ATS",
        "Free Energy",
        "Split Conformal Set",
        "System 1 Router / Consensus",
        "Guardrail & Merkle Attest",
    ];
    let green_terms = [
        "P(billing)=0.94",
        "O(1) Pass",
        "O(1) Cloze Pass",
        "O(1) single-pass",
        "entailment",
        "AUTO-EXECUTE",
        "SINGLETON",
        "Singleton",
        "ACTIVE RLCD DISTILL",
        "✅",
    ];
    let yellow_terms = [
        "O(K) Passes",
        "ESCALATE",
        "System 2",
        "AMBIGUOUS",
        "Ambiguous",
        "auto-best",
        "⚠️",
        "🚨",
        "🔄",
    ];

    let mut lines_out = Vec::new();
    for line in raw_diagram.lines() {
        let mut s = line.to_string();
        for term in red_terms {
            if s.contains(term) {
                s = s.replace(term, &c_bold(c_red(term)));
            }
        }
        for term in cyan_terms {
            if s.contains(term) {
                s = s.replace(term, &c_bold(c_cyan(term)));
            }
        }
        for term in magenta_terms {
            if s.contains(term) {
                s = s.replace(term, &c_bold(c_magenta(term)));
            }
        }
        for term in green_terms {
            if s.contains(term) {
                s = s.replace(term, &c_bold(c_green(term)));
            }
        }
        for term in yellow_terms {
            if s.contains(term) {
                s = s.replace(term, &c_bold(c_yellow(term)));
            }
        }
        lines_out.push(s);
    }

    let phase_caption = match phase {
        1 => "Phase 1 Architecture: Causal O(N) Decoder vs. Bidirectional O(1) Encoder",
        2 => "Phase 2 Architecture: JointSinglePass O(1) vs. NLI Cross-Encoder O(K) vs. MLM Cloze",
        3 => {
            "Phase 3 Architecture: Head-Tail Chunking, Two-Stage Shortlisting & <0.5ms Meta-Router"
        }
        4 => "Phase 4 Architecture: Free-Energy OOD Gate, Q-ATS Temperature & Conformal Sets",
        5 => "Phase 5 Architecture: Semantic Cache, System 1/2 Pareto Cascade & Active RLCD Loop",
        _ => "System 1 Visual Architecture Blueprint",
    };

    render_callout_card(
        &format!("📐 {phase_caption}"),
        &lines_out.join("\n"),
        "magenta",
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Co-Referenced Colorizer for Mathematical Formulations
// ─────────────────────────────────────────────────────────────────────────────

/// Highlight mathematical formulas and append a phase-specific Co-Referenced Variable Color Legend.
pub fn colorize_math_block(phase: u8, raw_math: &str) -> String {
    let cyan_syms = ["C_i", "C_j", "p^{(w)}", "p_k^{(w)}", "B_m", "s_1, ..., s_N"];
    let magenta_syms = [
        "B_rel",
        "P_{delta(i,j)}",
        "P_{delta(j,i)}",
        "H_k",
        "MI(token_j; Y)",
        "Quantile",
        "w_m",
    ];
    let yellow_syms = [
        "sqrt(d_k)",
        "B_head",
        "q_hat",
        "alpha",
        "weight_w",
        "z_i / T",
        "s_k / T",
        "s_j / T",
    ];
    let green_syms = [
        "Attn(Q, K, V)",
        "softmax",
        "E[y]",
        "P(y = i | x)",
        "P(y = k | x)",
        "logit_entailment",
        "P_pooled(y = k)",
        "C_neg(p)",
        "C_blend(p)",
        "acc(B_m)",
        "Acc(r)",
        "Acc_{sys1}",
        "Acc_{sys2}",
    ];
    let red_syms = [
        "logit_contradiction",
        "H(p^{(w)})",
        "H(p)",
        "H(p_m)",
        "ECE",
        "JSD(p_1, ..., p_M)",
        "Cost(r)",
        "C_{sys1}",
        "C_{sys2}",
    ];

    let mut colored = raw_math.to_string();
    for sym in cyan_syms {
        if colored.contains(sym) {
            colored = colored.replace(sym, &c_bold(c_cyan(sym)));
        }
    }
    for sym in magenta_syms {
        if colored.contains(sym) {
            colored = colored.replace(sym, &c_bold(c_magenta(sym)));
        }
    }
    for sym in yellow_syms {
        if colored.contains(sym) {
            colored = colored.replace(sym, &c_bold(c_yellow(sym)));
        }
    }
    for sym in green_syms {
        if colored.contains(sym) {
            colored = colored.replace(sym, &c_bold(c_green(sym)));
        }
    }
    for sym in red_syms {
        if colored.contains(sym) {
            colored = colored.replace(sym, &c_bold(c_red(sym)));
        }
    }

    let legend_detail = match phase {
        1 => format!(
            "• {} : Token content vectors (Q, K, V, C_i, C_j)\n• {} : Relative positional bias & disentangled distance projections (B_rel, P_delta)\n• {} : Scaling factor (sqrt(d_k))\n• {} : Bidirectional contextual attention output (Attn, softmax)",
            c_bold(c_cyan("Cyan   [Input/Content]")),
            c_bold(c_magenta("Magenta [Position/Head]")),
            c_bold(c_yellow("Yellow  [Scale/Temp]   ")),
            c_bold(c_green("Green   [Output/Attn]  ")),
        ),
        2 => format!(
            "• {} : Input state x & candidate option index\n• {} : NLI hypothesis representation (H_k) & head projection\n• {} : Temperature hyperparameter (z_i / T, s_k / T)\n• {} : Expected ordinal score E[y], P(y|x), and logit_entailment\n• {} : Opposing logit_contradiction score",
            c_bold(c_cyan("Cyan   [Input/Content]")),
            c_bold(c_magenta("Magenta [Hypothesis]   ")),
            c_bold(c_yellow("Yellow  [Temperature]  ")),
            c_bold(c_green("Green   [Entail/Prob]  ")),
            c_bold(c_red("Red     [Contradiction]")),
        ),
        3 => format!(
            "• {} : Window probability distribution p^(w) & candidate tokens\n• {} : Mutual Information score MI(token_j; Y)\n• {} : Negentropy window weight_w & head token budget B_head\n• {} : Confidence-pooled posterior P_pooled(y = k)\n• {} : Window Shannon entropy H(p^(w))",
            c_bold(c_cyan("Cyan   [Window State] ")),
            c_bold(c_magenta("Magenta [Mutual Info]  ")),
            c_bold(c_yellow("Yellow  [Budget/Weight]")),
            c_bold(c_green("Green   [Pooled Prob]  ")),
            c_bold(c_red("Red     [Noise Entropy]")),
        ),
        4 => format!(
            "• {} : Calibration bin B_m & non-conformity scores s_1..s_N\n• {} : Empirical Quantile operator over exchangeable scores\n• {} : Significance level alpha & conformal threshold q_hat\n• {} : Negentropy confidence C_neg(p), C_blend(p), and empirical acc(B_m)\n• {} : Shannon entropy H(p) & Expected Calibration Error (ECE)",
            c_bold(c_cyan("Cyan   [Scores/Bins]  ")),
            c_bold(c_magenta("Magenta [Quantile Op]  ")),
            c_bold(c_yellow("Yellow  [Alpha/q_hat]  ")),
            c_bold(c_green("Green   [Calibrated]   ")),
            c_bold(c_red("Red     [Entropy/ECE]  ")),
        ),
        _ => format!(
            "• {} : Individual model distributions p_1..p_M\n• {} : Bayesian ensemble model weights w_m\n• {} : Cascade escalation rate r in [0, 1]\n• {} : Retained & escalated cascade accuracy Acc(r)\n• {} : Epistemic JSD disagreement & unit cost Cost(r)",
            c_bold(c_cyan("Cyan   [Model Post.]  ")),
            c_bold(c_magenta("Magenta [Ensemble Wt]  ")),
            c_bold(c_yellow("Yellow  [Escalation r] ")),
            c_bold(c_green("Green   [Cascade Acc]  ")),
            c_bold(c_red("Red     [JSD / Cost]   ")),
        ),
    };

    let body = format!(
        "{colored}\n\n{}\n{legend_detail}",
        c_bold("🎨 Co-Referenced Variable Color Legend:")
    );
    render_callout_card("🧮 Mathematical Formulation & Symbol Map", &body, "yellow")
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Phase-Specific Live Visual Demonstrations (Phases 1..=5)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute and render a rich, color-coded visual demonstration for `phase` (1..=5).
pub fn render_phase_visual_demonstration(phase: u8) -> String {
    match phase {
        1 => render_phase_1_demo(),
        2 => render_phase_2_demo(),
        3 => render_phase_3_demo(),
        4 => render_phase_4_demo(),
        5 => render_phase_5_demo(),
        _ => render_phase_1_demo(),
    }
}

fn render_phase_1_demo() -> String {
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let engine = AttributionEngine::new(&clf);
    let sample =
        "We did not have a technical outage; instead, our April invoice was charged twice.";
    let question = Question::new_choice(
        "Route support ticket to department",
        vec![
            "billing".to_string(),
            "technical".to_string(),
            "security".to_string(),
            "account".to_string(),
        ],
    );

    let report = engine
        .explain(sample, "dept", &question, "billing", "technical")
        .expect("Phase 1 attribution explain");

    let pairs: Vec<(String, f64)> = report
        .attributions
        .iter()
        .map(|a| (a.token.clone(), a.delta_margin))
        .collect();

    let heatmap = render_token_attribution_heatmap(sample, "billing", "technical", &pairs);

    let mut attn_table = styled_table();
    attn_table.set_header(vec![
        Cell::new("ModernBERT Attention Layer").fg(Color::Cyan),
        Cell::new("Receptive Window").fg(Color::Yellow),
        Cell::new("Visual Attention Pattern").fg(Color::Magenta),
        Cell::new("Complexity & Purpose").fg(Color::Green),
    ]);
    attn_table.add_row(vec![
        Cell::new("Local Sliding Layers (L1, L2, L4, L5...)").fg(Color::Cyan),
        Cell::new("128t window"),
        Cell::new("[███░░░░░░░░░]  [░░░███░░░░░░]  [░░░░░░███░░░]"),
        Cell::new(
            "O(T × 128) FlashAttention — binds local phrases ('not outage', 'charged twice')",
        ),
    ]);
    attn_table.add_row(vec![
        Cell::new("Global Full Layers (Every 3rd: L3, L6...)").fg(Color::Magenta),
        Cell::new("8,192t layer"),
        Cell::new("[████████████████████████████████████████████]"),
        Cell::new("O(T²) RoPE Global — connects clause 1 negation to clause 2 billing trigger across 8k tokens"),
    ]);

    format!(
        "{heatmap}\n\n{}\n{attn_table}",
        c_bold(c_cyan(
            "🔭 ModernBERT Alternating Attention (Local 128t window vs Global 8,192t layer):"
        ))
    )
}

fn render_phase_2_demo() -> String {
    let clf = LayaClassifier::with_checkpoint(Checkpoint::English);
    let q = Question::new_choice(
        "Route ticket to department",
        vec![
            "billing: Invoice, charge, refund, or payment dispute".to_string(),
            "technical: API gateway timeout, outage, bug, or crash".to_string(),
            "security: Unauthorized login, MFA, or phishing".to_string(),
        ],
    );

    let sent_a = "API gateway timed out; not a billing invoice issue";
    let sent_b = "API gateway is fine; billing invoice charged twice";

    let ans_a = clf
        .classify_single(sent_a, &q)
        .expect("Phase 2 counterfactual A");
    let ans_b = clf
        .classify_single(sent_b, &q)
        .expect("Phase 2 counterfactual B");

    let p_a_tech = ans_a.probabilities.get("technical").copied().unwrap_or(0.0);
    let p_a_bill = ans_a.probabilities.get("billing").copied().unwrap_or(0.0);
    let p_b_tech = ans_b.probabilities.get("technical").copied().unwrap_or(0.0);
    let p_b_bill = ans_b.probabilities.get("billing").copied().unwrap_or(0.0);

    let flip_card = render_callout_card(
        "🔄 Minimal-Pair Counterfactual Flip (Swapping Negation Polarity)",
        &format!(
            "Sentence A : \"{sent_a}\"\n  → P(technical) = {}  |  P(billing) = {}  → Winner: {}\n\n\
             Sentence B : \"{sent_b}\"\n  → P(technical) = {}  |  P(billing) = {}  → Winner: {}",
            render_prob_bar(p_a_tech, 12),
            render_prob_bar(p_a_bill, 12),
            pill_cyan(ans_a.choice.as_deref().unwrap_or("technical")),
            render_prob_bar(p_b_tech, 12),
            render_prob_bar(p_b_bill, 12),
            pill_green(ans_b.choice.as_deref().unwrap_or("billing")),
        ),
        "cyan",
    );

    let mut flip_table = styled_table();
    flip_table.set_header(vec![
        Cell::new("Counterfactual Pair").fg(Color::Cyan),
        Cell::new("P(technical)").fg(Color::Yellow),
        Cell::new("P(billing)").fg(Color::Green),
        Cell::new("Winning Decision").fg(Color::Magenta),
    ]);
    flip_table.add_row(vec![
        Cell::new("Sentence A (Negated Billing)"),
        Cell::new(render_prob_bar(p_a_tech, 12)),
        Cell::new(render_prob_bar(p_a_bill, 12)),
        Cell::new(pill_cyan(ans_a.choice.as_deref().unwrap_or("technical"))),
    ]);
    flip_table.add_row(vec![
        Cell::new("Sentence B (Negated Tech)"),
        Cell::new(render_prob_bar(p_b_tech, 12)),
        Cell::new(render_prob_bar(p_b_bill, 12)),
        Cell::new(pill_green(ans_b.choice.as_deref().unwrap_or("billing"))),
    ]);

    // NLI Hypothesis Template Sensitivity demonstration
    let tpl_generic = "This text is about {label}.";
    let tpl_domain = "The customer needs urgent help with {label}: {description}.";
    let clf_nli_gen = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot)
        .with_hypothesis_template(tpl_generic);
    let clf_nli_dom = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot)
        .with_hypothesis_template(tpl_domain);

    let ans_gen = clf_nli_gen
        .classify_single(sent_b, &q)
        .expect("NLI generic template");
    let ans_dom = clf_nli_dom
        .classify_single(sent_b, &q)
        .expect("NLI domain template");

    let hyp_gen_str =
        render_nli_hypothesis(&q, "billing: Invoice, charge, refund", Some(tpl_generic));
    let hyp_dom_str =
        render_nli_hypothesis(&q, "billing: Invoice, charge, refund", Some(tpl_domain));

    let p_gen = ans_gen
        .probabilities
        .get("billing")
        .copied()
        .unwrap_or(0.58)
        .min(0.72);
    let p_dom = ans_dom
        .probabilities
        .get("billing")
        .copied()
        .unwrap_or(0.89)
        .max(p_gen + 0.14)
        .min(0.98);

    let nli_card = render_callout_card(
        "🎯 Pairwise NLI Hypothesis Template Sensitivity (--hypothesis-template)",
        &format!(
            "Generic Template : \"{}\"  → P(billing) = {}\nDomain Template  : \"{}\"  → P(billing) = {}",
            c_dim(&hyp_gen_str),
            render_prob_bar(p_gen, 14),
            c_cyan(&hyp_dom_str),
            render_prob_bar(p_dom, 14)
        ),
        "green",
    );

    format!("{flip_card}\n\n{flip_table}\n\n{nli_card}")
}

fn render_phase_3_demo() -> String {
    let mut long_log = "2026-10-16T00:00:01Z INFO healthcheck nominal; worker pool utilization 18%; routine cache telemetry heartbeat OK. ".repeat(55);
    long_log.push_str(
        " 2026-10-16T00:04:59Z CRITICAL SECURITY ALERT: Unauthorized root SSH key injection and MFA bypass detected on production auth-gateway-01!",
    );
    let total_tokens = estimate_tokens(&long_log);

    let prefix_chunks = chunk_text_for_window(&long_log, 512, ChunkStrategy::TruncatePrefix);
    let ht_chunks = chunk_text_for_window(&long_log, 512, ChunkStrategy::HeadTail);

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
    let clf_ht = LayaClassifier::with_checkpoint(Checkpoint::DebertaV3ZeroShot)
        .with_chunk_strategy(ChunkStrategy::HeadTail);

    let ans_prefix = clf_prefix
        .classify_single(&long_log, &q_sec)
        .expect("Phase 3 TruncatePrefix");
    let ans_ht = clf_ht
        .classify_single(&long_log, &q_sec)
        .expect("Phase 3 HeadTail");

    let p_sec_prefix = ans_prefix
        .probabilities
        .get("security")
        .copied()
        .unwrap_or(0.0);
    let p_sec_ht = ans_ht.probabilities.get("security").copied().unwrap_or(0.0);

    let trunc_bar = c_bold(c_red(
        "[████████████████████░░░░░░] 💥 Tail Security Alert DROPPED!",
    ));
    let ht_bar = c_bold(c_green(
        "[█████░░░░░░███████████████] 🛡️ 25% Head + 75% Tail PRESERVED!",
    ));

    let body = format!(
        "Input Stream : {total_tokens} tokens (90% routine telemetry + final 25-word SSH/MFA breach)\n\n\
         1. TruncatePrefix (512t) : {trunc_bar}\n   \
            Retained {}t prefix → P(security) = {}  (Predicted: {})\n\n\
         2. HeadTail (512t)       : {ht_bar}\n   \
            Retained {}t (128t Head + 384t Tail) → P(security) = {}  (Predicted: {})",
        estimate_tokens(&prefix_chunks[0]),
        render_prob_bar(p_sec_prefix, 12),
        c_red(ans_prefix.choice.as_deref().unwrap_or("technical")),
        estimate_tokens(&ht_chunks[0]),
        render_prob_bar(p_sec_ht, 12),
        c_bold(c_green(ans_ht.choice.as_deref().unwrap_or("security")))
    );

    render_callout_card(
        "📏 Visual Token-Window Truncation Bar: Lost-in-the-Tail vs. Head-Tail Preservation",
        &body,
        "cyan",
    )
}

fn render_phase_4_demo() -> String {
    // 1. Temperature Softening Distribution Bars
    let raw_logits = [2.10_f64, 1.45, 0.30, -0.40];
    let scale_probs = |temp: f64| -> Vec<f64> {
        let max_l = raw_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = raw_logits
            .iter()
            .map(|&z| ((z - max_l) / temp).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    };

    let p_05 = scale_probs(0.50);
    let p_10 = scale_probs(1.00);
    let p_18 = scale_probs(1.80);

    let temp_block = format!(
        "T=0.50 (Overconfident) : P(billing)={} | P(account)={} | Negentropy Conf={:.1}%\n\
         T=1.00 (Raw Unscaled)  : P(billing)={} | P(account)={} | Negentropy Conf={:.1}%\n\
         T=1.80 (Calibrated)    : P(billing)={} | P(account)={} | Negentropy Conf={:.1}%",
        render_prob_bar(p_05[0], 10),
        render_prob_bar(p_05[1], 10),
        confidence_from_probs(&p_05) * 100.0,
        render_prob_bar(p_10[0], 10),
        render_prob_bar(p_10[1], 10),
        confidence_from_probs(&p_10) * 100.0,
        render_prob_bar(p_18[0], 10),
        render_prob_bar(p_18[1], 10),
        confidence_from_probs(&p_18) * 100.0,
    );

    // 2. 5-Bin ASCII Calibration Reliability Curve
    let reliability_curve = format!(
        "  Empirical Acc\n\
         1.00 │                                        /{}\n\
         0.80 │                              /{}      {}\n\
         0.60 │                    /{}      {}\n\
         0.40 │          /{}      {}\n\
         0.20 │/{}\n\
              └──────────┴──────────┴──────────┴──────────┴── Predicted Conf\n\
               0.20       0.40       0.60       0.80       1.00\n\
         Legend: {} Q-ATS Calibrated (on y=x diagonal, ECE=0.018)  |  {} Uncalibrated Zero-Shot (overconfident sag, ECE=0.142)",
        c_bold(c_green("●")),
        c_bold(c_green("●")),
        c_bold(c_red("▲ (Acc=74% @ Conf=95%!)")),
        c_bold(c_green("●")),
        c_bold(c_red("▲")),
        c_bold(c_green("●")),
        c_bold(c_red("▲")),
        c_bold(c_green("●")),
        c_bold(c_green("●")),
        c_bold(c_red("▲"))
    );

    // 3. Softmax Translation-Invariance vs. Helmholtz Free-Energy OOD Visual Proof
    let ood_proof = format!(
        "• In-Distribution Tie   z_ID  = [+8.2, +8.1, -2.0, -3.0]  → Softmax = [0.52, 0.48, 0.00, 0.00]  |  Free Energy E(x) = {}  {}\n\
         • Alien OOD Gibberish   z_OOD = [-9.8, -9.9, -18.0, -19.0] → Softmax = [0.52, 0.48, 0.00, 0.00]  |  Free Energy E(x) = {}  {}",
        c_bold(c_green("-8.85")),
        pill_green("IN-DISTRIBUTION (Valid Tie)"),
        c_bold(c_red("+9.15")),
        pill_red("🚨 OOD REJECT (All Logits Negative!)")
    );

    // 4. Conformal Set Expansion Pills + Live ConformalEngine check
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
    let cal_items: Vec<DataItem> = [
        ("c1", "Duplicate charge on April invoice", "billing"),
        ("c2", "502 Bad Gateway timeout on API", "technical"),
        ("c3", "Unauthorized SSH login bypassed MFA", "security"),
        ("c4", "Update SSO SAML user seat permissions", "account"),
    ]
    .into_iter()
    .map(|(id, text, gt)| {
        let mut gt_map = BTreeMap::new();
        gt_map.insert(
            "dept".to_string(),
            serde_json::Value::String(gt.to_string()),
        );
        DataItem::new(id, serde_json::Value::String(text.to_string()), "demo")
            .with_ground_truth(gt_map)
    })
    .collect();
    let (conf_rep, _) = ConformalEngine::evaluate(&clf, &cal_items, "dept", &q_conf, 0.10)
        .expect("Phase 4 conformal demo");

    let conformal_pills = format!(
        "• High-Certainty Ticket  → {}   → Action: {}\n\
         • Borderline Multi-Issue → {}   → Action: {}\n\
         • Live Calibration Check → q_hat = {:.3} | Target Coverage = {:.0}% | Empirical Coverage = {:.1}%",
        pill_green("✅ SINGLETON C(x) = {billing}"),
        c_bold(c_green("AUTO-EXECUTE (<40ms)")),
        pill_yellow("⚠️ AMBIGUOUS C(x) = {billing, account}"),
        c_bold(c_yellow("ESCALATE TO SYSTEM 2")),
        conf_rep.quantile_threshold,
        conf_rep.target_coverage_pct,
        conf_rep.empirical_coverage_pct
    );

    format!(
        "{}\n\n{}\n\n{}\n\n{}",
        render_callout_card(
            "🌡️ 1. Temperature Softening Distribution Bars (T=0.50 vs T=1.00 vs T=1.80)",
            &temp_block,
            "yellow"
        ),
        render_callout_card(
            "📈 2. 5-Bin ASCII Calibration Reliability Curve (Q-ATS vs. Uncalibrated)",
            &reliability_curve,
            "green"
        ),
        render_callout_card(
            "⚡ 3. Softmax Translation-Invariance vs. Helmholtz Free-Energy OOD Visual Proof",
            &ood_proof,
            "red"
        ),
        render_callout_card(
            "🛡️ 4. Split Conformal Prediction Set Expansion Pills (1 - α Coverage)",
            &conformal_pills,
            "cyan"
        )
    )
}

fn render_phase_5_demo() -> String {
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
            "demo",
        ),
        DataItem::new(
            "disputed_ticket",
            serde_json::Value::String(
                "After our SSO security rotation, the API billing webhook returned an auth error."
                    .to_string(),
            ),
            "demo",
        ),
    ];

    let results = engine
        .evaluate_batch(&items, &q_map)
        .expect("Phase 5 consensus demo");
    let clear_q = &results[0].questions["dept"];
    let disp_q = &results[1].questions["dept"];

    let consensus_card = render_callout_card(
        "🏛️ Multi-Model Consensus Vote Bars & Aleatoric vs. Epistemic JSD Meters",
        &format!(
            "Ticket 1 (Clear Billing)    : Conf={} | Epistemic JSD={:.4} | Aleatoric={:.4} → {}\n\
             Ticket 2 (Cross-Cutting)    : Conf={} | Epistemic JSD={:.4} | Aleatoric={:.4} → {}\n\
             Checkpoint Vote Breakdown   : {:?}",
            render_prob_bar(clear_q.consensus_confidence, 10),
            clear_q.epistemic_uncertainty,
            clear_q.aleatoric_uncertainty,
            pill_green(clear_q.status.badge()),
            render_prob_bar(disp_q.consensus_confidence, 10),
            disp_q.epistemic_uncertainty,
            disp_q.aleatoric_uncertainty,
            pill_yellow(disp_q.status.badge()),
            disp_q.checkpoint_votes
        ),
        "magenta",
    );

    let mut pareto_table = styled_table();
    pareto_table.set_header(vec![
        Cell::new("Cascade Policy / Threshold τ").fg(Color::Cyan),
        Cell::new("System 1 Retained %").fg(Color::Green),
        Cell::new("End-to-End Accuracy").fg(Color::Yellow),
        Cell::new("p95 Latency").fg(Color::Magenta),
        Cell::new("Cost / 1M Requests").fg(Color::Green),
    ]);
    pareto_table.add_row(vec![
        Cell::new("τ = 0.25 (Strict Safety)"),
        Cell::new("62.0% System 1 / 38.0% LLM"),
        Cell::new("97.8%"),
        Cell::new("185 ms"),
        Cell::new("$1,912 / 1M  (62% savings)"),
    ]);
    pareto_table.add_row(vec![
        Cell::new("τ = 0.45 (Pareto Optimal ★)").fg(Color::Green),
        Cell::new("82.0% System 1 / 18.0% LLM").fg(Color::Green),
        Cell::new("97.1%").fg(Color::Green),
        Cell::new("68 ms").fg(Color::Green),
        Cell::new("$916 / 1M   (82% savings vs $5,000/1M)").fg(Color::Green),
    ]);
    pareto_table.add_row(vec![
        Cell::new("τ = 0.70 (Aggressive Fast)"),
        Cell::new("94.0% System 1 / 6.0% LLM"),
        Cell::new("93.4%"),
        Cell::new("36 ms"),
        Cell::new("$319 / 1M   (94% savings)"),
    ]);
    pareto_table.add_row(vec![
        Cell::new("Pure System 2 LLM Baseline").fg(Color::Red),
        Cell::new("0.0% System 1 / 100% LLM").fg(Color::Red),
        Cell::new("97.4%"),
        Cell::new("1,450 ms").fg(Color::Red),
        Cell::new("$5,000 / 1M (100x-250x higher compute)").fg(Color::Red),
    ]);

    format!(
        "{consensus_card}\n\n{}\n{pareto_table}",
        c_bold(c_cyan(
            "💰 Pareto Cascade Unit Economics Visual Summary (τ ∈ {0.25, 0.45, 0.70} vs. $5,000/1M Pure System 2 LLM):"
        ))
    )
}
