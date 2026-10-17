use comfy_table::Cell;
use lemmata::engine::learn::{CurriculumPhase, build_phase_lesson};
use lemmata::engine::learn_visuals::{
    c_blue, c_bold, c_cyan, c_dim, c_green, c_magenta, c_red, c_strike_red, c_yellow,
    colorize_ascii_diagram, colorize_math_block, pill_cyan, pill_green, pill_magenta, pill_red,
    pill_yellow, render_callout_card, render_phase_breadcrumb, render_phase_visual_demonstration,
    render_prob_bar, render_token_attribution_heatmap, strip_ansi_codes, styled_table,
};

#[test]
fn test_ansi_color_and_pill_helpers() {
    let helpers = [
        c_cyan("cyan"),
        c_green("green"),
        c_yellow("yellow"),
        c_red("red"),
        c_magenta("magenta"),
        c_blue("blue"),
        c_bold("bold"),
        c_dim("dim"),
        c_strike_red("strike"),
        pill_green("OK"),
        pill_yellow("WARN"),
        pill_red("ERR"),
        pill_magenta("HEAD"),
        pill_cyan("INFO"),
    ];

    for h in &helpers {
        assert!(h.contains("\x1b["), "Expected ANSI sequence in {h}");
        assert!(h.ends_with("\x1b[0m"), "Expected ANSI reset in {h}");
    }

    assert_eq!(strip_ansi_codes(&pill_green("SINGLETON")), "[ SINGLETON ]");
}

#[test]
fn test_styled_table_and_callout_card() {
    let mut table = styled_table();
    table.set_header(vec![Cell::new("Col A"), Cell::new("Col B")]);
    table.add_row(vec![Cell::new("Val 1"), Cell::new("Val 2")]);
    let rendered = table.to_string();
    assert!(rendered.contains('╭') && rendered.contains('╰'));
    assert!(rendered.contains("Col A") && rendered.contains("Val 2"));

    for accent in ["cyan", "green", "yellow", "red", "magenta", "blue"] {
        let card = render_callout_card("Mental Model", "Line 1\nLine 2", accent);
        assert!(card.contains("\x1b["));
        let plain = strip_ansi_codes(&card);
        assert!(plain.contains("╭── Mental Model"));
        assert!(plain.contains("│  Line 1"));
        assert!(plain.contains("╰──"));
    }
}

#[test]
fn test_render_phase_breadcrumb_all_phases() {
    for p in 1..=5 {
        let crumb = render_phase_breadcrumb(p);
        assert!(crumb.contains("\x1b["));
        let plain = strip_ansi_codes(&crumb);
        assert!(plain.contains(&format!("{}%", (p as usize) * 20)));
        assert!(plain.contains(&format!("(Phase {p} of 5)")));
        if p == 2 {
            assert!(plain.contains("[━━━━●━━━━━━━━]"));
            assert!(plain.contains("✓ 1.Foundations"));
            assert!(plain.contains("▸ 2.PARADIGMS"));
            assert!(plain.contains("○ 3.Scaling"));
        }
    }
}

#[test]
fn test_render_prob_bar_confidence_tiers() {
    let high = render_prob_bar(0.842, 10);
    let mid = render_prob_bar(0.55, 10);
    let low = render_prob_bar(0.20, 10);

    assert!(high.contains("\x1b[32m") && high.contains("84.2%"));
    assert!(mid.contains("\x1b[33m") && mid.contains("55.0%"));
    assert!(low.contains("\x1b[31m") && low.contains("20.0%"));
    assert!(strip_ansi_codes(&high).contains('█'));
}

#[test]
fn test_render_token_attribution_heatmap_salience_and_legend() {
    let text = "We did not have a technical outage; instead, our April invoice was charged twice.";
    let attributions = vec![
        ("not".to_string(), -0.05),
        ("outage;".to_string(), -0.18),
        ("instead,".to_string(), 0.04),
        ("invoice".to_string(), 0.42),
        ("charged".to_string(), 0.31),
    ];

    let heatmap = render_token_attribution_heatmap(text, "billing", "technical", &attributions);
    assert!(heatmap.contains("\x1b["));
    let plain = strip_ansi_codes(&heatmap);
    assert!(plain.contains("⚡instead,"));
    assert!(plain.contains("[🟢 invoice (+0.42)]"));
    assert!(plain.contains("[🟢 charged (+0.31)]"));
    assert!(plain.contains("~🔴 not (-0.05)~"));
    assert!(plain.contains("~🔴 outage; (-0.18)~"));
    assert!(plain.contains("[🟢 +Δm] Supports target"));
    assert!(plain.contains("~🔴 -Δm~ Negated/Contrast trap"));
    assert!(plain.contains("⚡ Pivot"));
}

#[test]
fn test_colorize_ascii_diagram_and_math_block_all_phases() {
    for phase in CurriculumPhase::all() {
        let p_num = phase.number();
        let lesson = build_phase_lesson(phase);

        let colored_diag = colorize_ascii_diagram(p_num, &lesson.ascii_diagram);
        assert!(
            colored_diag.contains("\x1b["),
            "Phase {p_num} diagram missing ANSI"
        );
        let plain_diag = strip_ansi_codes(&colored_diag);
        assert!(plain_diag.contains(&format!("Phase {p_num} Architecture")));

        let colored_math = colorize_math_block(p_num, &lesson.under_the_hood_math);
        assert!(
            colored_math.contains("\x1b["),
            "Phase {p_num} math missing ANSI"
        );
        let plain_math = strip_ansi_codes(&colored_math);
        assert!(plain_math.contains("🎨 Co-Referenced Variable Color Legend:"));
    }
}

#[test]
fn test_render_phase_visual_demonstration_all_phases() {
    let d1 = render_phase_visual_demonstration(1);
    assert!(d1.contains("\x1b["));
    let p1 = strip_ansi_codes(&d1);
    assert!(p1.contains("Contrastive Token Salience Heatmap"));
    assert!(p1.contains("ModernBERT Alternating Attention"));
    assert!(p1.contains("128t window") && p1.contains("8,192t layer"));

    let d2 = render_phase_visual_demonstration(2);
    assert!(d2.contains("\x1b["));
    let p2 = strip_ansi_codes(&d2);
    assert!(p2.contains("Minimal-Pair Counterfactual Flip"));
    assert!(p2.contains("API gateway timed out; not a billing invoice issue"));
    assert!(p2.contains("Pairwise NLI Hypothesis Template Sensitivity"));

    let d3 = render_phase_visual_demonstration(3);
    assert!(d3.contains("\x1b["));
    let p3 = strip_ansi_codes(&d3);
    assert!(p3.contains("TruncatePrefix (512t)"));
    assert!(p3.contains("Tail Security Alert DROPPED!"));
    assert!(p3.contains("HeadTail (512t)"));
    assert!(p3.contains("25% Head + 75% Tail PRESERVED!"));

    let d4 = render_phase_visual_demonstration(4);
    assert!(d4.contains("\x1b["));
    let p4 = strip_ansi_codes(&d4);
    assert!(p4.contains("T=0.50") && p4.contains("T=1.00") && p4.contains("T=1.80"));
    assert!(p4.contains("5-Bin ASCII Calibration Reliability Curve"));
    assert!(p4.contains("-8.85") && p4.contains("+9.15"));
    assert!(p4.contains("SINGLETON C(x) = {billing}"));
    assert!(p4.contains("AMBIGUOUS C(x) = {billing, account}"));

    let d5 = render_phase_visual_demonstration(5);
    assert!(d5.contains("\x1b["));
    let p5 = strip_ansi_codes(&d5);
    assert!(p5.contains("Multi-Model Consensus Vote Bars & Aleatoric vs. Epistemic JSD Meters"));
    assert!(p5.contains("Pareto Cascade Unit Economics Visual Summary"));
    assert!(p5.contains("0.25") && p5.contains("0.45") && p5.contains("0.70"));
    assert!(p5.contains("$5,000 / 1M"));
}
