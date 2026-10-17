use std::io::Cursor;
use std::str::FromStr;

use lemmata::engine::learn::{
    CurriculumPhase, build_all_quiz_questions, build_glossary, build_phase_lesson,
    build_reading_list, build_workflow_recipes, print_curriculum_overview, print_glossary,
    print_phase_lesson, print_reading_list, print_workflow_guide,
    run_interactive_session_with_reader, run_quiz_non_interactive,
};

#[test]
fn test_curriculum_phase_parsing_and_metadata() {
    let phase1_aliases = [
        "1",
        "phase1",
        "phase-1",
        "foundations",
        "encoders",
        "system1",
        "bidirectional",
        "modernbert",
    ];
    for alias in phase1_aliases {
        let p = CurriculumPhase::from_str(alias).expect("should parse phase 1 alias");
        assert_eq!(p, CurriculumPhase::Phase1Foundations);
        assert_eq!(p.number(), 1);
        assert_eq!(p.slug(), "foundations");
    }

    let phase2_aliases = [
        "2",
        "phase2",
        "paradigms",
        "heads",
        "nli",
        "mlm",
        "typed",
        "verbalizer",
    ];
    for alias in phase2_aliases {
        let p = CurriculumPhase::from_str(alias).expect("should parse phase 2 alias");
        assert_eq!(p, CurriculumPhase::Phase2HeadParadigms);
        assert_eq!(p.number(), 2);
        assert_eq!(p.slug(), "paradigms");
    }

    let phase3_aliases = [
        "3",
        "phase3",
        "routing",
        "chunking",
        "shortlist",
        "scaling",
        "context",
    ];
    for alias in phase3_aliases {
        let p = CurriculumPhase::from_str(alias).expect("should parse phase 3 alias");
        assert_eq!(p, CurriculumPhase::Phase3ScalingContextAndTaxonomies);
        assert_eq!(p.number(), 3);
        assert_eq!(p.slug(), "scaling");
    }

    let phase4_aliases = [
        "4",
        "phase4",
        "calibration",
        "conformal",
        "uncertainty",
        "ood",
        "qats",
    ];
    for alias in phase4_aliases {
        let p = CurriculumPhase::from_str(alias).expect("should parse phase 4 alias");
        assert_eq!(p, CurriculumPhase::Phase4CalibrationAndUncertainty);
        assert_eq!(p.number(), 4);
        assert_eq!(p.slug(), "calibration");
    }

    let phase5_aliases = [
        "5",
        "phase5",
        "production",
        "cascade",
        "consensus",
        "drift",
        "active",
    ];
    for alias in phase5_aliases {
        let p = CurriculumPhase::from_str(alias).expect("should parse phase 5 alias");
        assert_eq!(p, CurriculumPhase::Phase5ProductionCascadingAndOps);
        assert_eq!(p.number(), 5);
        assert_eq!(p.slug(), "production");
    }

    assert!(CurriculumPhase::from_str("invalid_phase_99").is_err());

    for p in CurriculumPhase::all() {
        assert!(!p.title().is_empty());
        assert!(!p.subtitle().is_empty());
        assert_eq!(p.key_takeaways().len(), 3);
    }
}

#[test]
fn test_build_all_five_phase_lessons_and_live_labs() {
    for phase in CurriculumPhase::all() {
        let lesson = build_phase_lesson(phase);
        assert_eq!(lesson.phase, phase.number());
        assert_eq!(lesson.slug, phase.slug());
        assert_eq!(lesson.title, phase.title());
        assert!(!lesson.mental_model.is_empty());
        assert!(!lesson.ascii_diagram.is_empty());
        assert!(lesson.core_concepts.len() >= 3);
        assert!(
            lesson.deep_intuitions.len() >= 2,
            "Phase {} should include at least 2 deep pedagogical intuitions",
            lesson.phase
        );
        assert!(
            lesson.further_reading.len() >= 3,
            "Phase {} should include at least 3 curated reading resources",
            lesson.phase
        );
        assert!(!lesson.under_the_hood_math.is_empty());
        assert!(!lesson.live_lab.title.is_empty());
        assert!(!lesson.live_lab.explanation.is_empty());
        assert!(
            lesson.live_lab.metrics_summary.len() >= 5,
            "Phase {} live lab should compute at least 5 runtime metrics",
            lesson.phase
        );
        assert!(lesson.live_lab.try_command.starts_with("lemmata "));
        assert_eq!(lesson.quiz.len(), 3);
        for q in &lesson.quiz {
            assert_eq!(q.phase, phase.number());
            assert_eq!(q.options.len(), 4);
            assert!(q.correct_index < q.options.len());
            assert!(!q.explanation.is_empty());
        }
        assert!(!lesson.next_step_commands.is_empty());
    }

    // Verify Phase 1 Contrastive Attribution Top Driver metric
    let p1 = build_phase_lesson(CurriculumPhase::Phase1Foundations);
    assert!(
        p1.live_lab
            .metrics_summary
            .iter()
            .any(|(k, _)| k == "Contrastive Attribution Top Driver")
    );

    // Verify Phase 2 Minimal-Pair Counterfactual Flip metric
    let p2 = build_phase_lesson(CurriculumPhase::Phase2HeadParadigms);
    let cf_metric = p2
        .live_lab
        .metrics_summary
        .iter()
        .find(|(k, _)| k == "Minimal-Pair Counterfactual Flip")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert!(
        cf_metric.contains("technical") && cf_metric.contains("billing"),
        "Counterfactual flip should demonstrate technical vs billing flip, got: {cf_metric}"
    );

    // Verify Phase 3 HeadTail vs TruncatePrefix live lab captures the tail security alert + Knapsack compression
    let p3 = build_phase_lesson(CurriculumPhase::Phase3ScalingContextAndTaxonomies);
    let ht_metric = p3
        .live_lab
        .metrics_summary
        .iter()
        .find(|(k, _)| k.contains("HeadTail"))
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert!(
        ht_metric.contains("pred=security"),
        "HeadTail live lab should recover the tail security alert, got: {ht_metric}"
    );
    assert!(
        p3.live_lab
            .metrics_summary
            .iter()
            .any(|(k, _)| k == "Knapsack Schema Compression")
    );

    // Verify Phase 4 Free-Energy Translation Proof (ID vs OOD)
    let p4 = build_phase_lesson(CurriculumPhase::Phase4CalibrationAndUncertainty);
    assert!(
        p4.live_lab
            .metrics_summary
            .iter()
            .any(|(k, _)| k == "Free-Energy Translation Proof (ID vs OOD)")
    );

    // Verify Phase 5 Cascade Unit Economics (80% Sys1 Retention)
    let p5 = build_phase_lesson(CurriculumPhase::Phase5ProductionCascadingAndOps);
    assert!(
        p5.live_lab.metrics_summary.iter().any(|(k, v)| k
            == "Cascade Unit Economics (80% Sys1 Retention)"
            && v.contains("$1,016"))
    );
}

#[test]
fn test_glossary_and_workflow_recipes_and_quizzes() {
    let glossary = build_glossary();
    assert!(glossary.len() >= 15);
    assert!(
        glossary
            .iter()
            .any(|g| g.term.contains("Conformal") && g.cli_flag_or_cmd.contains("conformal"))
    );
    assert!(
        glossary
            .iter()
            .any(|g| g.term.contains("Q-ATS") && g.cli_flag_or_cmd.contains("calibrate"))
    );

    let recipes = build_workflow_recipes();
    assert_eq!(recipes.len(), 5);
    let slugs: Vec<&str> = recipes.iter().map(|r| r.slug.as_str()).collect();
    assert_eq!(
        slugs,
        vec![
            "quickstart",
            "arena",
            "calibration",
            "scaling",
            "production"
        ]
    );

    let quizzes = build_all_quiz_questions();
    assert_eq!(quizzes.len(), 15);

    let reading = build_reading_list(None);
    assert!(reading.len() >= 18);
    for p in 1..=5 {
        let p_reading = build_reading_list(Some(p));
        assert!(p_reading.len() >= 3);
    }

    // Smoke-test terminal printing helpers
    print_curriculum_overview();
    print_phase_lesson(CurriculumPhase::Phase1Foundations);
    print_glossary(Some("conformal"));
    print_reading_list(None);
    print_reading_list(Some(1));
    print_workflow_guide(Some("calibration"));
    run_quiz_non_interactive(Some(2));
}

#[test]
fn test_interactive_session_handles_eof_and_graded_answers() {
    // 1. Immediate EOF does not hang
    let mut empty_input = Cursor::new(b"" as &[u8]);
    let (correct_eof, attempted_eof) = run_interactive_session_with_reader(
        &mut empty_input,
        Some(CurriculumPhase::Phase1Foundations),
        false,
    )
    .expect("EOF session should succeed gracefully");
    assert_eq!(correct_eof, 0);
    assert_eq!(attempted_eof, 0);

    // 2. Simulated quiz_only session for Phase 5 (answers: option 2, option 2, option 2)
    let mut simulated_input = Cursor::new(b"2\n2\n2\n" as &[u8]);
    let (correct, attempted) = run_interactive_session_with_reader(
        &mut simulated_input,
        Some(CurriculumPhase::Phase5ProductionCascadingAndOps),
        true,
    )
    .expect("Quiz session should succeed");
    assert_eq!(attempted, 3);
    assert_eq!(correct, 3);
}
