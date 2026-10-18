pub mod attest;
pub mod cache;
pub mod calibrate;
pub mod cascade;
pub mod classifier;
pub mod conformal;
pub mod consensus;
pub mod evaluator;
pub mod experiment;
pub mod explain;
pub mod features;
pub mod guardrail;
pub mod hierarchical;
pub mod learn;
pub mod learn_visuals;
pub mod metrics;
pub mod model_calibration;
pub mod ood;
pub mod probe;
pub mod shortlist;
pub mod slicing;
pub mod stats;
pub mod synthesis;

pub use attest::{
    AttestationEngine, AttestationVerificationReport, DecisionReceipt, RECEIPT_VERSION,
    schema_digest,
};
pub use cache::{
    CacheFileSummary, CacheStats, DecisionCache, cache_dir, cache_summary, clear_cache,
    compute_schema_hash, fingerprint,
};
pub use calibrate::{
    BucketCalibrationReport, CalibratedAgentConfig, CalibrationLoss, CalibrationMetadata,
    CalibrationReport, CalibrationTuner, LabelBiasFit, QatsProfile, fit_label_bias,
    parse_cost_weights,
};
pub use cascade::{
    CascadeConfig, CascadeDecision, CascadeRouter, DeferFit, DeferPoint, DeferPolicy, RISK_GRID,
    SystemTier, binomial_lower_bound,
};
pub use classifier::{ClassifierConfig, LayaClassifier};
pub use conformal::{ConformalEngine, ConformalReport};
pub use consensus::{ConsensusEngine, ConsensusResult, ConsensusStatus};
pub use evaluator::{BaselineComparison, Evaluator};
pub use experiment::{ArenaDisagreement, ExperimentRunner, ModelArenaEntry, ModelArenaReport};
pub use explain::AttributionEngine;
pub use guardrail::{AuditRecord, GuardAction, GuardrailEngine, GuardrailPolicy, PolicyRule};
pub use hierarchical::{
    HierarchicalBenchResult, HierarchicalDecisionResult, HierarchicalRouter, HierarchicalStep,
};
pub use learn::{
    CurriculumPhase, GlossaryEntry, LiveLabOutput, PhaseLesson, QuizQuestion, ReadingResource,
    SAMPLE_FILES, WorkflowRecipe, build_all_quiz_questions, build_glossary, build_phase_lesson,
    build_reading_list, build_workflow_recipes, print_all_phases, print_curriculum_overview,
    print_glossary, print_phase_lesson, print_reading_list, print_workflow_guide,
    run_interactive_session, run_interactive_session_with_reader, run_quiz_non_interactive,
    write_sample_files,
};
pub use learn_visuals::{
    c_blue, c_bold, c_cyan, c_dim, c_green, c_magenta, c_red, c_strike_red, c_yellow,
    colorize_ascii_diagram, colorize_math_block, pill_cyan, pill_green, pill_magenta, pill_red,
    pill_yellow, render_callout_card, render_phase_breadcrumb, render_phase_visual_demonstration,
    render_prob_bar, render_token_attribution_heatmap, strip_ansi_codes, styled_table,
};
pub use metrics::{
    EvaluationMetrics, calculate_brier_score, calculate_ece, resolve_ground_truth_for_question,
};
pub use model_calibration::{
    MODEL_CALIBRATION_FILENAME, apply_persisted_calibration, calibrate_and_persist_checkpoints,
    calibration_path_for, load_checkpoint_calibration, load_checkpoint_calibration_from,
    save_checkpoint_calibration, save_checkpoint_calibration_at,
};
pub use ood::{OodDetector, OodEvaluation, OodProfile, compute_free_energy};
pub use probe::{PROBE_KIND, Probe, choice_labels, fit_probe};
pub use shortlist::{ShortlistResult, expand_shortlisted_logits, shortlist_options};
pub use slicing::{
    ArenaSliceRow, SliceBy, SliceMetricEntry, compute_arena_slice_matrix, evaluate_slices,
    extract_item_slice, print_arena_slice_matrix, print_slice_report,
};
pub use stats::{
    AccuracyStats, ConfusedPair, CoveragePoint, LabelStats, PerformanceStats, QuestionStats,
    SpeedStats, StatsEngine, StatsEngineConfig, SystemStatsReport, UncertaintyStats,
};
pub use synthesis::{
    InferredSchemaResult, SchemaKnapsackOptimizer, SchemaOptimizationResult, SchemaSynthesizer,
};
