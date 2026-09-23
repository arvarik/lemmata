pub mod classifier;
pub mod conformal;
pub mod evaluator;
pub mod experiment;
pub mod explain;
pub mod metrics;

pub use classifier::{ClassifierConfig, LayaClassifier};
pub use conformal::{ConformalEngine, ConformalReport};
pub use evaluator::Evaluator;
pub use experiment::ExperimentRunner;
pub use explain::AttributionEngine;
pub use metrics::{EvaluationMetrics, calculate_brier_score, calculate_ece};
