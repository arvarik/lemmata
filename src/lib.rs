pub mod benchmark;
pub mod cli;
pub mod datasource;
pub mod engine;
pub mod model;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use anyhow::{Context, Result, bail};

use datasource::{DataItem, DataSource, DirectorySource, FileSource, MemorySource, StdinSource};
use model::{Question, QuestionType};

/// Load question definitions from a JSON file.
/// Format follows Laya / Jev questions dict:
/// {
///   "department": { "type": "choice", "instructions": "...", "criteria": { ... } },
///   "urgency": { "type": "score", "instructions": "...", "criteria": [ ... ] },
///   "churn_risk": { "type": "noul", "instructions": "..." }
/// }
pub fn load_questions_from_file(path: &Path) -> Result<BTreeMap<String, Question>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open questions schema file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let map: BTreeMap<String, Question> = serde_json::from_reader(reader)
        .with_context(|| format!("Malformed questions schema in: {}", path.display()))?;
    Ok(map)
}

/// Create a single default inline question for CLI usage.
pub fn create_inline_question(
    qtype_str: &str,
    instructions: &str,
    labels_csv: &str,
) -> Result<BTreeMap<String, Question>> {
    let mut map = BTreeMap::new();
    let qtype = match qtype_str.to_lowercase().as_str() {
        "choice" => QuestionType::Choice,
        "score" => QuestionType::Score,
        "noul" | "boolean" | "bool" => QuestionType::Noul,
        other => bail!("Invalid question type '{}'. Options: choice, score, noul", other),
    };

    let q = match qtype {
        QuestionType::Choice => {
            let labels: Vec<String> = labels_csv
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if labels.len() < 2 {
                bail!("Choice questions require at least 2 comma-separated labels");
            }
            Question::new_choice(instructions, labels)
        }
        QuestionType::Score => {
            let levels: Vec<String> = labels_csv
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            Question::new_score(instructions, levels)
        }
        QuestionType::Noul => Question::new_noul(instructions),
    };

    map.insert("classification".to_string(), q);
    Ok(map)
}

/// Resolve the appropriate DataSource based on CLI arguments.
pub fn resolve_data_source(
    data_dir: &Path,
    file: Option<&Path>,
    is_stdin: bool,
    inline_text: Option<&str>,
) -> Result<Box<dyn DataSource>> {
    if let Some(text) = inline_text {
        let item = DataItem::new("cli_input", serde_json::json!({ "text": text }), "cli_arg");
        return Ok(Box::new(MemorySource::new("InlineTextSource", vec![item])));
    }

    if is_stdin {
        return Ok(Box::new(StdinSource::new()));
    }

    if let Some(f) = file {
        return Ok(Box::new(FileSource::new(f)));
    }

    // Default: ingest from data directory
    Ok(Box::new(DirectorySource::new(data_dir)))
}
