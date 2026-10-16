use lemma::model::HeadParadigm;
use lemma::model::registry::{
    compute_dir_size_bytes, inspect_model_directory, print_models_inventory, scan_registered_models,
};
use std::path::Path;

fn write_minimal_safetensors(path: &Path, header_json: &str) {
    let header_bytes = header_json.as_bytes();
    let len_bytes = (header_bytes.len() as u64).to_le_bytes();
    let mut buf = Vec::with_capacity(8 + header_bytes.len());
    buf.extend_from_slice(&len_bytes);
    buf.extend_from_slice(header_bytes);
    std::fs::write(path, buf).expect("failed to write minimal safetensors");
}

#[test]
fn test_scan_all_seven_local_models_and_verify_paradigms() {
    let specs = scan_registered_models();
    assert_eq!(
        specs.len(),
        7,
        "Expected all 7 local checkpoints to be discovered"
    );

    print_models_inventory(&specs);

    // 1. Laya English (JointSinglePass)
    let laya_en = inspect_model_directory(Path::new("models/laya")).unwrap();
    assert_eq!(laya_en.model_id, "convaiinnovations/laya");
    assert_eq!(laya_en.detected_paradigm, HeadParadigm::JointSinglePass);
    assert_eq!(laya_en.max_position_embeddings, 8192);
    assert!(laya_en.total_parameters > 350_000_000);
    assert!(laya_en.disk_usage_bytes > 800_000_000);
    assert!(laya_en.is_valid);

    // 2. Laya Multilingual / mmBERT-base (JointSinglePass)
    let laya_multi = inspect_model_directory(Path::new("models/laya/multilingual")).unwrap();
    assert_eq!(laya_multi.model_id, "convaiinnovations/laya-multilingual");
    assert_eq!(laya_multi.detected_paradigm, HeadParadigm::JointSinglePass);
    assert_eq!(laya_multi.max_position_embeddings, 8192);
    assert_eq!(laya_multi.vocab_size, 256_000);
    assert!(laya_multi.total_parameters > 300_000_000);
    assert!(laya_multi.disk_usage_bytes > 600_000_000);
    assert!(laya_multi.is_valid);

    // 3. Laya Typed-Decisions (JointSinglePass)
    let laya_typed = inspect_model_directory(Path::new("models/laya/typed-decisions")).unwrap();
    assert_eq!(
        laya_typed.model_id,
        "convaiinnovations/laya-typed-decisions"
    );
    assert_eq!(laya_typed.detected_paradigm, HeadParadigm::JointSinglePass);
    assert_eq!(laya_typed.max_position_embeddings, 8192);
    assert!(laya_typed.total_parameters > 350_000_000);
    assert!(laya_typed.disk_usage_bytes > 800_000_000);
    assert!(laya_typed.is_valid);

    // Ensure non-recursive compute_dir_size_bytes on models/laya does not double-count subdirectories
    assert!(
        laya_en.disk_usage_bytes < 1_000_000_000,
        "models/laya disk_usage_bytes ({}) should not include multilingual + typed-decisions",
        laya_en.disk_usage_bytes
    );

    // 4. ModernBERT-large-zeroshot-v2.0 (PairwiseNliCrossEncoder)
    let modernbert = inspect_model_directory(Path::new("models/modernbert-zeroshot")).unwrap();
    assert_eq!(
        modernbert.model_id,
        "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
    );
    assert_eq!(modernbert.model_type, "modernbert");
    assert_eq!(
        modernbert.detected_paradigm,
        HeadParadigm::PairwiseNliCrossEncoder
    );
    assert_eq!(modernbert.max_position_embeddings, 8192);
    assert!(modernbert.total_parameters > 350_000_000);
    assert!(modernbert.disk_usage_bytes > 700_000_000);
    assert!(modernbert.is_valid);

    // 5. DeBERTa-v3-large-zeroshot-v2.0 (PairwiseNliCrossEncoder)
    let deberta = inspect_model_directory(Path::new("models/deberta-v3")).unwrap();
    assert_eq!(
        deberta.model_id,
        "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    );
    assert_eq!(deberta.model_type, "deberta-v2");
    assert_eq!(
        deberta.detected_paradigm,
        HeadParadigm::PairwiseNliCrossEncoder
    );
    assert_eq!(deberta.max_position_embeddings, 512);
    assert!(deberta.total_parameters > 400_000_000);
    assert!(deberta.disk_usage_bytes > 800_000_000);
    assert!(deberta.is_valid);

    // 6. EuroBERT-210m (GenerativeMlmVerbalizer)
    let eurobert = inspect_model_directory(Path::new("models/eurobert")).unwrap();
    assert_eq!(eurobert.model_id, "EuroBERT/EuroBERT-210m");
    assert_eq!(eurobert.model_type, "eurobert");
    assert_eq!(
        eurobert.detected_paradigm,
        HeadParadigm::GenerativeMlmVerbalizer
    );
    assert_eq!(eurobert.max_position_embeddings, 8192);
    assert!(eurobert.total_parameters > 200_000_000);
    assert!(eurobert.disk_usage_bytes > 800_000_000);
    assert!(eurobert.is_valid);

    // 7. NeoBERT (GenerativeMlmVerbalizer)
    let neobert = inspect_model_directory(Path::new("models/neobert")).unwrap();
    assert_eq!(neobert.model_id, "chandar-lab/NeoBERT");
    assert_eq!(neobert.model_type, "neobert");
    assert_eq!(
        neobert.detected_paradigm,
        HeadParadigm::GenerativeMlmVerbalizer
    );
    assert_eq!(neobert.max_position_embeddings, 4096);
    assert!(neobert.total_parameters > 200_000_000);
    assert!(neobert.disk_usage_bytes > 900_000_000);
    assert!(neobert.is_valid);
}

#[test]
fn test_synthetic_custom_gliclass_mmbert_directory() {
    let temp_dir = std::env::temp_dir().join(format!(
        "lemma_custom_gliclass_mmbert_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let config_json = r#"{
  "_name_or_path": "knowledgator/gliclass-mmbert-base-v2.0",
  "architectures": ["GLiClassModel"],
  "model_type": "GLiClass",
  "hidden_size": 768,
  "num_hidden_layers": 22,
  "num_attention_heads": 12,
  "max_position_embeddings": 8192,
  "vocab_size": 256000
}"#;
    std::fs::write(temp_dir.join("config.json"), config_json).unwrap();

    let gliclass_config_json = r#"{
  "encoder_model_name": "jhu-clsp/mmBERT-base",
  "scoring_type": "joint_single_pass"
}"#;
    std::fs::write(temp_dir.join("gliclass_config.json"), gliclass_config_json).unwrap();

    let safetensors_header = r#"{"encoder.tok_embeddings.weight":{"dtype":"BF16","shape":[256000,768],"data_offsets":[0,0]},"mask_scorer.weight":{"dtype":"BF16","shape":[768,768],"data_offsets":[0,0]}}"#;
    write_minimal_safetensors(&temp_dir.join("model.safetensors"), safetensors_header);

    let spec = inspect_model_directory(&temp_dir).unwrap();
    assert_eq!(spec.model_id, "knowledgator/gliclass-mmbert-base-v2.0");
    assert_eq!(spec.model_type, "GLiClass");
    assert_eq!(spec.detected_paradigm, HeadParadigm::JointSinglePass);
    assert_eq!(spec.max_position_embeddings, 8192);
    assert_eq!(spec.hidden_size, 768);
    assert_eq!(spec.num_hidden_layers, 22);
    assert_eq!(spec.num_attention_heads, 12);
    assert_eq!(spec.vocab_size, 256000);
    assert_eq!(spec.total_parameters, 256_000 * 768 + 768 * 768);
    assert_eq!(spec.disk_usage_bytes, compute_dir_size_bytes(&temp_dir));
    assert!(spec.disk_usage_bytes > 0);
    assert!(spec.is_valid);

    let _ = std::fs::remove_dir_all(&temp_dir);
}
