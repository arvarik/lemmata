use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use serde_json::json;

use lemmata::datasource::{DataItem, DataSource};
use lemmata::engine::{ClassifierConfig, LayaClassifier, fingerprint};
use lemmata::finetune::{
    CudaBackend, FinetuneMethod, PrepareOptions, TrainedInputsFilter, VerifyArtifactOptions,
    guard_eval_against_training_leakage, merge_finetuned_adapter, peft_defaults_for,
    prepare_bundle, pull_artifacts, push_bundle, verify_artifact,
};
use lemmata::model::native::LayaModel;
use lemmata::model::{
    Checkpoint, ModelSpec, PeftBackend, Question, QuestionType, TRAINING_MANIFEST_VERSION,
    merge_adapter_to_dir, safetensors_header_sha256, sha256_file,
};

fn unique_temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lemmata_ft_test_{}_{}_{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "mismatch at index {i}: got {g}, want {w} (diff {})",
            (g - w).abs()
        );
    }
}

#[test]
fn peft_backend_lora_rslora_dora_embedding_and_modules_to_save_parity() {
    let tmp = unique_temp_dir("peft_math");
    let base_dir = tmp.join("base");
    let adapter_dir = tmp.join("adapter");
    std::fs::create_dir_all(&base_dir).unwrap();
    std::fs::create_dir_all(&adapter_dir).unwrap();

    let dev = Device::Cpu;

    // 1. Base tensors:
    // - linear1.weight (3x4): standard rsLoRA target
    // - linear2.weight (2x3): DoRA target (`lora_magnitude_vector` + `alpha_pattern` override)
    // - tok_emb.weight (5x4): Embedding LoRA (`lora_embedding_A/B`)
    // - scorer.1.weight (2x4) & scorer.1.bias (2): `modules_to_save` override
    let w1_data: Vec<f32> = vec![
        1.0, -0.5, 0.25, 2.0, //
        -1.0, 0.75, 1.5, -0.25, //
        0.5, 0.5, -1.25, 0.0,
    ];
    let w2_data: Vec<f32> = vec![
        0.6, -0.8, 0.0, //
        -1.2, 0.5, 0.0,
    ];
    let emb_data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let scorer_base: Vec<f32> = vec![0.1; 8];
    let scorer_bias_base: Vec<f32> = vec![0.0, 0.0];

    let mut base_tensors = HashMap::new();
    base_tensors.insert(
        "encoder.layers.0.attn.Wqkv.weight".to_string(),
        Tensor::from_vec(w1_data.clone(), (3, 4), &dev).unwrap(),
    );
    base_tensors.insert(
        "encoder.layers.0.mlp.Wo.weight".to_string(),
        Tensor::from_vec(w2_data.clone(), (2, 3), &dev).unwrap(),
    );
    base_tensors.insert(
        "encoder.embeddings.tok_embeddings.weight".to_string(),
        Tensor::from_vec(emb_data.clone(), (5, 4), &dev).unwrap(),
    );
    base_tensors.insert(
        "scorer.1.weight".to_string(),
        Tensor::from_vec(scorer_base, (2, 4), &dev).unwrap(),
    );
    base_tensors.insert(
        "scorer.1.bias".to_string(),
        Tensor::from_vec(scorer_bias_base, 2, &dev).unwrap(),
    );
    base_tensors.insert(
        "act_head.weight".to_string(),
        Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap(),
    );
    candle_core::safetensors::save(&base_tensors, base_dir.join("model.safetensors")).unwrap();

    // 2. Adapter tensors (rank r=4 so sqrt(4)=2.0):
    // Global `lora_alpha = 8.0`, `use_rslora = true` -> scale = 8.0 / sqrt(4) = 4.0
    // Override `alpha_pattern` for `mlp.Wo`: `lora_alpha = 6.0` -> scale = 6.0 / sqrt(4) = 3.0
    let a1: Vec<f32> = (0..16).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect(); // (4, 4)
    let b1: Vec<f32> = (0..12).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect(); // (3, 4)

    let a2: Vec<f32> = (0..12).map(|i| (i as f32 - 5.0) * 0.05).collect(); // (4, 3)
    let b2: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 0.1).collect(); // (2, 4)
    let dora_mag: Vec<f32> = vec![2.5, 0.75]; // (2,)

    // Embedding LoRA (r=2): lora_embedding_A is (2, 5), lora_embedding_B is (4, 2); scale = 8 / sqrt(2)
    let emb_a: Vec<f32> = (0..10).map(|i| (i as f32) * 0.03).collect();
    let emb_b: Vec<f32> = (0..8).map(|i| (i as f32 - 2.0) * 0.04).collect();

    let scorer_trained: Vec<f32> = vec![0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2];
    let scorer_bias_trained: Vec<f32> = vec![0.42, -0.17];

    let mut ad_tensors = HashMap::new();
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_A.weight".to_string(),
        Tensor::from_vec(a1.clone(), (4, 4), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_B.weight".to_string(),
        Tensor::from_vec(b1.clone(), (3, 4), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.mlp.Wo.lora_A.default.weight".to_string(),
        Tensor::from_vec(a2.clone(), (4, 3), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.mlp.Wo.lora_B.default.weight".to_string(),
        Tensor::from_vec(b2.clone(), (2, 4), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.mlp.Wo.lora_magnitude_vector".to_string(),
        Tensor::from_vec(dora_mag.clone(), 2, &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.embeddings.tok_embeddings.lora_embedding_A".to_string(),
        Tensor::from_vec(emb_a.clone(), (2, 5), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.embeddings.tok_embeddings.lora_embedding_B".to_string(),
        Tensor::from_vec(emb_b.clone(), (4, 2), &dev).unwrap(),
    );
    // Test both `.modules_to_save.default.` infix and `.original_module.` shadow ignoring!
    ad_tensors.insert(
        "base_model.model.scorer.1.modules_to_save.default.weight".to_string(),
        Tensor::from_vec(scorer_trained.clone(), (2, 4), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.scorer.1.original_module.weight".to_string(),
        Tensor::from_vec(vec![999.0f32; 8], (2, 4), &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.scorer.1.bias".to_string(),
        Tensor::from_vec(scorer_bias_trained.clone(), 2, &dev).unwrap(),
    );
    // Optional ignored Laya `act_head` tensor should NOT fail `verify_all_consumed()`
    ad_tensors.insert(
        "base_model.model.act_head.weight".to_string(),
        Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &dev).unwrap(),
    );
    candle_core::safetensors::save(&ad_tensors, adapter_dir.join("adapter_model.safetensors"))
        .unwrap();

    let adapter_cfg = json!({
        "peft_type": "LORA",
        "base_model_name_or_path": "convaiinnovations/laya",
        "r": 4,
        "lora_alpha": 8.0,
        "use_rslora": true,
        "use_dora": false,
        "target_modules": ["attn.Wqkv", "mlp.Wo", "tok_embeddings"],
        "modules_to_save": ["scorer"],
        "alpha_pattern": {
            "mlp.Wo": 6.0
        }
    });
    std::fs::write(
        adapter_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_cfg).unwrap(),
    )
    .unwrap();

    let peft =
        Arc::new(PeftBackend::open(&base_dir.join("model.safetensors"), &adapter_dir).unwrap());
    let vb = peft.into_var_builder(DType::F32, &dev);

    // Check 1: rsLoRA linear (`scale = 8.0 / sqrt(4) = 4.0`)
    let got_w1 = vb
        .get((3, 4), "encoder.layers.0.attn.Wqkv.weight")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut want_w1 = vec![0.0f32; 12];
    for i in 0..3 {
        for j in 0..4 {
            let mut ba = 0.0f32;
            for k in 0..4 {
                ba += b1[i * 4 + k] * a1[k * 4 + j];
            }
            want_w1[i * 4 + j] = w1_data[i * 4 + j] + 4.0 * ba;
        }
    }
    assert_close(&got_w1, &want_w1, 1e-6);

    // Check 2: DoRA linear (`scale = 6.0 / sqrt(4) = 3.0`, row-wise L2 norm scaling by `dora_mag`)
    let got_w2 = vb
        .get((2, 3), "encoder.layers.0.mlp.Wo.weight")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut want_w2 = vec![0.0f32; 6];
    for i in 0..2 {
        let mut row = [0.0f32; 3];
        for j in 0..3 {
            let mut ba = 0.0f32;
            for k in 0..4 {
                ba += b2[i * 4 + k] * a2[k * 3 + j];
            }
            row[j] = w2_data[i * 3 + j] + 3.0 * ba;
        }
        let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        for j in 0..3 {
            want_w2[i * 3 + j] = dora_mag[i] * row[j] / norm;
        }
    }
    assert_close(&got_w2, &want_w2, 1e-6);

    // Check 3: Embedding rsLoRA (`scale = 8.0 / sqrt(2)`, `W + scale * A^T @ B^T`)
    let got_emb = vb
        .get((5, 4), "encoder.embeddings.tok_embeddings.weight")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scale_emb = (8.0f64 / 2.0f64.sqrt()) as f32;
    let mut want_emb = vec![0.0f32; 20];
    for v in 0..5 {
        for d in 0..4 {
            let mut ab = 0.0f32;
            for r in 0..2 {
                // emb_a is (2, 5), emb_b is (4, 2)
                ab += emb_a[r * 5 + v] * emb_b[d * 2 + r];
            }
            want_emb[v * 4 + d] = emb_data[v * 4 + d] + scale_emb * ab;
        }
    }
    assert_close(&got_emb, &want_emb, 1e-6);

    // Check 4: `modules_to_save` override (must use `modules_to_save.default`, NOT `original_module`)
    let got_scorer = vb
        .get((2, 4), "scorer.1.weight")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&got_scorer, &scorer_trained, 1e-6);
    let got_bias = vb
        .get(2, "scorer.1.bias")
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&got_bias, &scorer_bias_trained, 1e-6);

    // Check 5: All consumed (ignoring `act_head.weight`)
    peft.verify_all_consumed().unwrap();

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn peft_verify_all_consumed_rejects_mismatched_architecture() {
    let tmp = unique_temp_dir("peft_mismatch");
    let base_dir = tmp.join("base");
    let adapter_dir = tmp.join("adapter");
    std::fs::create_dir_all(&base_dir).unwrap();
    std::fs::create_dir_all(&adapter_dir).unwrap();

    let dev = Device::Cpu;
    let mut base_tensors = HashMap::new();
    base_tensors.insert(
        "model.layers.0.attn.Wqkv.weight".to_string(),
        Tensor::zeros((12, 4), DType::F32, &dev).unwrap(),
    );
    candle_core::safetensors::save(&base_tensors, base_dir.join("model.safetensors")).unwrap();

    // Adapter targets DeBERTa `query_proj`, which ModernBERT never queries!
    let mut ad_tensors = HashMap::new();
    ad_tensors.insert(
        "base_model.model.deberta.encoder.layer.0.attention.self.query_proj.lora_A.weight"
            .to_string(),
        Tensor::ones((2, 4), DType::F32, &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.deberta.encoder.layer.0.attention.self.query_proj.lora_B.weight"
            .to_string(),
        Tensor::ones((4, 2), DType::F32, &dev).unwrap(),
    );
    candle_core::safetensors::save(&ad_tensors, adapter_dir.join("adapter_model.safetensors"))
        .unwrap();
    std::fs::write(
        adapter_dir.join("adapter_config.json"),
        r#"{"peft_type": "LORA", "r": 2, "lora_alpha": 4.0}"#,
    )
    .unwrap();

    let peft =
        Arc::new(PeftBackend::open(&base_dir.join("model.safetensors"), &adapter_dir).unwrap());
    let vb = peft.into_var_builder(DType::F32, &dev);
    let _ = vb.get((12, 4), "model.layers.0.attn.Wqkv.weight").unwrap();

    let err = peft.verify_all_consumed().unwrap_err().to_string();
    assert!(
        err.contains("Adapter architecture mismatch"),
        "unexpected error: {err}"
    );
    assert!(err.contains("query_proj"));

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Build a tiny self-contained synthetic Laya (ModernBERT + Decision Head) checkpoint directory
/// so we can test real end-to-end Candle inference, live `--adapter` folding, and `finetune merge`
/// offline without downloading 800 MB weights.
fn write_tiny_laya_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir.join("encoder")).unwrap();
    std::fs::create_dir_all(dir.join("tokenizer")).unwrap();

    let rl_cfg = json!({
        "head_layers": 1,
        "max_len": 64,
        "head_max_len": 32,
        "temperature_by_options": {
            "choice:2": 1.95
        }
    });
    std::fs::write(
        dir.join("rl_agent_config.json"),
        serde_json::to_string_pretty(&rl_cfg).unwrap(),
    )
    .unwrap();

    let d = 64usize;
    let ff = 128usize;
    let vocab = 32usize;
    let enc_cfg = json!({
        "model_type": "modernbert",
        "vocab_size": vocab,
        "hidden_size": d,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": ff,
        "max_position_embeddings": 64,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "global_attn_every_n_layers": 1,
        "global_rope_theta": 10000.0,
        "local_attention": 16,
        "local_rope_theta": 10000.0
    });
    std::fs::write(
        dir.join("encoder/config.json"),
        serde_json::to_string_pretty(&enc_cfg).unwrap(),
    )
    .unwrap();

    let tok_cfg = json!({
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "pad_token": "[PAD]"
    });
    std::fs::write(
        dir.join("tokenizer/tokenizer_config.json"),
        serde_json::to_string_pretty(&tok_cfg).unwrap(),
    )
    .unwrap();

    let mut vocab_map = serde_json::Map::new();
    for (tok, id) in [
        ("[PAD]", 0),
        ("[UNK]", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[MASK]", 4),
        ("choice", 5),
        ("question:", 6),
        ("classify", 7),
        ("billing", 8),
        ("technical", 9),
        ("refund", 10),
        ("invoice", 11),
        ("crash", 12),
    ] {
        vocab_map.insert(tok.to_string(), json!(id));
    }
    for id in 13..vocab {
        vocab_map.insert(format!("tok_{id}"), json!(id));
    }
    let tok_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": { "type": "Lowercase" },
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_map,
            "unk_token": "[UNK]"
        }
    });
    std::fs::write(
        dir.join("tokenizer/tokenizer.json"),
        serde_json::to_string_pretty(&tok_json).unwrap(),
    )
    .unwrap();

    let dev = Device::Cpu;
    let det = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 31 + seed * 17) % 97) as f32 - 48.0) * 0.01)
            .collect()
    };

    let mut ts = HashMap::new();
    ts.insert(
        "encoder.embeddings.tok_embeddings.weight".to_string(),
        Tensor::from_vec(det(vocab * d, 1), (vocab, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.embeddings.norm.weight".to_string(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.attn.Wqkv.weight".to_string(),
        Tensor::from_vec(det(3 * d * d, 2), (3 * d, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.attn.Wo.weight".to_string(),
        Tensor::from_vec(det(d * d, 3), (d, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.mlp_norm.weight".to_string(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.mlp.Wi.weight".to_string(),
        Tensor::from_vec(det(2 * ff * d, 4), (2 * ff, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.mlp.Wo.weight".to_string(),
        Tensor::from_vec(det(d * ff, 5), (d, ff), &dev).unwrap(),
    );
    ts.insert(
        "encoder.final_norm.weight".to_string(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "type_emb.weight".to_string(),
        Tensor::from_vec(det(3 * d, 6), (3, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm1.weight".to_string(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm1.bias".to_string(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.in_proj_weight".to_string(),
        Tensor::from_vec(det(3 * d * d, 7), (3 * d, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.in_proj_bias".to_string(),
        Tensor::zeros(3 * d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.out_proj.weight".to_string(),
        Tensor::from_vec(det(d * d, 8), (d, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.out_proj.bias".to_string(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm2.weight".to_string(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm2.bias".to_string(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear1.weight".to_string(),
        Tensor::from_vec(det(4 * d * d, 9), (4 * d, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear1.bias".to_string(),
        Tensor::zeros(4 * d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear2.weight".to_string(),
        Tensor::from_vec(det(d * 4 * d, 10), (d, 4 * d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear2.bias".to_string(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.0.weight".to_string(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.0.bias".to_string(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.1.weight".to_string(),
        Tensor::from_vec(det(d * d, 11), (d, d), &dev).unwrap(),
    );
    ts.insert(
        "scorer.1.bias".to_string(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.3.weight".to_string(),
        Tensor::from_vec(det(d, 12), (1, d), &dev).unwrap(),
    );
    ts.insert(
        "scorer.3.bias".to_string(),
        Tensor::zeros(1, DType::F32, &dev).unwrap(),
    );

    candle_core::safetensors::save(&ts, dir.join("model.safetensors")).unwrap();
}

#[test]
fn laya_live_peft_adapter_matches_merged_checkpoint_and_resets_temperatures() {
    let tmp = unique_temp_dir("laya_merge_roundtrip");
    let base_dir = tmp.join("tiny_laya");
    let adapter_dir = tmp.join("adapter");
    let merged_dir = tmp.join("merged");
    write_tiny_laya_checkpoint(&base_dir);
    std::fs::create_dir_all(&adapter_dir).unwrap();

    let d = 64usize;
    let r = 4usize;
    let dev = Device::Cpu;
    let det = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 23 + seed * 13) % 89) as f32 - 44.0) * 0.05)
            .collect()
    };

    let mut ad = HashMap::new();
    ad.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_A.weight".to_string(),
        Tensor::from_vec(det(r * d, 1), (r, d), &dev).unwrap(),
    );
    ad.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_B.weight".to_string(),
        Tensor::from_vec(det(3 * d * r, 2), (3 * d, r), &dev).unwrap(),
    );
    ad.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_magnitude_vector".to_string(),
        Tensor::from_vec(vec![1.25f32; 3 * d], 3 * d, &dev).unwrap(),
    );
    ad.insert(
        "base_model.model.scorer.3.weight".to_string(),
        Tensor::from_vec(det(d, 9), (1, d), &dev).unwrap(),
    );
    ad.insert(
        "base_model.model.scorer.3.bias".to_string(),
        Tensor::from_vec(vec![0.35f32], 1, &dev).unwrap(),
    );
    candle_core::safetensors::save(&ad, adapter_dir.join("adapter_model.safetensors")).unwrap();

    let ad_cfg = json!({
        "peft_type": "DORA",
        "base_model_name_or_path": "convaiinnovations/laya",
        "r": r,
        "lora_alpha": 8.0,
        "use_rslora": true,
        "use_dora": true,
        "target_modules": ["attn.Wqkv"],
        "modules_to_save": ["scorer"]
    });
    std::fs::write(
        adapter_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&ad_cfg).unwrap(),
    )
    .unwrap();

    let options = vec!["billing".to_string(), "technical".to_string()];
    let base_model = LayaModel::load(&base_dir).unwrap();
    let base_logits = base_model
        .option_logits("refund invoice", QuestionType::Choice, "classify", &options)
        .unwrap();

    let adapted_model = LayaModel::load_with_adapter(&base_dir, Some(&adapter_dir)).unwrap();
    let adapted_logits = adapted_model
        .option_logits("refund invoice", QuestionType::Choice, "classify", &options)
        .unwrap();

    // Ensure the adapter actually changed the model's logits!
    assert!(
        (base_logits[0] - adapted_logits[0]).abs() > 1e-3
            || (base_logits[1] - adapted_logits[1]).abs() > 1e-3,
        "adapter should change logits: base={base_logits:?}, adapted={adapted_logits:?}"
    );

    // Merge via `merge_adapter_to_dir` and verify exact logit parity (<1e-6) with live `--adapter`!
    let rep = merge_adapter_to_dir(&base_dir, &adapter_dir, &merged_dir, true).unwrap();
    assert_eq!(rep.lora_tensors_merged, 1);
    assert_eq!(rep.override_tensors_applied, 2);

    let merged_model = LayaModel::load(&merged_dir).unwrap();
    let merged_logits = merged_model
        .option_logits("refund invoice", QuestionType::Choice, "classify", &options)
        .unwrap();
    for (g, w) in merged_logits.iter().zip(&adapted_logits) {
        assert!(
            (g - w).abs() < 1e-5,
            "merged vs live-adapter logit mismatch: {merged_logits:?} vs {adapted_logits:?}"
        );
    }

    // Verify `ModelSpec` + `ClassifierConfig::with_model_spec` resets stale zero-shot temperatures
    // (`base_dir/rl_agent_config.json` had `choice:2 = 1.95`, which must be stripped on merge and
    // ignored on adapted ModelSpec!) and changes cache fingerprint.
    let spec_base = ModelSpec::resolve(base_dir.to_str().unwrap(), None).unwrap();
    let spec_adapted = ModelSpec::resolve(base_dir.to_str().unwrap(), Some(&adapter_dir)).unwrap();
    let spec_merged = ModelSpec::resolve(merged_dir.to_str().unwrap(), None).unwrap();

    let cfg_adapted = ClassifierConfig::with_model_spec(&spec_adapted);
    assert!(
        cfg_adapted.temperature_by_options.is_empty(),
        "adapted model must not inherit zero-shot bucket temperatures"
    );
    let cfg_merged = ClassifierConfig::with_model_spec(&spec_merged);
    assert!(
        cfg_merged.temperature_by_options.is_empty(),
        "merged checkpoint must strip stale base rl_agent_config temperatures"
    );

    let fp_base = fingerprint(&LayaClassifier::with_model_spec(&spec_base)).unwrap();
    let fp_adapted = fingerprint(&LayaClassifier::with_model_spec(&spec_adapted)).unwrap();
    assert_ne!(
        fp_base, fp_adapted,
        "cache fingerprint must change when an adapter is attached"
    );

    // Verify cryptographic `@<short_sha>` provenance in `display_model_id`, `LayaClassifier`,
    // `MODELS` `(base, adapter)` singleton cache, and signed `DecisionReceipt`.
    lemmata::model::native::set_backend(lemmata::model::native::Backend::Auto);
    let base_clf = LayaClassifier::with_model_spec(&spec_base);
    let adapted_clf = LayaClassifier::with_model_spec(&spec_adapted);
    let merged_clf = LayaClassifier::with_model_spec(&spec_merged);
    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".to_string(),
        Question::new_choice("classify", options.clone()),
    );
    let base_dec = base_clf
        .classify(&json!("refund invoice"), &questions)
        .unwrap();
    let decision = adapted_clf
        .classify(&json!("refund invoice"), &questions)
        .unwrap();
    let merged_dec = merged_clf
        .classify(&json!("refund invoice"), &questions)
        .unwrap();
    assert_ne!(
        base_dec.answers["dept"].logits, decision.answers["dept"].logits,
        "MODELS singleton must not collide base vs adapted model!"
    );
    assert_eq!(
        decision.answers["dept"].probabilities,
        merged_dec.answers["dept"].probabilities
    );
    assert!(
        decision.model.contains("+adapter:adapter@"),
        "DecisionResult.model must embed adapter @short_sha provenance: {}",
        decision.model
    );
    let mut attester = lemmata::engine::AttestationEngine::new("test_attest_secret");
    let item = DataItem::new("i1", json!("refund invoice"), "test");
    let receipt = attester
        .create_receipt(
            &item,
            &decision,
            None,
            lemmata::engine::GuardAction::Allow,
            &lemmata::engine::schema_digest(&questions).unwrap(),
        )
        .unwrap();
    assert_eq!(receipt.model_id, decision.model);
    assert_eq!(
        receipt.model_id,
        lemmata::engine::model_provenance_id(&spec_adapted).unwrap()
    );

    // Also test `merge_finetuned_adapter` wrapper
    let merged2 = tmp.join("merged2");
    let rep2 = merge_finetuned_adapter(
        &adapter_dir,
        Some(base_dir.to_str().unwrap()),
        &merged2,
        false,
    )
    .unwrap();
    assert_eq!(rep2.merged_header_sha256, rep.merged_header_sha256);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn finetune_prepare_train_py_validation_and_five_layer_verify_leakage_guards() {
    let tmp = unique_temp_dir("prepare_and_verify");
    let splits_dir = tmp.join("splits");
    let out_dir = tmp.join("run_v1");
    let base_dir = tmp.join("tiny_laya");
    std::fs::create_dir_all(&splits_dir).unwrap();
    write_tiny_laya_checkpoint(&base_dir);

    // Schema
    let mut schema = BTreeMap::new();
    schema.insert(
        "dept".to_string(),
        Question::new_choice(
            "Which team handles this ticket?",
            vec!["billing".to_string(), "technical".to_string()],
        ),
    );
    let schema_path = tmp.join("schema.json");
    std::fs::write(&schema_path, serde_json::to_string_pretty(&schema).unwrap()).unwrap();

    let write_items = |path: &Path, rows: &[(&str, &str, &str)]| {
        let mut lines = String::new();
        for (id, text, lbl) in rows {
            let mut gt = BTreeMap::new();
            gt.insert("dept".to_string(), json!(lbl));
            let item = DataItem::new(*id, json!(text), "synthetic").with_ground_truth(gt);
            lines.push_str(&serde_json::to_string(&item).unwrap());
            lines.push('\n');
        }
        std::fs::write(path, lines).unwrap();
    };

    write_items(
        &splits_dir.join("train.jsonl"),
        &[
            ("tr1", "I was billed twice on invoice #901", "billing"),
            ("tr2", "Production API crashes with 500 error", "technical"),
        ],
    );
    write_items(
        &splits_dir.join("val.jsonl"),
        &[("vl1", "Refund my duplicate payment please", "billing")],
    );
    write_items(
        &splits_dir.join("cal.jsonl"),
        &[("cl1", "Server timeout after kernel upgrade", "technical")],
    );
    write_items(
        &splits_dir.join("test.jsonl"),
        &[
            ("ts1", "Need a copy of March receipt", "billing"),
            ("ts2", "Database connection pool exhausted", "technical"),
        ],
    );

    // 1. Check NeoBERT & DeBERTa PEFT target modules accuracy
    let neo = peft_defaults_for(Checkpoint::NeoBert, false).unwrap();
    assert_eq!(neo.target_modules, vec!["qkv", "wo", "w12", "w3"]);
    assert!(neo.modules_to_save.is_empty());

    // 2. Prepare bundle (including tar.gz)
    let prep = prepare_bundle(&PrepareOptions {
        split_dir: Some(splits_dir.clone()),
        train_file: None,
        val_file: None,
        cal_file: None,
        forbid_test_file: None,
        schema_path: schema_path.clone(),
        checkpoint: Checkpoint::English,
        method: FinetuneMethod::Lora,
        rank: 16,
        alpha: 32.0,
        use_rslora: true,
        lora_dropout: 0.05,
        unfreeze_laya_head: false,
        epochs: 4,
        lr: None,
        batch_size: 8,
        grad_accum: 2,
        label_smoothing: 0.02,
        neftune_alpha: 5.0,
        rdrop_alpha: 0.0,
        multi_label: false,
        cuda: CudaBackend::Cu126,
        seed: 42,
        create_tar: true,
        out_dir: out_dir.clone(),
    })
    .unwrap();

    assert_eq!(prep.train_rows, 2);
    assert_eq!(prep.val_rows, 1);
    assert_eq!(prep.cal_rows, 1);
    assert!(prep.bundle_dir.join("train.py").exists());
    assert!(prep.bundle_dir.join("bundle_manifest.json").exists());
    assert!(prep.bundle_dir.join("RUN.md").exists());
    assert!(prep.tar_path.as_ref().unwrap().exists());
    assert!(
        !prep.bundle_dir.join("data/test.jsonl").exists(),
        "bundle must NEVER contain test.jsonl!"
    );

    // 3. Validate `train.py` via `python3 -m py_compile` AND `--dry-validate` cross-language parity!
    let train_py = prep.bundle_dir.join("train.py");
    let py_compile = Command::new("python3")
        .env("CUDA_VISIBLE_DEVICES", "")
        .args(["-m", "py_compile", train_py.to_str().unwrap()])
        .output()
        .expect("python3 must be available");
    assert!(
        py_compile.status.success(),
        "py_compile failed: {}",
        String::from_utf8_lossy(&py_compile.stderr)
    );

    let artifacts_dir = out_dir.join("artifacts");
    let dry_val = Command::new("python3")
        .env("CUDA_VISIBLE_DEVICES", "")
        .args([
            train_py.to_str().unwrap(),
            "--dry-validate",
            "--bundle-dir",
            prep.bundle_dir.to_str().unwrap(),
            "--artifacts-dir",
            artifacts_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        dry_val.status.success(),
        "train.py --dry-validate failed: {}",
        String::from_utf8_lossy(&dry_val.stderr)
    );

    // Verify byte-for-byte parity between Python-emitted `trained_inputs.sha256.bin` and Rust `TrainedInputsFilter`!
    let py_filter =
        TrainedInputsFilter::read(&artifacts_dir.join("trained_inputs.sha256.bin")).unwrap();
    assert_eq!(py_filter.exact_hashes.len(), 4);
    assert_eq!(py_filter.norm_hashes.len(), 4);

    // 4. Simulate completed adapter + calibration + manifest in `artifacts_dir`
    let dev = Device::Cpu;
    let mut ad_tensors = HashMap::new();
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_A.weight".to_string(),
        Tensor::ones((4, 64), DType::F32, &dev).unwrap(),
    );
    ad_tensors.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_B.weight".to_string(),
        Tensor::zeros((192, 4), DType::F32, &dev).unwrap(),
    );
    let ad_st_path = artifacts_dir.join("adapter_model.safetensors");
    candle_core::safetensors::save(&ad_tensors, &ad_st_path).unwrap();
    std::fs::write(
        artifacts_dir.join("adapter_config.json"),
        r#"{"peft_type": "LORA", "base_model_name_or_path": "convaiinnovations/laya", "r": 4, "lora_alpha": 8.0}"#,
    )
    .unwrap();
    let cal_path = artifacts_dir.join("lemmata_calibration.json");
    std::fs::write(
        &cal_path,
        serde_json::to_string_pretty(&json!({
            "checkpoint": "convaiinnovations/laya",
            "temperature": 1.0,
            "temperature_by_options": { "choice:2": 1.15 },
            "loss_function": "nll",
            "metadata": {
                "dataset": "cal",
                "samples_evaluated": 30,
                "raw_mean_ece": 0.12,
                "calibrated_mean_ece": 0.04,
                "improvement_pct": 66.7,
                "calibrated_at": "2026-11-17T00:00:00Z"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let bundle_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(prep.bundle_dir.join("bundle_manifest.json")).unwrap(),
    )
    .unwrap();
    let base_hdr = safetensors_header_sha256(&base_dir.join("model.safetensors")).unwrap();

    let manifest = json!({
        "manifest_version": TRAINING_MANIFEST_VERSION,
        "run_id": prep.run_id,
        "status": "completed",
        "created_by_lemmata_version": "0.1.0",
        "base_model": {
            "checkpoint_slug": "english",
            "repo_id": "convaiinnovations/laya",
            "paradigm": "laya_joint_decision",
            "remote_base_header_sha256": base_hdr,
            "max_len": 64,
            "head_max_len": 32
        },
        "schema": bundle_json["schema"],
        "dataset_provenance": {
            "train": bundle_json["splits"]["train"],
            "val": bundle_json["splits"]["val"],
            "cal": bundle_json["splits"]["cal"],
            "locked_test_guard": {
                "expected_test_file_sha256": bundle_json["expected_test_file_sha256"],
                "trained_inputs_filter_file": "trained_inputs.sha256.bin",
                "trained_unique_exact_hashes": 4,
                "trained_unique_norm_hashes": 4
            }
        },
        "method": bundle_json["method"],
        "hyperparameters": bundle_json["hyperparameters"],
        "results": {},
        "environment": {},
        "artifacts": {
            "adapter_safetensors_sha256": sha256_file(&ad_st_path).unwrap(),
            "adapter_bytes": std::fs::metadata(&ad_st_path).unwrap().len(),
            "calibration_sha256": sha256_file(&cal_path).unwrap()
        }
    });
    std::fs::write(
        artifacts_dir.join("training_manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    // Clean verification passes all 5 layers!
    let ok_rep = verify_artifact(&VerifyArtifactOptions {
        artifact_dir: artifacts_dir.clone(),
        split_dir: Some(splits_dir.clone()),
        test_file: None,
        schema_path: Some(schema_path.clone()),
        base_dir: Some(base_dir.clone()),
    })
    .unwrap();
    assert!(
        ok_rep.valid,
        "expected clean pass, got errors: {:?}",
        ok_rep.errors
    );
    assert!(ok_rep.weights_verified);
    assert_eq!(ok_rep.base_header_verified, Some(true));
    assert_eq!(ok_rep.schema_verified, Some(true));
    assert_eq!(ok_rep.split_hashes_verified, Some(true));
    assert!(ok_rep.row_leakage_checked);
    assert!(ok_rep.leaked_item_ids.is_empty());

    // Passing `val.jsonl` as `--test-file` MUST fail Layer 4 & Layer 5!
    let split_leak_rep = verify_artifact(&VerifyArtifactOptions {
        artifact_dir: artifacts_dir.clone(),
        split_dir: None,
        test_file: Some(splits_dir.join("val.jsonl")),
        schema_path: Some(schema_path.clone()),
        base_dir: Some(base_dir.clone()),
    })
    .unwrap();
    assert!(!split_leak_rep.valid);
    assert_eq!(split_leak_rep.split_hashes_verified, Some(false));

    // Passing a whitespace/case-tweaked training item in a new test file MUST fail Layer 5 normalized check!
    let sneaky_test = tmp.join("sneaky_test.jsonl");
    write_items(
        &sneaky_test,
        &[
            ("clean_1", "Fresh unseen test message", "billing"),
            (
                "sneaky_1",
                "  I WAS   BILLED TWICE on invoice #901  ",
                "billing",
            ),
        ],
    );
    let row_leak_rep = verify_artifact(&VerifyArtifactOptions {
        artifact_dir: artifacts_dir.clone(),
        split_dir: None,
        test_file: Some(sneaky_test.clone()),
        schema_path: Some(schema_path.clone()),
        base_dir: Some(base_dir.clone()),
    })
    .unwrap();
    assert!(!row_leak_rep.valid);
    assert_eq!(row_leak_rep.leaked_item_ids, vec!["sneaky_1 (normalized)"]);

    // `guard_eval_against_training_leakage` blocks evaluation by default, and permits only with `allow_leakage = true`
    let spec = ModelSpec::resolve(base_dir.to_str().unwrap(), Some(&artifacts_dir)).unwrap();
    let sneaky_items = lemmata::datasource::FileSource::new(&sneaky_test)
        .read_items()
        .unwrap();
    assert!(
        guard_eval_against_training_leakage(
            &spec,
            Some(&sneaky_test),
            &sneaky_items,
            &schema,
            false
        )
        .is_err()
    );
    assert!(
        guard_eval_against_training_leakage(
            &spec,
            Some(&sneaky_test),
            &sneaky_items,
            &schema,
            true
        )
        .is_ok()
    );

    // 5. `push` & `pull --dry-run` checks (SSH rsync + S3)
    let push_ssh = push_bundle(&out_dir, "ubuntu@gpu-box:~/runs/r1", true).unwrap();
    assert_eq!(push_ssh.program, "rsync");
    assert!(push_ssh.args.iter().any(|a| a.contains("test.jsonl")));

    let pull_s3 = pull_artifacts(&out_dir, "s3://my-bucket/lemmata/r1", true).unwrap();
    assert_eq!(pull_s3.program, "aws");
    assert!(pull_s3.args.iter().any(|a| a == "checkpoints/*"));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn regression_preserve_order_state_parity_merge_force_and_nli_max_len() {
    let tmp = unique_temp_dir("reg_pr2_verifier");

    // 1. Bug #3: Non-alphabetical multi-field JSON `state` (`{"z_key":..., "a_key":...}`) parity
    // across `DataItem::state_text()`, `finetune`, `train.py`, and `paired_eval`.
    let splits_dir = tmp.join("splits");
    std::fs::create_dir_all(&splits_dir).unwrap();
    let schema_path = tmp.join("schema.json");
    let mut schema = BTreeMap::new();
    schema.insert(
        "dept".to_string(),
        Question::new_choice("Classify", vec!["billing".into(), "technical".into()]),
    );
    std::fs::write(&schema_path, serde_json::to_string_pretty(&schema).unwrap()).unwrap();

    // Write raw JSONL with key `"z_key"` strictly BEFORE `"a_key"`!
    std::fs::write(
        splits_dir.join("train.jsonl"),
        "{\"id\":\"tr1\",\"state\":{\"z_key\":\"refund\",\"a_key\":\"invoice\"},\"ground_truth\":{\"dept\":\"billing\"},\"source_uri\":\"raw\"}\n",
    )
    .unwrap();
    std::fs::write(
        splits_dir.join("val.jsonl"),
        "{\"id\":\"vl1\",\"state\":{\"z_key\":\"crash\",\"a_key\":\"server\"},\"ground_truth\":{\"dept\":\"technical\"},\"source_uri\":\"raw\"}\n",
    )
    .unwrap();

    let nli_out = tmp.join("nli_bundle");
    let prep_nli = prepare_bundle(&PrepareOptions {
        split_dir: Some(splits_dir.clone()),
        train_file: None,
        val_file: None,
        cal_file: None,
        forbid_test_file: None,
        schema_path: schema_path.clone(),
        checkpoint: Checkpoint::ModernBertZeroShot,
        method: FinetuneMethod::Lora,
        rank: 8,
        alpha: 16.0,
        use_rslora: true,
        lora_dropout: 0.05,
        unfreeze_laya_head: false,
        epochs: 2,
        lr: None,
        batch_size: 4,
        grad_accum: 1,
        label_smoothing: 0.0,
        neftune_alpha: 0.0,
        rdrop_alpha: 0.0,
        multi_label: false,
        cuda: CudaBackend::Cpu,
        seed: 7,
        create_tar: false,
        out_dir: nli_out.clone(),
    })
    .unwrap();

    // Bug #5a: `ModernBertZeroShot` pairwise NLI default `max_len` must be capped to 512 (not 8192).
    let bm_val: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(prep_nli.bundle_dir.join("bundle_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(bm_val["base_model"]["max_len"], 512);

    // Bug #5b: `train.py` strips stale `temperature` & `temperature_by_options` from copied `rl_agent_config.json`.
    let train_py_src = std::fs::read_to_string(prep_nli.bundle_dir.join("train.py")).unwrap();
    assert!(train_py_src.contains("rl_obj.pop(\"temperature\", None)"));
    assert!(train_py_src.contains("rl_obj.pop(\"temperature_by_options\", None)"));

    let art_dir = nli_out.join("artifacts");
    let dry = Command::new("python3")
        .env("CUDA_VISIBLE_DEVICES", "")
        .args([
            prep_nli.bundle_dir.join("train.py").to_str().unwrap(),
            "--dry-validate",
            "--bundle-dir",
            prep_nli.bundle_dir.to_str().unwrap(),
            "--artifacts-dir",
            art_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(dry.status.success());

    let leaked_test_path = tmp.join("leaked_multi_key.jsonl");
    std::fs::write(
        &leaked_test_path,
        "{\"id\":\"t_leak\",\"state\":{\"z_key\":\"refund\",\"a_key\":\"invoice\"},\"ground_truth\":{\"dept\":\"billing\"},\"source_uri\":\"raw\"}\n",
    )
    .unwrap();
    let leaked_items = lemmata::datasource::FileSource::new(&leaked_test_path)
        .read_items()
        .unwrap();
    assert_eq!(
        leaked_items[0].state_text(),
        "{\"z_key\":\"refund\",\"a_key\":\"invoice\"}"
    );
    let ds_filter = lemmata::engine::TrainedInputsFilter::load_from_file(
        &art_dir.join("trained_inputs.sha256.bin"),
    )
    .unwrap();
    let ov = ds_filter.check_overlap(&leaked_items);
    assert_eq!(
        ov.exact_overlap_ids,
        vec!["t_leak"],
        "Python train.py and Rust DataItem::state_text() must hash non-alphabetical JSON keys identically"
    );

    // 2. Bug #4: `merge_finetuned_adapter_with_options` `--force` overwrite guard vs `--f32` dtype control.
    let bf16_base = tmp.join("bf16_base");
    write_tiny_laya_checkpoint(&bf16_base);
    // Rewrite base `model.safetensors` in BF16 so we can verify dtype preservation vs `--f32` upcast!
    let orig =
        candle_core::safetensors::load(bf16_base.join("model.safetensors"), &Device::Cpu).unwrap();
    let bf16_map: HashMap<String, Tensor> = orig
        .into_iter()
        .map(|(k, t)| (k, t.to_dtype(DType::BF16).unwrap()))
        .collect();
    candle_core::safetensors::save(&bf16_map, bf16_base.join("model.safetensors")).unwrap();

    let ad_dir = tmp.join("bf16_adapter");
    std::fs::create_dir_all(&ad_dir).unwrap();
    let mut ad_map = HashMap::new();
    ad_map.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_A.weight".to_string(),
        Tensor::ones((2, 64), DType::F32, &Device::Cpu).unwrap(),
    );
    ad_map.insert(
        "base_model.model.encoder.layers.0.attn.Wqkv.lora_B.weight".to_string(),
        Tensor::zeros((192, 2), DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&ad_map, ad_dir.join("adapter_model.safetensors")).unwrap();
    std::fs::write(
        ad_dir.join("adapter_config.json"),
        r#"{"peft_type":"LORA","base_model_name_or_path":"convaiinnovations/laya","r":2,"lora_alpha":4.0}"#,
    )
    .unwrap();

    let merged_out = tmp.join("merged_bf16");
    lemmata::finetune::merge_finetuned_adapter_with_options(
        &ad_dir,
        Some(bf16_base.to_str().unwrap()),
        &merged_out,
        false,
        false,
    )
    .unwrap();
    let st_rep_bf16 = lemmata::model::inspect_safetensors(&merged_out).unwrap();
    assert!(
        st_rep_bf16.dtypes.contains_key("BF16") && !st_rep_bf16.dtypes.contains_key("F32"),
        "merge without --f32 must preserve BF16 base dtype: {:?}",
        st_rep_bf16.dtypes
    );

    // Second merge without `overwrite = true` (`--force`) must fail!
    let err_no_force = lemmata::finetune::merge_finetuned_adapter_with_options(
        &ad_dir,
        Some(bf16_base.to_str().unwrap()),
        &merged_out,
        false,
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(
        err_no_force.contains("--force"),
        "unexpected error: {err_no_force}"
    );

    // With `overwrite = true` and `force_f32 = false`, overwrites while STILL preserving BF16!
    lemmata::finetune::merge_finetuned_adapter_with_options(
        &ad_dir,
        Some(bf16_base.to_str().unwrap()),
        &merged_out,
        true,
        false,
    )
    .unwrap();
    let st_rep_still_bf16 = lemmata::model::inspect_safetensors(&merged_out).unwrap();
    assert!(st_rep_still_bf16.dtypes.contains_key("BF16"));

    // With `overwrite = true` and `force_f32 = true` (`--f32`), upcasts to F32!
    lemmata::finetune::merge_finetuned_adapter_with_options(
        &ad_dir,
        Some(bf16_base.to_str().unwrap()),
        &merged_out,
        true,
        true,
    )
    .unwrap();
    let st_rep_f32 = lemmata::model::inspect_safetensors(&merged_out).unwrap();
    assert!(st_rep_f32.dtypes.contains_key("F32") && !st_rep_f32.dtypes.contains_key("BF16"));

    let _ = std::fs::remove_dir_all(&tmp);
}
