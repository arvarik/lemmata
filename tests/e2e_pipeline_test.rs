//! End-to-end CLI integration test covering the complete supervised adaptation pipeline:
//! `dataset import -> dataset split -> dataset verify -> finetune prepare -> train.py --dry-validate -> finetune verify -> classify --adapter -> finetune merge -> experiment (+ zero-leakage rejection)`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_core::{DType, Device, Tensor};
use lemmata::model::{TRAINING_MANIFEST_VERSION, safetensors_header_sha256, sha256_file};
use serde_json::{Value, json};

fn unique_temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "lemmata_e2e_pipeline_{tag}_{}_{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a tiny self-contained native Laya (ModernBERT + Decision Head) checkpoint directory so
/// CLI invocations run real `candle` tensor inference (`--backend native`) completely offline.
fn write_tiny_laya_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir.join("encoder")).unwrap();
    std::fs::create_dir_all(dir.join("tokenizer")).unwrap();

    let rl_cfg = json!({
        "head_layers": 1,
        "max_len": 96,
        "head_max_len": 48,
        "temperature_by_options": {
            "choice:2": 1.90
        }
    });
    std::fs::write(
        dir.join("rl_agent_config.json"),
        serde_json::to_string_pretty(&rl_cfg).unwrap(),
    )
    .unwrap();

    let mut vocab_map = serde_json::Map::new();
    let tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "choice",
        "question:",
        "which",
        "team",
        "handles",
        "this",
        "ticket?",
        "billing:",
        "invoices",
        "charges",
        "refunds",
        "technical:",
        "crashes",
        "outages",
        "errors",
        "billing",
        "technical",
        "invoice",
        "refund",
        "charge",
        "charged",
        "twice",
        "payment",
        "database",
        "server",
        "crash",
        "crashed",
        "outage",
        "panic",
        "kernel",
        "latency",
    ];
    for (idx, tok) in tokens.iter().enumerate() {
        vocab_map.insert((*tok).to_string(), json!(idx));
    }
    while vocab_map.len() < 64 {
        let id = vocab_map.len();
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
    std::fs::write(
        dir.join("tokenizer/tokenizer_config.json"),
        serde_json::to_string_pretty(&json!({
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]"
        }))
        .unwrap(),
    )
    .unwrap();

    let d = 64usize;
    let inter = 128usize;
    let enc_cfg = json!({
        "model_type": "modernbert",
        "vocab_size": 64,
        "hidden_size": d,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": inter,
        "max_position_embeddings": 128,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "global_attn_every_n_layers": 1,
        "global_rope_theta": 10000.0,
        "local_attention": 64,
        "local_rope_theta": 10000.0,
        "classifier_activation": "gelu",
        "classifier_pooling": "cls"
    });
    std::fs::write(
        dir.join("encoder/config.json"),
        serde_json::to_string_pretty(&enc_cfg).unwrap(),
    )
    .unwrap();

    let dev = Device::Cpu;
    let det = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 19 + seed * 7) % 97) as f32 - 48.0) * 0.015)
            .collect()
    };

    let mut ts: HashMap<String, Tensor> = HashMap::new();
    ts.insert(
        "encoder.embeddings.tok_embeddings.weight".into(),
        Tensor::from_vec(det(64 * d, 1), (64, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.embeddings.norm.weight".into(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.attn.Wqkv.weight".into(),
        Tensor::from_vec(det(3 * d * d, 2), (3 * d, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.attn.Wo.weight".into(),
        Tensor::from_vec(det(d * d, 3), (d, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.mlp_norm.weight".into(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.mlp.Wi.weight".into(),
        Tensor::from_vec(det(2 * inter * d, 4), (2 * inter, d), &dev).unwrap(),
    );
    ts.insert(
        "encoder.layers.0.mlp.Wo.weight".into(),
        Tensor::from_vec(det(d * inter, 5), (d, inter), &dev).unwrap(),
    );
    ts.insert(
        "encoder.final_norm.weight".into(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "type_emb.weight".into(),
        Tensor::from_vec(det(3 * d, 6), (3, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm1.weight".into(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm1.bias".into(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.in_proj_weight".into(),
        Tensor::from_vec(det(3 * d * d, 7), (3 * d, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.in_proj_bias".into(),
        Tensor::zeros(3 * d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.out_proj.weight".into(),
        Tensor::from_vec(det(d * d, 8), (d, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.self_attn.out_proj.bias".into(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm2.weight".into(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.norm2.bias".into(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear1.weight".into(),
        Tensor::from_vec(det(4 * d * d, 9), (4 * d, d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear1.bias".into(),
        Tensor::zeros(4 * d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear2.weight".into(),
        Tensor::from_vec(det(d * 4 * d, 10), (d, 4 * d), &dev).unwrap(),
    );
    ts.insert(
        "head.layers.0.linear2.bias".into(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.0.weight".into(),
        Tensor::ones(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.0.bias".into(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.1.weight".into(),
        Tensor::from_vec(det(d * d, 11), (d, d), &dev).unwrap(),
    );
    ts.insert(
        "scorer.1.bias".into(),
        Tensor::zeros(d, DType::F32, &dev).unwrap(),
    );
    ts.insert(
        "scorer.3.weight".into(),
        Tensor::from_vec(det(d, 12), (1, d), &dev).unwrap(),
    );
    ts.insert(
        "scorer.3.bias".into(),
        Tensor::zeros(1, DType::F32, &dev).unwrap(),
    );

    candle_core::safetensors::save(&ts, dir.join("model.safetensors")).unwrap();
}

fn run_lemmata(args: &[&str], cache_dir: &Path) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_lemmata");
    Command::new(bin)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("LEMMATA_DEVICE", "cpu")
        .env("LEMMATA_CACHE_DIR", cache_dir)
        .args(args)
        .output()
        .expect("failed to invoke lemmata binary")
}

#[test]
fn end_to_end_dataset_finetune_adapter_merge_and_paired_experiment_cli() {
    let tmp = unique_temp_dir("full_cli");
    let cache_dir = tmp.join("cache");
    let base_dir = tmp.join("tiny_laya");
    write_tiny_laya_checkpoint(&base_dir);

    // 0. Write schema and raw UTF-8 BOM CSV with 80 distinct items across 2 classes + metadata
    let schema_path = tmp.join("questions.json");
    std::fs::write(
        &schema_path,
        serde_json::to_string_pretty(&json!({
            "dept": {
                "type": "choice",
                "instructions": "Which team handles this ticket?",
                "criteria": {
                    "billing": "invoices charges refunds",
                    "technical": "crashes outages errors"
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let raw_csv = tmp.join("raw_tickets.csv");
    let mut csv_body = String::from("\u{FEFF}ticket_id,message,category,region\n");
    let billing_openers = [
        "Please refund duplicate charge",
        "Monthly invoice total looks wrong",
        "Credit card billed twice yesterday",
        "Wire transfer receipt missing online",
        "Annual renewal fee overcharged account",
        "VAT tax calculation error on statement",
        "Subscription cancellation still charged card",
        "Payout settlement delayed from stripe",
        "Direct debit failed despite sufficient funds",
        "Prorated seat downgrade credit not applied",
        "ACH routing update rejected by billing",
    ];
    let billing_closers = [
        "for enterprise workspace alpha",
        "after upgrading seat tier",
        "on procurement purchase order",
        "inside finance dashboard",
        "during quarterly reconciliation",
        "under custom master contract",
        "for subsidiary ledger account",
        "following currency conversion",
        "before renewal cycle cutoff",
        "with net thirty payment terms",
    ];
    let tech_openers = [
        "Production database server crashed",
        "Kernel panic froze worker nodes",
        "TLS handshake failing on gateway",
        "Memory leak triggered container OOM",
        "Webhook queue stalled under traffic",
        "Search index replica out of sync",
        "gRPC stream timeout across regions",
        "SSO SAML assertion rejected login",
        "Deadlock observed inside connection pool",
        "Kubernetes ingress returning bad gateway",
        "Background job worker segfaulted",
    ];
    let tech_closers = [
        "after Tuesday helm rollout",
        "in Frankfurt availability zone",
        "when parsing large payloads",
        "on Linux arm64 runtime",
        "during peak batch ingestion",
        "behind envoy reverse proxy",
        "while compacting WAL segments",
        "under concurrent websocket load",
        "after upgrading rustls dependency",
        "inside isolated VPC subnet",
    ];
    for i in 0..110 {
        let b_msg = format!(
            "{} {} ref_{:03}",
            billing_openers[i % billing_openers.len()],
            billing_closers[i / billing_openers.len()],
            i * 17 + 3
        );
        let b_cat = if i % 3 == 0 { "fin" } else { "billing" };
        let reg = if i % 2 == 0 { "us" } else { "eu" };
        csv_body.push_str(&format!("B{i},\"{b_msg}\",{b_cat},{reg}\n"));

        let t_msg = format!(
            "{} {} trace_{:03}",
            tech_openers[i % tech_openers.len()],
            tech_closers[i / tech_openers.len()],
            i * 23 + 9
        );
        csv_body.push_str(&format!("T{i},\"{t_msg}\",technical,{reg}\n"));
    }
    // Include one blank row to exercise `--drop-empty`
    csv_body.push_str("EMPTY1,\"   \",billing,us\n");
    std::fs::write(&raw_csv, csv_body).unwrap();

    // 1. `lemmata dataset import` (via alias `data`)
    let canonical_jsonl = tmp.join("canonical.jsonl");
    let imp_out = run_lemmata(
        &[
            "data",
            "import",
            "--file",
            raw_csv.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "--id-col",
            "ticket_id",
            "--text-col",
            "message",
            "--meta-cols",
            "region",
            "--label-map",
            "dept=category",
            "--value-map",
            "dept:fin=billing",
            "--drop-empty",
            "--output",
            canonical_jsonl.to_str().unwrap(),
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        imp_out.status.success(),
        "dataset import failed: {}",
        String::from_utf8_lossy(&imp_out.stderr)
    );
    let imp_json: Value = serde_json::from_slice(&imp_out.stdout).unwrap();
    assert_eq!(imp_json["imported_items"], 220);
    assert_eq!(imp_json["dropped_empty"], 1);

    // 2. `lemmata dataset split`
    let splits_dir = tmp.join("splits");
    let split_out = run_lemmata(
        &[
            "dataset",
            "split",
            "--file",
            canonical_jsonl.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "--out-dir",
            splits_dir.to_str().unwrap(),
            "--seed",
            "42",
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        split_out.status.success(),
        "dataset split failed: {}",
        String::from_utf8_lossy(&split_out.stderr)
    );
    assert!(splits_dir.join("train.jsonl").exists());
    assert!(splits_dir.join("val.jsonl").exists());
    assert!(splits_dir.join("cal.jsonl").exists());
    assert!(splits_dir.join("test.jsonl").exists());
    assert!(splits_dir.join("dataset_manifest.json").exists());

    // 3. `lemmata dataset verify`
    let dver_out = run_lemmata(
        &[
            "dataset",
            "verify",
            "--split-dir",
            splits_dir.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        dver_out.status.success(),
        "dataset verify failed: {}",
        String::from_utf8_lossy(&dver_out.stderr)
    );
    let dver_json: Value = serde_json::from_slice(&dver_out.stdout).unwrap();
    assert_eq!(dver_json["ready"], true);
    assert_eq!(dver_json["fail_count"], 0);

    // 4. `lemmata finetune prepare` (via alias `ft bundle`)
    let run_dir = tmp.join("ft_run");
    let prep_out = run_lemmata(
        &[
            "ft",
            "bundle",
            "--split-dir",
            splits_dir.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "-c",
            "english",
            "--method",
            "dora",
            "--rank",
            "4",
            "--alpha",
            "8.0",
            "--out",
            run_dir.to_str().unwrap(),
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        prep_out.status.success(),
        "finetune prepare failed: {}",
        String::from_utf8_lossy(&prep_out.stderr)
    );
    let prep_json: Value = serde_json::from_slice(&prep_out.stdout).unwrap();
    let bundle_dir = run_dir.join("bundle");
    assert!(
        !bundle_dir.join("data/test.jsonl").exists(),
        "bundle/data/test.jsonl must never exist"
    );

    // 5. Execute `python3 bundle/train.py --dry-validate` to emit real `trained_inputs.sha256.bin`,
    // then materialize a valid DoRA adapter + calibration + `training_manifest.json` in `artifacts/`.
    let artifacts_dir = run_dir.join("artifacts");
    let py_dry = Command::new("python3")
        .env("CUDA_VISIBLE_DEVICES", "")
        .args([
            bundle_dir.join("train.py").to_str().unwrap(),
            "--dry-validate",
            "--bundle-dir",
            bundle_dir.to_str().unwrap(),
            "--artifacts-dir",
            artifacts_dir.to_str().unwrap(),
        ])
        .output()
        .expect("python3 must be available");
    assert!(
        py_dry.status.success(),
        "train.py --dry-validate failed: {}",
        String::from_utf8_lossy(&py_dry.stderr)
    );
    assert!(artifacts_dir.join("trained_inputs.sha256.bin").exists());

    let dev = Device::Cpu;
    let d = 64usize;
    let r = 4usize;
    let det = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 29 + seed * 11) % 83) as f32 - 41.0) * 0.04)
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
        Tensor::from_vec(vec![1.20f32; 3 * d], 3 * d, &dev).unwrap(),
    );
    ad.insert(
        "base_model.model.scorer.3.weight".to_string(),
        Tensor::from_vec(det(d, 5), (1, d), &dev).unwrap(),
    );
    ad.insert(
        "base_model.model.scorer.3.bias".to_string(),
        Tensor::from_vec(vec![0.25f32], 1, &dev).unwrap(),
    );
    let ad_st_path = artifacts_dir.join("adapter_model.safetensors");
    candle_core::safetensors::save(&ad, &ad_st_path).unwrap();
    std::fs::write(
        artifacts_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&json!({
            "peft_type": "DORA",
            "base_model_name_or_path": "convaiinnovations/laya",
            "r": r,
            "lora_alpha": 8.0,
            "use_rslora": true,
            "use_dora": true,
            "target_modules": ["attn.Wqkv"],
            "modules_to_save": ["scorer"]
        }))
        .unwrap(),
    )
    .unwrap();
    let cal_json_path = artifacts_dir.join("lemmata_calibration.json");
    std::fs::write(
        &cal_json_path,
        serde_json::to_string_pretty(&json!({
            "checkpoint": "convaiinnovations/laya",
            "temperature": 1.0,
            "temperature_by_options": { "choice:2": 1.12 },
            "loss_function": "nll",
            "metadata": {
                "dataset": "cal",
                "samples_evaluated": 8,
                "raw_mean_ece": 0.14,
                "calibrated_mean_ece": 0.05,
                "improvement_pct": 64.3,
                "calibrated_at": "2026-11-17T00:00:00Z"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let bundle_manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(bundle_dir.join("bundle_manifest.json")).unwrap(),
    )
    .unwrap();
    let base_hdr = safetensors_header_sha256(&base_dir.join("model.safetensors")).unwrap();
    std::fs::write(
        artifacts_dir.join("training_manifest.json"),
        serde_json::to_string_pretty(&json!({
            "manifest_version": TRAINING_MANIFEST_VERSION,
            "run_id": prep_json["run_id"],
            "status": "completed",
            "created_by_lemmata_version": "0.1.0",
            "base_model": {
                "checkpoint_slug": "english",
                "repo_id": "convaiinnovations/laya",
                "paradigm": "laya_joint_decision",
                "remote_base_header_sha256": base_hdr,
                "max_len": 96,
                "head_max_len": 48
            },
            "schema": bundle_manifest["schema"],
            "dataset_provenance": {
                "split_id": bundle_manifest["split_id"],
                "train": bundle_manifest["splits"]["train"],
                "val": bundle_manifest["splits"]["val"],
                "cal": bundle_manifest["splits"]["cal"],
                "locked_test_guard": {
                    "expected_test_file_sha256": bundle_manifest["expected_test_file_sha256"],
                    "trained_inputs_filter_file": "trained_inputs.sha256.bin",
                    "trained_unique_exact_hashes": 72,
                    "trained_unique_norm_hashes": 72
                }
            },
            "method": bundle_manifest["method"],
            "hyperparameters": bundle_manifest["hyperparameters"],
            "results": { "best_val_nll": 0.31 },
            "environment": {},
            "artifacts": {
                "adapter_safetensors_sha256": sha256_file(&ad_st_path).unwrap(),
                "adapter_bytes": std::fs::metadata(&ad_st_path).unwrap().len(),
                "calibration_sha256": sha256_file(&cal_json_path).unwrap()
            }
        }))
        .unwrap(),
    )
    .unwrap();

    // 6. `lemmata finetune verify` across all 5 layers
    let fver_out = run_lemmata(
        &[
            "finetune",
            "verify",
            "--artifact",
            artifacts_dir.to_str().unwrap(),
            "--split-dir",
            splits_dir.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "--base-dir",
            base_dir.to_str().unwrap(),
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        fver_out.status.success(),
        "finetune verify failed: {}",
        String::from_utf8_lossy(&fver_out.stderr)
    );
    let fver_json: Value = serde_json::from_slice(&fver_out.stdout).unwrap();
    assert_eq!(fver_json["valid"], true);
    assert_eq!(fver_json["weights_verified"], true);
    assert_eq!(fver_json["base_header_verified"], true);
    assert_eq!(fver_json["schema_verified"], true);
    assert_eq!(fver_json["split_hashes_verified"], true);
    assert_eq!(fver_json["row_leakage_checked"], true);

    // 7. `lemmata classify --adapter` vs `lemmata finetune merge` + `lemmata classify -c <merged_dir>`
    let merged_dir = tmp.join("merged_laya");
    let merge_out = run_lemmata(
        &[
            "finetune",
            "merge",
            "--adapter",
            artifacts_dir.to_str().unwrap(),
            "-c",
            base_dir.to_str().unwrap(),
            "--output",
            merged_dir.to_str().unwrap(),
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        merge_out.status.success(),
        "finetune merge failed: {}",
        String::from_utf8_lossy(&merge_out.stderr)
    );
    // Copy fitted calibration to merged_dir so probabilities match 1-for-1
    std::fs::copy(&cal_json_path, merged_dir.join("lemmata_calibration.json")).unwrap();

    let cls_ad_out = run_lemmata(
        &[
            "--backend",
            "native",
            "classify",
            "-c",
            base_dir.to_str().unwrap(),
            "--adapter",
            artifacts_dir.to_str().unwrap(),
            "--text",
            "Please refund duplicate charge on invoice",
            "--schema",
            schema_path.to_str().unwrap(),
            "--format",
            "json",
        ],
        &cache_dir,
    );
    assert!(
        cls_ad_out.status.success(),
        "classify --adapter failed: {}",
        String::from_utf8_lossy(&cls_ad_out.stderr)
    );
    let cls_mg_out = run_lemmata(
        &[
            "--backend",
            "native",
            "classify",
            "-c",
            merged_dir.to_str().unwrap(),
            "--text",
            "Please refund duplicate charge on invoice",
            "--schema",
            schema_path.to_str().unwrap(),
            "--format",
            "json",
        ],
        &cache_dir,
    );
    assert!(
        cls_mg_out.status.success(),
        "classify -c merged_dir failed: {}",
        String::from_utf8_lossy(&cls_mg_out.stderr)
    );
    let ad_res: Value = serde_json::from_slice(&cls_ad_out.stdout).unwrap();
    let mg_res: Value = serde_json::from_slice(&cls_mg_out.stdout).unwrap();
    assert_eq!(
        ad_res[0][1]["answers"]["dept"]["probabilities"],
        mg_res[0][1]["answers"]["dept"]["probabilities"]
    );
    assert!(
        ad_res[0][1]["model"].as_str().unwrap().contains("@"),
        "expected @short_sha provenance in model id: {}",
        ad_res[0][1]["model"]
    );

    // 8. `lemmata experiment` (alias `ab`) with clean held-out `test.jsonl` + `cal.jsonl` + slice analysis
    let exp_md = tmp.join("experiment_report.md");
    let cand_spec = format!("{}+{}", base_dir.display(), artifacts_dir.display());
    let exp_out = run_lemmata(
        &[
            "--backend",
            "native",
            "ab",
            "--base",
            base_dir.to_str().unwrap(),
            "--candidate",
            &cand_spec,
            "--split-dir",
            splits_dir.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "--bootstrap-samples",
            "200",
            "--permutations",
            "400",
            "--slice-by",
            "field:region",
            "--output",
            exp_md.to_str().unwrap(),
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        exp_out.status.success(),
        "lemmata experiment failed: {}",
        String::from_utf8_lossy(&exp_out.stderr)
    );
    let exp_json: Value = serde_json::from_slice(&exp_out.stdout).unwrap();
    assert_eq!(exp_json["provenance"]["verified"], true);
    assert_eq!(exp_json["provenance"]["leakage_detected"], false);
    assert!(exp_json["raw_comparison"].is_object());
    assert!(exp_json["calibrated_comparison"].is_object());
    assert!(exp_md.exists());
    let md_text = std::fs::read_to_string(&exp_md).unwrap();
    assert!(md_text.contains("Lemmata Paired Experiment"));

    // 9a. Leakage rejection when accidentally passing `train.jsonl` to `experiment`
    let leak_train_out = run_lemmata(
        &[
            "--backend",
            "native",
            "experiment",
            "--base",
            base_dir.to_str().unwrap(),
            "--candidate",
            &cand_spec,
            "--file",
            splits_dir.join("train.jsonl").to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
        ],
        &cache_dir,
    );
    assert!(
        !leak_train_out.status.success(),
        "experiment must reject train.jsonl passed as test file!"
    );

    // 9b. Row-level normalized state leakage rejection via `trained_inputs.sha256.bin`
    let train_first_line = std::fs::read_to_string(splits_dir.join("train.jsonl"))
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    let train_obj: Value = serde_json::from_str(&train_first_line).unwrap();
    let orig_state = train_obj["state"].as_str().unwrap();
    let perturbed_state = format!("  \t {}  \n ", orig_state.to_uppercase());
    let contaminated_path = tmp.join("contaminated_test.jsonl");
    let mut contam_content = std::fs::read_to_string(splits_dir.join("test.jsonl")).unwrap();
    contam_content.push_str(
        &serde_json::to_string(&json!({
            "id": "LEAKED_ROW_99",
            "state": perturbed_state,
            "ground_truth": train_obj["ground_truth"],
            "source_uri": "contaminated"
        }))
        .unwrap(),
    );
    contam_content.push('\n');
    std::fs::write(&contaminated_path, contam_content).unwrap();

    let leak_row_exp = run_lemmata(
        &[
            "--backend",
            "native",
            "experiment",
            "--base",
            base_dir.to_str().unwrap(),
            "--candidate",
            &cand_spec,
            "--file",
            contaminated_path.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
        ],
        &cache_dir,
    );
    assert!(
        !leak_row_exp.status.success(),
        "experiment must reject whitespace/case-perturbed training row via trained_inputs.sha256.bin!"
    );
    let stderr_exp = String::from_utf8_lossy(&leak_row_exp.stderr);
    assert!(
        stderr_exp.contains("LEAKED_ROW_99") || stderr_exp.contains("leakage"),
        "expected leaked item id in stderr: {stderr_exp}"
    );

    let leak_row_test = run_lemmata(
        &[
            "--backend",
            "native",
            "test",
            "-c",
            base_dir.to_str().unwrap(),
            "--adapter",
            artifacts_dir.to_str().unwrap(),
            "--file",
            contaminated_path.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
        ],
        &cache_dir,
    );
    assert!(
        !leak_row_test.status.success(),
        "lemmata test --adapter must also block contaminated test file!"
    );

    // Explicit `--allow-leakage` override permits diagnostic run and flags `leakage_overridden`
    let override_exp = run_lemmata(
        &[
            "--backend",
            "native",
            "experiment",
            "--base",
            base_dir.to_str().unwrap(),
            "--candidate",
            &cand_spec,
            "--file",
            contaminated_path.to_str().unwrap(),
            "--schema",
            schema_path.to_str().unwrap(),
            "--bootstrap-samples",
            "50",
            "--permutations",
            "100",
            "--allow-leakage",
            "--json",
        ],
        &cache_dir,
    );
    assert!(
        override_exp.status.success(),
        "--allow-leakage should succeed: {}",
        String::from_utf8_lossy(&override_exp.stderr)
    );
    let ov_json: Value = serde_json::from_slice(&override_exp.stdout).unwrap();
    assert_eq!(ov_json["provenance"]["leakage_detected"], true);
    assert_eq!(ov_json["provenance"]["leakage_overridden"], true);

    let _ = std::fs::remove_dir_all(&tmp);
}
