use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

use super::checkpoint::Checkpoint;

pub struct Downloader {
    pub cli_binary: String,
}

impl Downloader {
    /// Detect whether `hf` or `huggingface-cli` is installed and available in PATH.
    pub fn detect() -> Result<Self> {
        if Self::check_binary("hf") {
            Ok(Self {
                cli_binary: "hf".to_string(),
            })
        } else if Self::check_binary("huggingface-cli") {
            Ok(Self {
                cli_binary: "huggingface-cli".to_string(),
            })
        } else {
            bail!(
                "Neither `hf` nor `huggingface-cli` was found in PATH.\n\
                 Install the Hugging Face CLI via:\n\
                   brew install huggingface-cli\n\
                 or:\n\
                   pip install -U huggingface_hub"
            );
        }
    }

    fn check_binary(bin: &str) -> bool {
        Command::new(bin)
            .arg("--help")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }

    /// Check if the user is authenticated in Hugging Face CLI.
    pub fn check_auth(&self) -> Option<String> {
        let output = Command::new(&self.cli_binary)
            .args(["auth", "whoami"])
            .output()
            .ok()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Some(stdout)
        } else {
            None
        }
    }

    /// Download a System 1 checkpoint to a local directory.
    pub fn download(&self, checkpoint: Checkpoint, target_dir: &Path, dry_run: bool) -> Result<()> {
        let effective_dir = if target_dir == Path::new("models/laya") {
            checkpoint.local_dir()
        } else {
            target_dir.to_path_buf()
        };
        let repo_id = checkpoint.repo_id();
        let target_str = effective_dir.display().to_string();

        println!("⚡ System 1 Checkpoint: {}", repo_id);
        println!("   Backbone:        {}", checkpoint.backbone());
        println!("   Parameters:      {}", checkpoint.param_count());
        println!("   Context Window:  {} tokens", checkpoint.context_window());
        println!("   Destination:     {}", target_str);

        if let Some(user_info) = self.check_auth() {
            println!("   HF Auth:         Logged in ({})", user_info);
        } else {
            println!("   HF Auth:         No active token found (public download)");
        }

        if dry_run {
            println!("\n[DRY RUN] Would execute command:");
            println!(
                "   {} download {} --local-dir {} --exclude \"onnx/*\" --exclude \"*.bin\"",
                self.cli_binary, repo_id, target_str
            );
            return Ok(());
        }

        std::fs::create_dir_all(&effective_dir)
            .with_context(|| format!("Failed to create directory: {}", target_str))?;

        println!("\n📥 Downloading model files via {}...", self.cli_binary);
        let mut cmd = Command::new(&self.cli_binary);
        cmd.args([
            "download",
            repo_id,
            "--local-dir",
            &target_str,
            "--exclude",
            "onnx/*",
            "--exclude",
            "*.bin",
            "--exclude",
            "img/*",
            "--exclude",
            "*.jpeg",
        ]);

        let status = cmd
            .status()
            .with_context(|| format!("Failed to execute {} download", self.cli_binary))?;

        if !status.success() {
            bail!(
                "Download failed with exit code {:?}. Verify network connection and permissions.",
                status.code()
            );
        }

        let report = self.verify_checkpoint(checkpoint, &effective_dir)?;
        if !report.is_valid {
            bail!(
                "Verification failed after download in {}; missing required files: {:?}",
                report.path.display(),
                report.missing
            );
        }
        println!(
            "✅ Download and verification complete: {}",
            report.path.display()
        );
        Ok(())
    }

    /// Verify that essential model files are present for a specific checkpoint under `target_dir`.
    pub fn verify_checkpoint(
        &self,
        checkpoint: Checkpoint,
        target_dir: &Path,
    ) -> Result<VerificationReport> {
        let sub = checkpoint.local_subdir();
        if !sub.is_empty() {
            let candidate = target_dir.join(sub);
            if candidate.exists() {
                return self.verify(&candidate);
            }
        }
        if target_dir == Path::new("models/laya") {
            let ckpt_dir = checkpoint.local_dir();
            if ckpt_dir != target_dir && ckpt_dir.exists() {
                return self.verify(&ckpt_dir);
            }
        }
        self.verify(target_dir)
    }

    /// Verify that essential model files (Laya or standard Hugging Face Safetensors layout) are present.
    pub fn verify(&self, target_dir: &Path) -> Result<VerificationReport> {
        let resolved_dir = if !target_dir.join("rl_agent_config.json").exists()
            && !target_dir.join("config.json").exists()
        {
            ["multilingual", "typed-decisions"]
                .iter()
                .map(|sub| target_dir.join(sub))
                .find(|p| p.join("rl_agent_config.json").exists() || p.join("config.json").exists())
                .unwrap_or_else(|| target_dir.to_path_buf())
        } else {
            target_dir.to_path_buf()
        };

        let is_laya_layout = resolved_dir.join("rl_agent_config.json").exists()
            || resolved_dir.join("encoder/config.json").exists();

        let required_files: &[&str] = if is_laya_layout {
            &[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/tokenizer.json",
                "encoder/config.json",
            ]
        } else {
            &["config.json", "model.safetensors", "tokenizer.json"]
        };

        let mut found = Vec::new();
        let mut missing = Vec::new();

        for &rel in required_files {
            let p = resolved_dir.join(rel);
            if p.exists() {
                let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                found.push((rel.to_string(), size));
            } else {
                missing.push(rel.to_string());
            }
        }

        let is_valid = missing.is_empty();
        Ok(VerificationReport {
            path: resolved_dir,
            is_valid,
            found,
            missing,
        })
    }
}

pub struct VerificationReport {
    pub path: PathBuf,
    pub is_valid: bool,
    pub found: Vec<(String, u64)>,
    pub missing: Vec<String>,
}
