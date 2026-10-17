# Security Policy

## Supported Versions

Only the latest release version of `lemma` receives security updates and patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, report them privately via [GitHub Private Vulnerability Reporting](https://github.com/arvarik/lemma/security/advisories/new). You will receive an acknowledgment within 48 hours and a timeline for remediation once triaged.

## Security Model

These are the trust boundaries that lemma relies on. A report that breaks one of them is in scope.

- **Model files.** lemma loads weights only from `model.safetensors`, through a memory map, plus JSON configs and `tokenizer.json`. Safetensors holds only tensors, so loading a checkpoint does not run code. lemma never runs the Python files that ship with some checkpoints (for example `rl_agent_api.py`). A crafted model or tokenizer file that crashes lemma, reads outside its file, or runs code is a vulnerability.
- **Downloads.** `lemma` downloads checkpoints by running the Hugging Face CLI (`hf`, or `huggingface-cli`) as a subprocess. It uses your existing Hugging Face login and stores no credentials itself.
- **Network data.** `fetch`, `hn`, `reddit`, `wikipedia`, and `scrape` run `curl` on the URLs that you give. lemma treats their responses as untrusted text. lemma does not open a listening port.
- **Decision receipts.** lemma signs receipts with HMAC-SHA256 and chains them by hash. The key comes from `LEMMA_ATTEST_KEY` or `--attest-key`. A key on the command line stays in your shell history, so use the environment variable. A way to forge or reorder receipts without the key is a vulnerability.

Out of scope: wrong or low-confidence classifications. Those are model accuracy issues, so report them as normal issues.

## Verifying Release Artifacts

All official `lemma` release archives are built on GitHub Actions and signed with cryptographic SLSA build provenance attestations. You can verify any downloaded release archive using the GitHub CLI:

```sh
gh attestation verify lemma-<target>.tar.gz --repo arvarik/lemma
```
