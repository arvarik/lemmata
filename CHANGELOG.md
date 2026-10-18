# Changelog

All notable changes to lemmata are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

The first release.

### Added
- Native inference with candle for Laya (English, multilingual, typed decisions) and its router, the ModernBERT and DeBERTa-v3 zero-shot NLI models, ModernBERT NLI, EuroBERT, and NeoBERT. Models run on the CPU, and on the Metal GPU on macOS.
- Parity tests that compare each model with its Python reference code.
- `classify` for files, stdin, and inline text. `native` is the default backend: a checkpoint without weights fails and names the `lemmata download` command to run. `--backend auto` and `--backend mock` simulate missing models.
- Temperature calibration (`calibrate`), conformal prediction sets (`conformal`), and supervised probes (`probe`) in plain, residual, and stacked kinds.
- Evaluation and benchmarks: `test`, `bench`, `compare`, and `explain`.
- Signed, hash-chained decision receipts: `classify --receipts` writes them, and `attest verify` checks them.
- Ensembles (`ensemble`), label taxonomies (`taxonomy`), deferral to System 2 (`defer`), evaluation reports (`report`), a decision cache (`cache`), lessons (`learn`), and workflow recipes (`guide`).
- Prebuilt binaries for Linux (x86_64 glibc and musl, ARM64) and macOS (Intel and Apple Silicon) with build provenance attestations, and a container image on GHCR.

[0.1.0]: https://github.com/arvarik/lemmata/releases/tag/v0.1.0
