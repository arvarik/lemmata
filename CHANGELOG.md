# Changelog

All notable changes to lemmata are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-17

The first release.

### Added
- Native inference with candle for Laya (English, multilingual, typed decisions) and its router, the ModernBERT and DeBERTa-v3 zero-shot NLI models, ModernBERT NLI, EuroBERT, and NeoBERT. Models run on the CPU, and on the Metal GPU on macOS.
- Parity tests that compare each model with its Python reference code.
- `classify` for files, stdin, and inline text. `native` is the default backend: a checkpoint without weights fails and names the `lemmata download` command to run. `--backend auto` and `--backend mock` simulate missing models.
- Temperature calibration (`calibrate`), conformal prediction sets (`conformal`), and supervised probes (`probe`) in plain, residual, and stacked kinds.
- Evaluation and benchmarks: `test`, `bench`, `compare`, and `explain`.
- Signed, hash-chained decision receipts: `classify --receipts` writes them, and `attest verify` checks them.
- `report`: a full evaluation of one checkpoint, with per-label precision, recall, and F1, macro-F1, the most confused label pairs, and the risk-coverage curve. `--format md` writes Markdown.
- `defer fit` chooses the risk threshold for deferral to System 2 with an exact binomial bound, so the kept items reach a target accuracy at a stated confidence. `defer run` keeps or defers each item.
- `ensemble` pools diverse checkpoints, fits member weights on labelled data (`--fit-weights`), and reports their disagreement as mutual information.
- `taxonomy` classifies against a label taxonomy. It follows the top options at each level (`--beam`) and escalates on the best path probability.
- `--cache` saves exact decisions on disk, keyed by the input text, the questions, and a model fingerprint. `cache stats` and `cache clear` manage the files.
- `learn` runs every lesson without model downloads, and `guide --samples` writes the sample data and schemas that the recipes use.
- Prebuilt binaries for Linux (x86_64 glibc and musl, ARM64) and macOS (Intel and Apple Silicon) with build provenance attestations, and a container image on GHCR.

[0.1.0]: https://github.com/arvarik/lemmata/releases/tag/v0.1.0
