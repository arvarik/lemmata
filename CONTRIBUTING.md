# Contributing to Lemma

Thank you for contributing to `lemma`! We keep the codebase lean, fast, and minimal.

## Development Workflow

1. **Build & Test**:
   ```sh
   cargo test --workspace --all-targets
   ```
2. **Format & Lint**:
   ```sh
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   ```
3. **Documentation Check**:
   ```sh
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
   ```

## Design Principles

- **Standard Library First**: Avoid adding external crates when `std` covers the use case cleanly.
- **Extensible System 1 Engine**: New System 1 decision models should implement the `DecisionEngine` trait in `src/engine/classifier.rs` and register their checkpoint metadata in `src/model.rs`.
- **End-to-End Verification**: Prefer fast, deterministic integration tests in `tests/integration_test.rs` that exercise real CLI flags and output formats.
- **Release Titles**: GitHub releases must always be titled strictly in the format `vX.X.X` (e.g., `v0.1.0`).
