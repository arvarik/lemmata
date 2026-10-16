# Security Policy

## Supported Versions

Only the latest release version of `lemma` receives security updates and patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, report them privately via [GitHub Private Vulnerability Reporting](https://github.com/arvarik/lemma/security/advisories/new). You will receive an acknowledgment within 48 hours and a timeline for remediation once triaged.

## Verifying Release Artifacts

All official `lemma` release archives are built on GitHub Actions and signed with cryptographic SLSA build provenance attestations. You can verify any downloaded release archive using the GitHub CLI:

```sh
gh attestation verify lemma-<target>.tar.gz --repo arvarik/lemma
```
