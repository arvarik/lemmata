#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6.0,<2.9.0",
#   "transformers>=4.49.0,<5.0.0",
#   "peft>=0.14.0,<0.19.0",
#   "safetensors>=0.5.0",
#   "accelerate>=1.3.0",
#   "huggingface-hub>=0.28.0",
#   "sentencepiece>=0.2.0",
#   "protobuf>=4.25.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cuda"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cuda", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
#   { index = "pytorch-cpu", marker = "sys_platform != 'linux' or platform_machine != 'x86_64'" },
# ]
# ///
"""Self-bootstrapping external-GPU fine-tuning script for Lemmata System 1 models.

Supports Laya joint single-pass decision heads (ModernBERT-large / mmBERT-base),
pairwise NLI cross-encoders (ModernBERT-zeroshot, ModernBERT-NLI, DeBERTa-v3),
and masked-LM verbalizer encoders (EuroBERT, NeoBERT) with 100% sequence and
tensor parity with `lemmata`'s Rust/Candle inference engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "lemmata-ft-2026.1"
FILTER_MAGIC = b"LMFT1\x00\x00\x00"
MAX_OPTION_TOKENS = 48
MAX_LABEL_TOKENS = 8
MIN_TEMPERATURE = 0.5
MAX_TEMPERATURE = 5.0
MIN_BUCKET_SAMPLES = 30
EXIT_SPOT_INTERRUPTED = 75

QTYPE_INDEX = {"choice": 0, "score": 1, "noul": 2}


# ---------------------------------------------------------------------------
# Pure-stdlib helpers (usable offline via `--help` and `--dry-validate`)
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header_sha256(path: Path) -> str:
    target = path / "model.safetensors" if path.is_dir() else path
    size = target.stat().st_size
    if size < 8:
        raise ValueError(f"Safetensors file too small: {target}")
    with target.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if 8 + header_len > size:
            raise ValueError(f"Corrupt safetensors header length in {target}")
        header = f.read(header_len)
    json.loads(header.decode("utf-8"))
    h = hashlib.sha256()
    h.update(struct.pack("<Q", size))
    h.update(header)
    return h.hexdigest()


def extract_state_text(row: dict[str, Any]) -> str:
    if "state" in row:
        st = row["state"]
        if isinstance(st, str):
            return st
        if isinstance(st, dict) and len(st) == 1 and isinstance(st.get("text"), str):
            return st["text"]
        return json.dumps(st, separators=(",", ":"), ensure_ascii=False)
    if isinstance(row.get("text"), str):
        return row["text"]
    return ""


def normalize_state_text(text: str) -> str:
    return " ".join(text.replace("\ufeff", "").replace("\x00", "").lower().split())


def sha256_u64_le(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


def compute_split_merkle_and_hashes(
    path: Path,
) -> tuple[int, str, str, set[int], set[int], list[dict[str, Any]]]:
    file_sha = sha256_file(path)
    rows: list[dict[str, Any]] = []
    exact_u64: set[int] = set()
    norm_u64: set[int] = set()
    row_state_digests: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            st_text = extract_state_text(obj)
            norm_text = normalize_state_text(st_text)
            exact_u64.add(sha256_u64_le(st_text))
            if norm_text:
                norm_u64.add(sha256_u64_le(norm_text))
            row_state_digests.append(sha256_bytes(st_text.encode("utf-8")))
            rows.append(obj)

    row_state_digests.sort()
    merkle = sha256_bytes("\n".join(row_state_digests).encode("utf-8"))
    return len(rows), file_sha, merkle, exact_u64, norm_u64, rows


def write_trained_inputs_filter(
    out_path: Path, exact_hashes: set[int], norm_hashes: set[int]
) -> tuple[int, int]:
    exact_sorted = sorted(exact_hashes)
    norm_sorted = sorted(norm_hashes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".bin.tmp")
    with tmp.open("wb") as f:
        f.write(FILTER_MAGIC)
        f.write(struct.pack("<QQ", len(exact_sorted), len(norm_sorted)))
        for val in exact_sorted:
            f.write(struct.pack("<Q", val))
        for val in norm_sorted:
            f.write(struct.pack("<Q", val))
    os.replace(tmp, out_path)
    return len(exact_sorted), len(norm_sorted)


# ---------------------------------------------------------------------------
# Schema & Prompt Formatting (100% Parity with `src/model/schema.rs` & `native.rs`)
# ---------------------------------------------------------------------------


def noul_descriptions(q: dict[str, Any]) -> tuple[str, str] | None:
    crit = q.get("criteria")
    f_desc, t_desc = "", ""
    if isinstance(crit, dict):
        f_desc = str(crit.get("false") or "").strip()
        t_desc = str(crit.get("true") or "").strip()
    elif isinstance(crit, list):
        for item in crit:
            if isinstance(item, str) and ":" in item:
                k, d = item.split(":", 1)
                kl = k.strip().lower()
                if kl == "false":
                    f_desc = d.strip()
                elif kl == "true":
                    t_desc = d.strip()
    if f_desc or t_desc:
        return f_desc, t_desc
    return None


def render_options(q: dict[str, Any]) -> list[str]:
    qtype = q["type"].lower()
    crit = q.get("criteria")
    if qtype == "choice":
        if isinstance(crit, dict):
            return [
                f"{k}: {v}" if isinstance(v, str) else str(k) for k, v in crit.items()
            ]
        if isinstance(crit, list):
            return [s for s in crit if isinstance(s, str)]
        return []
    if qtype == "score":
        if isinstance(crit, list):
            return [
                f"level {i}: {v if isinstance(v, str) else ''}"
                for i, v in enumerate(crit)
            ]
        return ["level 0", "level 1", "level 2"]
    if qtype == "noul":
        pair = noul_descriptions(q) or ("", "")
        f_str = pair[0] if pair[0] else "no, the statement does not hold"
        t_str = pair[1] if pair[1] else "yes, the statement holds"
        return [f"false: {f_str}", f"true: {t_str}"]
    raise ValueError(f"Unsupported question type: {qtype}")


def choice_keys(q: dict[str, Any], options: list[str]) -> list[str]:
    crit = q.get("criteria")
    if isinstance(crit, dict):
        return [str(k) for k in crit.keys()]
    if isinstance(crit, list):
        return [str(s).split(":", 1)[0].strip() for s in crit if isinstance(s, str)]
    return [o.split(":", 1)[0].strip() for o in options]


def split_option_label_and_desc(q: dict[str, Any], option: str) -> tuple[str, str]:
    if ":" in option:
        lbl, desc = option.split(":", 1)
        if desc.strip():
            return lbl.strip(), desc.strip()
    label = option.strip()
    crit = q.get("criteria")
    if isinstance(crit, dict):
        if isinstance(crit.get(label), str):
            return label, crit[label]
        for k, v in crit.items():
            if str(k).lower() == label.lower() and isinstance(v, str):
                return label, v
    ins = str(q.get("instructions") or "").strip()
    return label, (ins if ins else label)


def render_nli_hypothesis(
    q: dict[str, Any], option: str, custom_template: str | None = None
) -> str:
    tpl = (
        custom_template
        or q.get("hypothesis_template")
        or "This text is about {label} ({description})."
    )
    label, desc = split_option_label_and_desc(q, option)
    return tpl.replace("{label}", label).replace("{description}", desc).replace("{}", label)


def level_description(q: dict[str, Any], option: str) -> str | None:
    if q["type"].lower() != "score" or ":" not in option:
        return None
    desc = option.split(":", 1)[1].strip()
    if ":" in desc:
        n_part, rest = desc.split(":", 1)
        if n_part.strip().isdigit():
            desc = rest.strip()
    desc = desc.rstrip(".")
    return desc if desc else None


def build_nli_hypothesis(
    q: dict[str, Any], option: str, template: str | None = None
) -> str:
    if template is not None or q.get("hypothesis_template") is not None:
        return render_nli_hypothesis(q, option, template)
    lvl = level_description(q, option)
    if lvl is not None:
        return f"This text is about {lvl}."
    raw = render_nli_hypothesis(q, option, "{label}")
    label = raw.replace("_", " ")
    desc = render_nli_hypothesis(q, option, "{description}")
    if desc == raw or desc == str(q.get("instructions") or "").strip():
        return f"This text is about {label}."
    return f"This text is about {label} ({desc})."


def resolve_ground_truth(
    row: dict[str, Any], qid: str, num_questions: int
) -> Any | None:
    gt = row.get("ground_truth")
    if isinstance(gt, dict):
        if qid in gt:
            return gt[qid]
        if num_questions == 1 and "default" in gt:
            return gt["default"]
        for k, v in gt.items():
            if k.lower() == qid.lower():
                return v
    if num_questions == 1 and "label" in row:
        return row["label"]
    return None


def parse_target_for_question(
    raw_target: Any,
    q: dict[str, Any],
    options: list[str],
    multi_label: bool = False,
) -> tuple[str, Any] | None:
    if raw_target is None:
        return None
    qtype = q["type"].lower()
    if qtype == "choice":
        keys = choice_keys(q, options)
        key_Lookup = {k.lower(): i for i, k in enumerate(keys)}
        opt_lookup = {o.lower(): i for i, o in enumerate(options)}
        if multi_label:
            items = (
                raw_target
                if isinstance(raw_target, list)
                else [s.strip() for s in str(raw_target).split(",") if s.strip()]
            )
            vec = [0.0] * len(keys)
            matched = False
            for item in items:
                s = str(item).strip().lower()
                idx = key_Lookup.get(s, opt_lookup.get(s))
                if idx is not None:
                    vec[idx] = 1.0
                    matched = True
            return ("multi_label", vec) if matched or items == [] else None
        s = str(raw_target).strip().lower()
        idx = key_Lookup.get(s, opt_lookup.get(s))
        return ("index", idx) if idx is not None else None

    if qtype == "score":
        try:
            val = int(round(float(raw_target)))
        except (TypeError, ValueError):
            return None
        clamped = max(0, min(val, len(options) - 1))
        return ("score", clamped)

    if qtype == "noul":
        if isinstance(raw_target, bool):
            b = raw_target
        elif isinstance(raw_target, (int, float)):
            b = raw_target != 0
        else:
            s = str(raw_target).strip().lower()
            if s in ("true", "1", "yes"):
                b = True
            elif s in ("false", "0", "no"):
                b = False
            else:
                return None
        return ("index", 1 if b else 0)
    return None


def get_temperature_bucket(qtype: str, k: int) -> str:
    qt = qtype.lower()
    if k <= 2:
        return f"{qt}:2"
    if k <= 5:
        return f"{qt}:3-5"
    if k <= 10:
        return f"{qt}:6-10"
    return f"{qt}:11+"


# ---------------------------------------------------------------------------
# AWS EC2 Spot Interruption Monitor (Stdlib IMDSv2 + SIGTERM/SIGINT)
# ---------------------------------------------------------------------------


class InterruptionMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.reason: str | None = None
        self._install_signals()
        self._thread = threading.Thread(target=self._poll_imdsv2, daemon=True)
        self._thread.start()

    def _install_signals(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            self.reason = f"signal:{sig_name}"
            self.stop_event.set()

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _fetch_imdsv2_token(self) -> str | None:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        try:
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8").strip()
        except Exception:
            return None
        return None

    def _poll_imdsv2(self) -> None:
        token = self._fetch_imdsv2_token()
        if not token:
            return  # Not running on AWS EC2; exit background monitor silently.
        refreshed_at = time.monotonic()
        while not self.stop_event.wait(5.0):
            if time.monotonic() - refreshed_at > 10800.0:
                token = self._fetch_imdsv2_token() or token
                refreshed_at = time.monotonic()
            req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/spot/instance-action",
                headers={"X-aws-ec2-metadata-token": token},
            )
            try:
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    if resp.status == 200:
                        payload = resp.read().decode("utf-8", errors="replace").strip()
                        self.reason = f"aws_spot_interruption:{payload}"
                        self.stop_event.set()
                        return
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    token = self._fetch_imdsv2_token() or token
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Post-Training Temperature Calibration (100% Parity with `calibrate.rs`)
# ---------------------------------------------------------------------------


def softmax_1d(logits: list[float], temp: float) -> list[float]:
    t = max(temp, 0.01)
    scaled = [z / t for z in logits]
    m = max(scaled)
    exps = [math.exp(z - m) for z in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def compute_bucket_metrics(
    samples: list[tuple[list[float], int]], temp: float, bins: int = 10
) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    n = len(samples)
    nll_sum = 0.0
    brier_sum = 0.0
    bin_counts = [0] * bins
    bin_acc = [0.0] * bins
    bin_conf = [0.0] * bins

    for raw_logits, target_idx in samples:
        probs = softmax_1d(raw_logits, temp)
        p_true = max(probs[target_idx], 1e-12)
        nll_sum += -math.log(p_true)
        brier_sum += sum(
            (p - (1.0 if i == target_idx else 0.0)) ** 2 for i, p in enumerate(probs)
        )
        pred_idx = max(range(len(probs)), key=probs.__getitem__)
        conf = probs[pred_idx]
        correct = 1.0 if pred_idx == target_idx else 0.0
        b = min(int(conf * bins), bins - 1)
        bin_counts[b] += 1
        bin_acc[b] += correct
        bin_conf[b] += conf

    ece = sum(
        abs(bin_acc[b] - bin_conf[b]) / n for b in range(bins) if bin_counts[b] > 0
    )
    return ece, nll_sum / n, brier_sum / n


def golden_section_search(
    fn: Any, lo: float, hi: float, tol: float = 1e-4, max_iter: int = 30
) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = fn(c)
    fd = fn(d)
    for _ in range(max_iter):
        if abs(hi - lo) <= tol:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = fn(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = fn(d)
    return 0.5 * (lo + hi)


def fit_temperature_buckets(
    bucket_samples: dict[str, list[tuple[list[float], int]]],
    checkpoint_repo: str,
    dataset_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fitted: dict[str, float] = {}
    total_n = 0
    raw_ece_weighted = 0.0
    cal_ece_weighted = 0.0
    cal_nll_weighted = 0.0

    ln_min = math.log(MIN_TEMPERATURE)
    ln_max = math.log(MAX_TEMPERATURE)
    base_grid = sorted(
        [1.0] + [math.exp(ln_min + (ln_max - ln_min) * (i / 39.0)) for i in range(40)]
    )

    for bucket, samples in sorted(bucket_samples.items()):
        n = len(samples)
        total_n += n
        raw_ece, raw_nll, _ = compute_bucket_metrics(samples, 1.0)
        if n >= MIN_BUCKET_SAMPLES:
            losses = [compute_bucket_metrics(samples, t)[1] for t in base_grid]
            best_idx = min(range(len(losses)), key=losses.__getitem__)
            lo = base_grid[max(0, best_idx - 1)]
            hi = base_grid[min(len(base_grid) - 1, best_idx + 1)]
            gss_local = golden_section_search(
                lambda t: compute_bucket_metrics(samples, t)[1], lo, hi
            )
            gss_global = golden_section_search(
                lambda t: compute_bucket_metrics(samples, t)[1],
                MIN_TEMPERATURE,
                MAX_TEMPERATURE,
            )
            candidates = base_grid + [gss_local, gss_global]
            best_t = 1.0
            best_pair = (raw_nll, raw_ece)
            for cand in candidates:
                t_round = round(cand, 3)
                c_ece, c_nll, _ = compute_bucket_metrics(samples, t_round)
                if c_ece <= raw_ece + 1e-9:
                    if c_nll < best_pair[0] - 1e-9 or (
                        abs(c_nll - best_pair[0]) <= 1e-9 and c_ece < best_pair[1]
                    ):
                        best_pair = (c_nll, c_ece)
                        best_t = t_round
        else:
            best_t = 1.0

        cal_ece, cal_nll, _ = compute_bucket_metrics(samples, best_t)
        fitted[bucket] = best_t
        raw_ece_weighted += raw_ece * n
        cal_ece_weighted += cal_ece * n
        cal_nll_weighted += cal_nll * n

    denom = max(total_n, 1)
    avg_raw_ece = raw_ece_weighted / denom
    avg_cal_ece = cal_ece_weighted / denom
    avg_cal_nll = cal_nll_weighted / denom
    improvement = (
        ((avg_raw_ece - avg_cal_ece) / avg_raw_ece) * 100.0
        if avg_raw_ece > 1e-6
        else 0.0
    )

    calibrated_cfg = {
        "checkpoint": checkpoint_repo,
        "temperature": 1.0,
        "temperature_by_options": fitted,
        "loss_function": "nll",
        "metadata": {
            "dataset": dataset_name,
            "samples_evaluated": total_n,
            "raw_mean_ece": round(avg_raw_ece, 6),
            "calibrated_mean_ece": round(avg_cal_ece, 6),
            "improvement_pct": round(improvement, 2),
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    summary = {
        "nll": round(avg_cal_nll, 6),
        "ece": round(avg_cal_ece, 6),
        "raw_ece": round(avg_raw_ece, 6),
        "fitted_temperatures": fitted,
    }
    return calibrated_cfg, summary


# ---------------------------------------------------------------------------
# Dry-Validate Preflight (Pure Stdlib)
# ---------------------------------------------------------------------------


def run_dry_validate(bundle_dir: Path, artifacts_dir: Path | None) -> int:
    manifest_path = bundle_dir / "bundle_manifest.json"
    schema_path = bundle_dir / "schema.json"
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    schema_sha = sha256_bytes(
        json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    all_exact: set[int] = set()
    all_norm: set[int] = set()
    split_counts: dict[str, int] = {}

    for split_name in ("train", "val", "cal"):
        info = bundle.get("splits", {}).get(split_name)
        if not info:
            continue
        p = bundle_dir / info["rel_path"]
        rows_n, f_sha, _, ex_set, nm_set, rows = compute_split_merkle_and_hashes(p)
        if f_sha != info["file_sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {split_name}: expected {info['file_sha256']}, got {f_sha}"
            )
        all_exact |= ex_set
        all_norm |= nm_set
        split_counts[split_name] = rows_n
        for row in rows[:5]:
            for qid, q in schema.items():
                opts = render_options(q)
                _ = resolve_ground_truth(row, qid, len(schema))
                if len(opts) < 2:
                    raise RuntimeError(f"Question '{qid}' rendered <2 options")

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        write_trained_inputs_filter(
            artifacts_dir / "trained_inputs.sha256.bin", all_exact, all_norm
        )

    print(
        json.dumps(
            {
                "valid": True,
                "run_id": bundle.get("run_id"),
                "paradigm": bundle.get("base_model", {}).get("paradigm"),
                "schema_sha256_checked": bool(schema_sha),
                "split_counts": split_counts,
                "unique_exact_hashes": len(all_exact),
                "unique_norm_hashes": len(all_norm),
            }
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Full GPU / Smoke-Test PyTorch + HuggingFace PEFT Training Engine
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    t0 = time.monotonic()
    monitor = InterruptionMonitor()

    bundle_dir = Path(args.bundle_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    bundle = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    schema: dict[str, Any] = json.loads(
        (bundle_dir / "schema.json").read_text(encoding="utf-8")
    )

    seed = int(bundle["hyperparameters"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Verify bundle split SHA-256 digests and build `trained_inputs.sha256.bin`
    all_exact: set[int] = set()
    all_norm: set[int] = set()
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_prov: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "val", "cal"):
        s_info = bundle.get("splits", {}).get(split_name)
        if not s_info:
            continue
        s_path = bundle_dir / s_info["rel_path"]
        n_rows, f_sha, merkle, ex_u64, nm_u64, rows = compute_split_merkle_and_hashes(
            s_path
        )
        if f_sha != s_info["file_sha256"]:
            raise RuntimeError(
                f"Data integrity failure on {split_name}: expected SHA-256 {s_info['file_sha256']}, got {f_sha}"
            )
        all_exact |= ex_u64
        all_norm |= nm_u64
        split_rows[split_name] = rows
        split_prov[split_name] = {
            "rows": n_rows,
            "file_sha256": f_sha,
            "state_merkle_sha256": merkle,
        }

    n_exact_u, n_norm_u = write_trained_inputs_filter(
        artifacts_dir / "trained_inputs.sha256.bin", all_exact, all_norm
    )

    # 2. Resolve base checkpoint directory (local override or HF Hub snapshot)
    bm = bundle["base_model"]
    paradigm = bm["paradigm"]
    repo_id = bm["repo_id"]
    sub_dir = bm.get("local_subdir") or ""

    if args.local_base_dir:
        base_dir = Path(args.local_base_dir).resolve()
    else:
        allow_patterns = None
        if sub_dir:
            allow_patterns = [f"{sub_dir}/*"]
        dl_root = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=bm.get("hf_revision"),
                allow_patterns=allow_patterns,
                ignore_patterns=["onnx/*", "*.bin", "img/*", "*.jpeg", "*.png"],
            )
        )
        base_dir = dl_root / sub_dir if sub_dir and (dl_root / sub_dir).exists() else dl_root

    remote_hdr_sha = safetensors_header_sha256(base_dir / "model.safetensors")
    if bm.get("local_base_header_sha256") and bm["local_base_header_sha256"] != remote_hdr_sha:
        print(
            f"[warn] Local base header SHA-256 ({bm['local_base_header_sha256'][:12]}) "
            f"differs from training host base ({remote_hdr_sha[:12]}).",
            file=sys.stderr,
        )

    # 3. Hardware & precision auto-detection (with DeBERTa-v3 FP16 guard!)
    is_deberta = "deberta" in bm["checkpoint_slug"].lower() or "deberta" in repo_id.lower()
    gpu_name = "cpu"
    vram_gb = 0.0
    driver_ver = "n/a"

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = round(props.total_memory / (1024**3), 2)
        cap = torch.cuda.get_device_capability(0)
        torch.backends.cuda.matmul.allow_tf32 = cap >= (8, 0)
        torch.backends.cudnn.allow_tf32 = cap >= (8, 0)
        if cap >= (8, 0):
            precision = "bf16"
            amp_dtype = torch.bfloat16
            use_scaler = False
        elif is_deberta:
            # DeBERTa-v3 disentangled c2p/p2c attention overflows FP16 (65504) -> force FP32 on T4!
            precision = "fp32"
            amp_dtype = torch.float32
            use_scaler = False
        else:
            precision = "fp16"
            amp_dtype = torch.float16
            use_scaler = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
        gpu_name = "apple_mps"
        precision = "fp32"
        amp_dtype = torch.float32
        use_scaler = False
    else:
        device = torch.device("cpu")
        precision = "fp32"
        amp_dtype = torch.float32
        use_scaler = False

    # 4. Construct native-parity PyTorch model for the checkpoint paradigm
    class LayaHeadContainer(nn.Module):
        def __init__(self, d: int, n_layers: int) -> None:
            super().__init__()
            n_heads = max(d // 64, 1)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d,
                        nhead=n_heads,
                        dim_feedforward=4 * d,
                        dropout=0.0,
                        activation="relu",
                        layer_norm_eps=1e-5,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )

        def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                h = layer(h, src_key_padding_mask=key_padding_mask)
            return h

    class LayaDecisionModule(nn.Module):
        """Exact PyTorch counterpart to `LayaModel` in `src/model/native.rs`."""

        def __init__(self, ckpt_dir: Path) -> None:
            super().__init__()
            agent_cfg = json.loads((ckpt_dir / "rl_agent_config.json").read_text("utf-8"))
            enc_cfg_dir = ckpt_dir / "encoder" if (ckpt_dir / "encoder").exists() else ckpt_dir
            enc_cfg = AutoConfig.from_pretrained(str(enc_cfg_dir), trust_remote_code=True)
            if hasattr(enc_cfg, "_attn_implementation"):
                enc_cfg._attn_implementation = "sdpa"
            self.config = enc_cfg
            self.encoder = AutoModel.from_config(enc_cfg, trust_remote_code=True)
            d = enc_cfg.hidden_size
            head_layers = int(agent_cfg["head_layers"])
            self.type_emb = nn.Embedding(3, d)
            self.head = LayaHeadContainer(d, head_layers)
            self.scorer = nn.Sequential(
                nn.LayerNorm(d, eps=1e-5),
                nn.Linear(d, d),
                nn.GELU(approximate="none"),
                nn.Linear(d, 1),
            )
            self.max_len = int(agent_cfg["max_len"])
            self.head_max_len = int(agent_cfg["head_max_len"])
            state = load_file(str(ckpt_dir / "model.safetensors"))
            filtered = {
                k: v
                for k, v in state.items()
                if not k.startswith("act_head.") and k != "temperature"
            }
            missing, unexpected = self.load_state_dict(filtered, strict=False)
            bad_missing = [m for m in missing if not m.startswith("act_head.")]
            if bad_missing:
                raise RuntimeError(
                    f"Laya state_dict missing required keys: {bad_missing[:8]}"
                )
            if unexpected:
                print(
                    f"[info] Ignored extra checkpoint tensors: {unexpected[:5]}",
                    file=sys.stderr,
                )

        def get_input_embeddings(self) -> nn.Module:
            return self.encoder.get_input_embeddings()

        def forward_markers(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            qtypes: torch.Tensor,
            markers_per_seq: list[list[int]],
        ) -> list[torch.Tensor]:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = enc_out.last_hidden_state + self.type_emb(qtypes).unsqueeze(1)
            pad_mask = attention_mask == 0
            h = self.head(h, pad_mask)
            b, t, d = h.shape
            flat = h.reshape(b * t, d)
            flat_indices: list[int] = []
            counts: list[int] = []
            for i, m_list in enumerate(markers_per_seq):
                counts.append(len(m_list))
                for m in m_list:
                    flat_indices.append(i * t + m)
            idx_t = torch.tensor(flat_indices, dtype=torch.long, device=h.device)
            all_scores = self.scorer(flat.index_select(0, idx_t)).squeeze(-1)
            return list(torch.split(all_scores, counts))

    tok_dir = (
        base_dir / "tokenizer"
        if (base_dir / "tokenizer" / "tokenizer.json").exists()
        else base_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)

    entail_idx = 0
    not_entail_idx = 1
    non_entail_indices: list[int] = [1]
    mlm_prefix_ids: list[int] = []
    mlm_suffix_ids: list[int] = []

    if paradigm == "laya_joint_decision":
        base_model: nn.Module = LayaDecisionModule(base_dir)
        max_seq_len = int(bm.get("max_len") or base_model.max_len)
        head_max_len = int(bm.get("head_max_len") or base_model.head_max_len)
    elif paradigm == "nli_cross_encoder":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            str(base_dir), trust_remote_code=True
        )
        cfg_id2label = {
            int(k): str(v).lower() for k, v in base_model.config.id2label.items()
        }
        entail_candidates = [i for i, l in cfg_id2label.items() if l.startswith("entail")]
        if not entail_candidates:
            raise RuntimeError(f"No entailment label in id2label: {cfg_id2label}")
        entail_idx = entail_candidates[0]
        contra_candidates = [
            i
            for i, l in cfg_id2label.items()
            if l.startswith("contra") or l.startswith("not")
        ]
        not_entail_idx = (
            contra_candidates[0]
            if contra_candidates
            else next(i for i in cfg_id2label if i != entail_idx)
        )
        non_entail_indices = [i for i in sorted(cfg_id2label) if i != entail_idx]
        max_seq_len = int(bm.get("max_len") or 512)
        head_max_len = 256
    elif paradigm == "mlm_cloze":
        base_model = AutoModelForMaskedLM.from_pretrained(
            str(base_dir), trust_remote_code=True
        )
        with_sp = tokenizer.encode("probe", add_special_tokens=True)
        bare_sp = tokenizer.encode("probe", add_special_tokens=False)
        start = next(
            i
            for i in range(len(with_sp) - len(bare_sp) + 1)
            if with_sp[i : i + len(bare_sp)] == bare_sp
        )
        mlm_prefix_ids = with_sp[:start]
        mlm_suffix_ids = with_sp[start + len(bare_sp) :]
        max_seq_len = min(int(bm.get("max_len") or 2048), 2048)
        head_max_len = 256
    else:
        raise RuntimeError(f"Unsupported paradigm: {paradigm}")

    # 5. Apply PEFT (`lora` / `dora`) or `full` fine-tuning
    method_cfg = bundle["method"]
    mode = method_cfg["mode"].lower()
    target_modules = list(method_cfg.get("target_modules") or [])
    modules_to_save = list(method_cfg.get("modules_to_save") or [])

    if mode in ("lora", "dora"):
        # Filter target_modules to linear/embedding suffixes actually present in base_model
        named_mod_names = [name for name, _ in base_model.named_modules() if name]
        valid_targets = [
            tm
            for tm in target_modules
            if any(n == tm or n.endswith(f".{tm}") for n in named_mod_names)
        ]
        if not valid_targets:
            raise RuntimeError(
                f"None of target_modules {target_modules} matched modules in {repo_id}"
            )
        valid_mts = [
            ms
            for ms in modules_to_save
            if any(n == ms or n.endswith(f".{ms}") for n in named_mod_names)
        ]
        lora_cfg = LoraConfig(
            base_model_name_or_path=repo_id,
            r=int(method_cfg["r"]),
            lora_alpha=float(method_cfg["lora_alpha"]),
            lora_dropout=float(method_cfg.get("lora_dropout", 0.05)),
            use_rslora=bool(method_cfg.get("use_rslora", True)),
            use_dora=(mode == "dora" or bool(method_cfg.get("use_dora", False))),
            bias="none",
            target_modules=valid_targets,
            modules_to_save=valid_mts if valid_mts else None,
        )
        model: nn.Module = get_peft_model(base_model, lora_cfg)
        target_modules = valid_targets
        modules_to_save = valid_mts
    elif mode == "full":
        model = base_model
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise RuntimeError(f"Unsupported fine-tune mode: {mode}")

    model.to(device)

    # 6. Register NEFTune noisy-embedding hook during `model.train()`
    hp = bundle["hyperparameters"]
    neftune_alpha = float(hp.get("neftune_alpha", 5.0))
    emb_module = (
        model.get_base_model().get_input_embeddings()
        if hasattr(model, "get_base_model")
        else model.get_input_embeddings()
    )

    def _neftune_hook(_mod: nn.Module, _inp: Any, output: torch.Tensor) -> torch.Tensor:
        if model.training and neftune_alpha > 0.0 and output.ndim == 3:
            dims = float(output.size(1) * output.size(2))
            mag = neftune_alpha / math.sqrt(max(dims, 1.0))
            noise = torch.empty_like(output).uniform_(-mag, mag)
            return output + noise
        return output

    if emb_module is not None and neftune_alpha > 0.0:
        emb_module.register_forward_hook(_neftune_hook)

    # 7. Tokenize & compile dataset examples matching `src/model/native.rs`
    multi_label_flag = bool(hp.get("multi_label", False))
    mask_token_str = str(tokenizer.mask_token or "[MASK]")
    cls_id = int(tokenizer.cls_token_id or 0)
    sep_id = int(tokenizer.sep_token_id or 0)
    mask_id = int(tokenizer.mask_token_id or 0)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    def encode_bare(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def build_laya_example(
        state_text: str, q: dict[str, Any], options: list[str]
    ) -> tuple[list[int], list[int], int]:
        qtype = q["type"].lower()
        ins = str(q.get("instructions") or "").replace(mask_token_str, " ")
        head_ids = encode_bare(f"{qtype} question: {ins}")
        opt_ids: list[list[int]] = []
        for o in options:
            toks = encode_bare(" " + o.replace(mask_token_str, " "))[:MAX_OPTION_TOKENS]
            opt_ids.append([mask_id] + toks)

        used = sum(len(o) for o in opt_ids)
        opt_budget = head_max_len - used
        if opt_budget < 16:
            per = max((head_max_len - 16) // max(len(opt_ids), 1), 4)
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        head_len = min(len(head_ids), max(opt_budget, 8))

        prefix = [cls_id] + head_ids[:head_len] + [sep_id]
        markers: list[int] = []
        for o in opt_ids:
            markers.append(len(prefix))
            prefix.extend(o)
        prefix.append(sep_id)

        room = max(max_seq_len - len(prefix) - 1, 0)
        st_ids = encode_bare(state_text.replace(mask_token_str, " "))[:room]
        ids = (prefix + st_ids + [sep_id])[:max_seq_len]
        markers = [m for m in markers if m < max_seq_len]
        return ids, markers, QTYPE_INDEX[qtype]

    def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compiled: list[dict[str, Any]] = []
        for row in rows:
            st_text = extract_state_text(row)
            if not st_text.strip():
                continue
            for qid, q in schema.items():
                raw_gt = resolve_ground_truth(row, qid, len(schema))
                options = render_options(q)
                parsed = parse_target_for_question(raw_gt, q, options, multi_label_flag)
                if parsed is None:
                    continue
                target_kind, target_val = parsed
                qtype = q["type"].lower()
                bucket = get_temperature_bucket(qtype, max(len(options), 2))

                if paradigm == "laya_joint_decision":
                    ids, markers, qt_idx = build_laya_example(st_text, q, options)
                    if len(markers) != len(options):
                        continue
                    compiled.append(
                        {
                            "paradigm": paradigm,
                            "ids": ids,
                            "markers": markers,
                            "qtype_idx": qt_idx,
                            "qtype": qtype,
                            "bucket": bucket,
                            "num_options": len(options),
                            "target_kind": target_kind,
                            "target": target_val,
                            "approx_len": len(ids),
                        }
                    )
                elif paradigm == "nli_cross_encoder":
                    noul_has_desc = qtype == "noul" and noul_descriptions(q) is not None
                    if qtype == "noul" and not noul_has_desc:
                        hyps = [str(q.get("instructions") or "")]
                        bare_noul = True
                    elif noul_has_desc:
                        hyps = [
                            (o.split(":", 1)[1] if ":" in o else o).strip().rstrip(".") + "."
                            for o in options
                        ]
                        bare_noul = False
                    else:
                        hyps = [build_nli_hypothesis(q, o) for o in options]
                        bare_noul = False
                    hyp_ids = [encode_bare(h) for h in hyps]
                    longest = max((len(h) for h in hyp_ids), default=0)
                    room = max(max_seq_len - longest - 3, 1)
                    st_ids = encode_bare(st_text)[:room]
                    pairs = [[cls_id] + st_ids + [sep_id] + h + [sep_id] for h in hyp_ids]
                    compiled.append(
                        {
                            "paradigm": paradigm,
                            "pairs": pairs,
                            "bare_noul": bare_noul,
                            "qtype": qtype,
                            "bucket": bucket,
                            "num_options": len(options),
                            "target_kind": target_kind,
                            "target": target_val,
                            "approx_len": sum(len(p) for p in pairs),
                        }
                    )
                elif paradigm == "mlm_cloze":
                    if qtype == "noul":
                        words = ["no", "yes"]
                    else:
                        words = [
                            level_description(q, o)
                            or render_nli_hypothesis(q, o, "{label}").replace("_", " ")
                            for o in options
                        ]
                    ins = str(q.get("instructions") or "").strip()
                    if qtype == "noul" and ins.endswith("?"):
                        q_prompt = f"\nQuestion: {ins}\nAnswer:"
                    elif qtype == "noul":
                        q_prompt = f"\nStatement: {ins}\nIs the statement true? Answer:"
                    else:
                        q_prompt = f"\nQuestion: {ins}\nAnswer:"
                    q_ids = encode_bare(q_prompt)
                    end_ids = encode_bare(".")
                    label_toks = [
                        encode_bare(f" {w}")[:MAX_LABEL_TOKENS] for w in words
                    ]
                    if any(len(lt) == 0 for lt in label_toks):
                        continue
                    longest = max(len(lt) for lt in label_toks)
                    overhead = (
                        len(mlm_prefix_ids)
                        + len(q_ids)
                        + longest
                        + len(end_ids)
                        + len(mlm_suffix_ids)
                    )
                    room = max(max_seq_len - overhead, 1)
                    st_ids = encode_bare(st_text)[:room]
                    na_ids = encode_bare("N/A")
                    compiled.append(
                        {
                            "paradigm": paradigm,
                            "st_ids": st_ids,
                            "na_ids": na_ids,
                            "q_ids": q_ids,
                            "end_ids": end_ids,
                            "label_toks": label_toks,
                            "qtype": qtype,
                            "bucket": bucket,
                            "num_options": len(options),
                            "target_kind": target_kind,
                            "target": target_val,
                            "approx_len": len(st_ids) + overhead,
                        }
                    )
        return compiled

    train_examples = build_examples(split_rows["train"])
    val_examples = build_examples(split_rows.get("val") or split_rows["train"][:32])
    cal_examples = build_examples(
        split_rows.get("cal") or split_rows.get("val") or split_rows["train"][:32]
    )

    if args.smoke_test:
        train_examples = train_examples[:16]
        val_examples = val_examples[:8]
        cal_examples = cal_examples[:8]

    if not train_examples:
        raise RuntimeError("Zero valid labeled training examples matched schema")

    # 8. Task-aligned loss computation (Choice CE, Multi-label BCE, Score Soft-KL + RPS, Noul CE+Brier)
    label_smoothing = float(hp.get("label_smoothing", 0.02))
    rdrop_alpha = float(hp.get("rdrop_alpha", 0.0))

    def compute_example_loss(logits: torch.Tensor, ex: dict[str, Any]) -> torch.Tensor:
        qtype = ex["qtype"]
        tk = ex["target_kind"]
        if tk == "multi_label":
            tgt = torch.tensor(ex["target"], dtype=torch.float32, device=logits.device)
            return F.binary_cross_entropy_with_logits(logits.float(), tgt)
        if qtype == "score":
            y = int(ex["target"])
            k = logits.size(-1)
            probs = F.softmax(logits.float(), dim=-1)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            idx = torch.arange(k, dtype=torch.float32, device=logits.device)
            sigma = 0.65
            soft_q = torch.softmax(-((idx - float(y)) ** 2) / (2.0 * sigma * sigma), dim=-1)
            kl = F.kl_div(log_probs, soft_q, reduction="sum")
            cdf_pred = torch.cumsum(probs, dim=-1)[:-1]
            cdf_true = (idx[:-1] >= float(y)).float()
            rps = torch.sum((cdf_pred - cdf_true) ** 2)
            return 0.5 * kl + 0.5 * rps
        if qtype == "noul":
            y_t = torch.tensor(int(ex["target"]), dtype=torch.long, device=logits.device)
            ce = F.cross_entropy(logits.float(), y_t)
            probs = F.softmax(logits.float(), dim=-1)
            onehot = F.one_hot(y_t, num_classes=logits.size(-1)).float()
            brier = torch.sum((probs - onehot) ** 2)
            return 0.8 * ce + 0.2 * brier
        y_t = torch.tensor(int(ex["target"]), dtype=torch.long, device=logits.device)
        return F.cross_entropy(
            logits.float(), y_t, label_smoothing=label_smoothing
        )

    def forward_batch_logits(batch: list[dict[str, Any]]) -> list[torch.Tensor]:
        if paradigm == "laya_joint_decision":
            max_t = max(len(ex["ids"]) for ex in batch)
            ids_mat = [ex["ids"] + [pad_id] * (max_t - len(ex["ids"])) for ex in batch]
            mask_mat = [
                [1] * len(ex["ids"]) + [0] * (max_t - len(ex["ids"])) for ex in batch
            ]
            qt_vec = [ex["qtype_idx"] for ex in batch]
            markers = [ex["markers"] for ex in batch]
            inp = torch.tensor(ids_mat, dtype=torch.long, device=device)
            msk = torch.tensor(mask_mat, dtype=torch.long, device=device)
            qts = torch.tensor(qt_vec, dtype=torch.long, device=device)
            inner = model.base_model.model if hasattr(model, "base_model") else model
            return inner.forward_markers(inp, msk, qts, markers)

        if paradigm == "nli_cross_encoder":
            all_pairs: list[list[int]] = [p for ex in batch for p in ex["pairs"]]
            max_t = max(len(p) for p in all_pairs)
            ids_mat = [p + [pad_id] * (max_t - len(p)) for p in all_pairs]
            mask_mat = [[1] * len(p) + [0] * (max_t - len(p)) for p in all_pairs]
            inp = torch.tensor(ids_mat, dtype=torch.long, device=device)
            msk = torch.tensor(mask_mat, dtype=torch.long, device=device)
            raw_out = model(input_ids=inp, attention_mask=msk).logits
            out_list: list[torch.Tensor] = []
            cursor = 0
            for ex in batch:
                cnt = len(ex["pairs"])
                chunk = raw_out[cursor : cursor + cnt]
                cursor += cnt
                if ex["bare_noul"]:
                    lp = F.log_softmax(chunk[0].float(), dim=-1)
                    non_ent = torch.logsumexp(lp[non_entail_indices], dim=0)
                    out_list.append(torch.stack([non_ent, lp[entail_idx]]))
                elif ex["target_kind"] == "multi_label":
                    out_list.append(chunk[:, entail_idx] - chunk[:, not_entail_idx])
                else:
                    out_list.append(chunk[:, entail_idx])
            return out_list

        # mlm_cloze
        out_list = []
        for ex in batch:
            lengths = sorted({len(lt) for lt in ex["label_toks"]})
            scores: list[torch.Tensor | None] = [None] * len(ex["label_toks"])
            for n_masks in lengths:
                seq = (
                    mlm_prefix_ids
                    + ex["st_ids"]
                    + ex["q_ids"]
                    + [mask_id] * n_masks
                    + ex["end_ids"]
                    + mlm_suffix_ids
                )
                first_m = len(mlm_prefix_ids) + len(ex["st_ids"]) + len(ex["q_ids"])
                inp = torch.tensor([seq], dtype=torch.long, device=device)
                logits_seq = model(input_ids=inp).logits[0, first_m : first_m + n_masks]
                lp = F.log_softmax(logits_seq.float(), dim=-1)

                with torch.no_grad():
                    na_seq = (
                        mlm_prefix_ids
                        + ex["na_ids"]
                        + ex["q_ids"]
                        + [mask_id] * n_masks
                        + ex["end_ids"]
                        + mlm_suffix_ids
                    )
                    na_m = len(mlm_prefix_ids) + len(ex["na_ids"]) + len(ex["q_ids"])
                    na_inp = torch.tensor([na_seq], dtype=torch.long, device=device)
                    na_lp = F.log_softmax(
                        model(input_ids=na_inp).logits[0, na_m : na_m + n_masks].float(),
                        dim=-1,
                    )

                for idx_l, lt in enumerate(ex["label_toks"]):
                    if len(lt) == n_masks:
                        tok_idx = torch.tensor(lt, dtype=torch.long, device=device)
                        rows_idx = torch.arange(n_masks, dtype=torch.long, device=device)
                        s_val = (lp[rows_idx, tok_idx] - na_lp[rows_idx, tok_idx]).sum()
                        scores[idx_l] = s_val
            out_list.append(torch.stack([s for s in scores if s is not None]))
        return out_list

    # 9. Differential AdamW optimizer & cosine warmup scheduler
    lr_lora = float(hp.get("lr_lora", 1.5e-4 if mode != "full" else 2e-5))
    lr_head = float(hp.get("lr_head", 1.0e-4))
    weight_decay = float(hp.get("weight_decay", 0.01))
    no_decay = ("bias", "LayerNorm", "layer_norm", "norm")

    head_markers = ("scorer", "type_emb", "classifier", "pooler", "head", "modules_to_save")
    lora_decay, lora_nodecay, head_decay, head_nodecay = [], [], [], []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head_p = any(hm in pname for hm in head_markers) and "lora_" not in pname
        is_nd = any(nd in pname for nd in no_decay)
        if is_head_p and is_nd:
            head_nodecay.append(param)
        elif is_head_p:
            head_decay.append(param)
        elif is_nd:
            lora_nodecay.append(param)
        else:
            lora_decay.append(param)

    param_groups = [
        g
        for g in [
            {"params": lora_decay, "lr": lr_lora, "weight_decay": weight_decay},
            {"params": lora_nodecay, "lr": lr_lora, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
            {"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0},
        ]
        if g["params"]
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98))

    batch_size = 4 if args.smoke_test else max(int(hp.get("batch_size", 16)), 1)
    grad_accum = 1 if args.smoke_test else max(int(hp.get("grad_accum", 2)), 1)
    epochs = 1 if args.smoke_test else max(int(hp.get("epochs", 5)), 1)
    steps_per_epoch = max(math.ceil(len(train_examples) / (batch_size * grad_accum)), 1)
    total_steps = 3 if args.smoke_test else epochs * steps_per_epoch
    warmup_steps = max(int(total_steps * float(hp.get("warmup_ratio", 0.08))), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        prog = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Evaluation helper
    @torch.no_grad()
    def evaluate_examples(
        examples: list[dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, list[tuple[list[float], int]]]]:
        model.eval()
        losses: list[float] = []
        correct = 0
        counted = 0
        bucket_collect: dict[str, list[tuple[list[float], int]]] = {}
        per_class_tp: dict[str, int] = {}
        per_class_fp: dict[str, int] = {}
        per_class_fn: dict[str, int] = {}

        for start_i in range(0, len(examples), batch_size):
            batch = examples[start_i : start_i + batch_size]
            with torch.autocast(
                device_type=device.type if device.type in ("cuda", "cpu") else "cpu",
                dtype=amp_dtype,
                enabled=(precision in ("bf16", "fp16") and device.type == "cuda"),
            ):
                logits_list = forward_batch_logits(batch)
            for logits, ex in zip(logits_list, batch):
                losses.append(float(compute_example_loss(logits, ex).item()))
                raw_vec = [float(x) for x in logits.detach().float().cpu().tolist()]
                if ex["target_kind"] != "multi_label":
                    tgt_i = int(ex["target"])
                    bucket_collect.setdefault(ex["bucket"], []).append((raw_vec, tgt_i))
                    pred_i = max(range(len(raw_vec)), key=raw_vec.__getitem__)
                    hit = pred_i == tgt_i
                    correct += int(hit)
                    counted += 1
                    t_key = f"{ex['bucket']}:{tgt_i}"
                    p_key = f"{ex['bucket']}:{pred_i}"
                    if hit:
                        per_class_tp[t_key] = per_class_tp.get(t_key, 0) + 1
                    else:
                        per_class_fn[t_key] = per_class_fn.get(t_key, 0) + 1
                        per_class_fp[p_key] = per_class_fp.get(p_key, 0) + 1

        all_keys = sorted(set(per_class_tp) | set(per_class_fn))
        f1s: list[float] = []
        for ck in all_keys:
            tp = per_class_tp.get(ck, 0)
            fp = per_class_fp.get(ck, 0)
            fn = per_class_fn.get(ck, 0)
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)

        flat_samples = [s for v in bucket_collect.values() for s in v]
        ece_val, _, _ = compute_bucket_metrics(flat_samples, 1.0)
        metrics = {
            "nll": round(sum(losses) / max(len(losses), 1), 6),
            "accuracy": round(correct / max(counted, 1), 6),
            "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
            "ece": round(ece_val, 6),
        }
        return metrics, bucket_collect

    def length_grouped_batches(
        examples: list[dict[str, Any]], ep_seed: int
    ) -> list[list[dict[str, Any]]]:
        rng = random.Random(ep_seed)
        pool = list(examples)
        rng.shuffle(pool)
        mega = batch_size * 16
        batches: list[list[dict[str, Any]]] = []
        for i in range(0, len(pool), mega):
            window = sorted(pool[i : i + mega], key=lambda x: x["approx_len"], reverse=True)
            for j in range(0, len(window), batch_size):
                batches.append(window[j : j + batch_size])
        rng.shuffle(batches)
        return batches

    def save_resumable_checkpoint(
        ckpt_dir: Path, epoch_i: int, batch_i: int, step_i: int, best_nll: float
    ) -> None:
        tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        if mode in ("lora", "dora"):
            model.save_pretrained(str(tmp), safe_serialization=True)
        else:
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                str(tmp / "model.safetensors"),
            )
        torch.save(
            {
                "epoch": epoch_i,
                "batch_idx": batch_i,
                "global_step": step_i,
                "best_val_nll": best_nll,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "py_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
            },
            str(tmp / "trainer_state.pt"),
        )
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        os.replace(tmp, ckpt_dir)
        if args.on_checkpoint_cmd:
            subprocess.run(args.on_checkpoint_cmd, shell=True, check=False)

    # Optional `--resume` from `checkpoints/last`
    start_epoch = 0
    resume_batch_idx = 0
    global_step = 0
    best_val_nll = float("inf")
    last_ckpt = checkpoints_dir / "last"

    if args.resume and (last_ckpt / "trainer_state.pt").exists():
        st = torch.load(str(last_ckpt / "trainer_state.pt"), map_location="cpu", weights_only=False)
        if mode in ("lora", "dora"):
            from peft import set_peft_model_state_dict

            ad_tensors = load_file(str(last_ckpt / "adapter_model.safetensors"))
            set_peft_model_state_dict(model, ad_tensors)
        else:
            model.load_state_dict(load_file(str(last_ckpt / "model.safetensors")))
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        start_epoch = int(st["epoch"])
        resume_batch_idx = int(st["batch_idx"])
        global_step = int(st["global_step"])
        best_val_nll = float(st["best_val_nll"])

    val_pre_metrics, _ = evaluate_examples(val_examples)
    best_val_metrics = dict(val_pre_metrics)
    best_step = 0
    best_state_cpu: dict[str, torch.Tensor] | None = None
    eval_every = max(args.eval_steps or max(steps_per_epoch // 2, 1), 1)
    patience = max(args.patience, 1)
    bad_evals = 0
    history_path = artifacts_dir / "metrics_history.jsonl"

    def snapshot_trainable_cpu() -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if v.requires_grad or any(ms in k for ms in modules_to_save) or mode == "full"
        }

    best_state_cpu = snapshot_trainable_cpu()
    stop_early = False

    for epoch in range(start_epoch, epochs):
        batches = length_grouped_batches(train_examples, seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_count = 0

        for b_idx, batch in enumerate(batches):
            if epoch == start_epoch and b_idx < resume_batch_idx:
                continue

            with torch.autocast(
                device_type=device.type if device.type in ("cuda", "cpu") else "cpu",
                dtype=amp_dtype,
                enabled=(precision in ("bf16", "fp16") and device.type == "cuda"),
            ):
                logits_1 = forward_batch_logits(batch)
                loss_1 = torch.stack(
                    [compute_example_loss(lg, ex) for lg, ex in zip(logits_1, batch)]
                ).mean()
                if rdrop_alpha > 0.0:
                    logits_2 = forward_batch_logits(batch)
                    loss_2 = torch.stack(
                        [compute_example_loss(lg, ex) for lg, ex in zip(logits_2, batch)]
                    ).mean()
                    kl_terms = []
                    for l1, l2 in zip(logits_1, logits_2):
                        p1, lp1 = F.softmax(l1.float(), -1), F.log_softmax(l1.float(), -1)
                        p2, lp2 = F.softmax(l2.float(), -1), F.log_softmax(l2.float(), -1)
                        kl_terms.append(
                            0.5 * (F.kl_div(lp1, p2, reduction="sum") + F.kl_div(lp2, p1, reduction="sum"))
                        )
                    loss = 0.5 * (loss_1 + loss_2) + rdrop_alpha * torch.stack(kl_terms).mean()
                else:
                    loss = loss_1
                scaled_loss = loss / grad_accum

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += float(loss.detach().item())
            micro_count += 1

            is_step_boundary = (micro_count % grad_accum == 0) or (b_idx + 1 == len(batches))
            if not is_step_boundary:
                continue

            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Safe step-boundary Spot / SIGTERM interruption save
            if monitor.stop_event.is_set():
                save_resumable_checkpoint(
                    last_ckpt, epoch, b_idx + 1, global_step, best_val_nll
                )
                interrupted_manifest = {
                    "manifest_version": MANIFEST_VERSION,
                    "run_id": bundle["run_id"],
                    "status": "interrupted_resumable",
                    "interruption_reason": monitor.reason,
                    "created_by_lemmata_version": bundle.get(
                        "created_by_lemmata_version", "0.1.0"
                    ),
                    "base_model": {**bm, "remote_base_header_sha256": remote_hdr_sha},
                    "schema": bundle["schema"],
                    "dataset_provenance": {
                        "split_id": bundle.get("split_id"),
                        **split_prov,
                    },
                }
                (artifacts_dir / "training_manifest.json").write_text(
                    json.dumps(interrupted_manifest, indent=2), encoding="utf-8"
                )
                return EXIT_SPOT_INTERRUPTED

            if global_step % eval_every == 0 or global_step >= total_steps:
                val_m, _ = evaluate_examples(val_examples)
                rec = {
                    "step": global_step,
                    "epoch": round(epoch + (b_idx + 1) / len(batches), 4),
                    "train_loss": round(accum_loss / max(micro_count, 1), 6),
                    "lr": round(float(scheduler.get_last_lr()[0]), 8),
                    **{f"val_{k}": v for k, v in val_m.items()},
                }
                with history_path.open("a", encoding="utf-8") as hf:
                    hf.write(json.dumps(rec) + "\n")
                if val_m["nll"] < best_val_nll - 1e-5:
                    best_val_nll = val_m["nll"]
                    best_val_metrics = dict(val_m)
                    best_step = global_step
                    best_state_cpu = snapshot_trainable_cpu()
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience and not args.smoke_test:
                        stop_early = True
                        break
                model.train()

            if global_step >= total_steps:
                stop_early = True
                break

        resume_batch_idx = 0
        if stop_early:
            break

    # 10. Restore best validation checkpoint & fit post-training temperature scaling on `cal`
    if best_state_cpu is not None:
        model.load_state_dict(best_state_cpu, strict=False)

    _, cal_buckets = evaluate_examples(cal_examples)
    cal_cfg, cal_summary = fit_temperature_buckets(
        cal_buckets, repo_id, "cal_split" if "cal" in split_rows else "val_split"
    )
    cal_out_path = artifacts_dir / "lemmata_calibration.json"
    cal_out_path.write_text(json.dumps(cal_cfg, indent=2), encoding="utf-8")
    cal_sha = sha256_file(cal_out_path)

    # 11. Export PEFT adapter (or full merged model) + `training_manifest.json`
    adapter_sha: str | None = None
    merged_hdr_sha: str | None = None
    adapter_bytes: int | None = None

    def copy_sanitized_model_configs(src_dir: Path, dst_dir: Path) -> None:
        for cfg_name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            src = src_dir / cfg_name
            if src.is_file():
                shutil.copy2(src, dst_dir / cfg_name)
        rl_src = src_dir / "rl_agent_config.json"
        if rl_src.is_file():
            rl_obj = json.loads(rl_src.read_text(encoding="utf-8"))
            if isinstance(rl_obj, dict):
                rl_obj.pop("temperature", None)
                rl_obj.pop("temperature_by_options", None)
            (dst_dir / "rl_agent_config.json").write_text(
                json.dumps(rl_obj, indent=2), encoding="utf-8"
            )
        for sub_n in ("encoder", "tokenizer"):
            if (src_dir / sub_n).is_dir():
                shutil.copytree(src_dir / sub_n, dst_dir / sub_n, dirs_exist_ok=True)

    if mode in ("lora", "dora"):
        model.save_pretrained(str(artifacts_dir), safe_serialization=True)
        ad_st = artifacts_dir / "adapter_model.safetensors"
        adapter_sha = sha256_file(ad_st)
        adapter_bytes = ad_st.stat().st_size
        if args.merge_full:
            merged_model = model.merge_and_unload()
            merged_st = artifacts_dir / "model.safetensors"
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in merged_model.state_dict().items()},
                str(merged_st),
            )
            merged_hdr_sha = safetensors_header_sha256(merged_st)
            copy_sanitized_model_configs(base_dir, artifacts_dir)
    else:
        merged_st = artifacts_dir / "model.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(merged_st),
        )
        merged_hdr_sha = safetensors_header_sha256(merged_st)
        copy_sanitized_model_configs(base_dir, artifacts_dir)

    wall_secs = round(time.monotonic() - t0, 2)
    final_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": bundle["run_id"],
        "status": "completed",
        "created_by_lemmata_version": bundle.get("created_by_lemmata_version", "0.1.0"),
        "git_commit": bundle.get("git_commit"),
        "base_model": {
            "checkpoint_slug": bm["checkpoint_slug"],
            "repo_id": repo_id,
            "hf_revision": bm.get("hf_revision"),
            "paradigm": paradigm,
            "local_base_header_sha256": bm.get("local_base_header_sha256"),
            "remote_base_header_sha256": remote_hdr_sha,
            "max_len": max_seq_len,
            "head_max_len": head_max_len,
        },
        "schema": bundle["schema"],
        "dataset_provenance": {
            "split_id": bundle.get("split_id"),
            **split_prov,
            "locked_test_guard": {
                "expected_test_file_sha256": bundle.get("expected_test_file_sha256"),
                "trained_inputs_filter_file": "trained_inputs.sha256.bin",
                "trained_unique_exact_hashes": n_exact_u,
                "trained_unique_norm_hashes": n_norm_u,
            },
        },
        "method": {
            **method_cfg,
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        },
        "hyperparameters": {
            **hp,
            "effective_batch_size": batch_size * grad_accum,
            "precision": precision,
            "attn_implementation": "sdpa",
            "early_stopping_metric": "val_nll",
        },
        "results": {
            "best_step": best_step,
            "val_pre_train": val_pre_metrics,
            "val_best_raw": best_val_metrics,
            "cal_post_temp": cal_summary,
        },
        "environment": {
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "driver_version": driver_ver,
            "torch_version": torch.__version__,
            "wall_time_secs": wall_secs,
        },
        "artifacts": {
            "adapter_safetensors_sha256": adapter_sha,
            "merged_safetensors_header_sha256": merged_hdr_sha,
            "adapter_bytes": adapter_bytes,
            "calibration_sha256": cal_sha,
        },
    }
    (artifacts_dir / "training_manifest.json").write_text(
        json.dumps(final_manifest, indent=2), encoding="utf-8"
    )
    print(f"Completed fine-tuning run {bundle['run_id']} in {wall_secs:.1f}s -> {artifacts_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_artifacts = script_dir.parent / "artifacts"
    default_ckpts = script_dir.parent / "checkpoints"
    p = argparse.ArgumentParser(
        description="Self-bootstrapping Lemmata System 1 GPU fine-tuning runner"
    )
    p.add_argument("--bundle-dir", default=str(script_dir), help="Path to bundle directory")
    p.add_argument(
        "--artifacts-dir", default=str(default_artifacts), help="Output directory for artifacts"
    )
    p.add_argument(
        "--checkpoints-dir",
        default=str(default_ckpts),
        help="Directory for resumable Spot/SIGTERM checkpoints",
    )
    p.add_argument("--local-base-dir", default=None, help="Optional local base checkpoint path")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/last")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 3-step smoke test on tiny slice to validate pipeline",
    )
    p.add_argument(
        "--dry-validate",
        action="store_true",
        help="Validate bundle digests, schema options, and leakage filter using Python stdlib only",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p.add_argument("--merge-full", action="store_true", help="Also emit merged model.safetensors")
    p.add_argument("--eval-steps", type=int, default=0, help="Validation interval in steps")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience on val_nll")
    p.add_argument(
        "--on-checkpoint-cmd",
        default=None,
        help="Optional shell command run after saving a resumable checkpoint",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.dry_validate:
        out_art = Path(args.artifacts_dir) if args.artifacts_dir else None
        return run_dry_validate(Path(args.bundle_dir), out_art)
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
