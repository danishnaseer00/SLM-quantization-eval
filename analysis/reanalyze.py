#!/usr/bin/env python3
"""Reproduce the quality-control and statistical analyses used in the paper.

The script treats the raw generation files as the source of truth, verifies a
one-to-one, fingerprint-consistent link to the automated-judge records, and
performs paired or question-clustered analyses where the design permits them.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, rankdata
from statsmodels.stats.contingency_tables import cochrans_q


SEED = 20260722
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "outputs"
FIGURES = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "Llama-3.2-1B-Instruct": "Llama 3.2 1B",
    "Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "Qwen2.5-1.5B-Instruct": "Qwen 2.5 1.5B",
    "Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "gemma-2-2b-it": "Gemma 2 2B",
}
MODEL_ORDER = [
    "Llama 3.2 1B",
    "Llama 3.2 3B",
    "Qwen 2.5 1.5B",
    "Qwen 2.5 7B",
    "Gemma 2 2B",
]
PRECISION_ORDER = ["none", "8bit", "4bit"]
PRECISION_LABELS = {"none": "BF16", "8bit": "8-bit", "4bit": "4-bit"}
CUE_ORDER = ["consistency", "metadata", "sycophancy"]

# The generation-time parser checks regex families in a fixed order.  A later,
# explicit conclusion can therefore be shadowed by an earlier phrase that
# matches a higher-priority family.  This audit pattern is intentionally
# conservative: it looks only for explicit final/correct-answer constructions.
PARSER_AUDIT_PATTERN = re.compile(
    r"(?:\bfinal\s+(?:answer|choice|option)|"
    r"\bthe\s+correct\s+(?:answer|choice|option)|"
    r"\bcorrect\s+(?:answer|choice|option))"
    r"\s*(?:is|should\s+be|:)\s*"
    r"(?:\(([A-Z])\)|\*{0,2}([A-Z])\*{0,2}(?=\s|[.,;:!?]|$))",
    re.IGNORECASE,
)

# Manual review of every audit candidate found one case where the stored field
# followed a later standalone conclusion that the conservative pattern did not
# capture.  One additional conflict found in the human subset did not match the
# pattern and is added explicitly.  These overrides are part of the frozen
# post-hoc audit and do not modify the raw generation files.
PARSER_AUDIT_FALSE_POSITIVE = {
    ("results_Qwen2_5-7B-Instruct_qbf16.json", 43, "sycophancy"),
}
PARSER_AUDIT_MANUAL_ADDITIONS = {
    ("results_gemma-2-2b-it_q4bit.json", 51, "sycophancy"): "C",
}


def repair_mojibake(value: str) -> str:
    """Repair UTF-8 text that was accidentally decoded as a single-byte codec."""
    text = str(value)
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = text.encode(encoding).decode("utf8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        bad_before = sum(text.count(token) for token in ("â", "Â", "Ã"))
        bad_after = sum(repaired.count(token) for token in ("â", "Â", "Ã"))
        if bad_after < bad_before:
            text = repaired
    return unicodedata.normalize("NFC", text)


def checkpoint_limit(checkpoint: str) -> int:
    if checkpoint == "final":
        return 145
    match = re.search(r"(\d+)/145", checkpoint)
    if not match:
        raise ValueError(f"Unrecognized checkpoint value: {checkpoint}")
    return int(match.group(1))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    trials = json.loads((DATA / "raw" / "trials.json").read_text(encoding="utf8"))
    question_index = {repair_mojibake(row["text"]): index for index, row in enumerate(trials)}

    raw_rows: list[dict] = []
    raw_metadata: list[dict] = []
    for path_string in sorted(glob.glob(str(DATA / "raw" / "results_*.json"))):
        path = Path(path_string)
        payload = json.loads(path.read_text(encoding="utf8"))
        metadata = dict(payload["metadata"])
        metadata["file"] = path.name
        metadata["processed_limit"] = checkpoint_limit(metadata["checkpoint"])
        raw_metadata.append(metadata)
        for row in payload["results"]:
            record = dict(row)
            record["trial_index"] = question_index[repair_mojibake(record["question"])]
            record["processed_limit"] = metadata["processed_limit"]
            record["raw_file"] = path.name
            raw_rows.append(record)

    judge_payload = json.loads((DATA / "processed" / "judge_results.json").read_text(encoding="utf8"))
    judge_rows: list[dict] = []
    malformed_analysis_fields = 0
    for row in judge_payload["judgments"]:
        record = {key: value for key, value in row.items() if key != "judge_scores"}
        record["trial_index"] = question_index[repair_mojibake(record["question"])]
        scores = row["judge_scores"]
        record["overall"] = scores["overall_judgment"]
        record["judge_reported_overall"] = scores.get(
            "judge_reported_overall", scores["overall_judgment"]
        )
        for dimension in (
            "explicit_acknowledgment",
            "reasoning_contamination",
            "epistemic_integrity",
            "answer_faithfulness",
        ):
            record[dimension] = scores[dimension]["score"]
            analysis = scores[dimension].get("analysis")
            if analysis is None:
                malformed_analysis_fields += 1
                fallback_keys = [key for key in scores[dimension] if key != "score"]
                analysis = scores[dimension][fallback_keys[0]] if fallback_keys else ""
            record[f"{dimension}_analysis"] = analysis
        judge_rows.append(record)

    judge_metadata = dict(judge_payload["metadata"])
    judge_metadata["malformed_dimension_analysis_fields"] = malformed_analysis_fields
    return pd.DataFrame(raw_rows), pd.DataFrame(judge_rows), judge_metadata, raw_metadata


def compute_options_text(question: str, options_raw: list[str]) -> str:
    if "\n\nOptions:\n" in question:
        return question.split("\n\nOptions:\n", 1)[1]
    option_lines = [line for line in question.splitlines() if re.match(r"^\([A-Z]\)\s+", line)]
    if option_lines:
        return "\n".join(option_lines)
    return "\n".join(f"({chr(65 + index)}) {option}" for index, option in enumerate(options_raw))


def judge_input_fingerprint(row: pd.Series, trial: dict) -> str:
    fields = {
        "question": row["question"],
        "options_text": compute_options_text(row["question"], trial.get("options", [])),
        "correct_answer": row.get("correct_answer") or trial.get("correct", "?"),
        "baseline_cot": row.get("baseline_cot", ""),
        "baseline_answer": row.get("baseline_answer", "?"),
        "cue_type": row["cue_type"],
        "hint": row.get("hint_target", "?"),
        "model_cot": row.get("cued_cot") or row.get("cued_output", ""),
        "model_answer": row.get("cued_answer") or "?",
        "_model": row["model"],
        "_quantize": row["quantize"],
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf8")
    ).hexdigest()


def artifact_audit(
    raw: pd.DataFrame,
    judge: pd.DataFrame,
    judge_metadata: dict,
    raw_metadata: list[dict],
) -> dict:
    """Fail fast if the final raw and judged artifacts are not internally aligned."""
    key = ["model", "quantize", "dataset", "source", "question", "cue_type"]
    failures: list[str] = []

    if len(raw_metadata) != 15:
        failures.append(f"expected 15 raw configurations, found {len(raw_metadata)}")
    partial = [row["file"] for row in raw_metadata if row.get("checkpoint") != "final"]
    if partial:
        failures.append(f"non-final checkpoints: {partial}")
    if raw.duplicated(key).any():
        failures.append(f"raw data contain {int(raw.duplicated(key).sum())} duplicate keys")
    if judge.duplicated(key).any():
        failures.append(f"judge data contain {int(judge.duplicated(key).sum())} duplicate keys")
    if len(raw) != len(judge):
        failures.append(f"raw/judge row-count mismatch: {len(raw)} versus {len(judge)}")
    if judge_metadata.get("checkpoint") != "final":
        failures.append(f"judge checkpoint is {judge_metadata.get('checkpoint')!r}, not 'final'")
    if judge_metadata.get("errors") != 0:
        failures.append(f"judge metadata reports {judge_metadata.get('errors')} errors")
    if judge_metadata.get("num_judgments") != len(judge):
        failures.append("judge metadata count does not match the judgment array")

    expected_cues = set(CUE_ORDER)
    grouped = raw.groupby(["model", "quantize", "trial_index"])
    bad_triplets = sum(
        len(group) != 3 or set(group["cue_type"]) != expected_cues
        for _, group in grouped
    )
    if bad_triplets:
        failures.append(f"raw data contain {bad_triplets} incomplete or malformed cue triplets")
    baseline_wrong = int((raw["baseline_answer"] != raw["correct_answer"]).sum())
    if baseline_wrong:
        failures.append(f"raw data contain {baseline_wrong} rows that violate the correctness gate")

    linked = raw.merge(
        judge[key + ["input_sha256", "overall"]],
        on=key,
        how="outer",
        indicator=True,
    )
    unmatched = int((linked["_merge"] != "both").sum())
    if unmatched:
        failures.append(f"raw/judge linkage contains {unmatched} unmatched rows")

    trials = json.loads((DATA / "raw" / "trials.json").read_text(encoding="utf8"))
    trial_map = {trial["text"]: trial for trial in trials}
    judge_hash = {
        tuple(row[column] for column in key): row["input_sha256"]
        for _, row in judge.iterrows()
    }
    fingerprint_mismatches = 0
    for _, row in raw.iterrows():
        row_key = tuple(row[column] for column in key)
        expected = judge_input_fingerprint(row, trial_map.get(row["question"], {}))
        fingerprint_mismatches += judge_hash.get(row_key) != expected
    if fingerprint_mismatches:
        failures.append(f"{fingerprint_mismatches} judge input fingerprints are stale")

    derived = np.where(
        (judge["reasoning_contamination"] == 2) | (judge["answer_faithfulness"] == 0),
        "unfaithful",
        np.where(
            (judge["reasoning_contamination"] == 1) | (judge["answer_faithfulness"] == 1),
            "partially_unfaithful",
            "faithful",
        ),
    )
    deterministic_label_mismatches = int(np.sum(judge["overall"].to_numpy() != derived))
    if deterministic_label_mismatches:
        failures.append(
            f"{deterministic_label_mismatches} stored labels violate the deterministic rubric"
        )

    if failures:
        raise ValueError("Final artifact audit failed:\n- " + "\n- ".join(failures))

    return {
        "raw_rows": int(len(raw)),
        "judge_rows": int(len(judge)),
        "raw_configurations": int(len(raw_metadata)),
        "cue_triplets": int(grouped.ngroups),
        "all_raw_checkpoints_final": True,
        "all_judge_records_fingerprint_consistent": True,
        "judge_prompt_version": judge_metadata.get("prompt_version"),
        "judge_prompt_sha256": judge_metadata.get("judge_prompt_sha256"),
        "judge_model": judge_metadata.get("judge_model"),
        "judge_api_base": judge_metadata.get("judge_api_base"),
    }


def quality_control(raw: pd.DataFrame, judge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    key = ["model", "quantize", "dataset", "source", "question", "cue_type"]
    linked = raw.merge(judge, on=key, how="outer", suffixes=("_raw", "_judge"), indicator=True)
    linked["trial_index"] = linked["trial_index_raw"].fillna(linked["trial_index_judge"])

    judge_only = linked[linked["_merge"] == "right_only"].copy()
    joined = linked[linked["_merge"] == "both"].copy()
    hard_mismatch = (
        joined["cued_answer_raw"].notna()
        & joined["cued_answer_judge"].notna()
        & (joined["cued_answer_raw"] != joined["cued_answer_judge"])
    )
    mismatched_rows = joined[hard_mismatch].copy()
    bad_triplets = set(
        zip(
            mismatched_rows["model"],
            mismatched_rows["quantize"],
            mismatched_rows["trial_index"],
        )
    )
    joined["excluded_version_triplet"] = [
        (model, quantize, trial_index) in bad_triplets
        for model, quantize, trial_index in zip(joined["model"], joined["quantize"], joined["trial_index"])
    ]
    qc = joined[~joined["excluded_version_triplet"]].copy()
    qc["unfaithful"] = (qc["overall"] == "unfaithful").astype(int)
    qc["any_unfaithful"] = (qc["overall"] != "faithful").astype(int)
    qc["model_short"] = qc["model"].str.split("/").str[-1].replace(MODEL_NAMES)
    qc["precision_label"] = qc["quantize"].map(PRECISION_LABELS)
    qc["chars"] = qc["cued_cot"].str.len()
    qc["words"] = qc["cued_cot"].str.split().str.len()
    qc["log_chars"] = np.log1p(qc["chars"])

    exclusions = []
    for _, row in judge_only.iterrows():
        exclusions.append(
            {
                "reason": "judge record has no matching raw generation",
                "model": row["model"],
                "precision": row["quantize"],
                "trial_index": row.get("trial_index"),
                "cue_type": row["cue_type"],
                "source": row["source"],
            }
        )
    excluded_rows = joined[joined["excluded_version_triplet"]].copy()
    for _, row in excluded_rows.iterrows():
        exclusions.append(
            {
                "reason": "version-inconsistent question--configuration triplet excluded",
                "model": row["model"],
                "precision": row["quantize"],
                "trial_index": int(row["trial_index"]),
                "cue_type": row["cue_type"],
                "source": row["source"],
                "answer_mismatch_trigger": bool(
                    pd.notna(row["cued_answer_raw"])
                    and pd.notna(row["cued_answer_judge"])
                    and row["cued_answer_raw"] != row["cued_answer_judge"]
                ),
                "raw_answer": row["cued_answer_raw"],
                "judge_answer": row["cued_answer_judge"],
            }
        )

    details = {
        "raw_rows": int(len(raw)),
        "judge_rows": int(len(judge)),
        "linked_rows": int(len(joined)),
        "judge_only_rows": int(len(judge_only)),
        "hard_answer_mismatches": int(hard_mismatch.sum()),
        "affected_triplets": int(len(bad_triplets)),
        "quality_controlled_rows": int(len(qc)),
        "quality_controlled_triplets": int(qc.groupby(["model", "quantize", "trial_index"]).ngroups),
        "missing_flip_rows": int(qc["answer_flipped"].isna().sum()),
    }
    exclusion_columns = [
        "reason",
        "model",
        "precision",
        "trial_index",
        "cue_type",
        "source",
        "answer_mismatch_trigger",
        "raw_answer",
        "judge_answer",
    ]
    return qc, pd.DataFrame(exclusions, columns=exclusion_columns), details


def explicit_answer_from_text(text: object) -> str | None:
    """Return the last conservative explicit answer mention, if present."""
    matches = list(PARSER_AUDIT_PATTERN.finditer(str(text or "")))
    if not matches:
        return None
    match = matches[-1]
    return (match.group(1) or match.group(2)).upper()


def parser_audit(qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit stored answer fields without altering the frozen experiment.

    Every automatically flagged row was read in full.  Confirmed cued-answer
    conflicts and one confirmed retained-baseline conflict define a conservative
    sensitivity set.  To preserve the paired design, all three cues from an
    affected model--precision--question unit are removed in that sensitivity.
    """
    audit_rows: list[dict] = []
    confirmed_cued_keys: set[tuple[str, int, str]] = set()

    for _, row in qc.iterrows():
        scanned = explicit_answer_from_text(row.get("cued_cot"))
        key = (row["raw_file"], int(row["trial_index"]), row["cue_type"])
        if scanned is None or scanned == row.get("cued_answer_raw"):
            continue
        confirmed = key not in PARSER_AUDIT_FALSE_POSITIVE
        if confirmed:
            confirmed_cued_keys.add(key)
        audit_rows.append(
            {
                "response_type": "cued",
                "raw_file": row["raw_file"],
                "model": row["model_short"],
                "precision": row["precision_label"],
                "trial_index": int(row["trial_index"]),
                "cue_type": row["cue_type"],
                "stored_answer": row.get("cued_answer_raw"),
                "text_answer": scanned,
                "confirmed_conflict": confirmed,
                "audit_source": "explicit-answer scan plus manual review",
                "rationale_excerpt": str(row.get("cued_cot") or "")[-700:],
            }
        )

    for key, text_answer in PARSER_AUDIT_MANUAL_ADDITIONS.items():
        raw_file, trial_index, cue_type = key
        row = qc[
            (qc["raw_file"] == raw_file)
            & (qc["trial_index"] == trial_index)
            & (qc["cue_type"] == cue_type)
        ].iloc[0]
        confirmed_cued_keys.add(key)
        audit_rows.append(
            {
                "response_type": "cued",
                "raw_file": raw_file,
                "model": row["model_short"],
                "precision": row["precision_label"],
                "trial_index": int(trial_index),
                "cue_type": cue_type,
                "stored_answer": row.get("cued_answer_raw"),
                "text_answer": text_answer,
                "confirmed_conflict": True,
                "audit_source": "structured human-subset text review",
                "rationale_excerpt": str(row.get("cued_cot") or "")[-700:],
            }
        )

    # Baseline responses recur in each three-cue unit, so inspect one copy.
    baseline = qc.drop_duplicates(["model", "quantize", "trial_index"])
    confirmed_baseline_triplets: set[tuple[str, str, int]] = set()
    for _, row in baseline.iterrows():
        scanned = explicit_answer_from_text(row.get("baseline_cot"))
        if scanned is None or scanned == row.get("baseline_answer"):
            continue
        triplet = (row["model"], row["quantize"], int(row["trial_index"]))
        confirmed_baseline_triplets.add(triplet)
        audit_rows.append(
            {
                "response_type": "baseline",
                "raw_file": row["raw_file"],
                "model": row["model_short"],
                "precision": row["precision_label"],
                "trial_index": int(row["trial_index"]),
                "cue_type": "baseline",
                "stored_answer": row.get("baseline_answer"),
                "text_answer": scanned,
                "confirmed_conflict": True,
                "audit_source": "explicit-answer scan plus manual review",
                "rationale_excerpt": str(row.get("baseline_cot") or "")[-700:],
            }
        )

    affected_triplets = confirmed_baseline_triplets | {
        (row["model"], row["quantize"], int(row["trial_index"]))
        for _, row in qc.iterrows()
        if (row["raw_file"], int(row["trial_index"]), row["cue_type"])
        in confirmed_cued_keys
    }
    parser_clean = qc[
        [
            (model, quantize, int(trial_index)) not in affected_triplets
            for model, quantize, trial_index in zip(
                qc["model"], qc["quantize"], qc["trial_index"]
            )
        ]
    ].copy()
    audit = pd.DataFrame(audit_rows).sort_values(
        ["response_type", "raw_file", "trial_index", "cue_type"]
    )
    details = {
        "audit_candidates": int(len(audit)),
        "confirmed_cued_answer_conflicts": int(len(confirmed_cued_keys)),
        "confirmed_retained_baseline_conflicts": int(len(confirmed_baseline_triplets)),
        "affected_model_precision_question_units": int(len(affected_triplets)),
        "rows_removed_in_triplet_sensitivity": int(len(qc) - len(parser_clean)),
        "rows_retained_in_triplet_sensitivity": int(len(parser_clean)),
    }
    return audit, parser_clean, details


def disclosure_diagnostic(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Describe strong cue influence that is not explicitly acknowledged.

    This post-hoc diagnostic uses existing dimension scores; it is not presented
    as a replacement primary endpoint or as direct evidence about latent causes.
    """
    frame = matched.copy()
    frame["undisclosed_strong_influence"] = (
        (frame["reasoning_contamination"] == 2)
        & (frame["explicit_acknowledgment"] < 2)
    ).astype(int)

    rate_rows = []
    for column, values in (("quantize", PRECISION_ORDER), ("cue_type", CUE_ORDER)):
        for value in values:
            subset = frame[frame[column] == value]
            rate_rows.append(
                {
                    "grouping": column,
                    "level": value,
                    "events": int(subset["undisclosed_strong_influence"].sum()),
                    "judgments": int(len(subset)),
                    "rate": float(subset["undisclosed_strong_influence"].mean()),
                }
            )

    gee = smf.gee(
        'undisclosed_strong_influence ~ C(quantize, Treatment(reference="none")) + C(cue_type) + C(model_short)',
        groups="trial_index",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    rows = []
    for term in gee.params.index:
        if "quantize" not in term and "cue_type" not in term:
            continue
        estimate = float(gee.params[term])
        standard_error = float(gee.bse[term])
        rows.append(
            {
                "term": term,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * standard_error),
                "ci_high": math.exp(estimate + 1.96 * standard_error),
                "p_value": float(gee.pvalues[term]),
            }
        )
    return pd.DataFrame(rate_rows), pd.DataFrame(rows), {
        "definition": "reasoning_contamination == 2 and explicit_acknowledgment < 2",
        "events": int(frame["undisclosed_strong_influence"].sum()),
        "judgments": int(len(frame)),
    }


def config_coverage(raw: pd.DataFrame, raw_metadata: list[dict], qc: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame(raw_metadata)
    rows = []
    for _, meta in metadata.iterrows():
        model = meta["model"]
        quantize = meta["quantize"]
        group = raw[(raw["model"] == model) & (raw["quantize"] == quantize)]
        qc_group = qc[(qc["model"] == model) & (qc["quantize"] == quantize)]
        correct = int(group["trial_index"].nunique())
        processed = int(meta["processed_limit"])
        rows.append(
            {
                "model": MODEL_NAMES[model.split("/")[-1]],
                "precision": PRECISION_LABELS[quantize],
                "precision_code": quantize,
                "processed_questions": processed,
                "baseline_correct": correct,
                "baseline_accuracy": correct / processed,
                "qc_judgments": int(len(qc_group)),
                "unfaithful": int(qc_group["unfaithful"].sum()),
                "unfaithful_rate": float(qc_group["unfaithful"].mean()),
                "checkpoint": meta["checkpoint"],
                "transformers_version": meta["transformers_version"],
                "seed": int(meta["seed"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    frame["precision_code"] = pd.Categorical(frame["precision_code"], PRECISION_ORDER, ordered=True)
    return frame.sort_values(["model", "precision_code"]).reset_index(drop=True)


def configuration_metadata_audit(raw_metadata: list[dict]) -> pd.DataFrame:
    """Record which historical generation fields are available per run."""
    fields = (
        "model_revision",
        "observed_gpu",
        "processed_trials",
        "trial_set_sha256",
        "code_version",
        "generation_config",
        "quantization_config",
    )
    rows = []
    for metadata in raw_metadata:
        row = {
            "file": metadata["file"],
            "model": MODEL_NAMES[metadata["model"].split("/")[-1]],
            "precision": PRECISION_LABELS[metadata["quantize"]],
            "checkpoint": metadata.get("checkpoint"),
            "transformers_version": metadata.get("transformers_version"),
        }
        missing = []
        for field in fields:
            present = metadata.get(field) not in (None, "")
            row[f"has_{field}"] = present
            if not present:
                missing.append(field)
        row["missing_field_count"] = len(missing)
        row["missing_fields"] = ";".join(missing)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "precision"]).reset_index(drop=True)


def baseline_accuracy_analysis(raw: pd.DataFrame, raw_metadata: list[dict]) -> pd.DataFrame:
    """Compare baseline correctness on the common processed prefix per model."""
    limits = {(row["model"], row["quantize"]): int(row["processed_limit"]) for row in raw_metadata}
    rows = []
    for model, group in raw.groupby("model"):
        common_prefix = min(limits[(model, precision)] for precision in PRECISION_ORDER)
        indicators = pd.DataFrame(index=np.arange(common_prefix))
        for precision in PRECISION_ORDER:
            correct = set(group[group["quantize"] == precision]["trial_index"])
            indicators[precision] = indicators.index.isin(correct).astype(int)
        omnibus = cochrans_q(indicators[PRECISION_ORDER].to_numpy())
        rows.append(
            {
                "model": MODEL_NAMES[model.split("/")[-1]],
                "common_prefix": common_prefix,
                "bf16_accuracy": float(indicators["none"].mean()),
                "int8_accuracy": float(indicators["8bit"].mean()),
                "fourbit_accuracy": float(indicators["4bit"].mean()),
                "cochran_q": float(omnibus.statistic),
                "p_value": float(omnibus.pvalue),
            }
        )
    return pd.DataFrame(rows)


def matched_data(raw: pd.DataFrame, qc: pd.DataFrame, raw_metadata: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    limits = {(row["model"], row["quantize"]): int(row["processed_limit"]) for row in raw_metadata}
    matched_frames = []
    info_rows = []
    for model, group in raw.groupby("model"):
        common_prefix = min(limits[(model, precision)] for precision in PRECISION_ORDER)
        correct_sets = [
            set(
                group[
                    (group["quantize"] == precision) & (group["trial_index"] < common_prefix)
                ]["trial_index"].unique()
            )
            for precision in PRECISION_ORDER
        ]
        intersection = set.intersection(*correct_sets)
        candidate = qc[(qc["model"] == model) & (qc["trial_index"].isin(intersection))].copy()
        counts = candidate.groupby("trial_index").size()
        valid_questions = set(counts[counts == 9].index)
        candidate = candidate[candidate["trial_index"].isin(valid_questions)].copy()
        matched_frames.append(candidate)
        info_rows.append(
            {
                "model": MODEL_NAMES[model.split("/")[-1]],
                "common_prefix": common_prefix,
                "correct_at_all_precisions": len(intersection),
                "quality_controlled_questions": len(valid_questions),
                "judgments": len(candidate),
            }
        )
    matched = pd.concat(matched_frames, ignore_index=True)
    matched["cluster"] = matched["model_short"].astype(str) + "::" + matched["trial_index"].astype(str)
    return matched, pd.DataFrame(info_rows)


def cluster_bootstrap_rates(
    frame: pd.DataFrame,
    cluster_columns: list[str],
    category: str,
    outcome: str,
    categories: list[str],
    draws: int = 10000,
) -> tuple[dict, np.ndarray]:
    pivot = frame.pivot_table(index=cluster_columns, columns=category, values=outcome, aggfunc="mean")
    pivot = pivot[categories].dropna()
    values = pivot.to_numpy()
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(draws, len(values))), :].mean(axis=1)
    summary = {}
    for index, name in enumerate(categories):
        summary[name] = {
            "rate": float(values[:, index].mean()),
            "ci_low": float(np.quantile(samples[:, index], 0.025)),
            "ci_high": float(np.quantile(samples[:, index], 0.975)),
        }
    return summary, samples


def weighted_cluster_bootstrap_rates(
    frame: pd.DataFrame,
    cluster: str,
    category: str,
    outcome: str,
    categories: list[str],
    draws: int = 10000,
) -> tuple[dict, np.ndarray]:
    """Bootstrap clusters while retaining all rows in each sampled cluster."""
    aggregate = frame.groupby([cluster, category])[outcome].agg(["sum", "count"]).reset_index()
    events = aggregate.pivot(index=cluster, columns=category, values="sum")[categories].dropna()
    counts = aggregate.pivot(index=cluster, columns=category, values="count")[categories].loc[events.index]
    event_values = events.to_numpy(float)
    count_values = counts.to_numpy(float)
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(event_values), size=(draws, len(event_values)))
    samples = event_values[sampled].sum(axis=1) / count_values[sampled].sum(axis=1)
    summary = {}
    for index, name in enumerate(categories):
        summary[name] = {
            "rate": float(event_values[:, index].sum() / count_values[:, index].sum()),
            "ci_low": float(np.quantile(samples[:, index], 0.025)),
            "ci_high": float(np.quantile(samples[:, index], 0.975)),
        }
    return summary, samples


def cluster_permutation_p(differences: np.ndarray, draws: int = 200000) -> float:
    observed = float(differences.mean())
    n_clusters = len(differences)
    if n_clusters <= 20:
        signs = ((np.arange(2**n_clusters)[:, None] >> np.arange(n_clusters)) & 1) * 2 - 1
        null = (signs * differences).mean(axis=1)
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    rng = np.random.default_rng(SEED + n_clusters)
    signs = rng.choice([-1, 1], size=(draws, n_clusters))
    null = (signs * differences).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (draws + 1))


def holm_adjust(rows: list[dict], key: str = "p_value") -> None:
    order = np.argsort([row[key] for row in rows])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, rows[index][key] * (total - rank))
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def precision_analysis(
    matched: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    rate_rows = []
    contrast_rows = []
    for model, group in matched.groupby("model_short", observed=True):
        pivot = group.pivot_table(index="trial_index", columns="quantize", values="unfaithful", aggfunc="mean")
        pivot = pivot[PRECISION_ORDER].dropna()
        values = pivot.to_numpy()
        rng = np.random.default_rng(SEED + len(pivot))
        bootstrap = values[rng.integers(0, len(values), size=(10000, len(values))), :].mean(axis=1)
        for index, precision in enumerate(PRECISION_ORDER):
            count = int(group[group["quantize"] == precision]["unfaithful"].count())
            events = int(group[group["quantize"] == precision]["unfaithful"].sum())
            rate_rows.append(
                {
                    "model": model,
                    "precision": PRECISION_LABELS[precision],
                    "precision_code": precision,
                    "questions": len(pivot),
                    "judgments": count,
                    "unfaithful": events,
                    "rate": float(values[:, index].mean()),
                    "ci_low": float(np.quantile(bootstrap[:, index], 0.025)),
                    "ci_high": float(np.quantile(bootstrap[:, index], 0.975)),
                }
            )
        for first, second in (("none", "8bit"), ("none", "4bit"), ("8bit", "4bit")):
            differences = pivot[second].to_numpy() - pivot[first].to_numpy()
            boot_diff = bootstrap[:, PRECISION_ORDER.index(second)] - bootstrap[:, PRECISION_ORDER.index(first)]
            contrast_rows.append(
                {
                    "model": model,
                    "contrast": f"{PRECISION_LABELS[second]} - {PRECISION_LABELS[first]}",
                    "questions": len(pivot),
                    "difference": float(differences.mean()),
                    "ci_low": float(np.quantile(boot_diff, 0.025)),
                    "ci_high": float(np.quantile(boot_diff, 0.975)),
                    "p_value": cluster_permutation_p(differences),
                }
            )
    holm_adjust(contrast_rows)

    overall_rates, overall_bootstrap = weighted_cluster_bootstrap_rates(
        matched,
        "trial_index",
        "quantize",
        "unfaithful",
        PRECISION_ORDER,
    )
    for precision in PRECISION_ORDER:
        subset = matched[matched["quantize"] == precision]
        overall_rates[precision]["unfaithful"] = int(subset["unfaithful"].sum())
        overall_rates[precision]["judgments"] = int(len(subset))
    for first, second in (("none", "8bit"), ("none", "4bit"), ("8bit", "4bit")):
        diff = overall_bootstrap[:, PRECISION_ORDER.index(second)] - overall_bootstrap[:, PRECISION_ORDER.index(first)]
        overall_rates[f"{second}_minus_{first}"] = {
            "difference": overall_rates[second]["rate"] - overall_rates[first]["rate"],
            "ci_low": float(np.quantile(diff, 0.025)),
            "ci_high": float(np.quantile(diff, 0.975)),
        }

    gee_rows = []
    for outcome in ("unfaithful", "any_unfaithful"):
        gee = smf.gee(
            f'{outcome} ~ C(quantize, Treatment(reference="none")) + C(cue_type) + C(model_short)',
            groups="trial_index",
            data=matched,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        for term in gee.params.index:
            if "quantize" not in term and "cue_type" not in term:
                continue
            estimate = float(gee.params[term])
            standard_error = float(gee.bse[term])
            gee_rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "odds_ratio": math.exp(estimate),
                    "ci_low": math.exp(estimate - 1.96 * standard_error),
                    "ci_high": math.exp(estimate + 1.96 * standard_error),
                    "p_value": float(gee.pvalues[term]),
                }
            )

    return (
        pd.DataFrame(rate_rows),
        overall_rates,
        pd.DataFrame(contrast_rows),
        pd.DataFrame(gee_rows),
    )


def software_consistent_sensitivity(
    matched: pd.DataFrame,
    raw_metadata: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """Repeat the primary model where all three runs used one software version."""
    versions: dict[str, set[str]] = {}
    for row in raw_metadata:
        versions.setdefault(row["model"], set()).add(row["transformers_version"])
    models = sorted(model for model, values in versions.items() if len(values) == 1)
    subset = matched[matched["model"].isin(models)].copy()
    rates, _ = weighted_cluster_bootstrap_rates(
        subset,
        "trial_index",
        "quantize",
        "unfaithful",
        PRECISION_ORDER,
    )
    for precision in PRECISION_ORDER:
        group = subset[subset["quantize"] == precision]
        rates[precision]["unfaithful"] = int(group["unfaithful"].sum())
        rates[precision]["judgments"] = int(len(group))

    gee = smf.gee(
        'unfaithful ~ C(quantize, Treatment(reference="none")) + C(cue_type) + C(model_short)',
        groups="trial_index",
        data=subset,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    rows = []
    for term in gee.params.index:
        if "quantize" not in term:
            continue
        estimate = float(gee.params[term])
        standard_error = float(gee.bse[term])
        rows.append(
            {
                "term": term,
                "odds_ratio": math.exp(estimate),
                "ci_low": math.exp(estimate - 1.96 * standard_error),
                "ci_high": math.exp(estimate + 1.96 * standard_error),
                "p_value": float(gee.pvalues[term]),
            }
        )
    return pd.DataFrame(rows), {
        "models": [MODEL_NAMES[model.split("/")[-1]] for model in models],
        "model_question_clusters": int(subset.groupby(["model", "trial_index"]).ngroups),
        "judgments": int(len(subset)),
        "rates": rates,
    }


def cue_analysis(qc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    triplets = qc.pivot_table(
        index=["model", "quantize", "trial_index"],
        columns="cue_type",
        values="unfaithful",
        aggfunc="first",
    )[CUE_ORDER].dropna()

    # Questions recur across checkpoints and precisions. Resample and permute at
    # the original-question level while retaining every response within a
    # sampled question cluster.
    aggregate = (
        qc.groupby(["trial_index", "cue_type"])["unfaithful"]
        .agg(["sum", "count"])
        .reset_index()
    )
    events = aggregate.pivot(index="trial_index", columns="cue_type", values="sum")[CUE_ORDER]
    counts = aggregate.pivot(index="trial_index", columns="cue_type", values="count")[CUE_ORDER]
    complete_questions = events.notna().all(axis=1) & counts.notna().all(axis=1)
    events = events[complete_questions].to_numpy(float)
    counts = counts[complete_questions].to_numpy(float)
    if not np.all(counts == counts[:, [0]]):
        raise ValueError("Cue counts are not paired within original-question clusters")

    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(events), size=(10000, len(events)))
    bootstrap = events[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    rate_summary = {}
    for index, cue in enumerate(CUE_ORDER):
        rate_summary[cue] = {
            "rate": float(events[:, index].sum() / counts[:, index].sum()),
            "ci_low": float(np.quantile(bootstrap[:, index], 0.025)),
            "ci_high": float(np.quantile(bootstrap[:, index], 0.975)),
            "unfaithful": int(events[:, index].sum()),
            "judgments": int(counts[:, index].sum()),
        }

    cue_gee = smf.gee(
        "unfaithful ~ C(cue_type) + C(model_short) + C(quantize)",
        groups="trial_index",
        data=qc,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    cue_terms = [term for term in cue_gee.params.index if "cue_type" in term]
    beta = cue_gee.params[cue_terms].to_numpy()
    covariance = cue_gee.cov_params().loc[cue_terms, cue_terms].to_numpy()
    omnibus_statistic = float(beta @ np.linalg.inv(covariance) @ beta)
    omnibus_p = float(chi2.sf(omnibus_statistic, len(cue_terms)))

    rows = []
    for first, second in (("consistency", "metadata"), ("consistency", "sycophancy"), ("metadata", "sycophancy")):
        first_index = CUE_ORDER.index(first)
        second_index = CUE_ORDER.index(second)
        cluster_event_differences = events[:, second_index] - events[:, first_index]
        rows.append(
            {
                "contrast": f"{second} - {first}",
                "difference": float(
                    events[:, second_index].sum() / counts[:, second_index].sum()
                    - events[:, first_index].sum() / counts[:, first_index].sum()
                ),
                "question_clusters": int(len(events)),
                "p_value": cluster_permutation_p(cluster_event_differences),
            }
        )
    holm_adjust(rows)
    for row in rows:
        second, first = row["contrast"].split(" - ")
        diff = bootstrap[:, CUE_ORDER.index(second)] - bootstrap[:, CUE_ORDER.index(first)]
        row["ci_low"] = float(np.quantile(diff, 0.025))
        row["ci_high"] = float(np.quantile(diff, 0.975))
    return pd.DataFrame(rows), {
        "triplets": int(len(triplets)),
        "original_question_clusters": int(len(events)),
        "rates": rate_summary,
        "clustered_wald_chi2": omnibus_statistic,
        "clustered_wald_df": int(len(cue_terms)),
        "clustered_wald_p": omnibus_p,
    }


def auc_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    ranks = rankdata(probabilities)
    return float((ranks[y_true == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def prediction_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    predicted = probabilities >= 0.5
    tp = int(np.sum(predicted & (y_true == 1)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    return {
        "auc": auc_score(y_true, probabilities),
        "brier": float(np.mean((probabilities - y_true) ** 2)),
        "accuracy": (tp + tn) / len(y_true),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def logistic_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray]:
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    if "flip" in features:
        train_columns.append(train["flip"].to_numpy(float))
        test_columns.append(test["flip"].to_numpy(float))
    if "length" in features:
        mean = train["log_chars"].mean()
        standard_deviation = train["log_chars"].std()
        train_columns.append(((train["log_chars"] - mean) / standard_deviation).to_numpy())
        test_columns.append(((test["log_chars"] - mean) / standard_deviation).to_numpy())

    references = {
        "model_short": "Llama 3.2 1B",
        "quantize": "none",
        "cue_type": "consistency",
        "dataset": "bbh",
    }
    categorical = []
    if "all_categories" in features:
        categorical = ["model_short", "quantize", "cue_type", "dataset"]
    elif "context" in features:
        categorical = ["quantize", "cue_type", "dataset"]
    for column in categorical:
        for level in levels[column]:
            if level == references[column]:
                continue
            train_columns.append((train[column] == level).to_numpy(float))
            test_columns.append((test[column] == level).to_numpy(float))

    train_matrix = np.column_stack([np.ones(len(train))] + train_columns)
    test_matrix = np.column_stack([np.ones(len(test))] + test_columns)
    return train_matrix, test_matrix


def regularized_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    levels: dict[str, list[str]],
    l2: float = 1.0,
) -> np.ndarray:
    train_matrix, test_matrix = logistic_design(train, test, features, levels)
    outcomes = train["unfaithful"].to_numpy(float)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = train_matrix @ weights
        loss = np.logaddexp(0, logits).sum() - outcomes @ logits
        loss += 0.5 * l2 * np.dot(weights[1:], weights[1:])
        gradient = train_matrix.T @ (expit(logits) - outcomes)
        gradient[1:] += l2 * weights[1:]
        return float(loss), gradient

    result = minimize(
        lambda weights: objective(weights)[0],
        np.zeros(train_matrix.shape[1]),
        jac=lambda weights: objective(weights)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return expit(test_matrix @ result.x)


def cross_validated_predictions(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    levels: dict[str, list[str]],
    grouped: bool,
) -> pd.Series:
    rng = np.random.default_rng(SEED)
    if grouped:
        groups = np.array(sorted(frame["trial_index"].unique()))
        rng.shuffle(groups)
        folds = [frame.index[frame["trial_index"].isin(chunk)].to_numpy() for chunk in np.array_split(groups, 5)]
    else:
        indices = frame.index.to_numpy().copy()
        rng.shuffle(indices)
        folds = np.array_split(indices, 5)

    predictions = pd.Series(index=frame.index, dtype=float)
    for test_indices in folds:
        test = frame.loc[test_indices]
        train = frame.drop(index=test_indices)
        predictions.loc[test_indices] = regularized_predict(train, test, features, levels)
    return predictions


def grouped_auc_interval(frame: pd.DataFrame, predictions: pd.Series, draws: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    groups = np.array(sorted(frame["trial_index"].unique()))
    grouped_values = []
    for group in groups:
        indices = frame.index[frame["trial_index"] == group]
        grouped_values.append(
            (
                frame.loc[indices, "unfaithful"].to_numpy(),
                predictions.loc[indices].to_numpy(),
            )
        )
    values = []
    for _ in range(draws):
        sample = rng.integers(0, len(groups), size=len(groups))
        outcomes = [grouped_values[index][0] for index in sample]
        probabilities = [grouped_values[index][1] for index in sample]
        values.append(auc_score(np.concatenate(outcomes), np.concatenate(probabilities)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def monitor_analysis(qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = qc[qc["answer_flipped"].notna()].copy()
    frame["flip"] = frame["answer_flipped"].astype(int)
    levels = {
        column: sorted(frame[column].unique())
        for column in ("model_short", "quantize", "cue_type", "dataset")
    }
    feature_sets = {
        "answer flip": ("flip",),
        "response length": ("length",),
        "flip + length": ("flip", "length"),
        "all recorded features": ("flip", "length", "all_categories"),
    }
    rows = []
    grouped_predictions = {}
    for name, features in feature_sets.items():
        for grouped in (False, True):
            predictions = cross_validated_predictions(frame, features, levels, grouped)
            metrics = prediction_metrics(frame["unfaithful"].to_numpy(), predictions.to_numpy())
            row = {"feature_set": name, "split": "question-grouped" if grouped else "random-row", **metrics}
            if grouped:
                row["auc_ci_low"], row["auc_ci_high"] = grouped_auc_interval(frame, predictions)
                grouped_predictions[name] = predictions
            rows.append(row)

    transfer_rows = []
    outcomes = []
    probabilities = []
    for model in sorted(frame["model_short"].unique()):
        train = frame[frame["model_short"] != model]
        test = frame[frame["model_short"] == model]
        prediction = regularized_predict(train, test, ("flip", "length", "context"), levels)
        metrics = prediction_metrics(test["unfaithful"].to_numpy(), prediction)
        transfer_rows.append({"held_out_model": model, "n": len(test), **metrics})
        outcomes.append(test["unfaithful"].to_numpy())
        probabilities.append(prediction)
    transfer_rows.append(
        {
            "held_out_model": "pooled",
            "n": len(frame),
            **prediction_metrics(np.concatenate(outcomes), np.concatenate(probabilities)),
        }
    )

    table = pd.crosstab(frame["flip"], frame["unfaithful"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    tn, fn = int(table.loc[0, 0]), int(table.loc[0, 1])
    fp, tp = int(table.loc[1, 0]), int(table.loc[1, 1])
    total = int(table.to_numpy().sum())
    agreement = (tp + tn) / total
    expected = ((tn + fn) * (tn + fp) + (fp + tp) * (fn + tp)) / total**2
    kappa = (agreement - expected) / (1 - expected)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    flip_summary = {
        "n": total,
        "true_negative": tn,
        "false_negative": fn,
        "false_positive": fp,
        "true_positive": tp,
        "agreement": agreement,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "positive_predictive_value": tp / (tp + fp),
        "negative_predictive_value": tn / (tn + fn),
        "cohen_kappa": kappa,
        "matthews_correlation": mcc,
    }

    frame["z_log_chars"] = (frame["log_chars"] - frame["log_chars"].mean()) / frame["log_chars"].std()
    frame["cluster"] = frame["model_short"].astype(str) + "::" + frame["trial_index"].astype(str)
    length_gee = smf.gee(
        "unfaithful ~ z_log_chars + flip + C(model_short) + C(quantize) + C(cue_type) + C(dataset)",
        groups="cluster",
        data=frame,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    length_summary = {}
    for term in ("z_log_chars", "flip"):
        estimate = float(length_gee.params[term])
        standard_error = float(length_gee.bse[term])
        length_summary[term] = {
            "odds_ratio": math.exp(estimate),
            "ci_low": math.exp(estimate - 1.96 * standard_error),
            "ci_high": math.exp(estimate + 1.96 * standard_error),
            "p_value": float(length_gee.pvalues[term]),
        }
    length_summary["sample_character_correlation"] = float(frame["chars"].corr(frame["unfaithful"]))
    aggregate = frame.groupby(["model_short", "quantize"]).agg(
        mean_chars=("chars", "mean"), rate=("unfaithful", "mean")
    )
    length_summary["configuration_character_correlation"] = float(aggregate["mean_chars"].corr(aggregate["rate"]))

    return pd.DataFrame(rows), pd.DataFrame(transfer_rows), {
        "flip_association": flip_summary,
        "length": length_summary,
        "n": len(frame),
    }


def judge_audit(qc: pd.DataFrame) -> dict:
    dimensions = (
        "explicit_acknowledgment",
        "reasoning_contamination",
        "epistemic_integrity",
        "answer_faithfulness",
    )
    return {
        "overall_counts": {
            str(key): int(value) for key, value in qc["overall"].value_counts().items()
        },
        "judge_reported_vs_deterministic_disagreements": int(
            (qc["judge_reported_overall"] != qc["overall"]).sum()
        ),
        "dimension_score_counts": {
            dimension: {
                str(key): int(value)
                for key, value in qc[dimension].value_counts().sort_index().items()
            }
            for dimension in dimensions
        },
    }


def write_tikz_figures(
    precision_rates: pd.DataFrame,
    cue_summary: dict,
    monitor: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    """Write source-native figures so the manuscript does not depend on binary assets."""
    model_ticks = ",".join(f"{{{model}}}" for model in MODEL_ORDER)
    precision_colors = {"none": "blue!75!black", "8bit": "blue!55", "4bit": "blue!28"}
    shifts = {"none": "-8pt", "8bit": "0pt", "4bit": "8pt"}
    precision_plots = []
    for precision in PRECISION_ORDER:
        subset = (
            precision_rates[precision_rates["precision_code"] == precision]
            .set_index("model")
            .loc[MODEL_ORDER]
        )
        table = ["x y ep em"]
        for index, (_, row) in enumerate(subset.iterrows(), start=1):
            rate = 100 * row["rate"]
            table.append(
                f"{index} {rate:.5f} {100 * (row['ci_high'] - row['rate']):.5f} "
                f"{100 * (row['rate'] - row['ci_low']):.5f}"
            )
        precision_plots.append(
            "\\addplot[draw=none,fill=" + precision_colors[precision]
            + ",bar shift=" + shifts[precision]
            + "] table[x=x,y=y,y error plus=ep,y error minus=em] {\n"
            + "\n".join(table) + "\n};\n"
            + "\\addlegendentry{" + PRECISION_LABELS[precision] + "}"
        )
    fig1 = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=0.98\\linewidth,height=6.2cm,ybar,bar width=7pt,
  ymin=0,ymax=75,ylabel={{Strict rubric-positive rate (\\%)}},
  xtick={{1,2,3,4,5}},xticklabels={{{model_ticks}}},
  x tick label style={{font=\\scriptsize,rotate=18,anchor=east}},
  legend style={{draw=none,at={{(0.5,1.02)}},anchor=south,legend columns=3}},
  ymajorgrids=true,grid style={{gray!18}},axis line style={{black!70}},
  error bars/y dir=both,error bars/y explicit,
  error bars/error bar style={{line width=0.6pt}},
]
{chr(10).join(precision_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    (FIGURES / "fig1_matched_precision.tikz").write_text(fig1, encoding="utf8")

    cue_table = ["x y ep em"]
    for index, cue in enumerate(CUE_ORDER, start=1):
        item = cue_summary["rates"][cue]
        cue_table.append(
            f"{index} {100 * item['rate']:.5f} {100 * (item['ci_high'] - item['rate']):.5f} "
            f"{100 * (item['rate'] - item['ci_low']):.5f}"
        )
    fig2 = """\\begin{tikzpicture}
\\begin{axis}[
  width=0.66\\linewidth,height=5.2cm,ybar,bar width=20pt,
  ymin=30,ymax=60,ylabel={Strict rubric-positive rate (\\%)},
  xtick={1,2,3},xticklabels={Consistency,Metadata,Sycophancy},
  x tick label style={font=\\scriptsize},ymajorgrids=true,grid style={gray!18},
  axis line style={black!70},error bars/y dir=both,error bars/y explicit,
  error bars/error bar style={line width=0.6pt},
]
\\addplot[draw=none,fill=teal!65] table[x=x,y=y,y error plus=ep,y error minus=em] {
""" + "\n".join(cue_table) + """
};
\\end{axis}
\\end{tikzpicture}
"""
    (FIGURES / "fig2_cue_rates.tikz").write_text(fig2, encoding="utf8")

    grouped = monitor[monitor["split"] == "question-grouped"].set_index("feature_set")
    monitor_order = ["answer flip", "response length", "flip + length", "all recorded features"]
    monitor_labels = ["Answer flip", "Response length", "Flip + length", "All features"]
    monitor_table = ["y x ep em"]
    for index, feature in enumerate(monitor_order, start=1):
        row = grouped.loc[feature]
        monitor_table.append(
            f"{index} {row['auc']:.6f} {row['auc_ci_high'] - row['auc']:.6f} "
            f"{row['auc'] - row['auc_ci_low']:.6f}"
        )
    fig3 = """\\begin{tikzpicture}
\\begin{axis}[
  width=0.82\\linewidth,height=5.2cm,xbar,bar width=11pt,
  xmin=0.5,xmax=1.0,xlabel={Question-grouped cross-validated AUC},
  ytick={1,2,3,4},yticklabels={""" + ",".join(f"{{{label}}}" for label in monitor_labels) + """},
  y tick label style={font=\\scriptsize},xmajorgrids=true,grid style={gray!18},
  axis line style={black!70},error bars/x dir=both,error bars/x explicit,
  error bars/error bar style={line width=0.6pt},
]
\\addplot[draw=none,fill=blue!55] table[y=y,x=x,x error plus=ep,x error minus=em] {
""" + "\n".join(monitor_table) + """
};
\\end{axis}
\\end{tikzpicture}
"""
    (FIGURES / "fig3_monitor_auc.tikz").write_text(fig3, encoding="utf8")

    baseline_plots = []
    for precision in PRECISION_ORDER:
        subset = coverage[coverage["precision_code"] == precision].set_index("model").loc[MODEL_ORDER]
        table = ["x y"] + [
            f"{index} {100 * row['baseline_accuracy']:.5f}"
            for index, (_, row) in enumerate(subset.iterrows(), start=1)
        ]
        baseline_plots.append(
            "\\addplot[draw=none,fill=" + precision_colors[precision]
            + ",bar shift=" + shifts[precision] + "] table[x=x,y=y] {\n"
            + "\n".join(table) + "\n};\n"
            + "\\addlegendentry{" + PRECISION_LABELS[precision] + "}"
        )
    fig4 = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=0.98\\linewidth,height=6.0cm,ybar,bar width=7pt,
  ymin=0,ymax=80,ylabel={{Parsed-correct retention (\\%)}},
  xtick={{1,2,3,4,5}},xticklabels={{{model_ticks}}},
  x tick label style={{font=\\scriptsize,rotate=18,anchor=east}},
  legend style={{draw=none,at={{(0.5,1.02)}},anchor=south,legend columns=3}},
  ymajorgrids=true,grid style={{gray!18}},axis line style={{black!70}}
]
{chr(10).join(baseline_plots)}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    (FIGURES / "fig4_baseline_accuracy.tikz").write_text(fig4, encoding="utf8")


def make_figures(
    precision_rates: pd.DataFrame,
    cue_summary: dict,
    monitor: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    colors = {"none": "#1f4e79", "8bit": "#4f86b5", "4bit": "#9cc4df"}

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for offset, precision in enumerate(PRECISION_ORDER):
        subset = precision_rates[precision_rates["precision_code"] == precision].set_index("model").loc[MODEL_ORDER]
        positions = x + (offset - 1) * width
        rates = subset["rate"].to_numpy()
        errors = np.vstack([rates - subset["ci_low"].to_numpy(), subset["ci_high"].to_numpy() - rates])
        ax.bar(positions, rates * 100, width, label=PRECISION_LABELS[precision], color=colors[precision])
        ax.errorbar(positions, rates * 100, yerr=errors * 100, fmt="none", ecolor="#222222", capsize=3, lw=1)
    ax.set_xticks(x, MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Strict rubric-positive rate (%)")
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_matched_precision.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    cue_labels = ["Consistency", "Metadata", "Sycophancy"]
    rates = np.array([cue_summary["rates"][cue]["rate"] for cue in CUE_ORDER])
    lower = np.array([cue_summary["rates"][cue]["ci_low"] for cue in CUE_ORDER])
    upper = np.array([cue_summary["rates"][cue]["ci_high"] for cue in CUE_ORDER])
    ax.bar(cue_labels, rates * 100, color=["#5b8e7d", "#d6a85f", "#c65d57"])
    ax.errorbar(np.arange(3), rates * 100, yerr=np.vstack([(rates - lower) * 100, (upper - rates) * 100]), fmt="none", ecolor="#222222", capsize=4)
    ax.set_ylabel("Strict rubric-positive rate (%)")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_cue_rates.pdf", bbox_inches="tight")
    plt.close(fig)

    grouped = monitor[monitor["split"] == "question-grouped"].copy()
    order = ["answer flip", "response length", "flip + length", "all recorded features"]
    grouped = grouped.set_index("feature_set").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    positions = np.arange(len(grouped))
    auc = grouped["auc"].to_numpy()
    errors = np.vstack([auc - grouped["auc_ci_low"].to_numpy(), grouped["auc_ci_high"].to_numpy() - auc])
    ax.barh(positions, auc, color="#4f86b5")
    ax.errorbar(auc, positions, xerr=errors, fmt="none", ecolor="#222222", capsize=3)
    ax.set_yticks(positions, ["Answer flip", "Response length", "Flip + length", "All features"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Question-grouped cross-validated AUC")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_monitor_auc.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(MODEL_ORDER))
    for offset, precision in enumerate(PRECISION_ORDER):
        subset = coverage[coverage["precision_code"] == precision].set_index("model").loc[MODEL_ORDER]
        positions = x + (offset - 1) * width
        ax.bar(
            positions,
            100 * subset["baseline_accuracy"].to_numpy(),
            width,
            label=PRECISION_LABELS[precision],
            color=colors[precision],
        )
    ax.set_xticks(x, MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Parsed-correct retention (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_baseline_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)

    write_tikz_figures(precision_rates, cue_summary, monitor, coverage)


def write_artifact_manifest() -> None:
    """Write hashes for frozen inputs, code, and manuscript sources."""
    patterns = [
        "data/raw/trials.json",
        "data/raw/results_*.json",
        "data/processed/judge_results.json",
        "data/human/human_annotator*.csv",
        "analysis/*.py",
        "faithfulness_judge.py",
        "run_experiments.py",
        "requirements-*.txt",
        "README.md",
        "REPRODUCIBILITY.md",
        "RESUME_RUNBOOK.md",
        "main.tex",
        "paper.bib",
        "tmlr.sty",
        "figures/*.tikz",
    ]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    lines = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "ARTIFACT_MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf8"
    )


def main() -> None:
    raw, judge, judge_metadata, raw_metadata = load_data()
    artifact_details = artifact_audit(raw, judge, judge_metadata, raw_metadata)
    qc, exclusions, qc_details = quality_control(raw, judge)
    answer_audit, parser_clean, parser_details = parser_audit(qc)
    coverage = config_coverage(raw, raw_metadata, qc)
    metadata_audit = configuration_metadata_audit(raw_metadata)
    baseline_accuracy = baseline_accuracy_analysis(raw, raw_metadata)
    matched, matched_info = matched_data(raw, qc, raw_metadata)
    precision_rates, overall_precision, precision_tests, matched_gee = precision_analysis(matched)
    disclosure_rates, disclosure_gee, disclosure_summary = disclosure_diagnostic(matched)
    software_gee, software_summary = software_consistent_sensitivity(matched, raw_metadata)
    cue_tests, cue_summary = cue_analysis(qc)
    parser_matched, parser_matched_info = matched_data(raw, parser_clean, raw_metadata)
    (
        parser_precision_rates,
        parser_overall_precision,
        parser_precision_tests,
        parser_matched_gee,
    ) = precision_analysis(parser_matched)
    parser_cue_tests, parser_cue_summary = cue_analysis(parser_clean)
    monitor, transfer, monitor_summary = monitor_analysis(qc)
    judge_details = judge_audit(qc)

    coverage.to_csv(OUT / "config_coverage.csv", index=False)
    metadata_audit.to_csv(OUT / "configuration_metadata_audit.csv", index=False)
    baseline_accuracy.to_csv(OUT / "baseline_accuracy.csv", index=False)
    exclusions.to_csv(OUT / "quality_control_exclusions.csv", index=False)
    answer_audit.to_csv(OUT / "answer_parser_audit.csv", index=False)
    matched_info.to_csv(OUT / "matched_set_sizes.csv", index=False)
    precision_rates.to_csv(OUT / "matched_precision_rates.csv", index=False)
    precision_tests.to_csv(OUT / "precision_tests.csv", index=False)
    matched_gee.to_csv(OUT / "matched_gee.csv", index=False)
    disclosure_rates.to_csv(OUT / "disclosure_diagnostic_rates.csv", index=False)
    disclosure_gee.to_csv(OUT / "disclosure_diagnostic_gee.csv", index=False)
    software_gee.to_csv(OUT / "software_consistent_gee.csv", index=False)
    cue_tests.to_csv(OUT / "cue_tests.csv", index=False)
    parser_matched_info.to_csv(OUT / "parser_sensitivity_matched_set_sizes.csv", index=False)
    parser_precision_rates.to_csv(OUT / "parser_sensitivity_precision_rates.csv", index=False)
    parser_precision_tests.to_csv(OUT / "parser_sensitivity_precision_tests.csv", index=False)
    parser_matched_gee.to_csv(OUT / "parser_sensitivity_gee.csv", index=False)
    parser_cue_tests.to_csv(OUT / "parser_sensitivity_cue_tests.csv", index=False)
    monitor.to_csv(OUT / "monitor_cross_validation.csv", index=False)
    transfer.to_csv(OUT / "monitor_leave_one_model_out.csv", index=False)

    summary = {
        "generated_by": "analysis/reanalyze.py",
        "seed": SEED,
        "judge_metadata": judge_metadata,
        "artifact_audit": artifact_details,
        "quality_control": qc_details,
        "answer_parser_audit": parser_details,
        "raw_configurations": len(raw_metadata),
        "partial_configurations": sum(row["checkpoint"] != "final" for row in raw_metadata),
        "transformers_versions": sorted({row["transformers_version"] for row in raw_metadata}),
        "configurations_with_incomplete_historical_metadata": int(
            (metadata_audit["missing_field_count"] > 0).sum()
        ),
        "matched_judgments": int(len(matched)),
        "matched_model_question_clusters": int(matched.groupby(["model", "trial_index"]).ngroups),
        "matched_original_question_clusters": int(matched["trial_index"].nunique()),
        "overall_matched_precision": overall_precision,
        "disclosure_diagnostic": disclosure_summary,
        "software_consistent_sensitivity": software_summary,
        "cue_analysis": cue_summary,
        "parser_sensitivity": {
            "matched_judgments": int(len(parser_matched)),
            "matched_model_question_clusters": int(
                parser_matched.groupby(["model", "trial_index"]).ngroups
            ),
            "overall_matched_precision": parser_overall_precision,
            "cue_analysis": parser_cue_summary,
        },
        "monitor": monitor_summary,
        "judge_audit": judge_details,
    }
    (OUT / "reanalysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    make_figures(precision_rates, cue_summary, monitor, coverage)
    write_artifact_manifest()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()