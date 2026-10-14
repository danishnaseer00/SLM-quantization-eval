"""Revision analyses for the EACL resubmission.

Adds four analyses that are absent from the frozen pipeline but address
reviewer-requested and self-identified gaps:

1. answer_change   -- judge-free replication of the headline configuration
                       and cue contrasts on parsed answer change
2. gate_selection  -- selection induced by the parsed-correct gate
3. judge_by_group  -- judge/annotator agreement by configuration and cue
4. severity        -- prevalence and severity-threshold diagnostic

Run from the repository root:

    python analysis/revision_analyses.py

Writes to analysis/outputs/. Reads only frozen inputs; never modifies
data/raw, data/processed, or data/human.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import reanalyze as R  # noqa: E402

OUT = ROOT / "analysis" / "outputs"

CUE_LABELS = {
    "consistency": "Consistency",
    "metadata": "Metadata",
    "sycophancy": "Sycophancy",
}

DIMENSIONS = [
    "explicit_acknowledgment",
    "reasoning_contamination",
    "epistemic_integrity",
    "answer_faithfulness",
]

BROAD_LABELS = {"unfaithful", "partially_unfaithful"}

QUANT_TERM = {
    "8bit": 'C(quantize, Treatment(reference="none"))[T.8bit]',
    "4bit": 'C(quantize, Treatment(reference="none"))[T.4bit]',
}
CUE_TERM = {
    "metadata": "C(cue_type)[T.metadata]",
    "sycophancy": "C(cue_type)[T.sycophancy]",
}


def mde_odds_ratio(beta_se: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable odds ratio at the given power (two-sided Wald)."""
    return float(math.exp((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * beta_se))


def or_to_pp(odds_ratio: float, base_rate: float) -> float:
    """Absolute percentage-point shift implied by an odds ratio at base_rate."""
    base_odds = base_rate / (1 - base_rate)
    shifted = odds_ratio * base_odds / (1 + odds_ratio * base_odds)
    return float(100 * (shifted - base_rate))


def extract(fit, term: str) -> dict:
    b = float(fit.params[term])
    se = float(fit.bse[term])
    return {
        "term": term,
        "odds_ratio": float(math.exp(b)),
        "ci_low": float(math.exp(b - 1.96 * se)),
        "ci_high": float(math.exp(b + 1.96 * se)),
        "p_value": float(fit.pvalues[term]),
        "beta": b,
        "beta_se": se,
    }


# ---------------------------------------------------------------------------
# 1. Answer change: a judge-free replication
# ---------------------------------------------------------------------------
def answer_change_report(qc: pd.DataFrame, raw: pd.DataFrame, raw_metadata: list[dict]) -> dict:
    report: dict = {}

    matched, _ = R.matched_data(raw, qc, raw_metadata)
    report["matched_model_question_units"] = int(
        matched.groupby(["model", "trial_index"]).ngroups
    )
    report["matched_models"] = int(matched["model"].nunique())
    report["matched_rows"] = int(len(matched))

    # --- matched set restricted to rows with a parsed answer on both sides ---
    m = matched[matched["answer_flipped"].notna()].copy()
    m["flipped"] = m["answer_flipped"].astype(int)
    report["matched_rows_with_parsed_answers"] = int(len(m))

    # --- GEE on parsed answer change, mirroring the paper's specification ---
    fit_flip = smf.gee(
        'flipped ~ C(quantize, Treatment(reference="none")) + C(cue_type) + C(model_short)',
        groups="trial_index",
        data=m,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()

    report["configuration_contrasts"] = {
        "8bit_vs_bf16": extract(fit_flip, QUANT_TERM["8bit"]),
        "4bit_vs_bf16": extract(fit_flip, QUANT_TERM["4bit"]),
    }
    report["cue_contrast_sycophancy_vs_consistency"] = extract(
        fit_flip, CUE_TERM["sycophancy"]
    )

    # --- the same GEE on the judge label, for a matched comparison ---
    fit_label = smf.gee(
        'unfaithful ~ C(quantize, Treatment(reference="none")) + C(cue_type) + C(model_short)',
        groups="trial_index",
        data=matched,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    report["configuration_controunds_judge_label"] = {
        "8bit_vs_bf16": extract(fit_label, QUANT_TERM["8bit"]),
        "4bit_vs_bf16": extract(fit_label, QUANT_TERM["4bit"]),
    }

    # --- descriptive flip rates ---
    rates = {}
    for code in R.PRECISION_ORDER:
        sub = m[m["quantize"] == code]
        rates[R.PRECISION_LABELS[code]] = {
            "n": int(len(sub)),
            "flips": int(sub["flipped"].sum()),
            "rate": float(100 * sub["flipped"].mean()),
        }
    report["matched_flip_rates"] = rates

    # --- minimum detectable effect on both endpoints ---
    base_flip = rates["BF16"]["rate"] / 100
    se_flip = report["configuration_contrasts"]["4bit_vs_bf16"]["beta_se"]
    mde_flip = mde_odds_ratio(se_flip)
    se_label = report["configuration_controunds_judge_label"]["4bit_vs_bf16"]["beta_se"]
    mde_label = mde_odds_ratio(se_label)
    report["mde"] = {
        "beta_se_4bit_answer_change": se_flip,
        "mde_or_answer_change": mde_flip,
        "mde_pp_answer_change": or_to_pp(mde_flip, base_flip),
        "observed_pp_4bit_answer_change": float(
            rates["4-bit"]["rate"] - rates["BF16"]["rate"]
        ),
        "beta_se_4bit_judge_label": se_label,
        "mde_or_judge_label": mde_label,
        "mde_pp_judge_label": or_to_pp(mde_label, 0.340),
    }

    # --- cue contrasts on answer change, matching Table 8's cluster machinery ---
    #
    # 34 rows lack a parsed cued option, so cue-specific counts are not exactly
    # paired within question. We therefore form a per-question cue rate and
    # contrast paired question-level differences, which keeps the original
    # question as the resampling and permutation unit.
    def cue_contrasts(frame: pd.DataFrame) -> list[dict]:
        frame = frame[frame["answer_flipped"].notna()].copy()
        frame["flipped"] = frame["answer_flipped"].astype(int)
        per_question = (
            frame.groupby(["trial_index", "cue_type"])["flipped"].mean().unstack()
        )
        per_question = per_question[R.CUE_ORDER].dropna()
        values = per_question.to_numpy(float)
        n_questions = len(values)

        rng = np.random.default_rng(R.SEED)
        sampled = rng.integers(0, n_questions, size=(10000, n_questions))
        bootstrap = values[sampled].mean(axis=1)

        rows = []
        for first, second in (
            ("consistency", "metadata"),
            ("consistency", "sycophancy"),
            ("metadata", "sycophancy"),
        ):
            i, j = R.CUE_ORDER.index(first), R.CUE_ORDER.index(second)
            observed = 100 * (values[:, j].mean() - values[:, i].mean())
            boot_diff = 100 * (bootstrap[:, j] - bootstrap[:, i])
            rows.append(
                {
                    "contrast": f"{CUE_LABELS[second]} - {CUE_LABELS[first]}",
                    "difference_pp": float(observed),
                    "ci_low_pp": float(np.quantile(boot_diff, 0.025)),
                    "ci_high_pp": float(np.quantile(boot_diff, 0.975)),
                    "p_value": R.cluster_permutation_p(values[:, j] - values[:, i]),
                    "n_questions": int(n_questions),
                }
            )
        R.holm_adjust(rows)
        return rows

    flipped_all = qc.copy()
    flipped_all["answer_flipped"] = pd.to_numeric(
        flipped_all["answer_flipped"], errors="coerce"
    )
    report["cue_frozen"] = cue_contrasts(flipped_all)

    _, parser_clean, audit_details = R.parser_audit(qc)
    report["parser_audit"] = {
        k: v for k, v in audit_details.items() if "affected" in k or "conflict" in k
    }
    report["cue_audited"] = cue_contrasts(parser_clean)

    matched_audited, _ = R.matched_data(raw, parser_clean, raw_metadata)
    parsed_audited = matched_audited[
        matched_audited["answer_flipped"].notna()
    ].copy()
    parsed_audited["flipped"] = parsed_audited["answer_flipped"].astype(int)
    fit_flip_audited = smf.gee(
        'flipped ~ C(quantize, Treatment(reference="none")) + C(cue_type) + C(model_short)',
        groups="trial_index",
        data=parsed_audited,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    report["matched_audited"] = {
        "model_question_units": int(
            matched_audited.groupby(["model", "trial_index"]).ngroups
        ),
        "original_question_clusters": int(matched_audited["trial_index"].nunique()),
        "judgments": int(len(matched_audited)),
        "parsed_answer_rows": int(len(parsed_audited)),
        "parsed_answer_rows_by_configuration": {
            R.PRECISION_LABELS[precision]: int(
                (parsed_audited["quantize"] == precision).sum()
            )
            for precision in R.PRECISION_ORDER
        },
    }
    report["configuration_contrasts_audited"] = {
        "8bit_vs_bf16": extract(fit_flip_audited, QUANT_TERM["8bit"]),
        "4bit_vs_bf16": extract(fit_flip_audited, QUANT_TERM["4bit"]),
    }

    # --- how redundant is the rubric with parsed answer change? ---
    sub = flipped_all.dropna(subset=["answer_flipped"]).copy()
    sub["flipped"] = sub["answer_flipped"].astype(int)
    sub["strict"] = (sub["overall"] == "unfaithful").astype(int)
    report["rubric_redundancy"] = {
        "p_strict_given_flip": float(sub.loc[sub["flipped"] == 1, "strict"].mean()),
        "p_strict_given_no_flip": float(sub.loc[sub["flipped"] == 0, "strict"].mean()),
        "n_flip": int((sub["flipped"] == 1).sum()),
        "n_no_flip": int((sub["flipped"] == 0).sum()),
        "agreement": float(100 * (sub["flipped"] == sub["strict"]).mean()),
    }

    cramers = {}
    for dim in DIMENSIONS:
        table = pd.crosstab(sub[dim].fillna(-1), sub["flipped"])
        stat = float(chi2_contingency(table)[0])
        n = float(table.to_numpy().sum())
        cramers[dim] = float(math.sqrt(stat / (n * (min(table.shape) - 1))))
    report["cramers_v_vs_flip"] = cramers

    pos = qc[qc["overall"] == "unfaithful"]
    report["positive_decomposition"] = {
        "contamination_only": int(
            ((pos["reasoning_contamination"] == 2) & (pos["answer_faithfulness"] != 0)).sum()
        ),
        "answer_faithfulness_only": int(
            ((pos["reasoning_contamination"] != 2) & (pos["answer_faithfulness"] == 0)).sum()
        ),
        "both": int(
            ((pos["reasoning_contamination"] == 2) & (pos["answer_faithfulness"] == 0)).sum()
        ),
        "total_positives": int(len(pos)),
    }

    # --- selection induced by the parsed-correct gate ---
    matched_units = set(
        map(tuple, matched[["model", "trial_index"]].drop_duplicates().to_numpy())
    )
    frame = flipped_all.copy()
    frame["in_matched_set"] = [
        (mdl, ti) in matched_units
        for mdl, ti in zip(frame["model"], frame["trial_index"])
    ]
    frame["strict"] = (frame["overall"] == "unfaithful").astype(int)

    strata = {}
    for flag, name in ((True, "in_matched_set"), (False, "retained_not_matched")):
        s = frame[frame["in_matched_set"] == flag]
        entry = {
            "n_rows": int(len(s)),
            "n_units": int(s.groupby(["model", "trial_index"]).ngroups),
            "strict_positive_rate": float(100 * s["strict"].mean()),
            "answer_flip_rate": float(100 * s["answer_flipped"].mean()),
            "flip_rates_by_precision": {},
        }
        for code in R.PRECISION_ORDER:
            sp = s[s["quantize"] == code]
            entry["flip_rates_by_precision"][R.PRECISION_LABELS[code]] = float(
                100 * sp["answer_flipped"].mean()
            )
        strata[name] = entry

    excluded = strata["retained_not_matched"]["flip_rates_by_precision"]
    included = strata["in_matched_set"]["flip_rates_by_precision"]
    strata["contrast_8bit_minus_bf16_pp"] = {
        "in_matched_set": float(included["8-bit"] - included["BF16"]),
        "retained_not_matched": float(excluded["8-bit"] - excluded["BF16"]),
    }
    strata["contrast_4bit_minus_bf16_pp"] = {
        "in_matched_set": float(included["4-bit"] - included["BF16"]),
        "retained_not_matched": float(excluded["4-bit"] - excluded["BF16"]),
    }
    report["gate_selection"] = strata

    # --- retention vs positive-rate correlation across model x config cells ---
    coverage = R.config_coverage(raw, raw_metadata, qc)
    xs = coverage["baseline_correct"].astype(float).to_numpy()
    ys = coverage["unfaithful_rate"].astype(float).to_numpy() * 100
    report["retention_vs_positive_rate_r"] = float(np.corrcoef(xs, ys)[0, 1])
    report["n_model_config_cells"] = int(len(coverage))

    return report


# ---------------------------------------------------------------------------
# 2. Judge/annotator agreement by configuration and cue
# ---------------------------------------------------------------------------
HUMAN_COLUMN = {"annotator1": "human1_overall", "annotator2": "human2_overall"}


def judge_by_group_report() -> list[dict]:
    comp = pd.read_csv(OUT / "human_annotation_comparison.csv")

    comp["judge_broad"] = comp["judge_overall"].isin(BROAD_LABELS)
    for who in HUMAN_COLUMN:
        comp[f"{who}_broad"] = comp[HUMAN_COLUMN[who]].isin(BROAD_LABELS)

    rows = []
    for field, order in (("precision_code", R.PRECISION_ORDER), ("cue_type", R.CUE_ORDER)):
        for level in order:
            sub = comp[comp[field] == level]
            labels = (
                R.PRECISION_LABELS[level]
                if field == "precision_code"
                else CUE_LABELS[level]
            )
            rec = {
                "group_type": field,
                "level": labels,
                "n": int(len(sub)),
                "judge_strict_rate": float(
                    100 * (sub["judge_overall"] == "unfaithful").mean()
                ),
                "judge_broad_rate": float(100 * sub["judge_broad"].mean()),
            }
            for who in HUMAN_COLUMN:
                col = HUMAN_COLUMN[who]
                rec[f"{who}_strict_rate"] = float(
                    100 * (sub[col] == "unfaithful").mean()
                )
                rec[f"{who}_broad_rate"] = float(100 * sub[f"{who}_broad"].mean())
                j = (sub["judge_overall"] == "unfaithful").astype(int)
                a = (sub[col] == "unfaithful").astype(int)
                rec[f"judge_vs_{who}_strict_agreement"] = float(100 * (j == a).mean())
                rec[f"judge_vs_{who}_broad_agreement"] = float(
                    100
                    * (
                        sub["judge_broad"].astype(int)
                        == sub[f"{who}_broad"].astype(int)
                    ).mean()
                )
            rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# 3. Prevalence and severity-threshold diagnostic
# ---------------------------------------------------------------------------
def severity_report() -> dict:
    comp = pd.read_csv(OUT / "human_annotation_comparison.csv")
    n = int(len(comp))

    strict = {"judge": int((comp["judge_overall"] == "unfaithful").sum())}
    broad = {"judge": int(comp["judge_overall"].isin(BROAD_LABELS).sum())}
    for who in HUMAN_COLUMN:
        col = HUMAN_COLUMN[who]
        strict[who] = int((comp[col] == "unfaithful").sum())
        broad[who] = int(comp[col].isin(BROAD_LABELS).sum())

    return {
        "n_rows": n,
        "strict_counts": strict,
        "broad_counts": broad,
        "strict_rates": {k: 100 * v / n for k, v in strict.items()},
        "broad_rates": {k: 100 * v / n for k, v in broad.items()},
    }


def main() -> None:
    raw, judge, _judge_metadata, raw_metadata = R.load_data()
    qc, _exclusions, _qc_details = R.quality_control(raw, judge)

    report = {
        "answer_change": answer_change_report(qc, raw, raw_metadata),
        "judge_by_group": judge_by_group_report(),
        "severity_threshold": severity_report(),
    }

    (OUT / "revision_summary.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame(report["judge_by_group"]).to_csv(
        OUT / "judge_by_group_agreement.csv", index=False
    )
    pd.DataFrame(report["answer_change"]["cue_frozen"]).to_csv(
        OUT / "answer_change_cue_tests_frozen.csv", index=False
    )
    pd.DataFrame(report["answer_change"]["cue_audited"]).to_csv(
        OUT / "answer_change_cue_tests_audited.csv", index=False
    )

    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()