# Quantization and cue-conditioned reasoning artifact

This repository accompanies the TMLR manuscript in `main.tex`. It contains
the frozen 145-question trial set, 15 completed generation files, 3,078
automated judgments, two 360-row human annotation files, and the analysis code
used for the reported tables and figures.

## Reproduce the reported analysis

Use the pinned analysis environment in `requirements-analysis.txt`, then run:

```bash
python analysis/reanalyze.py
python analysis/analyze_human_annotations.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The first script verifies the raw-to-judge fingerprints, reconstructs the
matched analysis, performs the answer-field and metadata audits, writes the
sensitivity analyses, regenerates figure sources, and creates
`ARTIFACT_MANIFEST.sha256`. The second script verifies the two canonical human
annotation files and regenerates the human-agreement outputs.

## Run new generations

`run_experiments.py` is the maintained Modal entry point. It embeds and checks
the frozen trial file before launching GPU jobs:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal run run_experiments.py::test
modal run run_experiments.py --batch t4 --allow-restart
modal run run_experiments.py --batch l4 --allow-restart
```

The frozen JSON outputs, rather than a fresh run, are authoritative for the
paper. Eight older configurations omit some provenance fields, the Qwen runs
span two Transformers versions, and requested software dtypes do not by
themselves establish native GPU arithmetic. New runs record additional device,
counter, and parser-version metadata. See `RESUME_RUNBOOK.md`.

## Automated judging

The judge uses the official DeepSeek OpenAI-compatible endpoint. Preview the
inputs before making paid API calls:

```bash
export DEEPSEEK_API_KEY=...
python faithfulness_judge.py --dry-run
python faithfulness_judge.py
```

The released `data/processed/judge_results.json` is the frozen judgment
artifact. Rejudging can change outputs when the hosted model or service
changes.

## Important audit files

- `analysis/outputs/answer_parser_audit.csv`: 41 confirmed cued conflicts,
  one confirmed uncued conflict, and one rejected automated candidate.
- `analysis/outputs/configuration_metadata_audit.csv`: provenance-field
  completeness for all 15 result files.
- `analysis/outputs/reanalysis_summary.json`: machine-readable analysis
  summary.
- `analysis/outputs/human_annotation_summary.json`: human-label validation and
  agreement summary.

The paper treats the evaluator output as a specified rubric-based behavioral
measure, not as direct access to latent chain-of-thought computation.