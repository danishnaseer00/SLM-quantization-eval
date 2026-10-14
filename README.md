<div align="center">

# Does Quantization Change Small Language Models' Responses to Misleading Context?

**A reproducibility artifact studying how post-training quantization interacts with cue-conditioned reasoning in small language models.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.1-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

</div>

This repository is a reproducibility artifact for a study of how post-training quantization changes how small language models respond to misleading contextual cues. It contains a frozen 145-question trial set, 15 completed generation files, 3,078 automated judgments, two 360-row human annotation files, and the full analysis code used for the reported tables and figures.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproduce the Reported Analysis](#reproduce-the-reported-analysis)
- [Run New Generations](#run-new-generations)
- [Automated Judging](#automated-judging)
- [Data Integrity & Audits](#data-integrity--audits)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

We study how **quantization** interacts with **cue-conditioned reasoning** in small language models (SLMs). The artifact includes:

- A frozen **145-question trial set** drawn from BIG-Bench Hard and MMLU.
- **15 generation files** covering 5 model checkpoints (Llama-3.2-1B, Llama-3.2-3B, Qwen2.5-1.5B, Qwen2.5-7B, Gemma-2-2b) × 3 precisions (BF16, 8-bit LLM.int8, 4-bit NF4).
- **3,078 automated rubric-based judgments** scored through a DeepSeek OpenAI-compatible endpoint.
- **Two 360-row human annotation files** used to validate the automated labels.
- A complete **re-analysis pipeline** that regenerates every table, figure, and audit from the frozen inputs.

Cued responses are generated only when the uncued answer parses as correct, so all reported rates are conditional on passing that gate. Comparisons across precisions therefore use the intersection of questions answered correctly under every configuration rather than pooling unmatched rows.

> **Note:** This artifact treats the evaluator output as a specified rubric-based behavioral measure, not as direct access to latent chain-of-thought computation.

## Repository Structure

```
.
├── data/
│   ├── raw/                  # trials.json + 15 model generation files
│   ├── processed/            # judge_results.json, judge_summary.json (frozen judgment artifacts)
│   └── human/                # human_annotator1.csv, human_annotator2.csv
├── analysis/
│   ├── reanalyze.py          # Main re-analysis: fingerprints, audits, figures
│   ├── analyze_human_annotations.py
│   ├── adjudicate_humans.py
│   └── outputs/              # Regenerated audit tables and summaries
├── figures/                  # Regenerated figure sources (.tikz) and previews (.pdf)
├── faithfulness_judge.py     # Rubric-based faithfulness judge (DeepSeek API)
├── run_experiments.py        # Modal entry point for GPU generation jobs
├── requirements-analysis.txt # Pinned analysis environment
├── requirements-judge.txt    # Judge client requirements
├── data.zip                  # Zipped archive of the data/ directory
├── ARTIFACT_MANIFEST.sha256  # SHA-256 integrity manifest (frozen)
├── REPRODUCIBILITY.md        # Detailed verification instructions
├── LICENSE                   # Code license (MIT)
├── LICENSE-DATA              # Data license (CC BY 4.0)
└── README.md
```

## Getting Started

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
```

## Reproduce the Reported Analysis

Run the analysis pipeline, then verify the artifact manifest:

```bash
python analysis/reanalyze.py
python analysis/analyze_human_annotations.py
sha256sum --check ARTIFACT_MANIFEST.sha256
```

The first script verifies the raw-to-judge fingerprints, reconstructs the matched analysis, performs the answer-field and metadata audits, writes the sensitivity analyses, and regenerates figure sources. The second script verifies the two canonical human annotation files and regenerates the human-agreement outputs.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full verification walkthrough.

## Run New Generations

`run_experiments.py` is the maintained [Modal](https://modal.com/) entry point. It embeds and checks the frozen trial file before launching GPU jobs:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal run run_experiments.py::test
modal run run_experiments.py --batch t4 --allow-restart
modal run run_experiments.py --batch l4 --allow-restart
```

> **Note:** The frozen JSON outputs in `data/raw/` — not a fresh run — are authoritative for the reported analysis. Eight older configurations omit some provenance fields, the Qwen runs span two Transformers versions, and requested software dtypes do not by themselves establish native GPU arithmetic. New runs record additional device, counter, and parser-version metadata.

## Automated Judging

The judge uses an OpenAI-compatible DeepSeek endpoint. Preview the inputs before making paid API calls:

```bash
export DEEPSEEK_API_KEY=...
python faithfulness_judge.py --dry-run
python faithfulness_judge.py
```

> **Note:** The released `data/processed/judge_results.json` is the frozen judgment artifact. Rejudging can change outputs when the hosted model or service changes.

## Data Integrity & Audits

The `ARTIFACT_MANIFEST.sha256` file contains SHA-256 checksums for every frozen input, source script, and figure file. Verify with:

```bash
sha256sum --check ARTIFACT_MANIFEST.sha256
```

Key audit outputs produced by the pipeline:

| File | Contents |
| --- | --- |
| `analysis/outputs/answer_parser_audit.csv` | 41 confirmed cued conflicts, one confirmed uncued conflict, one rejected automated candidate |
| `analysis/outputs/configuration_metadata_audit.csv` | Provenance-field completeness for all 15 result files |
| `analysis/outputs/reanalysis_summary.json` | Machine-readable analysis summary |
| `analysis/outputs/human_annotation_summary.json` | Human-label validation and agreement summary |

## Contributing

Bug reports, issue triage, and reproducibility notes are welcome. Please open an issue before proposing changes to the frozen data, the analysis pipeline, or the integrity manifest.

## License

The analysis code and scripts are released under the MIT License (see `LICENSE`). The research data are released under the Creative Commons Attribution 4.0 International License (see `LICENSE-DATA`), intended for research use consistent with the study's reproducibility goals.