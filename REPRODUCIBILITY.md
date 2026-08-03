# Reproducibility scope and provenance

## Frozen numerical reproduction

The manuscript's numerical results are reproduced from the files under
`data/raw/`, `data/processed/`, and `data/human/`. The primary script fails if
the raw generations and judge records do not have a one-to-one,
fingerprint-consistent match. Randomized resampling and permutation procedures
use the seed recorded in `analysis/reanalyze.py`.

The answer-field audit does not rewrite the frozen records. Its conservative
sensitivity removes all three cue rows from every affected
checkpoint--configuration--question unit. This preserves the paired design and
leaves the primary configuration conclusion unchanged.

## Fresh generation

`run_experiments.py` records the model revision for every new run, pins the
library versions inside the Modal image, verifies the frozen trial count and
SHA256, and uses atomic checkpoint replacement. It retains the historical
answer extractor so a new run does not silently change the protocol; the
extractor version is written to metadata and the known failure mode is handled
by the downstream audit.

A fresh run is not guaranteed to be bitwise identical to the released files.
Hosted model repositories, GPU kernels, and platform behavior can change, and
eight historical files lack the full provenance schema used by the maintained
runner. The frozen output analysis is therefore the strongest reproducibility
claim made by the paper.

## Human annotations

The canonical files are:

- `data/human/human_annotator1.csv`
- `data/human/human_annotator2.csv`

Each contains 360 rows. The analysis keeps annotators separate and does not
construct an automated consensus. The artifact does not preserve the original
selection procedure for the 40 audited model--question units or a complete
annotator protocol; the manuscript reports this limitation and treats the
human results descriptively.

## Integrity verification

After running `python analysis/reanalyze.py`, verify the manifest with:

```bash
sha256sum --check ARTIFACT_MANIFEST.sha256
```

API keys, Modal credentials, local `.env` files, editor uploads, LaTeX build
auxiliaries, and the historical similarity/AI-detection reports should not be
included in the anonymous supplementary archive.