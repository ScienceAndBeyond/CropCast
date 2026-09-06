# CropCast — AGU 2025 archive

This folder holds the project exactly as presented at AGU 2025 (poster
GC13F-0713). Nothing in it has been edited.

**These results are superseded.** Both the data pipeline and the way models are
scored have changed since — see the [current README](../README.md).

The original README is preserved verbatim as
[`ORIGINAL_README.md`](ORIGINAL_README.md).

## Contents

| | |
|---|---|
| `ORIGINAL_README.md` | the README as published |
| `src/` | the code that produced these results |
| `data/processed/` | the processed inputs |
| `results/` | model outputs |
| `poster/` | the AGU 2025 poster |
| `requirements.txt` | dependencies as of 2025 |

Kept so the poster's figures stay reproducible: `src/ml.py` run against
`data/processed/` reproduces `results/`.