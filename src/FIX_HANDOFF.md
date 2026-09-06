# Fix handoff — CropCast `src`

**Date:** 2026-09-06
**For:** an implementing agent (GPT). Findings below were raised by GPT and
**independently reproduced** before being written up. Each has a runnable
failing case and an acceptance test.

**Audit contract:** implement the fixes; do not restructure anything else. Every
item will be re-audited against the acceptance criteria stated here, plus a
whole-repo regression pass (§6). If you disagree with a spec, say so rather than
implementing something different.

---

## 0. Orientation

```
C:\Arit\Projects\Climate\
  src/          the pipeline — ALL work happens here
  data/processed/  yield.csv climate.csv vegetation.csv soil.csv
                   merged.csv  yield_split.csv
  data_raw/        monthly climate + vegetation, raw NASS, checkpoints
  results/         aggregate study     results_split/  irrigation study
  archive/         superseded material — do not read as current
```

**Run python from `src/`.** `DATA_PATH = Path("../data")` resolves against the
process cwd; running from the repo root silently writes to a sibling directory.

```powershell
cd C:\Arit\Projects\Climate\src
python -X utf8 <script>.py
```

`HANDOFF.md` (same folder) holds the scientific context, the three headline
findings, and a corrections log. Read §2 and the corrections section before
touching statistics — two claims in this project have already been retracted.

**Environment:** `.venv` at the repo root, `uv`-managed, Python 3.14.
Earth Engine is authenticated. **No fix below requires a re-download.**

---

## 1. HIGH — irrigation detrending uses globally-centred years

**File:** `irrigation_contrast.py`, in `load_pairs()`, the block beginning
`# Sensitivity check: additionally remove a pooled linear year trend`.

**Defect.** County-centred variables are regressed on *globally* centred years.
On an unbalanced panel — which this is; counties enter and leave across
2008–2018 — the year deviation is confounded with county identity, so the
fitted trend is badly attenuated and county means are reintroduced into a
quantity that is supposed to have them removed.

**Reproduction** (exact shared trend of 2.0/yr, no weather signal at all):

```python
import numpy as np, pandas as pd
rows=[]
for c,yrs in [("A",range(2000,2010)),("B",range(2010,2020))]:
    for y in yrs: rows.append(dict(county_fips=c,year=y,v=100+2.0*(y-2000)))
df=pd.DataFrame(rows)
df["v_a"]=df.v-df.groupby("county_fips").v.transform("mean")

yr=df.year.to_numpy(float); yr_c=yr-yr.mean()                    # CURRENT
slope=float(np.sum(yr_c*df.v_a)/np.sum(yr_c**2))
df["dt"]=df.v_a-slope*yr_c
print(slope)                                  # 0.496   (truth 2.0)
print(df.groupby("county_fips").dt.mean())    # A +2.481, B -2.481  (must be 0)
```

**Required fix.** Centre year **within county**, exactly as
`evaluate.py::predict_baseline("county_mean_trend")` already does:

```python
yr   = df["year"].to_numpy(dtype=float)
yrm  = df.groupby("county_fips")["year"].transform("mean").to_numpy(dtype=float)
yr_c = yr - yrm                      # county-centred, NOT global
denom = float(np.sum(yr_c ** 2))
for col in PRACTICES + WEATHER:
    a = df[col + "_a"].to_numpy(dtype=float)
    slope = float(np.sum(yr_c * a) / denom) if denom > 0 else 0.0
    df[col + "_dt"] = a - slope * yr_c
```

**Acceptance.** With the reproduction data above: fitted slope `2.000 ± 0.05`,
and residual county means `0.0` to within `1e-9` for every county.

**Note.** `evaluate.py` has the correct form. This is a regression I introduced
when writing `irrigation_contrast.py` — please keep the two consistent, and add
a comment saying why global centring is wrong so it does not come back a third
time.

---

## 2. MEDIUM — irrigation bootstrap does not refit the detrend

**File:** `irrigation_contrast.py::contrast()`.

**Defect.** `load_pairs()` estimates the pooled trend once; `contrast()` then
bootstraps the already-adjusted `_dt` columns. Intervals for `trend=linear`
therefore condition on a single trend estimate and omit the uncertainty from
having estimated it.

**Required fix.** Inside each bootstrap iteration, for `suffix == "_dt"` only,
re-estimate the trend **on the resampled rows** (using county-centred years per
§1) and re-detrend before computing slopes. Leave `trend=none` unchanged.

**Acceptance.** The trend is demonstrably re-estimated on each resample (not
read from the pre-computed `_dt` columns), and the before/after interval widths
are reported. `trend=none` output is byte-identical to before the change.

**Corrected 2026-09-06.** This originally required intervals to be "no
narrower". That was wrong. Refitting adds uncertainty from estimating the trend
but also lets the trend absorb resample-specific variation, so the net change in
width can go either way. Verified by simulation on clustered panel data with a
planted trend: detrend-once gave width 0.1229, refit-inside 0.1222. Observed
changes here were ~0.003 on widths of 2-5, below Monte-Carlo noise at 2000
resamples. Report the direction; do not force it.

**Do not** widen the API. Keep the returned column names as they are.

---

## 3. MEDIUM — absent coverage metadata bypasses validation

**Files:** `ml.py::filter_climate_complete()` (~line 378),
`irrigation_contrast.py::load_pairs()`, `evaluate.py` (uses the shared helper).

**Defect.** When `COVERAGE_MIN` is missing, the code logs a warning and keeps
every row. An older `climate.csv` therefore silently bypasses the completeness
requirement — the failure mode is invisible in the results.

**Required fix.** Raise instead of warning, with an actionable message naming
the rebuild command:

```
COVERAGE_MIN is absent from the climate data, so season completeness cannot be
enforced. Rebuild with:  python download_climate.py --aggregate-only
```

Add a module-level `REQUIRE_COVERAGE = True` in `ml.py`; when set to `False` the
old warn-and-continue behaviour applies, for deliberate work with legacy files.
Import that flag in `irrigation_contrast.py` rather than defining a second one.

**Acceptance.** Loading a `merged.csv` with `COVERAGE_MIN` dropped raises
`KeyError`; with `REQUIRE_COVERAGE = False` it warns and proceeds. Current data
has the column, so normal runs are unaffected.

---

## 4. LOW — forced-download checkpoint reset is not durable

**Files:** `download_climate.py::run()`, `download_soil.py::run()`.

**Defect.** `--force` clears the completion set **in memory** and deletes the
outputs, but the stale checkpoint *file* survives until the first successful
write. If every download fails, or the process is interrupted first, a later
plain resume reads completion entries pointing at deleted data and skips those
states.

**Required fix.** Delete (or truncate to empty) the checkpoint file at the same
point the outputs are deleted, before any download is attempted.

**Acceptance.** Plant a checkpoint containing `ZZ_STALE`, run with `--force`
against a state list that fails immediately, confirm the on-disk checkpoint no
longer contains `ZZ_STALE`.

---

## 5. MEDIUM — coverage cannot detect missing days inside a month

**File:** `download_climate.py::build_monthly_climate_image()` (~line 240).

`N_DAYS` is the calendar length of the month. `COVERAGE_*` detects absent
*months* and absent *variables*, but a month built from an incomplete daily
collection still reports full coverage.

**Verified as not currently biting:** gridMET image counts equal calendar days
across six sampled month-years spanning 2008–2025, including February 2020's
leap day.

**This is the one item that WOULD require a re-download.** Do **not** implement
it now. Instead, add a short note in the `build_monthly_climate_image` docstring
recording the assumption, the verification above, and the intended future fix
(emit `col.size()` as `N_IMAGES` alongside `N_DAYS` and fold it into coverage).

---

## 6. Regression requirements — all fixes

Run and report:

```powershell
cd C:\Arit\Projects\Climate\src
python -X utf8 -m py_compile config.py utils.py ml.py evaluate.py `
    irrigation_contrast.py download_climate.py download_soil.py `
    download_vegetation.py download_yield.py
python -X utf8 irrigation_contrast.py          # ~1 min
```

**Must still hold after your changes** (`results_split/irrigation_contrast.csv`,
`trend=none` rows — these must not move at all):

| weather | rainfed slope | irrigated slope | p(\|irr\| ≥ \|rain\|) |
|---|---|---|---|
| PRCP | +32.95 | +3.31 | 0.0 |
| EDD_TMAX | −21.61 | −5.74 | 0.0 |
| VPD | −73.44 | −17.02 | 0.0 |
| TMAX | −13.39 | −3.35 | 0.0 |

867 pairs · 114 counties · 2008–2018 · gap +81.4 BU/AC.

`trend=linear` rows **are expected to change** — that is the point of §1 and §2.
Report the before/after for those explicitly rather than burying it.

**Duplicate-definition check** — this repo has repeatedly acquired duplicated
constants from bad string replacements. Must print `clean` for all nine files:

```python
import ast, collections, glob
for f in sorted(glob.glob("*.py")):
    c = collections.Counter()
    for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
        if isinstance(n, ast.FunctionDef): c[n.name] += 1
        if isinstance(n, ast.Assign) and n.col_offset == 0 and isinstance(n.targets[0], ast.Name):
            c[n.targets[0].id] += 1
    d = [k for k, v in c.items() if v > 1]
    print(f, "DUPS " + str(d) if d else "clean")
```

---

## 7. Editing protocol — please follow

Three separate bugs in this project came from `str.replace` patches that
**matched nothing, changed nothing, and still compiled**, leaving the old
behaviour in place while the log said "applied". Two of them survived a full
audit round because compilation succeeded.

Assert the match count on every textual patch:

```python
n = s.count(old)
assert n == 1, f"expected 1 match, found {n}"
s = s.replace(old, new)
```

A fourth bug came from a replacement that matched **inside a comment** as well
as the target line, producing duplicated constants. Check what you matched.

---

## 8. Out of scope

Do not change these; they are known, documented, and deliberate:

- `TMAX_MAX` aggregation (documented custom statistic).
- Generic rather than crop-specific CDL masks (would need one vegetation table
  per crop).
- Soil stability mask using the full 2008–2025 window including held-out years
  (stated retrospective-design choice).
- NASS offset pagination (never triggered — all state queries return < 50k rows).
- Mixed-resolution spatial averaging across the three sources.
- Anything in `archive/`.

---

## 9. Git baseline — please work on top of it

The repo was initialised and committed immediately before this handoff:

```
97019ff  Corrected CropCast pipeline: data bugs, evaluation protocol, irrigation contrast
```

That commit is the audit baseline. **Do not amend or rebase it.** Make your
changes as one or more commits on top, so the review surface is exactly
`git diff 97019ff..HEAD`. Small, separately-titled commits per numbered finding
are easier to audit than one large one.

`data/`, `data_raw/`, `archive/`, `.venv/`, `*.log` and **all `.env` files** are
gitignored. `src/.env` holds your own NASS API key — never stage it, and do not
paste its contents anywhere.

---

## 10. Do not run two pipelines at once

`ml.py` and `evaluate.py` write to fixed paths (`data/processed/merged.csv`,
`results/*.csv`). Launching a second run while one is going makes them race:
run 2's `ml.py` replaces `merged.csv` underneath run 1's `evaluate.py`, and both
append to the same log. This happened during preparation of this handoff and
produced a `results/` directory holding files from two different runs. It is not
detectable from the numbers — only from the file timestamps.

Before starting a pipeline run, check nothing is already running:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, CreationDate, CommandLine | Format-List
```

None of the fixes in this document require `ml.py` or `evaluate.py`.
`irrigation_contrast.py` (§6) writes only to `results_split/` and is safe to run
at any time.

---

## 11. State at handoff

- A pipeline run (`ml.py` then `evaluate.py`) was in progress when this was
  written; `results/` may hold a partially-written set. **Do not judge any
  number from `results/`**, and do not commit that directory.
- `results_split/` is current and is the baseline for the §6 regression table.
- Nine files compile; AST duplicate check clean.
- Out of scope and unrelated: README rewrite; removing the placeholder `src/.env`
  that earlier commits tracked (it is untracked as of this commit, and `.gitignore`
  now covers it).
