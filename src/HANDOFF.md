# CropCast — project state and decisions

Last updated: 2026-09-05

This file is the durable context for the project. Chat sessions are scoped to a
single directory and do not survive relaunching elsewhere; this does.

---

## 1. Which code is live

**`src/` is the pipeline.** As of 2026-09-05 the repo root holds only
live material; everything else moved under `archive/`:

```
Climate/
  src/     THE pipeline (this folder)
  data/           processed inputs + merged.csv
  data_raw/       raw NASS yield, monthly vegetation, checkpoints
  results/        current outputs
  .venv/ pyproject.toml uv.lock
  archive/
    agu2025/            src_agu (poster code), data_agu (4.2 GB), poster PDFs,
                        results-agu / results_agu / results - Copy, zips
    snapshots_20260905/ pre-fix safety copies: data - Copy, data_raw - Copy,
                        src_v2_wip_backup_20260905, results_wip_jan05_recovered
    unrelated/          sealevel/, empty wip/
```

`archive/agu2025/src_agu/` shares `../data` and `../results` path strings with
the live code, but from inside `archive/agu2025/` those now resolve to
`archive/agu2025/data` — so it can no longer overwrite live outputs. It did
destroy the Jan 5 results once, before the move.

Restore anything with a plain move back to the repo root.

### Always run python FROM `src/`

`DATA_PATH = Path("../data")` resolves against the process working directory:

```
cwd = Climate      ->  ../data = C:\Arit\Projects\data           WRONG
cwd = src   ->  ../data = C:\Arit\Projects\Climate\data   correct
```

Running from the repo root silently writes to a sibling of `Climate/`. Editors
and agents may open the project at `C:\Arit\Projects\Climate` (that is the right
scope, since data/, results/ and both source trees live under it) — but every
`python ...` invocation still needs `cd src` first.

---

## 2. The research strategy

The paper is an **evaluation-protocol paper, not a better-model paper**.

Thesis (REVISED 2026-09-06): *county-level yield models report high R² that is
insensitive to serious defects in the underlying data, because conventional R²
on a county-year panel is dominated by between-county variance, a monotonic
technology trend, and a same-season outcome variable rather than by weather
response. Scored against a county-mean-plus-trend baseline with the trend
removed, the same models show no skill.*

**RETRACTED 2026-09-06 — the insensitivity result.** An earlier version of this
section claimed that re-running the poster's analysis on corrected data moved
corn only 0.764 -> 0.772 (later 0.753), and concluded the metric is insensitive
to data defects. Both halves were wrong:

- `matched_rerun.py` matched the poster's configuration but not its ROWS. Corn
  trained on 9,374 observations against the poster's 5,448 and was scored on a
  different test set, so R2's denominator changed with the sample.
- The defects it named are NOT in the poster's code. `archive/src_agu/
  download_vegetation.py` has no crop mask; `download_climaate.py` uses a fixed
  Apr-Sep season for every state so its totals are comparable; `download_soil.py`
  uses the SoilGrids API at 0-5cm with no Earth Engine masking. The inverted
  `cultivated.eq(1)` lives only in `download_vegetation-wip.py`, which is later
  development work and never produced the poster.

**Never write that the poster's pipeline had an inverted crop mask, confounded
precipitation totals, or an inert soil mask.** Check `archive/src_agu/` before
attributing any defect to the AGU work. This was asserted repeatedly across four
documents and a pushed commit before external review caught it.

`paired_rerun.py` replaces it. On 11,786 shared county-years with identical
yields, training the poster's model on the SAME rows with each dataset's
features: corn +0.0890, soybeans +0.0787, spring wheat +0.0330, oats -0.0001
(seed sd <= 0.0109). The current data predicts these county-years materially
better. Which change is responsible is NOT isolated.

The surviving framing: *similar headline scores across unmatched samples
concealed a material difference in predictive performance.*

### The three headline results

**These replace an earlier set that was WRONG. See "Corrections" below.**

1. **Detrended, the models do not beat a county average.** 95% CI on
   `skill_vs_county_mean` (county-detrended, temporal split):

   | crop | all_features | county_mean_trend |
   |---|---|---|
   | CORN | **[-0.233, -0.046]** | [0.006, 0.010] |
   | SOYBEANS | **[-0.200, -0.063]** | [0.005, 0.006] |
   | OATS | [-0.046, 0.114] | [0.008, 0.013] |
   | SORGHUM | [-0.040, 0.124] | [-0.077, -0.063] |
   | WHEAT_SPRING | [0.081, 0.257] | [0.005, 0.015] |

   (All five crops. An earlier version of this table quoted three crops and
   numbers from a superseded run; these match `results/bootstrap_ci.csv`.)

   For the two highest-volume crops the full climate+soil+satellite model is
   *significantly worse* than predicting each county's mean, while county mean
   plus a straight line is significantly better. Undetrended, `all_features`
   scores +0.606 on corn; detrended it is -0.144. That gap is the technology
   trend, not weather skill.

2. **Soil is NOT merely a county fingerprint.** With the corrected placebo
   (`climate+soil` vs `climate+noise`), county identity explains only 7-13% of
   soil's contribution, and the difference clears zero even after detrending for
   corn [0.026, 0.065], soybeans [0.060, 0.098] and oats [0.051, 0.120].
   Spring wheat (p=0.14) and sorghum (p=0.60) are indistinguishable.

3. **Variance decomposition.** 40-61% of yield variance is between-county
   (corn 57%, soybeans 61%, wheat 54%, oats 52%, sorghum 40%). That
   is what conventional R2 is mostly measuring. (It is NOT a ceiling on anomaly
   skill - a perfect anomaly prediction scores 1 regardless.)


### 5. The irrigation contrast — the mechanism, measured directly

`irrigation_contrast.py`. NASS reports county corn yield separately for
irrigated and non-irrigated land. Where both appear for the SAME county in the
SAME year, county, soil and season are held constant and only management
differs — so the between-county confounding that wrecked the poster's
"29 C threshold" cannot operate.

**867 pairs · 114 counties · 2008–2018 · NE 665, KS 202.**
Mean irrigated-minus-rainfed gap **+81.4 BU/AC** (0.5% negative).

In-sample weather R² of the yield anomaly: **rainfed 0.685, irrigated 0.166**.

| weather | rainfed slope | irrigated slope | \|ratio\| slope | corr | %mean |
|---|---|---|---|---|---|
| PRCP | +32.95 | +3.31 | 0.10 | 0.19 | 0.06 |
| EDD_TMAX | −21.61 | −5.74 | 0.27 | 0.51 | 0.15 |
| VPD | −73.44 | −17.02 | 0.23 | 0.45 | 0.13 |
| TMAX | −13.39 | −3.35 | 0.25 | 0.48 | 0.14 |

All p(|irrigated| ≥ |rainfed|) = 0.0, county-clustered bootstrap, 2000 resamples.
Correlation is unit-free, so this is not an artefact of rainfed yields being
more variable (sd 29.5 vs 15.3) — the obvious objection, and it does not hold.

**Linear-detrended sensitivity** (removes a pooled year trend from every
anomaly, so a technology trend cannot masquerade as weather response). The
result strengthens:

| weather | rainfed | irrigated | \|ratio\| slope | corr | %mean |
|---|---|---|---|---|---|
| PRCP | +32.56 | +1.72 | **0.053** | 0.111 | 0.030 |
| EDD_TMAX | −21.53 | −5.64 | 0.262 | 0.549 | 0.150 |
| VPD | −73.46 | −17.04 | 0.232 | 0.487 | 0.133 |
| TMAX | −13.64 | −3.61 | 0.264 | 0.555 | 0.151 |

In-sample weather R²: rainfed 0.685 → **0.699**, irrigated 0.166 → **0.253**.
All p(|irrigated| ≥ |rainfed|) = 0.0 under both trend modes.

The trend is re-estimated inside each bootstrap resample, so these intervals
include the uncertainty of having estimated it. Years are centred **within
county** — global centring attenuates the trend on an unbalanced panel and
reintroduces county means (an exact 2.0/yr trend fitted as 0.496). An earlier
version of this file reported PRCP ratio 0.056 from that buggy estimator.

**Read as WITHIN-COUNTY ASSOCIATIONS, in sample.** County-demeaning removes each
county's level; it does not remove year shocks, the technology trend (hence the
detrended column), or year-to-year changes in which fields are irrigated. The R²
figures are in-sample fit, not predictive validation. Year fixed effects are
deliberately not used — within a state, a year's weather is shared across
counties, so year dummies would absorb the signal being measured.

**Why this matters:** the main analysis shows models losing to a county mean once
detrended, i.e. weak weather skill. This supplies a reason: where management has
removed the weather signal, there is little left to predict. And the +81 BU/AC
management gap is more than double the 38 BU/AC the poster attributed to heat
from a model that never saw irrigation.

**Limits:** 2008–2018 only (NASS stopped publishing the county split after 2018;
coverage thinned from 105 counties to 51), corn only, effectively two states,
and irrigation is a management choice rather than a randomised treatment.

### 6. Operational integrity fixes (2026-09-06, final audit round)

| Issue | Fix |
|---|---|
| `ml.py` computed `MERGED_FILE` but still wrote hardcoded `merged.csv`, so split mode overwrote the aggregate table | Write honours `MERGED_FILE`; aggregate→`merged.csv`, split→`merged_split.csv` |
| `--force` loaded the old checkpoint, deleted all outputs, then processed only the requested states — stale entries pointed at deleted data and a later resume skipped them | Completion set starts empty whenever outputs are cleared (`resume and not force`) |
| A subset `--states` yield run REPLACED the whole table with that subset (the documented recovery path destroyed the successful states) | Fetched states are MERGED into the existing table; other states kept |
| `--reuse-raw --states X` shrank the cleaned table to X | `--states` now controls fetching only; the cleaned table is always the full union (ml.py applies `STUDY_STATES`) |
| `COVERAGE_FRAC` measured precipitation only, against rows present | Per-variable `COVERAGE_<VAR>` + `COVERAGE_MIN`, against calendar `EXPECTED_DAYS`; enforced by `MIN_CLIMATE_COVERAGE` in ml.py |
| Duplicated constant block in ml.py from a bad edit | Removed; AST-level duplicate-definition check now clean across all 9 files |
| `p_no_decoupling` compared the SIGN of the slope difference, not magnitude — it reported "decoupled" for rainfed −1 vs irrigated +10 | Now `P(\|irrigated\| ≥ \|rainfed\|)`; unit-tested against that counterexample |
| Irrigation analysis ignored `COVERAGE_MIN` and did not drop missing weather | Applies the same completeness rule as ml.py before any slope |
| `COVERAGE_MIN` excluded `TMAX_MAX` | Now 12 coverage columns; `TMAX_MAX` included |
| Climate completeness enforced in ml.py but not evaluate.py | Shared `filter_climate_complete()`, imported by both |
| A partial yield refresh deleted cached rows for states whose fetch FAILED | Only states actually refreshed are replaced; failures keep prior data |
| Variance-decomposition wording implied within-county % caps weather R² | Reworded: weather explains between-county variance too, via climatology |

**Editing lesson recorded for whoever works on this next:** several bugs in this
file's history came from `str.replace` patches that silently matched nothing
and left the old code in place, while still compiling. Every patch is now
applied with an asserted match count. Do the same.

### Corrections to earlier versions of this file

- **"Soil is a county fingerprint; its advantage vanishes when detrended"** -
  WRONG, an artefact of a bug. The placebo arm was the random features *alone*,
  with no climate, so it was compared against `climate+soil`. Numbers quoted at
  the time (38% / 70% / 134% of the soil gain) are invalid.
- **"anomaly R2 <= 0 means no skill beyond county identity"** - WRONG. That
  metric was `r2_score` on residuals, which centres on the mean TEST anomaly,
  not zero. A predictor that tied the county mean exactly scored **-33.8**.
  Replaced by `skill_vs_county_mean` = 1 - SSE(model)/SSE(county mean).
- Any figure computed before 2026-09-05 22:30 predates these fixes.

⚠️ Computed on data with the CDL mask, per-day rates and 2025 window all
correct. Still pending at the time of writing: the "group 2" download
refinements (native GEE scales, thickness-weighted soil depth, MODIS QA mask,
GDD/EDD renaming). Measured effect of those is small - soil SOC moved -2.0%,
other properties <0.5% - so the conclusions above are not expected to change,
but the numbers will shift slightly.

---

## 3. Bugs found and fixed (2026-09-05)

| Area | Problem | Fix |
|---|---|---|
| `utils.get_crop_mask` | CDL `cultivated` band is 1=Non-cultivated, 2=Cultivated. Code used `.eq(1)`, selecting **non-farmland** for 2013–2023. Confirmed empirically: cross-county NDVI dispersion separates perfectly at the 2013/2024 boundaries, and the by-state shift runs +0.13 (AR, forest) to −0.06 (CA, desert) — a mask flip, not a trend. | Use the `cropland` code ranges for **all** years. The two branches also used different *definitions*, so even a corrected `.eq(2)` would leave a step at 2013/2024. |
| `download_soil` | "Cropland union" mask (≥1 year of 17) covered ~the whole county. Verified: masked vs unmasked soil agreed to r=0.9999, median relative difference **0.00%** — the masking did nothing. | `get_stable_crop_mask(min_frac=0.5)`. |
| `download_soil` | Depth `b0` only — a 0 cm point estimate, not a layer. | Mean of `b0`/`b10`/`b30` = 0–30 cm rooting zone (also what the poster claimed). |
| `download_climate` | PRCP/ETO as season **totals** while growing-season windows differ by state. MN's May–Sep is 153 d vs 183 d, understating MN by 16% by construction → model learns "low PRCP = Minnesota". | Per-day rates. Unit-tested: identical weather in IA and MN now gives identical features. |
| `download_climate` | `TMAX` is a season mean of daily maxima — 85% of its variance is between-county, so the poster's "29 °C heat threshold" is a geography contrast (TX/AR/LA vs Corn Belt), not a dose-response. | Added `GDD`, `EDD`, `HOT_DAYS`, `TMAX_MAX` from daily data. |
| all downloaders | `--no-resume`/`--force` reset the checkpoint but left the output file, and the writer appends → every fresh run silently doubled the rows. | Delete output when not resuming. |
| `ml.py` | Impurity importance on training data — biased toward continuous high-cardinality features. Changing to permutation importance on the test set materially reorders results (corn `ph_mean` 13.6% → 24.5%, `PRCP` 7.7% → 2.3%). | `compute_permutation_importance()`. |
| `ml.py` | PDP under severe collinearity (VPD–ETO r=0.97) evaluates combinations that never occur. | Added ALE + `feature_collinearity.csv`. |
| `ml.py` | `if causal_r2` — a real R² of exactly 0.0 is falsy and became `None`; then `f"{None:.3f}"` crashed. | `is not None`. |
| `ml.py` | Percent-of-R² exploded on weak baselines (oats "+319.8%" was +0.27 off a 0.086 baseline). | Suppressed below baseline 0.2; report ΔR². |
| `ml.py` | `n_months` computed then discarded — partial growing seasons mixed with full ones. | Drop county-years with < 5 months. |
| `utils`, `download_climate` | `geopandas`/`pygris`/`ee` imported at module scope, so `ml.py` and `--aggregate-only` needed the whole geospatial stack. | Lazy imports. |

### Architecture change

`download_climate.py` now emits **monthly** rows (`data_raw/climate_monthly.csv`)
and aggregates to a season offline. Changing the growing season — crop-specific
windows, winter crops, sensitivity tests — is now `--aggregate-only`, never
another download. Bands were chosen to aggregate cleanly: `TMAX_P95` was dropped
(a season percentile cannot be rebuilt from monthly percentiles) in favour of
`HOT_DAYS` and `TMAX_MAX`, which can.

`--year-chunk` default dropped 6 → 2: monthly output is ~12× the payload per
call (TX at chunk 6 would be 18k features per `getInfo`).

`SEASON_DAYS` is deliberately **excluded** from the feature list — it is
constant per state, i.e. another state fingerprint.

---

## 4. Data status

| Dataset | Re-download? | Why |
|---|---|---|
| **Yield** | **No** | `data_raw/crop_yield_raw_2005_2024.csv` is the full raw NASS response — 150,380 rows, 21 states, 2005–2024. All filtering is pandas. |
| **Climate** | Yes | GDD/EDD need daily gridMET, never saved. (The season-length fix alone could be done in place.) |
| **Vegetation** | Yes | Mask applied server-side *before* `reduceRegions`; county means cannot be un-masked. |
| **Soil** | Yes | Same — mask and depth are baked into the county means. |

Rejected as substitutes: `data_raw/vegetation_monthly.csv` (inverted mask baked
in), `data_agu/raw/ndvi_gee_partial/` (county-year, no CDL masking at all),
`data_agu/backup/NDVI-*-MOD13A1-061-results.csv` (AppEEARS — 999 IDs each with
one lat/lon, i.e. a single point per county, and MOD13A1 not MOD13A3).

### Three state lists — do not conflate

```
DEFAULT_STATES (21)  download scope, includes CA/TX/AR/LA/NE/ID/OR/WA/MS/GA/NC
STUDY_STATES   (10)  model scope: IA IL IN OH MN WI MO KS ND SD
AGU poster     (11)  AR CA IA ID IL LA MN NE OR TX WA
```

**The poster's region and the current study region overlap by only 3 states**
(IA, IL, MN). The WIP is therefore a *different study*, not a corrected version
of the poster — do not write "we improved R² from 0.76 to X"; the samples are
not comparable. Either present it as new, or re-run the poster's 11 states for a
like-for-like comparison.

Known inconsistency: `STUDY_STATES` is labelled "rainfed" but western Kansas is
Ogallala-irrigated.

---

## 5. Next steps

1. **Smoke-test the mask before committing hours to downloads.** Story County IA
   (19169) is ~90% row crop; expect 0.85–0.95 for every year and no step
   between 2012 and 2013:
   ```python
   import ee; ee.Initialize(project=GEE_PROJECT_ID)
   from utils import get_crop_mask, get_tiger_counties_fc
   story = get_tiger_counties_fc("19").filter(ee.Filter.eq("county_fips", "19169"))
   for yr in (2012, 2013, 2020, 2024):
       print(yr, get_crop_mask(yr).unmask(0).reduceRegion(
           ee.Reducer.mean(), story.geometry(), 30, maxPixels=1e9).get("mask").getInfo())
   ```

2. **Download.** Use `--study` — never retype the state list. It resolves to
   `config.STUDY_STATES`, so adding a state to the study cannot silently miss
   the download. Destructive flags on the FIRST batch only; passing them again
   wipes the previous batch.
   ```bash
   python download_yield.py      --study                 # writes the irrigation column
   python download_soil.py       --study --force
   python download_vegetation.py --study --no-resume
   python download_climate.py    --study --no-resume
   # later, robustness batch - NO destructive flags, NO --study:
   python download_soil.py --states CA TX AR LA WA OR ID MS GA NC
   ```
   Soil takes no year arguments. Vegetation defaults to 2008-2024. Climate
   defaults to 2005-2024 (`YEARS_CAUSAL`), 3 years wider than vegetation.

3. **Irrigation covariate — needs no downloads, highest value outstanding.**
   `clean_yield_data()` currently keeps only `ALL PRODUCTION PRACTICES`, which
   collapses irrigated and rainfed. The raw file has 9,225 IRRIGATED and 11,485
   NON-IRRIGATED rows (corn 2,271/2,471; soybeans 1,311/1,314). Irrigation is
   the prime confounder behind the spurious heat threshold.

4. Re-run `ml.py`, then `evaluate.py`. Point `RESULTS_DIR` somewhere fresh first.

### Irrigation (added 2026-09-05)

`config.NASS_PRACTICE_MODE` controls the unit of analysis — `"aggregate"`
(default, one row per county-year-crop, irrigation invisible), `"split"`
(IRRIGATED + NON-IRRIGATED only; `ml.py` scores `CORN__IRRIGATED` and
`CORN__NON_IRRIGATED` as separate strata), or `"both"` (inspection only; rows
duplicate by design and `merge_datasets` raises).

NE was added to `STUDY_STATES` for this. NE and KS are the only states with a
large balanced within-state split (NE 1020/999 corn records over 92 counties;
KS 556/578).

**The number that matters:** 1,577 county-years report BOTH practices for corn
across 223 counties — the same county, the same year, so climate and soil are
held exactly constant. The irrigated-minus-rainfed gap is **+79 BU/AC in NE and
+101 in KS**. The AGU poster's headline heat effect was 38 BU/AC, from a model
that did not control for irrigation at all.

### Not done, deliberately

- **Crop-specific NDVI masking.** Would need one vegetation table per crop
  (~6× the download). Currently generic cropland — corn NDVI is averaged with
  soybean pixels. Ship as a stated limitation.
- **Detrending yields.** Random Forests cannot extrapolate a trend; corn train
  mean 164 → test mean 179 BU/AC.
- **Dropping Kansas** from `STUDY_STATES`.

---

## 6. Security

A live `NASS_API_KEY` was committed publicly at `src/.env` in
`github.com/scienceandbeyond/CropCast`. **Rotate the key**, add `.env` to
`.gitignore`, and purge it from history (`git filter-repo`).
