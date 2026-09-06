# Archive — AGU 2025 material

Everything here backs the AGU 2025 poster (GC13F-0713) and is **superseded**.
Retained for provenance; tagged `agu2025-poster`.

`src_agu/`, `results_agu/`, `data_agu/` are NOT reproducible with the current
pipeline and contain known defects, most importantly:

- The CDL crop mask was inverted for 2013-2023 (`cultivated.eq(1)` selects
  *Non*-cultivated), so vegetation indices for those years were averaged over
  land that was not farmland.
- Precipitation and reference ET were season totals while growing-season
  windows differ by state, letting the model learn "low rainfall = Minnesota".
- Soil used a "cropland union" mask that covered essentially the whole county,
  and a 0 cm surface value rather than a rooting-zone depth.
- No irrigation control, which is the largest single confounder for the
  poster's "29 C heat threshold" claim.

**The poster reported its own results accurately, and its headline figures
survive these corrections.** Both reconcile exactly to
`results_agu/improvement_summary.csv` as committed here: CORN `best_r2` = 0.764,
and the mean of `pct_improvement` across the six crops is 74.1%. Re-running the
same analysis on fully corrected data gives CORN 0.772 and +82%.

The defects listed above are real and worth fixing, but they are not what
changes the conclusion. That a crop mask can be inverted for 11 of 18 years
while the headline R2 moves by 0.008 is the finding, not a footnote.

Two caveats on the "+74%" that are matters of method rather than of data, and
would apply even to a clean pipeline:

- It is a mean of per-crop ratios, so crops with weak baselines dominate. Oats
  (+137%, baseline R2 0.189) and spring wheat (+162%, baseline 0.265) pull the
  average up; corn, the largest crop, improved 28%.
- Test R2 on a county-year panel largely measures persistent differences
  *between* counties. The current analysis scores against a county-mean
  baseline instead, which is the change that reverses the conclusion.

The poster's crop panel also included upland cotton, which the current study
drops: the 11-state Corn Belt / Northern Plains sample does not contain
meaningful cotton acreage.

See `../src/HANDOFF.md`.
