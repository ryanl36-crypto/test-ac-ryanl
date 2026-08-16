# Is Inflammation Associated with Depression Independent of Body Fat?

An analysis of hs-CRP and depressive symptoms in U.S. adults, using NHANES
2017–2018 data.

CSE 163 (Intermediate Data Programming), University of Washington.
Author: Ryan Liu.

This is a cross-sectional association study. **All findings are associations,
never causal claims.** Reverse causation is plausible — depression itself
raises inflammation.

## Research questions

1. Does hs-CRP associate with PHQ-9 depression severity, adjusting for age,
   sex, and income?
2. Does that association survive adjustment for body fat percentage and
   physical activity, or is it explained by adiposity?
3. Does the association differ by sex or by age group?

## Requirements

Python 3.9 or newer (developed on 3.13.9), plus four libraries:

```
pip install pandas numpy matplotlib statsmodels
```

Developed against pandas 3.0.3, numpy 2.5.1, matplotlib 3.11.0, and
statsmodels 0.14.6.

## Getting the data (required — the repository does not contain it)

The six NHANES data files are **not committed**. They total roughly 10 MB, and
`.gitignore` excludes `*.xpt` so that public federal data is not duplicated in
version control. You must download them before the code will run.

Download all six from the CDC's NHANES 2017–2018 release and place them in the
**same folder as the `.py` files** — the code resolves paths relative to its own
location, so no configuration is needed.

| File | Direct download | Contains |
|---|---|---|
| `DEMO_J.xpt` | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT | Age, sex, income-to-poverty ratio |
| `HSCRP_J.xpt` | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HSCRP_J.XPT | High-sensitivity C-reactive protein (mg/L) |
| `DPQ_J.xpt` | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DPQ_J.XPT | The nine PHQ-9 depression items |
| `DXX_J.xpt` | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DXX_J.XPT | DXA total body fat percentage |
| `BMX_J.xpt` | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT | Body mass index |
| `PAQ_J.xpt` | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT | Physical activity, including sedentary minutes/day |

**Save each file with a lowercase `.xpt` extension.** The CDC serves them as
`.XPT`. Windows filenames are case-insensitive so this makes no difference
there, but on macOS or Linux a file saved as `DEMO_J.XPT` will not be found and
the script will raise `FileNotFoundError`.

Variable documentation for each file is at the same address with `.htm` in
place of `.XPT`.

## Running the code

From the project folder:

```
python analysis.py
```

That is the full run. It cleans the data, builds the three analytic samples,
writes every figure and table, and fits and prints all six regression models.
Expect it to take well under a minute.

To run only the data-cleaning half and its summary tables:

```
python merge.py
```

To run the tests:

```
python test_merge.py
```

## What each file does

### `merge.py` — data

Loading, merging, cleaning, and sample building. Nothing in this file plots or
models.

Loads the six transport files with `pd.read_sas`, trims each to the columns the
project needs, and **left-joins** the other five onto `DEMO_J`. The join is
deliberately left rather than inner: a global inner join would collapse every
research question down to the smallest file (DXA) and would destroy the
missingness evidence the report depends on.

Cleaning steps, in an order that matters:

- Missingness is counted on the full merged table **before** any rows are
  dropped, because it is required report evidence and cannot be recovered
  afterwards.
- Adults only (`RIDAGEYR >= 18`); the PHQ-9 is administered to 18+ only.
- PHQ-9 items coded 7 (refused) and 9 (don't know) become `NaN` before summing.
  Left alone, a single refusal would silently add 7 points to a score with a
  maximum of 27.
- `PAD680` sedentary minutes coded 7777/9999 likewise become `NaN`.
- The PHQ-9 total is the sum of the nine items with `skipna=False`, so a
  partially answered questionnaire scores `NaN` rather than a falsely low total.
- hs-CRP above 10 mg/L is set to `NaN` (that range indicates acute infection
  rather than the chronic low-grade inflammation this project is about), then
  log-transformed with `log1p`, which tolerates the zeros that `log` would not.
  This happens before the samples are built so all three inherit it.

Three complete-case samples are then built, one per question, dropping
incomplete rows only for the variables that question actually needs:

| Sample | Used for | n | Adds |
|---|---|---|---|
| A | Q1 and Q3 | 3,847 | — |
| B | Q2, primary | 1,842 | DXA body fat %, sedentary minutes |
| C | Q2, robustness | 3,787 | BMI, sedentary minutes |

### `analysis.py` — figures and models

Imports the cleaned data from `merge.py` and produces everything the report
cites. The six linear models are fitted with `statsmodels` using HC3
heteroscedasticity-robust standard errors, because the PHQ-9's hard floor at
zero makes the residual variance visibly non-constant. HC3 leaves the
coefficients untouched and corrects only the inference.

Q1 is additionally fitted with classical standard errors so the two can be
compared directly; they turn out to agree closely.

A seventh model refits Q1 as a negative binomial count model. HC3 fixes the
inference for a linear model but does not make a linear model the right shape
for a bounded integer outcome, so this checks whether the Q1 conclusion depends
on that choice. It also yields an incidence rate ratio, which is a more
interpretable effect size than a change in points.

Q2 refits the Q1 model on Sample B and Sample C before adding the adiposity
controls. Comparing against Sample A's coefficient instead would confuse a
change of controls with a change of which people are included.

### `test_merge.py` — tests

Tests the PHQ-9 scorer, which carries the highest correctness risk in the
project: all-zero answers must score 0, all-threes must score 27, and a refusal
coded 7 must produce a missing score rather than adding 7 points.

## Output

Running `analysis.py` writes the following into the project folder.

Figures: `plot_hscrp.png`, `plot_phq9.png`, `plot_crp_vs_phq9_sampleA.png`,
`plot_crp_vs_phq9_sampleB.png`, `plot_bodyfat.png`,
`plot_dxa_missingness.png`, `plot_interactions.png`, and residual and Q-Q
diagnostic plots for the Q1 and Q2 models.

Tables: per-stage missingness counts, seven-number summaries and sex
`value_counts()` for each sample, and predictor correlation matrices used as
the multicollinearity assumption check.

Model summaries: the full `statsmodels` output for each model, saved as
`model_q1.txt`, `model_q1_negbin.txt`, `model_q2_base.txt`,
`model_q2_full.txt`, `model_q2c_base.txt`, `model_q2c_full.txt`,
`model_q3_sex.txt`, and `model_q3_age.txt`.

## Known limitations

- **The DXA subsample is not representative.** DXA was administered only to
  ages 8–59, and scan validity falls with increasing age and BMI, so Sample B
  skews younger and leaner than Sample A. `plot_dxa_missingness.png` documents
  this. It is a property of the survey, not a defect in the cleaning.
- **The design is cross-sectional.** hs-CRP and PHQ-9 are measured at the same
  visit, so nothing here can establish direction.
- **Complete-case analysis.** Rows with missing values are dropped rather than
  imputed, which assumes the missingness is unrelated to the outcome — an
  assumption the DXA data visibly violates.
- **Survey weights are not applied.** NHANES uses a complex multistage sample
  design; the estimates here are unweighted and so describe the analytic
  samples rather than the U.S. adult population.
