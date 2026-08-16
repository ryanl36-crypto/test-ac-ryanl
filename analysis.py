"""
CSE 163 Final Project - analysis module.

Fits the regression models for the three research questions and produces
every figure in the report. The cleaned data and the per-question samples
come from merge.py; nothing in this file touches the raw .xpt files.

All findings are associations. The data are cross-sectional, so no
coefficient here supports a causal claim in either direction.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.base.wrapper import ResultsWrapper

from merge import DIR, prepare_data, sampleA, sampleB, sampleC

# Part 2 feedback asked for larger, more readable figures, so every figure
# in this module is sized explicitly and saved at 150 dpi.
plt.rcParams.update({'font.size': 12})

# The Q1 adjustment set, reused by every later model so that the models
# stay nested and the log_crp coefficient stays comparable across them.
BASE_TERMS = 'log_crp + RIDAGEYR + C(RIAGENDR) + INDFMPIR'

# Q3's age-group model swaps continuous age for the binned version rather
# than including both, since the two would be almost perfectly collinear.
AGE_GROUP_TERMS = 'log_crp + C(age_group) + C(RIAGENDR) + INDFMPIR'


def save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Saves the given figure into the project folder under filename, at the
    150 dpi the report needs. Centralised so every figure is saved the
    same way, and closed afterwards so repeated calls do not accumulate
    open figures.
    """
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, filename), dpi=150)
    plt.close(fig)


def save_correlation(data: pd.DataFrame, columns: list[str],
                     filename: str) -> pd.DataFrame:
    """
    Saves the pairwise correlation matrix of the given predictors as a CSV.
    This is the multicollinearity assumption check: OLS coefficients become
    unstable when predictors are strongly correlated with each other.
    """
    corr = data[columns].corr()
    corr.to_csv(os.path.join(DIR, filename))
    return corr


def add_age_group(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an age_group column binning age into 18-39, 40-59 and 60+.
    Q3 asks whether the hs-CRP association differs by age group, which
    needs age as a categorical variable rather than a continuous one.
    """
    data = data.copy()
    data['age_group'] = pd.cut(
        data['RIDAGEYR'], bins=[17, 39, 59, 120],
        labels=['18-39', '40-59', '60+'])
    return data


def plot_hscrp(data: pd.DataFrame) -> None:
    """
    Saves a two-panel histogram comparing raw (trimmed) hs-CRP against its
    log transform, justifying the transform. Assumes trim_and_log_crp has
    already been applied.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(data['LBXHSCRP'].dropna(), bins=30)
    axes[0].set_title('hs-CRP, raw (trimmed > 10 mg/L)')
    axes[0].set_xlabel('hs-CRP (mg/L)')
    axes[0].set_ylabel('count')
    axes[1].hist(data['log_crp'].dropna(), bins=30)
    axes[1].set_title('hs-CRP, log-transformed')
    axes[1].set_xlabel('log(1 + hs-CRP)')
    axes[1].set_ylabel('count')
    save_figure(fig, 'plot_hscrp.png')


def plot_phq9(data: pd.DataFrame) -> None:
    """
    Saves a histogram of PHQ-9 total scores, showing the floor-at-zero
    right skew that drives the normality and homoscedasticity violations.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data['phq9'].dropna(), bins=range(0, 29))
    ax.set_title('PHQ-9 total score distribution')
    ax.set_xlabel('PHQ-9 score (0-27)')
    ax.set_ylabel('count')
    save_figure(fig, 'plot_phq9.png')


def plot_crp_vs_phq9(data: pd.DataFrame, name: str) -> None:
    """
    Saves a scatter plot of log-transformed hs-CRP against PHQ-9 score for
    the named sample, the core relationship behind every research question.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data['log_crp'], data['phq9'], alpha=0.3, s=10)
    ax.set_title(f'log(hs-CRP) vs PHQ-9 score, {name}')
    ax.set_xlabel('log(1 + hs-CRP)')
    ax.set_ylabel('PHQ-9 score')
    save_figure(fig, f'plot_crp_vs_phq9_{name}.png')


def plot_dxa_missingness(data: pd.DataFrame) -> None:
    """
    Compares age and BMI distributions between adults with a valid DXA
    scan and those without, as evidence DXA missingness is non-random
    (skews toward younger, leaner participants).
    """
    has_dxa = data['DXDTOPF'].notna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(
        data.loc[has_dxa, 'RIDAGEYR'], bins=20, alpha=0.5, label='has DXA')
    axes[0].hist(
        data.loc[~has_dxa, 'RIDAGEYR'], bins=20, alpha=0.5,
        label='missing DXA')
    axes[0].set_title('Age by DXA availability')
    axes[0].set_xlabel('age (years)')
    axes[0].set_ylabel('count')
    axes[0].legend()
    axes[1].hist(
        data.loc[has_dxa, 'BMXBMI'].dropna(), bins=20, alpha=0.5,
        label='has DXA')
    axes[1].hist(
        data.loc[~has_dxa, 'BMXBMI'].dropna(), bins=20, alpha=0.5,
        label='missing DXA')
    axes[1].set_title('BMI by DXA availability')
    axes[1].set_xlabel('BMI')
    axes[1].set_ylabel('count')
    axes[1].legend()
    save_figure(fig, 'plot_dxa_missingness.png')


def plot_bodyfat(data: pd.DataFrame) -> None:
    """
    Sample B figure. Panel 1 is the distribution of DXA total body fat %;
    panel 2 plots body fat against log hs-CRP. Panel 2 is the reason Q2
    exists: if adiposity drives inflammation, controlling for body fat
    should absorb some of the hs-CRP coefficient.
    """
    corr = data['DXDTOPF'].corr(data['log_crp'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(data['DXDTOPF'], bins=30)
    axes[0].set_title('DXA total body fat %, Sample B')
    axes[0].set_xlabel('body fat (%)')
    axes[0].set_ylabel('count')
    axes[1].scatter(data['DXDTOPF'], data['log_crp'], alpha=0.3, s=10)
    axes[1].set_title(f'Body fat % vs log(hs-CRP), r = {corr:.2f}')
    axes[1].set_xlabel('body fat (%)')
    axes[1].set_ylabel('log(1 + hs-CRP)')
    save_figure(fig, 'plot_bodyfat.png')


def plot_interactions(data: pd.DataFrame, sex_model: ResultsWrapper,
                      age_model: ResultsWrapper) -> None:
    """
    Q3 figure. Plots the model-implied PHQ-9 slope on log(hs-CRP)
    separately by sex and by age group, over the observed data in grey.
    The other covariates are held at their sample means (and sex held at
    male in the right panel), which shifts the lines up or down but does
    not change their slopes - and the slopes are what Q3 is asking about.
    """
    grid = np.linspace(data['log_crp'].min(), data['log_crp'].max(), 50)
    mean_age = data['RIDAGEYR'].mean()
    mean_income = data['INDFMPIR'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        ax.scatter(data['log_crp'], data['phq9'], alpha=0.08, s=8,
                   color='grey')
        ax.set_xlabel('log(1 + hs-CRP)')
        ax.set_ylabel('PHQ-9 score')

    for code, label in ((1.0, 'male'), (2.0, 'female')):
        frame = pd.DataFrame({
            'log_crp': grid, 'RIDAGEYR': mean_age,
            'RIAGENDR': code, 'INDFMPIR': mean_income})
        axes[0].plot(grid, sex_model.predict(frame), linewidth=2,
                     label=label)
    axes[0].set_title('hs-CRP slope by sex')
    axes[0].legend()

    for label in data['age_group'].cat.categories:
        frame = pd.DataFrame({
            'log_crp': grid, 'age_group': label,
            'RIAGENDR': 1.0, 'INDFMPIR': mean_income})
        axes[1].plot(grid, age_model.predict(frame), linewidth=2,
                     label=label)
    axes[1].set_title('hs-CRP slope by age group')
    axes[1].legend()

    save_figure(fig, 'plot_interactions.png')


def model_diagnostics(model: ResultsWrapper, name: str) -> None:
    """
    Saves the two Result Validity assumption checks for a fitted model,
    labelled with name: a residuals-vs-fitted plot for constant variance,
    and a Q-Q plot of the residuals for normality.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(model.fittedvalues, model.resid, alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f'Residuals vs fitted values, {name}')
    ax.set_xlabel('fitted PHQ-9 score')
    ax.set_ylabel('residual')
    save_figure(fig, f'plot_residuals_{name}.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    # qqplot builds its own figure unless handed an axis, and a figure it
    # builds itself cannot be titled afterwards.
    sm.qqplot(model.resid, line='s', ax=ax)
    ax.set_title(f'Q-Q plot of residuals, {name}')
    save_figure(fig, f'plot_qq_{name}.png')


def fit_model(formula: str, data: pd.DataFrame, label: str,
              slug: str) -> ResultsWrapper:
    """
    Fits formula by OLS with HC3 heteroscedasticity-robust standard errors
    and returns the fitted model. Prints the summary under label and saves
    it to model_<slug>.txt for the report.

    HC3 is used because the PHQ-9 floor at zero makes the residual
    variance clearly non-constant. It leaves the coefficients untouched
    and only corrects the standard errors, t statistics, p values and
    confidence intervals, which is exactly the assumption that is violated.
    """
    model = smf.ols(formula, data=data).fit(cov_type='HC3')
    header = f'{label}\nformula: {formula}\nn = {int(model.nobs)}\n'
    print('\n' + '=' * 70)
    print(header)
    print(model.summary())
    with open(os.path.join(DIR, f'model_{slug}.txt'), 'w') as f:
        f.write(header + '\n' + str(model.summary()) + '\n')
    return model


def report_attenuation(base: ResultsWrapper, full: ResultsWrapper,
                       label: str) -> None:
    """
    Prints how much the log_crp coefficient moves once the extra controls
    are added. Both models are fitted on the same sample, so the change is
    attributable to the controls and not to a change in who is included.
    """
    before = base.params['log_crp']
    after = full.params['log_crp']
    p_before = base.pvalues['log_crp']
    p_after = full.pvalues['log_crp']
    change = (after - before) / before * 100
    print(f'\n--- {label}: change in the log_crp coefficient ---')
    print(f'before added controls: {before:.4f}  (p = {p_before:.4g})')
    print(f'after added controls:  {after:.4f}  (p = {p_after:.4g})')
    print(f'change: {change:+.1f}%')


def test_interaction(model: ResultsWrapper, prefix: str) -> None:
    """
    Prints a joint Wald test of the null that every coefficient in model
    whose name starts with prefix is zero at once. One combined p value is
    needed because the age-group interaction spans two coefficients, and
    testing them separately would inflate the chance of a false positive.
    """
    names = list(model.params.index)
    terms = [n for n in names if n.startswith(prefix)]
    # The constraint matrix is built directly rather than from a formula
    # string, because coefficient names such as log_crp:C(age_group)[T.60+]
    # contain brackets and hyphens the string parser would misread.
    constraint = np.zeros((len(terms), len(names)))
    for i, term in enumerate(terms):
        constraint[i, names.index(term)] = 1
    result = model.f_test(constraint)
    print(f'\n--- joint Wald test, {prefix}* = 0 ---')
    print(f'terms tested: {terms}')
    print(f'F = {float(result.fvalue):.4f}, p = {float(result.pvalue):.4g}')


def fit_q1_model(data: pd.DataFrame) -> ResultsWrapper:
    """
    RQ1: does hs-CRP associate with PHQ-9, adjusting for age, sex and
    income? Takes Sample A and returns the HC3 fit, alongside saving the
    model's assumption-check figures and predictor correlation matrix.

    Null hypothesis: the coefficient on log_crp is zero.
    Test: two-sided t test on that coefficient in a multiple OLS model.

    Both a classical and an HC3 fit are reported. The classical one
    reproduces the standard errors quoted in the Part 2 report; the HC3
    one is the Part 3 result, since Part 2 documented the
    heteroscedasticity violation and promised robust errors here.
    """
    formula = f'phq9 ~ {BASE_TERMS}'
    classical = smf.ols(formula, data=data).fit()
    print('\n' + '=' * 70)
    print('Q1, classical OLS errors (reproduces the Part 2 numbers)\n')
    print(classical.summary())

    robust = fit_model(formula, data, 'Q1: hs-CRP and PHQ-9, Sample A, HC3',
                       'q1')
    model_diagnostics(classical, 'q1')
    save_correlation(data, ['log_crp', 'RIDAGEYR', 'INDFMPIR'],
                     'predictor_correlation.csv')
    return robust


def fit_count_model(data: pd.DataFrame,
                    linear_model: ResultsWrapper) -> ResultsWrapper:
    """
    Refits the Q1 model on Sample A as a negative binomial count model and
    returns it, as a robustness check on the linear model's assumption
    violations. linear_model is the fitted Q1 model to compare against.

    HC3 corrects the standard errors of a linear model but does not make a
    linear model appropriate for this outcome. PHQ-9 is a bounded integer
    count with a hard floor at zero, and OLS treats it as continuous and
    unbounded. The negative binomial instead models the mean on a log
    scale, treats predictors as multiplicative rather than additive, and
    lets the variance grow with the mean rather than assuming it is
    constant, which is what the residual plot actually shows.

    Also reports the range of the linear model's fitted values, since an
    unbounded model can in principle predict negative scores that are
    impossible on a 0-27 scale, and the effect on log_crp as an incidence
    rate ratio: the multiplicative change in the expected PHQ-9 score per
    one unit of log(1 + hs-CRP).
    """
    fitted = linear_model.fittedvalues
    below_zero = int((fitted < 0).sum())
    print(f'\nOLS fitted PHQ-9 range: {fitted.min():.2f} to '
          f'{fitted.max():.2f}, with {below_zero} of {len(data)} below '
          f'zero (observed scores run 0 to {data["phq9"].max():.0f})')

    formula = f'phq9 ~ {BASE_TERMS}'
    model = smf.negativebinomial(formula, data=data).fit(disp=0)
    irr = np.exp(model.params['log_crp'])
    low, high = np.exp(model.conf_int().loc['log_crp'])
    p_value = model.pvalues['log_crp']
    rate = (irr - 1) * 100

    header = (f'Q1 robustness: negative binomial count model\n'
              f'formula: {formula}\nn = {int(model.nobs)}\n')
    summary = (f'log_crp incidence rate ratio: {irr:.4f} '
               f'[{low:.4f}, {high:.4f}], p = {p_value:.4g}\n'
               f'i.e. each unit of log(1 + hs-CRP) is associated with a '
               f'{rate:.1f}% higher expected PHQ-9 score.\n')
    print('\n' + '=' * 70)
    print(header)
    print(model.summary())
    print(summary)
    with open(os.path.join(DIR, 'model_q1_negbin.txt'), 'w') as f:
        f.write(header + '\n' + str(model.summary()) + '\n\n' + summary)
    return model


def fit_q2_model(data: pd.DataFrame) -> tuple[ResultsWrapper,
                                              ResultsWrapper]:
    """
    RQ2: does the hs-CRP association survive adjustment for body fat and
    physical activity, or is it explained by adiposity? Takes Sample B and
    returns the model without the adiposity controls and the model with
    them, in that order.

    Null hypothesis: the coefficient on log_crp is zero once body fat %
    and sedentary minutes are also in the model.
    Test: two-sided t test on that coefficient, HC3 errors.

    Both models are fitted on Sample B. Sample B is a different and much
    smaller set of people than Sample A, so comparing the full model
    against Sample A's coefficient would confuse a change of controls with
    a change of sample. Returning both fits from the same sample holds the
    people fixed and varies only the controls.
    """
    base = fit_model(f'phq9 ~ {BASE_TERMS}', data,
                     'Q2 base: Q1 model refitted on Sample B', 'q2_base')
    full = fit_model(
        f'phq9 ~ {BASE_TERMS} + DXDTOPF + PAD680', data,
        'Q2 full: adding DXA body fat % and sedentary minutes', 'q2_full')
    report_attenuation(base, full, 'Q2, Sample B, DXA body fat %')
    model_diagnostics(full, 'q2')
    save_correlation(
        data, ['log_crp', 'RIDAGEYR', 'INDFMPIR', 'DXDTOPF', 'PAD680'],
        'predictor_correlation_q2.csv')
    return base, full


def fit_q2_robustness(data: pd.DataFrame) -> tuple[ResultsWrapper,
                                                   ResultsWrapper]:
    """
    RQ2 robustness check. Takes Sample C and returns the model without the
    adiposity controls and the model with them, substituting BMI for DXA
    body fat. Same null hypothesis and test as fit_q2_model.

    BMI is a cruder measure of adiposity, but it was measured on roughly
    twice as many adults and across the whole age range, so this check
    asks whether the Q2 answer is an artefact of the younger, leaner DXA
    subsample.
    """
    base = fit_model(f'phq9 ~ {BASE_TERMS}', data,
                     'Q2 robustness base: Q1 model refitted on Sample C',
                     'q2c_base')
    full = fit_model(
        f'phq9 ~ {BASE_TERMS} + BMXBMI + PAD680', data,
        'Q2 robustness full: adding BMI and sedentary minutes', 'q2c_full')
    report_attenuation(base, full, 'Q2 robustness, Sample C, BMI')
    save_correlation(
        data, ['log_crp', 'RIDAGEYR', 'INDFMPIR', 'BMXBMI', 'PAD680'],
        'predictor_correlation_q2c.csv')
    return base, full


def fit_q3_model(data: pd.DataFrame) -> tuple[ResultsWrapper,
                                              ResultsWrapper]:
    """
    RQ3: does the hs-CRP association differ by sex or by age group? Takes
    Sample A, which must already have an age_group column, and returns the
    sex model and the age-group model in that order.

    Null hypothesis: every interaction coefficient between log_crp and the
    grouping variable is zero, i.e. the slope on log_crp is the same in
    every group.
    Test: joint Wald F test on the interaction terms, HC3 errors.

    Sex and age group are fitted as two separate models so that each
    interaction is tested against the same Q1 adjustment set, rather than
    each one soaking up variation the other would have explained.
    """
    sex_model = fit_model(
        f'phq9 ~ {BASE_TERMS} + log_crp:C(RIAGENDR)', data,
        'Q3a: does the hs-CRP slope differ by sex?', 'q3_sex')
    test_interaction(sex_model, 'log_crp:C(RIAGENDR)')

    age_model = fit_model(
        f'phq9 ~ {AGE_GROUP_TERMS} + log_crp:C(age_group)', data,
        'Q3b: does the hs-CRP slope differ by age group?', 'q3_age')
    test_interaction(age_model, 'log_crp:C(age_group)')

    plot_interactions(data, sex_model, age_model)
    return sex_model, age_model


def main() -> None:
    """
    Runs the whole analysis: builds the three samples from the cleaned
    data, then saves every figure and fits every model the report cites.
    """
    data = prepare_data(DIR)
    plot_dxa_missingness(data)

    sample_a = add_age_group(sampleA(data))
    print('\nSample A:', sample_a.shape)
    plot_hscrp(sample_a)
    plot_phq9(sample_a)
    plot_crp_vs_phq9(sample_a, 'sampleA')
    q1_model = fit_q1_model(sample_a)
    fit_count_model(sample_a, q1_model)
    fit_q3_model(sample_a)

    sample_b = sampleB(data)
    print('\nSample B:', sample_b.shape)
    plot_bodyfat(sample_b)
    plot_crp_vs_phq9(sample_b, 'sampleB')
    fit_q2_model(sample_b)

    sample_c = sampleC(data)
    print('\nSample C:', sample_c.shape)
    fit_q2_robustness(sample_c)


if __name__ == '__main__':
    main()
