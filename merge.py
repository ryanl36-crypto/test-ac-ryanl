import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
# changes the directory assuming that the user has all of the files in the
# same folder as the py file
DIR = os.path.dirname(os.path.abspath(__file__))

# The nine PHQ-9 item columns, DPQ010-DPQ090, reused across several functions
DPQ_COLUMNS = [f"DPQ{n:03d}" for n in range(10, 100, 10)]

# Dictionary with each file name as the key and a list of columns needed from
# each file
COLUMNS = {
    'DEMO_J.xpt': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'RIDRETH3'],
    'HSCRP_J.xpt': ['SEQN', 'LBXHSCRP'],
    'DPQ_J.xpt': ['SEQN'] + DPQ_COLUMNS,
    'DXX_J.xpt': ['SEQN', 'DXDTOPF'],
    'BMX_J.xpt': ['SEQN', 'BMXBMI'],
    'PAQ_J.xpt': None
}


def load_data(folder_path: str) -> dict[str, pd.DataFrame]:
    """
    takes in a str folder path of .xpt files and return a dict of file names
    as the key and filtered dataframes as the value
    """
    result = {}
    for file in COLUMNS:
        file_name = os.path.join(folder_path, file)
        if COLUMNS[file] is None:
            result[file] = pd.read_sas(file_name, format='xport').round(6)
        else:
            result[file] = pd.read_sas(
                file_name, format='xport')[COLUMNS[file]].round(6)
    return result


def merge_all_files(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Takes in the dict of file names as the key and dataframes filtered for the
    needed columns as the value. Then merges all of the dataframes into one,
    maintaining all data
    """
    result = data['DEMO_J.xpt']
    for file in data:
        if file == 'DEMO_J.xpt':
            continue
        result = pd.merge(result, data[file], on='SEQN', how='left')
    return result


def count_missing(data: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Takes in a dataframe and a csv filename and returns dataframe that
    describes the counts of missing entries and percentage of missing
    entries per column. Saves the missingness dataframe as a csv to use in
    EDA report.
    """
    missing_count = data.isna().sum()  # should give a number per column
    missing_percentage = missing_count / len(data) * 100
    result = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percentage': missing_percentage
        })
    result.to_csv(os.path.join(DIR, filename))
    return result


def filter_age(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the merged dataframe and filters out individuals under 18.
    Returns the dataframe of only adults.
    """
    return data.loc[data['RIDAGEYR'] >= 18]


def recode_phq9(data: pd.DataFrame) -> pd.DataFrame:
    """
    takes in the adult merged dataframe and replaces out any "I don't know"
    or "I refuse to answer" values with NaN
    """
    return data[DPQ_COLUMNS].mask(data[DPQ_COLUMNS].isin([7, 9]))


def score_phq9(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the 18+ merged dataframe and scores the PHQ9 scores.
    In the case of NaN values, the PHQ9 score will also be NaN
    """
    data['phq9'] = data[DPQ_COLUMNS].sum(axis=1, skipna=False)
    return data


def sampleA(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the 18+ merged PHQ9 scored dataframe and returns a dataframe
    to be used in statistical analysis for research question 1. The dataframe
    is filtered to the required variables
    """
    sampleA_data = data[[
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'LBXHSCRP', 'phq9',
        'log_crp'
        ]]
    return sampleA_data.dropna()


def sampleB(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the 18+ merged PHQ9 scored dataframe and returns a dataframe
    to be used in statistical analysis for research question 2. The dataframe
    is filtered to the required variables
    """
    sampleB_data = data[[
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'LBXHSCRP', 'phq9',
        'log_crp', 'DXDTOPF', 'PAD680'
        ]]
    return sampleB_data.dropna()


def sampleC(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the 18+ merged PHQ9 scored dataframe and returns a dataframe
    to be used in statistical analysis for research question 2 as a robustness
    check. The dataframe is filtered to the required variables.
    """
    sampleC_data = data[[
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR', 'LBXHSCRP', 'phq9',
        'log_crp', 'BMXBMI', 'PAD680'
        ]]
    return sampleC_data.dropna()


def describe_sample(data: pd.DataFrame, name: str) -> None:
    """
    Saves a seven-number summary (mean, std, min, Q1, median, Q3, max) for
    each quantitative variable of interest, and a value count for each
    categorical variable of interest (RIAGENDR), labeled with the sample
    name. SEQN is excluded since it's a participant ID, not a variable of
    interest, and RIAGENDR is excluded from the quantitative summary since
    it's categorical, not quantitative.
    """
    quant_columns = [c for c in data.columns if c not in ('SEQN', 'RIAGENDR')]
    seven_number_summary = data[quant_columns].describe().drop('count')
    seven_number_summary.to_csv(os.path.join(DIR, f'describe_{name}.csv'))
    data['RIAGENDR'].value_counts().to_csv(
        os.path.join(DIR, f'gender_counts_{name}.csv'))


def trim_and_log_crp(data: pd.DataFrame) -> pd.DataFrame:
    """
    Trims hs-CRP values above 10 mg/L (acute infection, not chronic
    inflammation) to NaN and adds a log-transformed log_crp column.
    Returns the dataframe with both changes applied.
    """
    data = data.copy()
    data.loc[data['LBXHSCRP'] > 10, 'LBXHSCRP'] = np.nan
    data['log_crp'] = np.log1p(data['LBXHSCRP'])
    return data


def plot_hscrp(data: pd.DataFrame) -> None:
    """
    Saves a two-panel histogram comparing raw (trimmed) hs-CRP against its
    log transform, justifying the transform. Assumes trim_and_log_crp has
    already been applied.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data['LBXHSCRP'].dropna(), bins=30)
    axes[0].set_title('hs-CRP, raw (trimmed > 10 mg/L)')
    axes[0].set_xlabel('mg/L')
    axes[1].hist(data['log_crp'].dropna(), bins=30)
    axes[1].set_title('hs-CRP, log-transformed')
    axes[1].set_xlabel('log(1 + mg/L)')
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'plot_hscrp.png'))
    plt.close(fig)
    return data


def plot_phq9(data: pd.DataFrame) -> None:
    """
    Saves a histogram of PHQ-9 total scores, showing the floor-at-zero
    right skew.
    """
    fig, ax = plt.subplots()
    ax.hist(data['phq9'].dropna(), bins=range(0, 29))
    ax.set_title('PHQ-9 total score distribution')
    ax.set_xlabel('PHQ-9 score (0-27)')
    ax.set_ylabel('count')
    fig.savefig(os.path.join(DIR, 'plot_phq9.png'))
    plt.close(fig)


def plot_crp_vs_phq9(data: pd.DataFrame) -> None:
    """
    Saves a scatter plot of log-transformed hs-CRP against PHQ-9 score,
    the core relationship for research question 1.
    """
    fig, ax = plt.subplots()
    ax.scatter(data['log_crp'], data['phq9'], alpha=0.3, s=10)
    ax.set_title('log(hs-CRP) vs PHQ-9 score')
    ax.set_xlabel('log(1 + hs-CRP)')
    ax.set_ylabel('PHQ-9 score')
    fig.savefig(os.path.join(DIR, 'plot_crp_vs_phq9.png'))
    plt.close(fig)


def plot_dxa_missingness(data: pd.DataFrame) -> None:
    """
    Compares age and BMI distributions between adults with a valid DXA
    scan and those without, as evidence DXA missingness is non-random
    (skews toward younger, leaner participants).
    """
    has_dxa = data['DXDTOPF'].notna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        data.loc[has_dxa, 'RIDAGEYR'], bins=20, alpha=0.5, label='has DXA')
    axes[0].hist(
        data.loc[~has_dxa, 'RIDAGEYR'], bins=20, alpha=0.5,
        label='missing DXA')
    axes[0].set_title('Age by DXA availability')
    axes[0].set_xlabel('age')
    axes[0].legend()
    axes[1].hist(
        data.loc[has_dxa, 'BMXBMI'].dropna(), bins=20, alpha=0.5,
        label='has DXA')
    axes[1].hist(
        data.loc[~has_dxa, 'BMXBMI'].dropna(), bins=20, alpha=0.5,
        label='missing DXA')
    axes[1].set_title('BMI by DXA availability')
    axes[1].set_xlabel('BMI')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'plot_dxa_missingness.png'))
    plt.close(fig)


def fit_q1_model(data: pd.DataFrame):
    """
    Fits the Q1 model (phq9 ~ log_crp + age + sex + income) with
    statsmodels, saves a residuals-vs-fitted plot, a Q-Q plot, and a
    predictor correlation matrix as Result Validity assumption checks.
    """
    model = smf.ols(
        'phq9 ~ log_crp + RIDAGEYR + C(RIAGENDR) + INDFMPIR', data=data
        ).fit()
    print(model.summary())

    fig, ax = plt.subplots()
    ax.scatter(model.fittedvalues, model.resid, alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title('Residuals vs fitted values')
    ax.set_xlabel('fitted values')
    ax.set_ylabel('residuals')
    fig.savefig(os.path.join(DIR, 'plot_residuals.png'))
    plt.close(fig)

    qq_fig = sm.qqplot(model.resid, line='s')
    qq_fig.savefig(os.path.join(DIR, 'plot_qq.png'))
    plt.close(qq_fig)

    corr = data[['log_crp', 'RIDAGEYR', 'INDFMPIR']].corr()
    corr.to_csv(os.path.join(DIR, 'predictor_correlation.csv'))
    return model


def main():
    data = load_data(DIR)
    data = merge_all_files(data)
    print(data.shape)
    count_missing(data, 'missingness.csv')

    data = filter_age(data)
    print(len(data))
    count_missing(data, 'missing_adults.csv')

    data['PAD680'] = data['PAD680'].replace([7777, 9999], np.nan)
    data[DPQ_COLUMNS] = recode_phq9(data)
    data = score_phq9(data)
    data = trim_and_log_crp(data)

    plot_dxa_missingness(data)

    sampleA_data = sampleA(data)
    print(sampleA_data.shape)
    count_missing(sampleA_data, 'missing_sampleA.csv')
    describe_sample(sampleA_data, 'sampleA')
    plot_hscrp(sampleA_data)
    plot_phq9(sampleA_data)
    plot_crp_vs_phq9(sampleA_data)
    fit_q1_model(sampleA_data)

    sampleB_data = sampleB(data)
    print(sampleB_data.shape)
    count_missing(sampleB_data, 'missing_sampleB.csv')
    describe_sample(sampleB_data, 'sampleB')

    sampleC_data = sampleC(data)
    print(sampleC_data.shape)


if __name__ == '__main__':
    main()
