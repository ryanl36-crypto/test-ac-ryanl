"""
CSE 163 Final Project - data module.

Loads the six NHANES 2017-2018 transport files, merges them on SEQN,
cleans them, and builds the three per-question analytic samples. The
figures and regression models live in analysis.py.
"""
import pandas as pd
import os
import numpy as np
# changes the directory assuming that the user has all of the files in the
# same folder as the py file
DIR = os.path.dirname(os.path.abspath(__file__))

# The nine PHQ-9 item columns, DPQ010-DPQ090, reused across several functions
DPQ_COLUMNS = [f"DPQ{n:03d}" for n in range(10, 100, 10)]

# Dictionary with each file name as the key and a list of columns needed from
# each file
COLUMNS = {
    'DEMO_J.xpt': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR'],
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


def prepare_data(folder_path: str) -> pd.DataFrame:
    """
    Runs the whole cleaning pipeline on the six .xpt files in folder_path
    and returns the adult (18+) merged table, with phq9 and log_crp added
    and every sentinel code recoded to NaN. The returned table is what the
    three sample builders expect as input.

    Also saves missingness counts for the full merged table and for the
    adult table as csv files, since both are required report evidence.
    """
    data = load_data(folder_path)
    data = merge_all_files(data)
    print('Merged table:', data.shape)
    # Missingness has to be measured before any rows are dropped - it is
    # required report evidence and cannot be recovered afterwards.
    count_missing(data, 'missingness.csv')

    data = filter_age(data)
    print('Adults 18+:', len(data))
    count_missing(data, 'missing_adults.csv')

    data['PAD680'] = data['PAD680'].replace([7777, 9999], np.nan)
    data[DPQ_COLUMNS] = recode_phq9(data)
    data = score_phq9(data)
    # Trimming and logging happens here rather than per sample, so that
    # all three samples inherit the same transformed variable.
    data = trim_and_log_crp(data)
    return data


def main():
    data = prepare_data(DIR)
    builders = {'sampleA': sampleA, 'sampleB': sampleB, 'sampleC': sampleC}
    for name in builders:
        sample = builders[name](data)
        print(f'{name}: {sample.shape}')
        count_missing(sample, f'missing_{name}.csv')
        describe_sample(sample, name)


if __name__ == '__main__':
    main()
