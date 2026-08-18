import pandas as pd
from merge import (DPQ_COLUMNS, filter_age, merge_all_files, recode_phq9,
                   score_phq9, trim_and_log_crp)
from analysis import add_age_group


def score(row_values: list[int]) -> float:
    """
    Helper: takes nine PHQ-9 answers for one person and returns the total
    score the pipeline gives them, so each test can be a single call.
    """
    data = pd.DataFrame([row_values], columns=DPQ_COLUMNS)
    data[DPQ_COLUMNS] = recode_phq9(data)
    data = score_phq9(data)
    return data['phq9'].iloc[0]


def test_all_zeros_scores_zero() -> None:
    """
    Answering 0 to every item is the lowest possible score, so the total
    must be exactly 0 and not NaN.
    """
    assert score([0] * 9) == 0


def test_all_threes_scores_27() -> None:
    """
    Answering 3 to every item is the highest possible score, which checks
    that all nine items are counted and none is dropped.
    """
    assert score([3] * 9) == 27


def test_refused_item_is_missing_not_added() -> None:
    """
    A refusal is coded 7, which is outside the valid 0-3 range. It must
    make the whole score missing rather than adding 7 points to it - the
    highest correctness risk in the project.
    """
    # eight valid answers (summing to 8) plus one refused (7) answer
    result = score([1, 1, 1, 1, 1, 1, 1, 1, 7])
    assert pd.isna(result)


def test_crp_above_ten_is_trimmed() -> None:
    """
    hs-CRP above 10 mg/L signals acute infection rather than the chronic
    inflammation this project measures, so it must become NaN. Exactly
    10.0 is kept, since the cut is strictly greater than 10. This trim
    once ran too late in the pipeline and left an untrimmed maximum of
    109.81 in a summary table, so the boundary is worth pinning down.
    """
    data = pd.DataFrame({'LBXHSCRP': [0.0, 10.0, 10.01, 109.81]})
    result = trim_and_log_crp(data)
    assert result['LBXHSCRP'].iloc[0] == 0.0
    assert result['LBXHSCRP'].iloc[1] == 10.0
    assert pd.isna(result['LBXHSCRP'].iloc[2])
    assert pd.isna(result['LBXHSCRP'].iloc[3])
    # log1p is used rather than log so that a valid hs-CRP of 0 survives
    assert result['log_crp'].iloc[0] == 0.0


def test_age_groups_split_at_forty_and_sixty() -> None:
    """
    Q3's answer depends on these bins being right, and the ages either
    side of each boundary are where a binning error would hide.
    """
    data = pd.DataFrame({'RIDAGEYR': [18, 39, 40, 59, 60, 80]})
    result = add_age_group(data)
    assert result['age_group'].tolist() == [
        '18-39', '18-39', '40-59', '40-59', '60+', '60+']


def test_filter_age_keeps_eighteen() -> None:
    """
    The PHQ-9 is only administered to adults, so 18 must be kept and 17
    dropped.
    """
    data = pd.DataFrame({'RIDAGEYR': [17, 18, 80]})
    assert filter_age(data)['RIDAGEYR'].tolist() == [18, 80]


def test_merge_keeps_every_participant() -> None:
    """
    The merge is a left join onto DEMO_J, so a participant missing from a
    lab file keeps their row and gains a NaN. An inner join would silently
    shrink every sample to whoever appears in all six files and destroy
    the missingness evidence the report depends on.
    """
    data = {
        'DEMO_J.xpt': pd.DataFrame({'SEQN': [1.0, 2.0, 3.0]}),
        'HSCRP_J.xpt': pd.DataFrame({'SEQN': [1.0], 'LBXHSCRP': [2.5]}),
    }
    result = merge_all_files(data)
    assert result['SEQN'].tolist() == [1.0, 2.0, 3.0]
    assert result['LBXHSCRP'].notna().sum() == 1


if __name__ == '__main__':
    test_all_zeros_scores_zero()
    test_all_threes_scores_27()
    test_refused_item_is_missing_not_added()
    test_crp_above_ten_is_trimmed()
    test_age_groups_split_at_forty_and_sixty()
    test_filter_age_keeps_eighteen()
    test_merge_keeps_every_participant()
    print('All tests passed.')
