import pandas as pd
from merge import DPQ_COLUMNS, recode_phq9, score_phq9


def score(row_values):
    """
    Helper: builds a one-row dataframe with the given DPQ answers, runs it
    through recode_phq9 + score_phq9, and returns the resulting phq9 value.
    """
    data = pd.DataFrame([row_values], columns=DPQ_COLUMNS)
    data[DPQ_COLUMNS] = recode_phq9(data)
    data = score_phq9(data)
    return data['phq9'].iloc[0]


def test_all_zeros_scores_zero():
    assert score([0] * 9) == 0


def test_all_threes_scores_27():
    assert score([3] * 9) == 27


def test_refused_item_is_missing_not_added():
    # eight valid answers (summing to 8) plus one refused (7) answer
    result = score([1, 1, 1, 1, 1, 1, 1, 1, 7])
    assert pd.isna(result)


if __name__ == '__main__':
    test_all_zeros_scores_zero()
    test_all_threes_scores_27()
    test_refused_item_is_missing_not_added()
    print('All PHQ-9 scorer tests passed.')
