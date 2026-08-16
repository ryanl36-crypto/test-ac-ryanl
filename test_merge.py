import pandas as pd
from merge import DPQ_COLUMNS, recode_phq9, score_phq9


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


if __name__ == '__main__':
    test_all_zeros_scores_zero()
    test_all_threes_scores_27()
    test_refused_item_is_missing_not_added()
    print('All PHQ-9 scorer tests passed.')
