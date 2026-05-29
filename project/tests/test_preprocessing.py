"""Юнит-тесты трансформеров предобработки."""
import pandas as pd
import numpy as np
from src.features.preprocessing import create_transformer, PastDueAggregator

def test_past_due_aggregator():
    df = pd.DataFrame({
        "NumberOfTime30-59DaysPastDueNotWorse": [1, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 1],
        "NumberOfTimes90DaysLate": [0, 0],
        "age": [30, 40]
    })
    agg = PastDueAggregator()
    result = agg.transform(df)
    assert "TotalPastDue" in result.columns
    assert list(result["TotalPastDue"]) == [1, 3]
    assert "NumberOfTime30-59DaysPastDueNotWorse" not in result.columns

def test_full_pipeline_shape():
    df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.2],
        "age": [30, 40],
        "DebtRatio": [0.1, 0.2],
        "MonthlyIncome": [5000, 6000],
        "NumberOfOpenCreditLinesAndLoans": [2, 3],
        "NumberRealEstateLoansOrLines": [1, 0],
        "NumberOfDependents": [0, 1]
    })
    pipeline = create_transformer()
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[1] > 0  # Проверка, что трансформер не ломается