from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_constant_columns():
    """
    Тест проверяет эвристику has_constant_columns.
    Создаем датафрейм с константной колонкой и проверяем,
    что соответствующий флаг устанавливается в True.
    """
    # Создаем DataFrame с константной колонкой
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "constant_feature": [5, 5, 5, 5],  # Константная колонка
        "almost_constant": [10, 10, 10, None],  # Почти константная (с пропуском)
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем флаг константных колонок
    assert flags["has_constant_columns"] is True
    
    # Проверяем, что скор качества снижен из-за константной колонки
    assert flags["quality_score"] < 1.0


def test_quality_flags_high_cardinality():
    """
    Тест проверяет эвристику has_high_cardinality_categoricals.
    Создаем датафрейм с категориальной колонкой высокой кардинальности
    и проверяем, что соответствующий флаг устанавливается в True.
    """
    # Создаем DataFrame с колонкой высокой кардинальности
    df = pd.DataFrame({
        "id": range(1, 200),  # 199 строк - достаточно для анализа
        "feature1": [i % 2 for i in range(1, 200)],  # Числовая колонка
        "high_card_feature": [f"item_{i}" for i in range(1, 200)]  # 199 уникальных значений
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем флаг высокой кардинальности
    assert flags["has_high_cardinality_categoricals"] is True
    
    # Проверяем, что колонка определена как нечисловая с высокой кардинальностью
    high_card_col = next((col for col in summary.columns if col.name == "high_card_feature"), None)
    assert high_card_col is not None
    assert not high_card_col.is_numeric
    assert high_card_col.non_null == 199
    assert high_card_col.unique == 199
    
    # Проверяем, что скор качества снижен из-за высокой кардинальности
    assert flags["quality_score"] < 1.0