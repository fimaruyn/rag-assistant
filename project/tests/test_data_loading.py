"""Тесты для модулей загрузки и валидации данных."""
import pandas as pd
import pytest
from pathlib import Path
from pandas.api.types import is_integer_dtype

from src.data.loader import load_data
from src.data.validator import validate_data


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Создаёт минимальный валидный CSV для изолированного тестирования."""
    # Создаём DataFrame со всеми колонками схемы
    data = {
        "SeriousDlqin2yrs": [0, 1, 0],
        "RevolvingUtilizationOfUnsecuredLines": [0.5, 0.1, 1.2],
        "age": [30, 45, 60],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 1, None],
        "DebtRatio": [0.2, 0.5, 0.1],
        "MonthlyIncome": [5000.0, None, 3000.0],
        "NumberOfOpenCreditLinesAndLoans": [3, 5, 2],
        "NumberOfTimes90DaysLate": [0, 2, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, None, 1],
        "NumberOfDependents": [0, 2, None]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_credit.csv"
    # index=True имитирует ID из реального датасета (первый столбец — индекс)
    df.to_csv(file_path, index=True)
    return file_path


def test_load_data_success(valid_csv: Path):
    """Проверка успешной загрузки и валидации корректных данных."""
    df = load_data(valid_csv.parent, valid_csv.name)
    assert isinstance(df, pd.DataFrame)
    assert "SeriousDlqin2yrs" in df.columns
    assert len(df) == 3
    assert is_integer_dtype(df["age"])


def test_load_data_file_not_found(tmp_path: Path):
    """Проверка обработки отсутствия файла."""
    with pytest.raises(FileNotFoundError, match="Training data file not found"):
        load_data(tmp_path, "nonexistent.csv")


def test_validate_data_missing_target():
    """Проверка отказа при отсутствии целевой переменной."""
    # Минимальный DataFrame без обязательных колонок
    df = pd.DataFrame({
        "age": [30],
        "RevolvingUtilizationOfUnsecuredLines": [0.5],
        "DebtRatio": [0.1],
        "MonthlyIncome": [5000.0],
        "NumberOfOpenCreditLinesAndLoans": [1],
        "NumberOfTimes90DaysLate": [0],
        "NumberRealEstateLoansOrLines": [0]
        # Намеренно отсутствуют: SeriousDlqin2yrs и другие
    })
    with pytest.raises(ValueError, match="validation"):
        validate_data(df)


def test_validate_data_invalid_age():
    """Проверка отказа при недопустимом возрасте (<18)."""
    df = pd.DataFrame({
        "SeriousDlqin2yrs": [0],
        "age": [10],  # Нарушение: возраст < 18
        "RevolvingUtilizationOfUnsecuredLines": [0.5],
        "DebtRatio": [0.1],
        "MonthlyIncome": [5000.0],
        "NumberOfOpenCreditLinesAndLoans": [1],
        "NumberOfTimes90DaysLate": [0],
        "NumberRealEstateLoansOrLines": [0],
        "NumberOfTime30-59DaysPastDueNotWorse": [0],
        "NumberOfTime60-89DaysPastDueNotWorse": [0],
        "NumberOfDependents": [0]
    })
    with pytest.raises(ValueError, match="validation"):
        validate_data(df)


def test_validate_data_negative_debt_ratio():
    """Проверка отказа при отрицательном DebtRatio."""
    df = pd.DataFrame({
        "SeriousDlqin2yrs": [0],
        "age": [30],
        "RevolvingUtilizationOfUnsecuredLines": [0.5],
        "DebtRatio": [-0.1],  # Нарушение: отрицательное значение
        "MonthlyIncome": [5000.0],
        "NumberOfOpenCreditLinesAndLoans": [1],
        "NumberOfTimes90DaysLate": [0],
        "NumberRealEstateLoansOrLines": [0],
        "NumberOfTime30-59DaysPastDueNotWorse": [0],
        "NumberOfTime60-89DaysPastDueNotWorse": [0],
        "NumberOfDependents": [0]
    })
    with pytest.raises(ValueError, match="validation"):
        validate_data(df)