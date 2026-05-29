"""Загрузка и валидация тренировочных данных."""
import logging
import pandas as pd
from pathlib import Path

from src.data.validator import validate_data

logger = logging.getLogger(__name__)
DEFAULT_TRAIN_FILE = "cs-training.csv"
TARGET_COLUMN = "SeriousDlqin2yrs"


def load_data(data_dir: Path, filename: str = DEFAULT_TRAIN_FILE) -> pd.DataFrame:
    """
    Загружает, нормализует, очищает и валидирует тренировочный датасет.
    
    Args:
        data_dir: Путь к директории с сырыми данными.
        filename: Имя CSV-файла.
        
    Returns:
        Валидированный DataFrame с нормализованными именами колонок.
        
    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если целевая колонка отсутствует после нормализации.
    """
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Training data file not found at: {file_path}")

    logger.info("Loading raw data from %s", file_path)
    
    df = pd.read_csv(
        file_path,
        index_col=0,
        na_values=[' NA', 'NA', 'na', ''],
        skipinitialspace=True
    )

    # Нормализация имён колонок
    df.columns = df.columns.str.strip()

    # Приведение имени целевой переменной
    if TARGET_COLUMN not in df.columns:
        similar_cols = [c for c in df.columns if TARGET_COLUMN.lower() in c.lower()]
        if similar_cols:
            df = df.rename(columns={similar_cols[0]: TARGET_COLUMN})
            logger.info("Renamed '%s' to '%s'", similar_cols[0], TARGET_COLUMN)
        else:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

    # Приведение числовых колонок к числовому типу
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        if col != TARGET_COLUMN:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Фильтрация аномального возраста
    if 'age' in df.columns:
        initial_count = len(df)
        df = df[(df['age'] >= 18) & (df['age'] <= 120)]
        removed = initial_count - len(df)
        if removed > 0:
            logger.warning(
                "Filtered out %d rows with invalid age (<18 or >120). "
                "Remaining: %d rows",
                removed, len(df)
            )

    logger.info(
        "Dataset loaded: %d rows, %d columns. Target distribution: %s",
        len(df),
        len(df.columns),
        df[TARGET_COLUMN].value_counts(normalize=True).round(3).to_dict(),
    )

    return validate_data(df)