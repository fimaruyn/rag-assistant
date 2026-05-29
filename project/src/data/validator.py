"""Валидация сырых данных из папки data/raw"""
import logging
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

CREDIT_SCHEMA = DataFrameSchema(
    columns={
        "SeriousDlqin2yrs": Column(
            pd.Int64Dtype(),
            Check.isin([0, 1]),
            description="Целевая переменная: 1 = дефолт 90+ дней",
        ),
        "RevolvingUtilizationOfUnsecuredLines": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            description="Отношение баланса к кредитному лимиту",
        ),
        "age": Column(
            pd.Int64Dtype(),  
            Check.in_range(min_value=18, max_value=120),  
            description="Возраст заёмщика",
        ),
        "NumberOfTime30-59DaysPastDueNotWorse": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Просрочки 30-59 дней",
        ),
        "DebtRatio": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            description="Отношение долгов к доходу",
        ),
        "MonthlyIncome": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Ежемесячный доход",
        ),
        "NumberOfOpenCreditLinesAndLoans": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Открытые кредитные линии",
        ),
        "NumberOfTimes90DaysLate": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Просрочки 90+ дней",
        ),
        "NumberRealEstateLoansOrLines": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Ипотечные кредиты",
        ),
        "NumberOfTime60-89DaysPastDueNotWorse": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Просрочки 60-89 дней",
        ),
        "NumberOfDependents": Column(
            pd.Float64Dtype(),
            Check.ge(0),
            nullable=True,
            description="Количество иждивенцев",
        ),
    },
    strict=False,  # Разрешает дополнительные колонки на промежуточных этапах пайплайна
    coerce=True,   # Безопасно приводит типы (int -> float при наличии NaN)
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Валидирует DataFrame по схеме CREDIT_SCHEMA.
    
    Args:
        df: Сырой DataFrame.
        
    Returns:
        Валидированный DataFrame.
        
    Raises:
        ValueError: При нарушении схемы данных.
    """
    try:
        validated_df = CREDIT_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed successfully.")
        return validated_df
    except pa.errors.SchemaErrors as e:
        failure_cases = e.failure_cases
        logger.error("Data validation failed. Aggregated issues:")
        for col, cases in failure_cases.groupby("column"):
            unique_failures = cases["failure_case"].unique()[:3].tolist()
            logger.error(
                "  Column '%s': %d violations. Examples: %s",
                col,
                len(cases),
                unique_failures,
            )
        raise ValueError(
            "Raw data failed validation checks. See logs for detailed failure cases."
        ) from e