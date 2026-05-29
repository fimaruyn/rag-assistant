"""
Модуль предобработки данных для кредитного скоринга.
Реализует гипотезы из EDA: импутация, трансформация распределений, масштабирование.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

logger = logging.getLogger(__name__)

# Названия колонок, определенные на этапе EDA
TARGET_COL = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0" # Индекс, который будет отброшен

# Группы признаков для трансформаций
NUMERIC_FEATURES = [
    "age",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Признаки с экстремальным перекосом (требуют log1p)
SKEWED_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
]

# Признаки, требующие медианной импутации
MEDIAN_IMPUTE_FEATURES = ["MonthlyIncome"]

# Признаки, требующие константной импутации (0)
CONSTANT_IMPUTE_FEATURES = ["NumberOfDependents"]

# Признаки для агрегации (высокая корреляция)
PAST_DUE_FEATURES = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


class PastDueAggregator(BaseEstimator, TransformerMixin):
    """
    Трансформер для агрегации признаков просрочек.
    Создает новый признак 'TotalPastDue' и удаляет исходные.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        # Суммирование просрочек
        X_transformed["TotalPastDue"] = X_transformed[PAST_DUE_FEATURES].sum(axis=1)
        # Удаление исходных колонок
        X_transformed = X_transformed.drop(columns=PAST_DUE_FEATURES, errors="ignore")
        logger.info("PastDueAggregator: Created 'TotalPastDue' and dropped original columns.")
        return X_transformed


def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Сборка полного пайплайна предобработки.
    
    Returns:
        ColumnTransformer: Готовый к fit/transform объект sklearn.
    """
    logger.info("Building preprocessing pipeline...")

    # 1. Обработка пропусков
    median_imputer = SimpleImputer(strategy="median")
    const_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # 2. Лог-трансформация (применяется после импутации для избежания log(0))
    # Imputation -> Log1p
    log_transform_steps = [
        ("impute_median", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p))
    ]
    log_pipeline = Pipeline(log_transform_steps)

    # 3. Агрегация признаков просрочек
    past_due_aggregator = PastDueAggregator()

    # Сборка ColumnTransformer
  
    preprocessor = ColumnTransformer(
        transformers=[
            # Агрегированные признаки (лог-трансформация после агрегации)
            ("skewed_transformed", log_pipeline, SKEWED_FEATURES),
            
            # Остальные числовые признаки (простая импутация)
            ("numeric_basic", median_imputer, 
             [f for f in NUMERIC_FEATURES if f not in SKEWED_FEATURES and f not in PAST_DUE_FEATURES]),
             
            # Иждивенцы (заполняем 0)
            ("dependents_impute", const_imputer, CONSTANT_IMPUTE_FEATURES),
            
        ],
        remainder="drop", # Отбрасываем ID и Target
    )

    return preprocessor


def build_full_pipeline() -> Pipeline:
    """
    Создает полный пайплайн: Агрегация -> Предобработка колонок -> Масштабирование.
    """
    logger.info("Building full feature engineering pipeline...")
    
    preprocessor = get_preprocessing_pipeline()
    
    full_pipeline = Pipeline(
        steps=[
            ("aggregator", PastDueAggregator()),
            ("preprocessor", preprocessor),
        ]
    )
    
    return full_pipeline


def create_transformer() -> ColumnTransformer:
    """
    Возвращает готовый ColumnTransformer с логикой импутации и масштабирования.
    Агрегация должна выполняться отдельным шагом перед этим трансформером.
    """
    transformers = [
        # 1. Skewed features: Impute -> Log1p -> RobustScale
        (
            "skewed_num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p)),
                ("scaler", RobustScaler())
            ]),
            SKEWED_FEATURES
        ),
        # 2. Other numeric: Impute -> RobustScale
        (
            "other_num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ]),
            [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES and c not in PAST_DUE_FEATURES and c not in CONSTANT_IMPUTE_FEATURES]
        ),
        # 3. Dependents: Impute 0
        (
            "dependents",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0))
            ]),
            CONSTANT_IMPUTE_FEATURES
        )
    ]
    
    return ColumnTransformer(transformers=transformers, remainder="drop")

def get_transformed_feature_names(preprocessor, original_columns):
    """
    Получает имена признаков после трансформации.
    """
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers:
        if columns == 'passthrough':
            feature_names.extend(original_columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            try:
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            except:
                feature_names.extend(columns if isinstance(columns, list) else [columns])
        else:
            feature_names.extend(columns if isinstance(columns, list) else [columns])
    
    return feature_names