"""Схемы валидации входных и выходных данных."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ScoringRequest(BaseModel):
    """Входные данные для предсказания. Имена колонок должны совпадать с датасетом."""
    
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0.0)
    age: int = Field(ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: float = Field(ge=0.0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(ge=0.0)
    MonthlyIncome: Optional[float] = Field(ge=0.0, default=None)
    NumberOfOpenCreditLinesAndLoans: float = Field(ge=0.0)
    NumberOfTimes90DaysLate: float = Field(ge=0.0)
    NumberRealEstateLoansOrLines: float = Field(ge=0.0)
    NumberOfTime60_89DaysPastDueNotWorse: float = Field(ge=0.0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: Optional[float] = Field(ge=0.0, default=None)

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=()
    )


class ScoringResponse(BaseModel):
    """Структура ответа API."""
    
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_category: str
    decision: str
    model_version: str
    request_id: str
    
    model_config = ConfigDict(protected_namespaces=())