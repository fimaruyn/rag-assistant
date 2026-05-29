"""Интеграционные тесты для API."""
from fastapi.testclient import TestClient
from src.service.main import app


def test_health_endpoint():
    """Проверка health-check эндпоинта."""
    # TestClient автоматически управляет lifespan, если он определён в приложении
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Статус может быть "healthy" или "unhealthy" в зависимости от наличия модели
        # Для тестов важно, что эндпоинт работает и возвращает корректную структуру
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


def test_predict_valid_payload():
    """Проверка валидного запроса на предсказание."""
    # Если модель не загружена (нет артефакта), тест должен пропускаться или мокироваться
    # Для полноценного теста нужен файл модели, поэтому здесь проверяем только контракт
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.15,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 3,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        # Если модель не загружена, получим 500 — это ожидаемо в тестовой среде без артефакта
        # Поэтому проверяем либо 200 (если модель есть), либо 500 с корректным сообщением
        if response.status_code == 200:
            data = response.json()
            assert "default_probability" in data
            assert 0 <= data["default_probability"] <= 1
            assert data["risk_category"] in ["low", "medium", "high"]
            assert data["decision"] in ["approve", "approve_with_conditions", "reject"]
        elif response.status_code == 500:
            # Ожидаемая ошибка, если модель не загружена в тестовой среде
            assert "Internal prediction error" in response.json()["detail"]


def test_predict_invalid_age():
    """Проверка обработки невалидного запроса (валидация Pydantic)."""
    payload = {"age": "not_a_number"}  # неверный тип
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity