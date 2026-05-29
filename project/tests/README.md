# Тесты проекта

В этой директории находятся автоматизированные тесты для проверки корректности работы модулей проекта «Кредитный скоринг».

## Структура тестов

```
tests/
├── __init__.py                    # Инициализация пакета тестов
├── README.md                      # Этот файл
├── test_data_loading.py          # Тесты загрузки и валидации данных
├── test_preprocessing.py         # Тесты пайплайна предобработки
└── test_api.py                   # Интеграционные тесты FastAPI-сервиса
```

## Описание тестовых модулей

### `test_data_loading.py`
Проверяет модуль `src/data/loader.py` и `src/data/validator.py`:

| Тест | Описание |
|------|----------|
| `test_load_data_success` | Успешная загрузка валидного CSV с корректной схемой |
| `test_load_data_file_not_found` | Обработка отсутствия файла данных |
| `test_validate_data_missing_target` | Отказ при отсутствии целевой переменной |
| `test_validate_data_invalid_age` | Отказ при недопустимом возрасте (<18 или >120) |
| `test_validate_data_negative_debt_ratio` | Отказ при отрицательном значении `DebtRatio` |

### `test_preprocessing.py`
Проверяет модуль `src/features/preprocessing.py`:

| Тест | Описание |
|------|----------|
| `test_past_due_aggregator` | Корректность агрегации признаков просрочек в `TotalPastDue` |
| `test_full_pipeline_shape` | Проверка формы выходных данных после `ColumnTransformer` |

### `test_api.py`
Интеграционные тесты для `src/service/main.py` (FastAPI):

| Тест | Описание |
|------|----------|
| `test_health_endpoint` | Проверка endpoint `/health`: статус, версия, загрузка модели |
| `test_predict_valid_payload` | Успешный запрос с валидными данными, проверка структуры ответа |
| `test_predict_invalid_age` | Проверка валидации входных данных через Pydantic (ошибка 422) |

## Запуск тестов

### Базовый запуск
```bash
cd project
uv run pytest tests/ -v
```

### Запуск с отчётом о покрытии
```bash
uv run pytest tests/ -v --cov=src --cov-report=html
# Отчёт откроется в браузере: open htmlcov/index.html (macOS/Linux)
# или: start htmlcov/index.html (Windows)
```

### Запуск конкретного теста
```bash
# По файлу
uv run pytest tests/test_api.py -v

# По функции
uv run pytest tests/test_api.py::test_health_endpoint -v

# По ключевому слову
uv run pytest -k "predict" -v
```

### Запуск в режиме fast (без кэша, с остановкой на первой ошибке)
```bash
uv run pytest tests/ -x --cache-clear
```

## Примечания

- Тесты не требуют доступа к интернету или внешним сервисам.
- Для тестов API используется `TestClient` из `fastapi.testclient`, который автоматически управляет жизненным циклом приложения (`lifespan`).
- Артефакты модели (`*.joblib`) не требуются для большинства юнит-тестов; интеграционные тесты могут пропускаться, если модель отсутствует (проверка через `if response.status_code == 500`).
- Предупреждения от сторонних библиотек (`matplotlib`, `pyparsing`) можно игнорировать — они не влияют на корректность тестов.

---

> **Совет**: Запускайте тесты перед каждым коммитом (`git commit`) — это поможет избежать регрессий и сэкономить время на отладке перед защитой.