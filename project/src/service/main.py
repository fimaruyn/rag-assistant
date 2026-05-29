"""Точка входа FastAPI-сервиса кредитного скоринга."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.service.config import settings
from src.service.schemas import ScoringRequest, ScoringResponse

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Глобальное состояние: загруженная модель
_model = None
_model_loaded = False


def _load_model() -> None:
    """Загрузка сериализованного пайплайна при старте."""
    global _model, _model_loaded
    if not settings.model_artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {settings.model_artifact_path}. "
            "Check MODEL_ARTIFACT_PATH in .env or configs/.env.example."
        )
    
    logger.info("Loading model from %s", settings.model_artifact_path)
    _model = joblib.load(settings.model_artifact_path)
    _model_loaded = True
    logger.info("Model loaded successfully. Version: %s", settings.app_version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения: загрузка/выгрузка модели."""
    # Startup
    try:
        _load_model()
    except Exception as exc:
        logger.critical("Failed to load model: %s", str(exc))
        raise RuntimeError("Service cannot start without a model.") from exc
    
    yield
    
    # Shutdown (опционально: освободить ресурсы)
    global _model, _model_loaded
    _model = None
    _model_loaded = False
    logger.info("Model unloaded, service shutting down.")


# Инициализация приложения с lifespan
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Monitoring"])
async def health_check() -> dict:
    """Endpoint для проверки работоспособности сервиса."""
    return {
        "status": "healthy" if _model_loaded else "unhealthy",
        "model_loaded": _model_loaded,
        "version": settings.app_version
    }


@app.post("/predict", response_model=ScoringResponse, tags=["Scoring"])
async def predict(request: ScoringRequest, req: Request) -> ScoringResponse:
    """Endpoint для получения кредитного скоринга."""
    start_time = time.perf_counter()
    request_id = req.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    try:
        # Проверка, что модель загружена
        if not _model_loaded or _model is None:
            raise RuntimeError("Model not loaded. Check service startup logs.")
        
        # Преобразование в DataFrame для совместимости с sklearn-пайплайном
        features_df = pd.DataFrame([request.model_dump(by_alias=True)])
        
        # Предсказание
        proba = _model.predict_proba(features_df)[0, 1]
        
        # Бизнес-логика категоризации риска
        if proba < 0.10:
            risk_category = "low"
            decision = "approve"
        elif proba < 0.35:
            risk_category = "medium"
            decision = "approve_with_conditions"
        else:
            risk_category = "high"
            decision = "reject"
        
        latency = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "Prediction completed | request_id=%s | probability=%.4f | decision=%s | latency_ms=%.2f",
            request_id, proba, decision, latency
        )
        
        return ScoringResponse(
            default_probability=round(float(proba), 4),
            risk_category=risk_category,
            decision=decision,
            model_version=settings.app_version,
            request_id=request_id
        )
        
    except Exception as exc:
        logger.error("Prediction failed | request_id=%s | error=%s", request_id, str(exc))
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """Middleware для добавления заголовка времени обработки и логирования."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response