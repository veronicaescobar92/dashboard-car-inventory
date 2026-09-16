"""API local. Iniciar: python -m uvicorn repuestos.api:app --host 127.0.0.1."""
import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import joblib
from fastapi import Depends, FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from repuestos.analytics import Analytics, DomainError
from repuestos.data import read
from repuestos.ml import predict
from repuestos.settings import DB_PATH, MODEL_PATH


class PredictionRequest(BaseModel):
    product_ids: list[str] = Field(min_length=1, max_length=100, examples=[["P001", "P005"]])
    target_month: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    @field_validator("product_ids")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Los productos no deben repetirse")
        return value


class Envelope(BaseModel):
    metadata: dict
    data: dict | list


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def filters(start_date: date | None = None, end_date: date | None = None,
            product_ids: Annotated[list[str] | None, Query()] = None,
            brand: Annotated[list[str] | None, Query()] = None,
            vehicle_model: Annotated[list[str] | None, Query()] = None):
    return dict(start_date=start_date, end_date=end_date, product_ids=product_ids, brand=brand, vehicle_model=vehicle_model)


def create_app(db_path=DB_PATH, model_path=MODEL_PATH):
    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = app.state.analytics = app.state.artifact = None
        app.state.db_signature = None
        try:
            bundle = read(db_path)
            app.state.bundle, app.state.analytics = bundle, Analytics(bundle)
            app.state.db_signature = (Path(db_path).stat().st_mtime_ns, Path(db_path).stat().st_size)
        except Exception:
            logging.exception("No se pudo preparar el conjunto local")
        try:
            app.state.artifact = joblib.load(model_path)
        except Exception:
            logging.warning("Modelo no disponible; ejecute python -m repuestos.ml")
        yield

    errors = {code: {"model": ErrorResponse, "description": description} for code, description in
              [(404, "Producto desconocido"), (422, "Solicitud inválida"), (503, "Datos o modelo no preparados")]}
    app = FastAPI(title="Repuestos | API de demostración", version="0.1.0", lifespan=lifespan,
                  description="Datos y compatibilidades exclusivamente sintéticos. Sin operaciones sobre empresas reales.",
                  responses=errors)

    @app.exception_handler(DomainError)
    async def domain_error(_, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_, exc):
        fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "Revise los campos: " + fields}})

    @app.exception_handler(Exception)
    async def internal_error(_, exc):
        logging.error("Error interno", exc_info=exc)
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "No se pudo completar la solicitud."}})

    def ready():
        try:
            signature = (Path(db_path).stat().st_mtime_ns, Path(db_path).stat().st_size)
        except OSError:
            signature = None
        if app.state.bundle is None or signature != app.state.db_signature:
            raise DomainError(503, "data_not_ready", "Prepare los datos y reinicie la API antes de consultar.")
        return app.state.bundle

    def model_ready():
        bundle = ready()
        artifact = app.state.artifact
        if artifact is None:
            raise DomainError(503, "model_not_ready", "Entrene el modelo y reinicie la API.")
        # Validate schema, data version and history before metrics or inference.
        predict(bundle, artifact, [bundle["products"][0]["product_id"]])
        return artifact

    def response(data, metadata=None):
        return dict(metadata=metadata or ready()["metadata"], data=data)

    @app.get("/health", tags=["Estado"])
    def health():
        data_ok = model_ok = False
        try:
            ready()
            data_ok = True
            model_ready()
            model_ok = True
        except DomainError:
            pass
        return dict(status="ok", data_ready=data_ok, model_ready=model_ok, synthetic=True)

    @app.get("/catalog", response_model=Envelope, tags=["Catálogo"])
    def catalog():
        bundle = ready()
        return response({k: bundle[k] for k in ["products", "vehicles", "compatibility"]})

    @app.get("/analytics/summary", response_model=Envelope, tags=["Análisis"])
    def summary(f: dict = Depends(filters)):
        ready()
        return response(app.state.analytics.summary(**f))

    @app.get("/analytics/ranking", response_model=Envelope, tags=["Análisis"])
    def ranking(metric: Literal["units", "revenue"] = "units", order: Literal["asc", "desc"] = "desc",
                f: dict = Depends(filters)):
        ready()
        return response(dict(rows=app.state.analytics.ranking(metric=metric, order=order, **f),
                             signals=app.state.analytics.signals(**f)))

    @app.get("/analytics/monthly", response_model=Envelope, tags=["Análisis"])
    def monthly(f: dict = Depends(filters)):
        ready()
        return response(dict(rows=app.state.analytics.monthly(**f), signals=app.state.analytics.signals(**f)))

    @app.get("/inventory/overview", response_model=Envelope, tags=["Inventario"])
    def inventory_overview(as_of: date | None = None,
                           window_days: Annotated[int, Query(ge=30, le=730)] = 90,
                           slow_days: Annotated[int, Query(ge=1, le=730)] = 90,
                           holding_rate: Annotated[float | None, Query(ge=0, le=5)] = None,
                           expiry_days: Annotated[int, Query(ge=0, le=730)] = 90,
                           f: dict = Depends(filters)):
        ready()
        return response(app.state.analytics.inventory_overview(
            as_of=as_of, window_days=window_days, slow_days=slow_days,
            holding_rate=holding_rate, expiry_days=expiry_days, **f))

    @app.get("/model/metrics", response_model=Envelope, tags=["Modelo"])
    def metrics():
        artifact = model_ready()
        return response(artifact["report"], artifact["metadata"])

    @app.post("/predictions", response_model=Envelope, tags=["Modelo"])
    def predictions(request: PredictionRequest):
        artifact = model_ready()
        return response(predict(ready(), artifact, request.product_ids, request.target_month), artifact["metadata"])

    return app


app = create_app()
