"""FastAPI entry point for the AI-dependency and burnout screening system."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from runtime import ArtifactRuntime, ArtifactUnavailable, PredictionFailed
from schemas import PredictionRequest, PredictionResponse


LOGGER = logging.getLogger("burnout_api")
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "03_数据分析" / "artifacts"
DEFAULT_FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
MAX_PREDICTION_BODY_BYTES = 16 * 1024


def _problem(
    status: int,
    title: str,
    detail: str,
    *,
    problem_type: str,
    errors: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "type": f"https://example.edu/problems/{problem_type}",
        "title": title,
        "status": status,
        "detail": detail,
    }
    if errors is not None:
        payload["errors"] = errors
    return JSONResponse(payload, status_code=status, media_type="application/problem+json")


def _validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for error in exc.errors():
        # Do not echo the request value; this avoids retaining student inputs in errors/logs.
        location = [str(item) for item in error.get("loc", ()) if item != "body"]
        safe.append(
            {
                "field": ".".join(location) or "body",
                "message": str(error.get("msg", "输入无效")),
                "code": str(error.get("type", "validation_error")),
            }
        )
    return safe


def create_app(
    *,
    runtime: ArtifactRuntime | Any | None = None,
    artifact_dir: Path | None = None,
    frontend_dir: Path | None = None,
    mount_frontend: bool = True,
) -> FastAPI:
    selected_artifact_dir = Path(
        os.environ.get("BURNOUT_ARTIFACT_DIR", artifact_dir or DEFAULT_ARTIFACT_DIR)
    )
    selected_frontend_dir = Path(
        os.environ.get("BURNOUT_FRONTEND_DIR", frontend_dir or DEFAULT_FRONTEND_DIR)
    )
    runtime_instance = runtime or ArtifactRuntime(selected_artifact_dir)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime_instance.load()
        application.state.runtime = runtime_instance
        yield

    application = FastAPI(
        title="基于生成式AI使用与感知依赖特征的学业倦怠风险预测服务",
        summary="研究性、支持性三分类早期筛查接口",
        description=(
            "使用固定12项筛查时点特征，返回Low/Medium/High模型估计概率、局部SHAP解释与分层支持建议；输出不是真实患病概率。"
            "输出不是医学或心理诊断。"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        return _problem(
            422,
            "Invalid prediction input",
            "输入未通过严格校验，请检查缺失字段、数据类型、枚举值和取值范围。",
            problem_type="validation-error",
            errors=_validation_errors(exc),
        )

    @application.middleware("http")
    async def security_and_size_headers(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/api/predict":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    too_large = int(content_length) > MAX_PREDICTION_BODY_BYTES
                except ValueError:
                    too_large = True
                if too_large:
                    return _problem(
                        413,
                        "Request body too large",
                        "预测请求体超过16 KiB限制。",
                        problem_type="request-too-large",
                    )
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.get("/api/health", tags=["service"])
    def health():
        return runtime_instance.health()

    @application.get("/api/metadata", tags=["service"])
    def metadata():
        try:
            return runtime_instance.metadata_response()
        except ArtifactUnavailable as exc:
            return _problem(
                503,
                "Metadata unavailable",
                str(exc),
                problem_type="metadata-unavailable",
            )

    @application.get("/api/dashboard", tags=["analytics"])
    def dashboard():
        try:
            return runtime_instance.dashboard_response()
        except ArtifactUnavailable as exc:
            return _problem(
                503,
                "Dashboard unavailable",
                str(exc),
                problem_type="dashboard-unavailable",
            )

    @application.get("/api/fairness", tags=["analytics"])
    def fairness():
        try:
            return runtime_instance.fairness_response()
        except ArtifactUnavailable as exc:
            return _problem(
                503,
                "Fairness audit unavailable",
                str(exc),
                problem_type="fairness-unavailable",
            )

    @application.post(
        "/api/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
    )
    def predict(payload: PredictionRequest):
        try:
            return runtime_instance.predict(payload)
        except ArtifactUnavailable as exc:
            return _problem(
                503,
                "Model unavailable",
                str(exc),
                problem_type="model-unavailable",
            )
        except PredictionFailed as exc:
            # The input is intentionally never written to logs.
            LOGGER.error("Prediction failed because loaded artifacts are incompatible")
            return _problem(
                503,
                "Prediction unavailable",
                str(exc),
                problem_type="prediction-unavailable",
            )

    # API routes must be registered before the catch-all static mount.
    if mount_frontend and selected_frontend_dir.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=selected_frontend_dir, html=True),
            name="frontend",
        )

    return application


app = create_app()
