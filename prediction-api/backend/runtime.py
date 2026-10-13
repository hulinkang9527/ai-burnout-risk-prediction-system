"""Artifact loading, API adaptation, inference, and local SHAP explanations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from schemas import (
    API_TO_MODEL_FIELD,
    FEATURE_LABELS,
    MODEL_FEATURE_ORDER,
    MODEL_TO_API_FIELD,
    PredictionRequest,
    PredictionResponse,
    RiskProbabilities,
    ShapExplanation,
    ShapFeature,
    VALUE_LABELS,
    public_feature_schema,
)

try:  # Keep /api/health available when SHAP is not installed correctly.
    import shap
except Exception as exc:  # pragma: no cover - depends on the host environment
    shap = None
    SHAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    SHAP_IMPORT_ERROR = None


DISPLAY_CLASS_ORDER = ("Low", "Medium", "High")
CHINA_TZ = timezone(timedelta(hours=8))

ARTIFACT_FILES = {
    "model": "model.joblib",
    "metrics": "metrics.json",
    "metadata": "metadata.json",
    "dashboard": "dashboard.json",
    "fairness": "fairness.json",
    "shap_background": "shap_background.npy",
    "transformed_feature_names": "transformed_feature_names.json",
}
PREDICTION_ARTIFACTS = {
    "model",
    "metadata",
    "shap_background",
    "transformed_feature_names",
}

MAJOR_LABELS = VALUE_LABELS["major_category"]
YEAR_LABELS = VALUE_LABELS["year_of_study"]


class ArtifactUnavailable(RuntimeError):
    """The requested result cannot be produced from the real artifacts."""


class PredictionFailed(RuntimeError):
    """Inference or explanation failed after artifacts were loaded."""


def _now_iso() -> str:
    return datetime.now(CHINA_TZ).isoformat()


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _first(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _safe_ratio(value: Any, *, percentage: bool = False) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    if percentage:
        number /= 100.0
    if -1e-12 <= number <= 1.0 + 1e-12:
        return min(max(number, 0.0), 1.0)
    return None


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _original_model_feature(transformed_name: str) -> str | None:
    suffix = transformed_name.split("__", 1)[-1]
    for feature in sorted(MODEL_FEATURE_ORDER, key=len, reverse=True):
        if suffix == feature or suffix.startswith(feature + "_"):
            return feature
    return None


class ArtifactRuntime:
    """Owns immutable trained artifacts and exposes contract-shaped results."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir).resolve()
        self.model: Any | None = None
        self.metrics: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None
        self.dashboard: dict[str, Any] | None = None
        self.fairness: dict[str, Any] | None = None
        self.shap_background: np.ndarray | None = None
        self.transformed_feature_names: list[str] | None = None
        self.model_sha256: str | None = None
        self.artifact_status: dict[str, dict[str, Any]] = {}
        self.load_warnings: list[str] = []
        self.runtime_error: str | None = None
        self._explainer: Any | None = None
        self._shap_method = ""
        self._lock = threading.RLock()

    def load(self) -> None:
        """Load every declared artifact independently so health remains observable."""

        with self._lock:
            self.model = None
            self.metrics = None
            self.metadata = None
            self.dashboard = None
            self.fairness = None
            self.shap_background = None
            self.transformed_feature_names = None
            self.model_sha256 = None
            self.artifact_status = {}
            self.load_warnings = []
            self.runtime_error = None
            self._explainer = None
            self._shap_method = ""

            for key, filename in ARTIFACT_FILES.items():
                path = self.artifact_dir / filename
                state: dict[str, Any] = {
                    "file": filename,
                    "required": True,
                    "loaded": False,
                }
                try:
                    if not path.is_file():
                        raise FileNotFoundError(f"未找到产物文件：{filename}")
                    if key == "model":
                        value = joblib.load(path)
                        if not callable(getattr(value, "predict_proba", None)):
                            raise TypeError("model.joblib 不含可调用的 predict_proba")
                        self.model = value
                        self.model_sha256 = _sha256(path)
                    elif key == "shap_background":
                        value = np.load(path, allow_pickle=False)
                        value = np.asarray(value, dtype=np.float64)
                        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
                            raise ValueError("SHAP 背景必须是非空二维数值数组")
                        if not np.isfinite(value).all():
                            raise ValueError("SHAP 背景含 NaN 或无穷值")
                        self.shap_background = value
                        state["shape"] = list(value.shape)
                    else:
                        value = _json_file(path)
                        if key == "metrics":
                            if not isinstance(value, dict):
                                raise TypeError("metrics.json 顶层必须是对象")
                            self.metrics = value
                        elif key == "metadata":
                            if not isinstance(value, dict):
                                raise TypeError("metadata.json 顶层必须是对象")
                            self.metadata = value
                        elif key == "dashboard":
                            if not isinstance(value, dict):
                                raise TypeError("dashboard.json 顶层必须是对象")
                            self.dashboard = value
                        elif key == "fairness":
                            if not isinstance(value, dict):
                                raise TypeError("fairness.json 顶层必须是对象")
                            self.fairness = value
                        elif key == "transformed_feature_names":
                            if not isinstance(value, list) or not value or not all(
                                isinstance(item, str) and item for item in value
                            ):
                                raise TypeError("转换后特征名必须是非空字符串数组")
                            self.transformed_feature_names = list(value)
                            state["count"] = len(value)
                    state["loaded"] = True
                    state["bytes"] = path.stat().st_size
                except Exception as exc:
                    state["error"] = f"{type(exc).__name__}: {exc}"
                self.artifact_status[key] = state

            self._validate_compatibility()

    def _invalidate(self, key: str, message: str) -> None:
        state = self.artifact_status.setdefault(
            key,
            {"file": ARTIFACT_FILES[key], "required": True, "loaded": False},
        )
        state["loaded"] = False
        state["error"] = message

    def _validate_compatibility(self) -> None:
        if self.shap_background is not None and self.transformed_feature_names is not None:
            if self.shap_background.shape[1] != len(self.transformed_feature_names):
                message = (
                    "SHAP背景列数与转换后特征名数量不一致："
                    f"{self.shap_background.shape[1]} != {len(self.transformed_feature_names)}"
                )
                self._invalidate("shap_background", message)
                self._invalidate("transformed_feature_names", message)

        if self.transformed_feature_names is not None:
            unknown = [
                name
                for name in self.transformed_feature_names
                if _original_model_feature(name) is None
            ]
            if unknown:
                self._invalidate(
                    "transformed_feature_names",
                    "存在无法聚合回12个原字段的转换特征：" + ", ".join(unknown[:5]),
                )

        if self.metadata is not None:
            declared = self.metadata.get("input_feature_order")
            if isinstance(declared, list) and declared != MODEL_FEATURE_ORDER:
                if set(declared) == set(MODEL_FEATURE_ORDER):
                    self.load_warnings.append("元数据输入顺序不同；在线推理已按训练字段显式重排。")
                else:
                    self._invalidate("metadata", "元数据声明的输入字段与在线12字段集合不一致")

        if self.model is not None:
            classes = [str(item) for item in getattr(self.model, "classes_", [])]
            if set(classes) != set(DISPLAY_CLASS_ORDER):
                self._invalidate(
                    "model",
                    "模型类别必须且只能包含 Low、Medium、High；实际为 " + repr(classes),
                )
            named_steps = getattr(self.model, "named_steps", None)
            if not isinstance(named_steps, dict) and not hasattr(named_steps, "get"):
                self._invalidate("model", "模型不是包含预处理器和分类器的 sklearn Pipeline")
            elif named_steps.get("preprocessor") is None or named_steps.get("model") is None:
                self._invalidate("model", "模型 Pipeline 缺少 preprocessor 或 model 步骤")

        if SHAP_IMPORT_ERROR:
            self.load_warnings.append("SHAP 运行库不可用：" + SHAP_IMPORT_ERROR)

    @property
    def prediction_ready(self) -> bool:
        return (
            all(self.artifact_status.get(key, {}).get("loaded") for key in PREDICTION_ARTIFACTS)
            and shap is not None
            and self.runtime_error is None
        )

    @property
    def model_version(self) -> str:
        metadata = self.metadata or {}
        candidates = [
            metadata.get("model_version"),
            metadata.get("version"),
        ]
        model_block = metadata.get("model")
        if isinstance(model_block, dict):
            candidates.insert(0, model_block.get("version"))
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()
        family = str(metadata.get("model_family") or "burnout-model")
        if self.model_sha256:
            return f"{_slug(family)}-{self.model_sha256[:12]}"
        return "unavailable"

    def health(self) -> dict[str, Any]:
        missing = [
            state["file"]
            for state in self.artifact_status.values()
            if not state.get("loaded")
        ]
        status = "ok" if not missing and self.prediction_ready and not self.load_warnings else "degraded"
        return {
            "status": status,
            "service_name": "AI依赖与学业倦怠预测服务",
            "api_version": "1.0",
            "checked_at": _now_iso(),
            "artifact_directory": str(self.artifact_dir),
            "model_ready": bool(self.artifact_status.get("model", {}).get("loaded")),
            "prediction_ready": self.prediction_ready,
            "model_version": self.model_version if self.model is not None else None,
            "artifacts": self.artifact_status,
            "missing_or_invalid": missing,
            "warnings": self.load_warnings,
            "runtime_error": self.runtime_error,
        }

    def metadata_response(self) -> dict[str, Any]:
        if self.metadata is None or not self.artifact_status.get("metadata", {}).get("loaded"):
            raise ArtifactUnavailable("模型元数据不可用，无法提供可核验的模型信息。")
        classes = self.metadata.get("target_classes_display_order")
        if not isinstance(classes, list) or set(map(str, classes)) != set(DISPLAY_CLASS_ORDER):
            classes = list(DISPLAY_CLASS_ORDER)
        schema = self.metadata.get("input_schema")
        return {
            "api_version": "1.0",
            "service_name": "AI依赖与学业倦怠预测服务",
            "status": "ready" if self.prediction_ready else "degraded",
            "model": {
                "version": self.model_version,
                "available": bool(self.artifact_status.get("model", {}).get("loaded")),
                "family": self.metadata.get("model_family"),
                "target": self.metadata.get("target") or "Burnout_Risk_Level",
                "classes": [str(item) for item in classes],
                "trained_at": self.metadata.get("created_at_utc"),
            },
            "feature_schema": public_feature_schema(schema if isinstance(schema, dict) else None),
            "data": self.metadata.get("data"),
            "intended_use": self.metadata.get("intended_use"),
            "leakage_controls": self.metadata.get("leakage_controls"),
        }

    def dashboard_response(self) -> dict[str, Any]:
        source = self.dashboard
        if source is None or not self.artifact_status.get("dashboard", {}).get("loaded"):
            raise ArtifactUnavailable("仪表盘统计产物不可用，系统不会返回演示或估算数据。")

        dataset = source.get("dataset") if isinstance(source.get("dataset"), dict) else {}
        means = (
            source.get("key_numeric_means")
            if isinstance(source.get("key_numeric_means"), dict)
            else {}
        )
        total = _integer(_first(dataset, ("rows", "total_students", "count")))
        high_rate: float | None = None
        distribution = source.get("burnout_distribution")
        if isinstance(distribution, list):
            for row in distribution:
                if not isinstance(row, dict):
                    continue
                level = str(_first(row, ("level", "class", "risk_level", "group")) or "")
                if level.lower() != "high":
                    continue
                count = _integer(row.get("count"))
                if count is not None and total:
                    high_rate = count / total
                else:
                    high_rate = _safe_ratio(row.get("percentage"), percentage=True)
                    if high_rate is None:
                        high_rate = _safe_ratio(_first(row, ("rate", "proportion")))
                break

        selected_holdout = (
            source.get("selected_model_holdout")
            if isinstance(source.get("selected_model_holdout"), dict)
            else {}
        )
        metrics_best = {}
        if isinstance(self.metrics, dict) and isinstance(
            self.metrics.get("best_model_holdout"), dict
        ):
            metrics_best = self.metrics["best_model_holdout"]
        macro_f1 = _finite_number(
            _first(selected_holdout, ("f1_macro", "macro_f1"))
            if selected_holdout
            else _first(metrics_best, ("f1_macro", "macro_f1"))
        )

        return {
            "updated_at": (self.metadata or {}).get("created_at_utc"),
            "summary": {
                "total_students": total,
                "high_risk_rate": high_rate,
                "avg_ai_dependency": _finite_number(
                    _first(means, ("perceived_ai_dependency", "avg_ai_dependency"))
                ),
            },
            "model_metrics": {
                "macro_f1": macro_f1,
                "selected_model": source.get("selected_model")
                or (self.metrics or {}).get("selected_model"),
            },
            "major_comparison": self._comparison_rows(
                source.get("major_comparison"), "Major_Category", MAJOR_LABELS
            ),
            "year_comparison": self._comparison_rows(
                source.get("year_comparison"), "Year_of_Study", YEAR_LABELS
            ),
            "global_shap": {
                "method": "mean absolute TreeSHAP",
                "explained_output": "Low/Medium/High 三类概率输出的跨样本、跨类别平均绝对归因强度",
                "features": self._global_shap_features(source.get("top_shap_features")),
                "caveat": "全局 SHAP 对三类概率的绝对归因在解释样本与类别维度求均值；它不保留正负方向，不表示因果关系，也不直接给出单个学生的风险原因。",
            },
            "provenance": {
                "dataset": dataset,
                "artifact": ARTIFACT_FILES["dashboard"],
            },
        }

    @staticmethod
    def _global_shap_features(rows: Any) -> list[dict[str, Any]]:
        """Adapt the offline SHAP artifact to a small, typed dashboard payload."""

        if not isinstance(rows, list):
            return []
        adapted: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            source_field = _first(row, ("original_feature", "source_field", "feature"))
            mean_abs_shap = _finite_number(
                _first(row, ("mean_abs_shap", "mean_absolute_shap", "importance"))
            )
            normalized = _finite_number(
                _first(row, ("normalized_importance", "normalized", "share"))
            )
            if source_field is None or mean_abs_shap is None or normalized is None:
                continue
            if mean_abs_shap < 0 or normalized < 0 or normalized > 1:
                continue
            source_field_text = str(source_field)
            api_name = MODEL_TO_API_FIELD.get(source_field_text, source_field_text)
            adapted.append(
                {
                    "rank": len(adapted) + 1,
                    "feature": api_name,
                    "source_field": source_field_text,
                    "label": FEATURE_LABELS.get(api_name, source_field_text),
                    "mean_abs_shap": mean_abs_shap,
                    "normalized_importance": normalized,
                }
            )
            if len(adapted) == 10:
                break
        return adapted

    @staticmethod
    def _comparison_rows(
        rows: Any,
        source_group_key: str,
        labels: dict[Any, str],
    ) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        adapted: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            group = _first(row, (source_group_key, "group", "category", "name"))
            if group is None:
                continue
            item = {
                "group": str(group),
                "label": labels.get(group, str(group)),
                "count": _integer(_first(row, ("n", "count", "sample_size"))),
                "high_risk_rate": _safe_ratio(
                    _first(row, ("high_burnout_rate", "high_risk_rate"))
                ),
                "avg_ai_dependency": _finite_number(
                    _first(row, ("mean_dependency", "avg_ai_dependency"))
                ),
                "high_dependency_rate": _safe_ratio(row.get("high_dependency_rate")),
                "joint_high_rate": _safe_ratio(row.get("joint_high_rate")),
            }
            adapted.append(item)
        return adapted

    def fairness_response(self) -> dict[str, Any]:
        source = self.fairness
        if source is None or not self.artifact_status.get("fairness", {}).get("loaded"):
            raise ArtifactUnavailable("公平性审计产物不可用，系统不会虚构群体指标。")

        thresholds = {"max_tpr_gap": 0.10, "max_fpr_gap": 0.10, "min_group_size": 50}
        dimensions = (
            ("major_category", "专业类别", source.get("by_major"), MAJOR_LABELS),
            ("year_of_study", "学习年级", source.get("by_year"), YEAR_LABELS),
        )
        groups: list[dict[str, Any]] = []
        dimension_audits: list[dict[str, Any]] = []
        for dimension, dimension_label, rows, labels in dimensions:
            valid_rows: list[dict[str, Any]] = []
            if isinstance(rows, list):
                for row in rows:
                    adapted = self._fairness_group(
                        row,
                        dimension,
                        dimension_label,
                        labels,
                        thresholds["min_group_size"],
                    )
                    if adapted is not None:
                        valid_rows.append(adapted)
            tpr_gap = self._metric_gap(valid_rows, "true_positive_rate")
            fpr_gap = self._metric_gap(valid_rows, "false_positive_rate")
            dimension_status = (
                "warning"
                if (tpr_gap is not None and tpr_gap > thresholds["max_tpr_gap"])
                or (fpr_gap is not None and fpr_gap > thresholds["max_fpr_gap"])
                else "pass"
            )
            for row in valid_rows:
                row["status"] = dimension_status
            groups.extend(valid_rows)
            dimension_audits.append(
                {
                    "dimension": dimension,
                    "dimension_label": dimension_label,
                    "group_count": len(valid_rows),
                    "tpr_gap": tpr_gap,
                    "fpr_gap": fpr_gap,
                    "status": dimension_status,
                }
            )

        gap_candidates: list[tuple[str, float]] = []
        for audit in dimension_audits:
            if audit["tpr_gap"] is not None:
                gap_candidates.append(
                    (f"{audit['dimension']}.true_positive_rate", audit["tpr_gap"])
                )
            if audit["fpr_gap"] is not None:
                gap_candidates.append(
                    (f"{audit['dimension']}.false_positive_rate", audit["fpr_gap"])
                )
        if gap_candidates:
            max_metric, max_gap = max(gap_candidates, key=lambda item: item[1])
        else:
            max_metric, max_gap = "unavailable", None
        overall_status = (
            "warning"
            if any(item["status"] == "warning" for item in dimension_audits)
            else "pass"
        )
        if not groups:
            overall_status = "fail"

        summary = (
            "至少一个专业或年级维度的 TPR/FPR 最大组间差异超过预警阈值，"
            "需人工复核样本代表性、阈值和支持资源分配。"
            if overall_status == "warning"
            else "已报告的专业与年级分组差异未超过当前预警阈值；这不等于证明模型绝对公平。"
        )
        if overall_status == "fail":
            summary = "没有达到最小样本量且指标完整的群体，当前无法作出公平性判断。"

        return {
            "evaluated_at": (self.metadata or {}).get("created_at_utc"),
            "status": overall_status,
            "summary": summary,
            "overall": {
                "status": overall_status,
                "max_gap": max_gap,
                "metric": max_metric,
            },
            "thresholds": thresholds,
            "groups": groups,
            "dimensions": dimension_audits,
            "model": source.get("model"),
            "evaluation_set": source.get("evaluation_set"),
            "unavailable_attributes": source.get("unavailable_attributes", {}),
            "limitations": source.get("limitations", []),
        }

    @staticmethod
    def _fairness_group(
        row: Any,
        dimension: str,
        dimension_label: str,
        labels: dict[Any, str],
        min_group_size: int,
    ) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        group = _first(row, ("group", "category", "name"))
        n = _integer(_first(row, ("n", "sample_size", "count")))
        if group is None or n is None or n < min_group_size:
            return None

        tp = _finite_number(_first(row, ("tp_high", "tp")))
        fn = _finite_number(_first(row, ("fn_high", "fn")))
        fp = _finite_number(_first(row, ("fp_high", "fp")))
        tn = _finite_number(_first(row, ("tn_high", "tn")))
        tpr = _safe_ratio(
            _first(row, ("high_risk_recall_tpr", "true_positive_rate", "tpr"))
        )
        fpr = _safe_ratio(
            _first(row, ("high_risk_false_positive_rate", "false_positive_rate", "fpr"))
        )
        selection = _safe_ratio(
            _first(row, ("predicted_high_rate", "selection_rate", "positive_prediction_rate"))
        )
        if tpr is None and tp is not None and fn is not None and tp + fn:
            tpr = tp / (tp + fn)
        if fpr is None and fp is not None and tn is not None and fp + tn:
            fpr = fp / (fp + tn)
        if selection is None and tp is not None and fp is not None and n:
            selection = (tp + fp) / n
        if tpr is None or fpr is None or selection is None:
            return None

        return {
            "dimension": dimension,
            "dimension_label": dimension_label,
            "group": str(group),
            "group_label": labels.get(group, str(group)),
            "sample_size": n,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "selection_rate": selection,
            "status": "pass",
        }

    @staticmethod
    def _metric_gap(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if _finite_number(row.get(key)) is not None]
        return max(values) - min(values) if len(values) >= 2 else None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self.prediction_ready:
            missing = [
                ARTIFACT_FILES[key]
                for key in PREDICTION_ARTIFACTS
                if not self.artifact_status.get(key, {}).get("loaded")
            ]
            reason = "、".join(missing) if missing else self.runtime_error or "SHAP运行库"
            raise ArtifactUnavailable(f"预测服务尚未就绪：{reason}。")

        assert self.model is not None
        frame = pd.DataFrame([request.to_model_record()], columns=MODEL_FEATURE_ORDER)
        try:
            raw_probabilities = np.asarray(self.model.predict_proba(frame), dtype=np.float64)
            if raw_probabilities.shape != (1, 3):
                raise ValueError(f"predict_proba 返回形状 {raw_probabilities.shape}，期望 (1, 3)")
            if not np.isfinite(raw_probabilities).all() or (raw_probabilities < 0).any():
                raise ValueError("模型返回了非法概率")
            model_classes = [str(item) for item in self.model.classes_]
            probability_by_class = {
                class_name: float(raw_probabilities[0, index])
                for index, class_name in enumerate(model_classes)
            }
            if set(probability_by_class) != set(DISPLAY_CLASS_ORDER):
                raise ValueError("模型类别与API三分类契约不一致")
            total = sum(probability_by_class.values())
            if not math.isfinite(total) or total <= 0:
                raise ValueError("模型概率和无效")
            probability_by_class = {
                name: probability_by_class[name] / total for name in DISPLAY_CLASS_ORDER
            }
            # Correct the final bit of floating-point drift without hiding model errors.
            probability_by_class["High"] += 1.0 - sum(probability_by_class.values())
            predicted_class = max(DISPLAY_CLASS_ORDER, key=probability_by_class.__getitem__)
            explanation = self._explain(frame, request, probability_by_class, model_classes)
        except ArtifactUnavailable:
            raise
        except Exception as exc:
            self.runtime_error = f"{type(exc).__name__}: {exc}"
            raise PredictionFailed("模型推理或局部 SHAP 解释失败，请检查产物兼容性。") from exc

        return PredictionResponse(
            request_id=str(uuid4()),
            model_version=self.model_version,
            generated_at=_now_iso(),
            predicted_class=predicted_class,  # type: ignore[arg-type]
            probabilities=RiskProbabilities(**probability_by_class),
            summary=self._summary(predicted_class),
            explanation=explanation,
            recommendations=self._recommendations(request, predicted_class),
            disclaimer=(
                "本结果仅供教育支持与研究性早期筛查参考，不构成医学或心理诊断，"
                "不得作为处分、录取、评奖或其他不利决定的唯一依据。"
            ),
        )

    @staticmethod
    def _summary(predicted_class: str) -> str:
        return {
            "Low": "模型提示当前处于较低风险区间；仍建议保持规律学习，并结合真实体验持续观察。",
            "Medium": "模型提示当前处于中等风险区间，建议结合近期学习负荷、情绪与睡眠情况进行人工复核。",
            "High": "模型提示当前处于较高风险区间，建议尽快由本人、辅导员或专业支持人员共同复核并制定支持方案。",
        }[predicted_class]

    @staticmethod
    def _recommendations(request: PredictionRequest, predicted_class: str) -> list[str]:
        recommendations: list[str] = []
        if predicted_class == "High":
            recommendations.append("尽快与辅导员或可信任的教师进行非惩罚性复核，共同评估近期学习负荷与支持需求。")
        elif predicted_class == "Medium":
            recommendations.append("在一至两周内记录学习负荷、疲惫感与任务完成情况，再与教师或辅导员复核变化。")
        else:
            recommendations.append("保持当前学习节奏，每周做一次简短自查；若疲惫或回避持续加重，应主动寻求支持。")

        if request.perceived_ai_dependency >= 6:
            recommendations.append("设置固定的“无 AI 学习时段”，用主动回忆、手写推导或口头讲解检验是否真正掌握知识。")
        if request.weekly_genai_hours > request.traditional_study_hours:
            recommendations.append("逐步调整 AI 与传统学习时间比例：先独立作答，再用 AI 核验，并保留自己的推理过程。")
        if request.primary_use_case == "Direct_Answer_Generation":
            recommendations.append("将直接索取答案改为请求提示、反例和分步反馈，避免跳过必要的认知加工。")
        if request.anxiety_level_during_exams >= 7:
            recommendations.append("与学校心理健康或学习支持资源联系，讨论考试焦虑管理；如感到紧急危险，应立即联系当地紧急援助。")
        if request.pre_semester_gpa < 2.0:
            recommendations.append("把近期课程拆分为可执行的小目标，并与任课教师核对优先补强的基础知识。")

        # Keep the response focused while retaining the most safety-relevant rules.
        unique = list(dict.fromkeys(recommendations))
        return unique[:4]

    def _ensure_explainer(self) -> None:
        if self._explainer is not None:
            return
        if shap is None or self.shap_background is None or self.model is None:
            raise ArtifactUnavailable("SHAP解释器或背景产物不可用。")
        named_steps = self.model.named_steps
        classifier = named_steps["model"]
        background = self.shap_background
        with self._lock:
            if self._explainer is not None:
                return
            try:
                self._explainer = shap.TreeExplainer(
                    classifier,
                    data=background,
                    feature_perturbation="interventional",
                    model_output="probability",
                )
                self._shap_method = "TreeSHAP（背景条件概率，序数期望风险）"
                return
            except Exception as tree_error:
                try:
                    self._explainer = shap.LinearExplainer(classifier, background)
                    self._shap_method = "LinearSHAP（序数风险输出）"
                    return
                except Exception:
                    try:
                        masker = shap.maskers.Independent(background[: min(100, len(background))])
                        self._explainer = shap.Explainer(
                            classifier.predict_proba,
                            masker,
                            algorithm="permutation",
                        )
                        self._shap_method = "PermutationSHAP（序数期望风险）"
                        self.load_warnings.append(
                            "模型专用SHAP不可用，已采用真实的PermutationSHAP；"
                            f"原因为 {type(tree_error).__name__}。"
                        )
                        return
                    except Exception as fallback_error:
                        raise ArtifactUnavailable(
                            "当前模型与SHAP运行库不兼容，无法生成真实局部解释。"
                        ) from fallback_error

    def _explain(
        self,
        frame: pd.DataFrame,
        request: PredictionRequest,
        probabilities: dict[str, float],
        model_classes: list[str],
    ) -> ShapExplanation:
        assert self.model is not None
        assert self.transformed_feature_names is not None
        self._ensure_explainer()
        transformed = self.model.named_steps["preprocessor"].transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float64)
        if transformed.shape != (1, len(self.transformed_feature_names)):
            raise ValueError(
                "预处理输出形状与特征名不一致："
                f"{transformed.shape} vs (1, {len(self.transformed_feature_names)})"
            )

        with self._lock:
            try:
                raw = self._explainer.shap_values(transformed, check_additivity=False)
            except (AttributeError, TypeError):
                raw = self._explainer(transformed).values
        normalized = self._normalize_shap_values(
            raw,
            samples=1,
            features=transformed.shape[1],
            classes=len(model_classes),
        )
        weights = np.asarray(
            [{"Low": 0.0, "Medium": 0.5, "High": 1.0}[name] for name in model_classes],
            dtype=np.float64,
        )
        if normalized.shape[2] != len(weights):
            raise ValueError("SHAP输出类别维与模型类别数量不一致")
        transformed_contribution = normalized[0] @ weights
        if not np.isfinite(transformed_contribution).all():
            raise ValueError("SHAP输出含NaN或无穷值")

        by_model_feature = {name: 0.0 for name in MODEL_FEATURE_ORDER}
        for name, contribution in zip(
            self.transformed_feature_names,
            transformed_contribution,
            strict=True,
        ):
            original = _original_model_feature(name)
            if original is None:
                raise ValueError(f"无法将SHAP特征聚合回原字段：{name}")
            by_model_feature[original] += float(contribution)

        output_value = probabilities["Medium"] * 0.5 + probabilities["High"]
        base_value = output_value - sum(by_model_feature.values())
        request_values = request.model_dump()
        features = []
        for model_name in MODEL_FEATURE_ORDER:
            api_name = MODEL_TO_API_FIELD[model_name]
            value = request_values[api_name]
            display_value: str | float | int | bool = value
            labels = VALUE_LABELS.get(api_name)
            if labels and value in labels:
                display_value = labels[value]
            features.append(
                ShapFeature(
                    feature=api_name,
                    label=FEATURE_LABELS[api_name],
                    value=display_value,
                    shap_value=by_model_feature[model_name],
                )
            )

        return ShapExplanation(
            method=self._shap_method,
            explained_output="序数期望风险：Low=0、Medium=0.5、High=1；正值推动更高风险，负值推动更低风险",
            base_value=float(base_value),
            output_value=float(output_value),
            features=features,
            warning="SHAP解释的是模型在当前数据中的数学行为和关联，不证明因果关系。",
        )

    @staticmethod
    def _normalize_shap_values(
        raw: Any,
        *,
        samples: int,
        features: int,
        classes: int,
    ) -> np.ndarray:
        if isinstance(raw, list):
            arrays = [np.asarray(item, dtype=np.float64) for item in raw]
            if len(arrays) != classes or any(item.shape != (samples, features) for item in arrays):
                raise ValueError("列表形式的SHAP输出维度异常")
            return np.stack(arrays, axis=2)

        values = np.asarray(raw, dtype=np.float64)
        if values.ndim == 2:
            if values.shape != (samples, features) or classes != 1:
                raise ValueError(f"二维SHAP输出维度异常：{values.shape}")
            return values[:, :, np.newaxis]
        if values.ndim != 3:
            raise ValueError(f"SHAP输出必须为二维或三维；实际为 {values.shape}")
        if values.shape == (samples, features, classes):
            return values
        if values.shape == (classes, samples, features):
            return np.moveaxis(values, 0, 2)
        if values.shape == (samples, classes, features):
            return np.swapaxes(values, 1, 2)
        raise ValueError(f"无法识别SHAP输出维度：{values.shape}")
