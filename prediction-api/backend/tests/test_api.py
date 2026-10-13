from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import create_app
from runtime import ArtifactRuntime, ArtifactUnavailable
from schemas import PredictionRequest, PredictionResponse


VALID_PAYLOAD = {
    "major_category": "STEM",
    "year_of_study": "Junior",
    "pre_semester_gpa": 3.25,
    "weekly_genai_hours": 8.5,
    "primary_use_case": "Debugging/Troubleshooting",
    "prompt_engineering_skill": "Intermediate",
    "tool_diversity": 3,
    "paid_subscription": False,
    "traditional_study_hours": 12.5,
    "perceived_ai_dependency": 6,
    "institutional_policy": "Allowed_With_Citation",
    "anxiety_level_during_exams": 7,
}


class FakeRuntime:
    def load(self) -> None:
        pass

    def health(self):
        return {
            "status": "ok",
            "model_ready": True,
            "prediction_ready": True,
            "artifacts": {},
        }

    def metadata_response(self):
        return {
            "api_version": "1.0",
            "service_name": "AI依赖与学业倦怠预测服务",
            "model": {
                "version": "test-model",
                "target": "Burnout_Risk_Level",
                "classes": ["Low", "Medium", "High"],
            },
            "feature_schema": {},
        }

    def dashboard_response(self):
        return {
            "summary": {
                "total_students": 100,
                "high_risk_rate": 0.25,
                "avg_ai_dependency": 5.2,
            },
            "model_metrics": {"macro_f1": 0.6},
            "major_comparison": [],
            "year_comparison": [],
        }

    def fairness_response(self):
        return {
            "status": "pass",
            "overall": {"status": "pass", "max_gap": 0.05, "metric": "tpr"},
            "thresholds": {
                "max_tpr_gap": 0.1,
                "max_fpr_gap": 0.1,
                "min_group_size": 50,
            },
            "groups": [],
        }

    def predict(self, request: PredictionRequest):
        features = [
            {
                "feature": name,
                "label": name,
                "value": value,
                "shap_value": 0.01 if index == 0 else 0.0,
            }
            for index, (name, value) in enumerate(request.model_dump().items())
        ]
        return PredictionResponse.model_validate(
            {
                "request_id": "00000000-0000-4000-8000-000000000001",
                "model_version": "test-model",
                "generated_at": "2026-07-16T12:00:00+08:00",
                "predicted_class": "Medium",
                "probabilities": {"Low": 0.2, "Medium": 0.6, "High": 0.2},
                "summary": "测试摘要",
                "explanation": {
                    "method": "TreeSHAP",
                    "explained_output": "ordinal expected risk",
                    "base_value": 0.49,
                    "output_value": 0.5,
                    "features": features,
                    "warning": "不代表因果关系。",
                },
                "recommendations": ["测试建议"],
                "disclaimer": "非诊断。",
            }
        )


@pytest.fixture()
def client():
    app = create_app(runtime=FakeRuntime(), mount_frontend=False)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_read_endpoints(client: TestClient):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/metadata").json()["model"]["version"] == "test-model"
    assert client.get("/api/dashboard").json()["summary"]["total_students"] == 100
    assert client.get("/api/fairness").json()["status"] == "pass"


def test_valid_prediction_has_three_probabilities_and_twelve_shap_features(client: TestClient):
    response = client.post("/api/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_class"] == "Medium"
    assert set(body["probabilities"]) == {"Low", "Medium", "High"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert len(body["explanation"]["features"]) == 12


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"weekly_genai_hours": 40.1}, "weekly_genai_hours"),
        ({"tool_diversity": 3.0}, "tool_diversity"),
        ({"paid_subscription": "false"}, "paid_subscription"),
        ({"major_category": "Engineering"}, "major_category"),
        ({"unexpected": "not allowed"}, "unexpected"),
    ],
)
def test_strict_validation_rejects_invalid_payload(
    client: TestClient, change: dict, field: str
):
    payload = {**VALID_PAYLOAD, **change}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert field in {item["field"] for item in response.json()["errors"]}


def test_missing_field_is_rejected(client: TestClient):
    payload = dict(VALID_PAYLOAD)
    payload.pop("year_of_study")
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert any(item["field"] == "year_of_study" for item in response.json()["errors"])


def test_missing_artifacts_degrade_health_and_prediction_returns_503(tmp_path: Path):
    runtime = ArtifactRuntime(tmp_path)
    app = create_app(runtime=runtime, mount_frontend=False)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "degraded"
        assert health["prediction_ready"] is False
        response = client.post("/api/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/problem+json")


def test_dashboard_schema_adaptation_uses_real_counts_and_ratios(tmp_path: Path):
    runtime = ArtifactRuntime(tmp_path)
    runtime.dashboard = {
        "dataset": {"rows": 200},
        "burnout_distribution": [
            {"level": "Low", "count": 100, "percentage": 50.0},
            {"level": "Medium", "count": 60, "percentage": 30.0},
            {"level": "High", "count": 40, "percentage": 20.0},
        ],
        "key_numeric_means": {"perceived_ai_dependency": 5.5},
        "major_comparison": [
            {
                "Major_Category": "STEM",
                "n": 80,
                "high_burnout_rate": 0.25,
                "mean_dependency": 5.8,
            }
        ],
        "year_comparison": [],
        "selected_model": "Random Forest",
        "selected_model_holdout": {"f1_macro": 0.55},
    }
    runtime.metadata = {"created_at_utc": "2026-07-16T01:00:00+00:00"}
    runtime.artifact_status["dashboard"] = {"loaded": True, "file": "dashboard.json"}
    result = runtime.dashboard_response()
    assert result["summary"]["high_risk_rate"] == pytest.approx(0.2)
    assert result["major_comparison"][0]["group"] == "STEM"
    assert result["major_comparison"][0]["count"] == 80


def test_dashboard_exposes_top_ten_global_shap_features(tmp_path: Path):
    runtime = ArtifactRuntime(tmp_path)
    feature_names = [
        "Weekly_GenAI_Hours",
        "Year_of_Study",
        "Perceived_AI_Dependency",
        "Pre_Semester_GPA",
        "Traditional_Study_Hours",
        "Institutional_Policy",
        "Major_Category",
        "Paid_Subscription",
        "Anxiety_Level_During_Exams",
        "Primary_Use_Case",
        "Prompt_Engineering_Skill",
        "Tool_Diversity",
    ]
    runtime.dashboard = {
        "dataset": {"rows": 200},
        "top_shap_features": [
            {
                "original_feature": name,
                "mean_abs_shap": 0.12 / (index + 1),
                "normalized_importance": 0.30 / (index + 1),
            }
            for index, name in enumerate(feature_names)
        ],
    }
    runtime.artifact_status["dashboard"] = {"loaded": True, "file": "dashboard.json"}

    result = runtime.dashboard_response()
    features = result["global_shap"]["features"]

    assert len(features) == 10
    assert features[0]["feature"] == "weekly_genai_hours"
    assert features[0]["label"] == "每周生成式 AI 使用时长"
    assert features[0]["normalized_importance"] == pytest.approx(0.30)
    assert result["global_shap"]["explained_output"] == (
        "Low/Medium/High 三类概率输出的跨样本、跨类别平均绝对归因强度"
    )
    assert "不保留正负方向" in result["global_shap"]["caveat"]
    assert result["global_shap"]["caveat"].startswith("全局 SHAP")


def test_fairness_adaptation_and_small_group_suppression(tmp_path: Path):
    runtime = ArtifactRuntime(tmp_path)
    runtime.fairness = {
        "model": "Random Forest",
        "by_major": [
            {
                "group": "STEM",
                "n": 100,
                "high_risk_recall_tpr": 0.8,
                "high_risk_false_positive_rate": 0.1,
                "predicted_high_rate": 0.3,
            },
            {
                "group": "Arts",
                "n": 100,
                "high_risk_recall_tpr": 0.6,
                "high_risk_false_positive_rate": 0.12,
                "predicted_high_rate": 0.2,
            },
            {
                "group": "Medical",
                "n": 12,
                "high_risk_recall_tpr": 0.9,
                "high_risk_false_positive_rate": 0.2,
                "predicted_high_rate": 0.4,
            },
        ],
        "by_year": [],
    }
    runtime.artifact_status["fairness"] = {"loaded": True, "file": "fairness.json"}
    result = runtime.fairness_response()
    assert result["status"] == "warning"
    assert {row["group"] for row in result["groups"]} == {"STEM", "Arts"}
    assert all("true_positive_rate" in row for row in result["groups"])
    assert all("false_positive_rate" in row for row in result["groups"])
    assert all("selection_rate" in row for row in result["groups"])


class UnavailableRuntime(FakeRuntime):
    def predict(self, request: PredictionRequest):
        del request
        raise ArtifactUnavailable("模型缺失")
