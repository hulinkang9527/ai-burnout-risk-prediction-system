"""API schemas and the single source of truth for online input validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RiskClass = Literal["Low", "Medium", "High"]

MAJOR_VALUES = ("Humanities", "Medical", "Business", "STEM", "Arts")
YEAR_VALUES = ("Freshman", "Sophomore", "Junior", "Senior", "Graduate")
USE_CASE_VALUES = (
    "Copywriting/Drafting",
    "Ideation",
    "Summarizing_Reading",
    "Debugging/Troubleshooting",
    "Direct_Answer_Generation",
)
SKILL_VALUES = ("Beginner", "Intermediate", "Advanced")
POLICY_VALUES = ("Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged")

API_TO_MODEL_FIELD = {
    "major_category": "Major_Category",
    "year_of_study": "Year_of_Study",
    "pre_semester_gpa": "Pre_Semester_GPA",
    "weekly_genai_hours": "Weekly_GenAI_Hours",
    "primary_use_case": "Primary_Use_Case",
    "prompt_engineering_skill": "Prompt_Engineering_Skill",
    "tool_diversity": "Tool_Diversity",
    "paid_subscription": "Paid_Subscription",
    "traditional_study_hours": "Traditional_Study_Hours",
    "perceived_ai_dependency": "Perceived_AI_Dependency",
    "institutional_policy": "Institutional_Policy",
    "anxiety_level_during_exams": "Anxiety_Level_During_Exams",
}
MODEL_TO_API_FIELD = {value: key for key, value in API_TO_MODEL_FIELD.items()}
MODEL_FEATURE_ORDER = list(API_TO_MODEL_FIELD.values())

FEATURE_LABELS = {
    "major_category": "专业类别",
    "year_of_study": "学习年级",
    "pre_semester_gpa": "上学期 GPA",
    "weekly_genai_hours": "每周生成式 AI 使用时长",
    "primary_use_case": "AI 主要使用场景",
    "prompt_engineering_skill": "提示工程能力",
    "tool_diversity": "AI 工具多样性",
    "paid_subscription": "是否付费订阅",
    "traditional_study_hours": "每周传统学习时长",
    "perceived_ai_dependency": "自评 AI 依赖程度",
    "institutional_policy": "学校 AI 使用政策",
    "anxiety_level_during_exams": "考试焦虑程度",
}

VALUE_LABELS: dict[str, dict[Any, str]] = {
    "major_category": {
        "Humanities": "人文社科",
        "Medical": "医学",
        "Business": "商科",
        "STEM": "理工科",
        "Arts": "艺术类",
    },
    "year_of_study": {
        "Freshman": "大一",
        "Sophomore": "大二",
        "Junior": "大三",
        "Senior": "大四",
        "Graduate": "研究生",
    },
    "primary_use_case": {
        "Copywriting/Drafting": "写作与起草",
        "Ideation": "创意构思",
        "Summarizing_Reading": "阅读总结",
        "Debugging/Troubleshooting": "调试与问题排查",
        "Direct_Answer_Generation": "直接生成答案",
    },
    "prompt_engineering_skill": {
        "Beginner": "初级",
        "Intermediate": "中级",
        "Advanced": "高级",
    },
    "paid_subscription": {False: "否", True: "是"},
    "institutional_policy": {
        "Allowed_With_Citation": "允许但须引用",
        "Strict_Ban": "严格禁止",
        "Actively_Encouraged": "积极鼓励",
    },
}


class PredictionRequest(BaseModel):
    """Twelve screening-time fields; coercion and unknown fields are forbidden."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=False,
        allow_inf_nan=False,
    )

    major_category: Literal[MAJOR_VALUES]  # type: ignore[valid-type]
    year_of_study: Literal[YEAR_VALUES]  # type: ignore[valid-type]
    pre_semester_gpa: float = Field(ge=0.0, le=4.0, allow_inf_nan=False)
    weekly_genai_hours: float = Field(ge=0.0, le=40.0, allow_inf_nan=False)
    primary_use_case: Literal[USE_CASE_VALUES]  # type: ignore[valid-type]
    prompt_engineering_skill: Literal[SKILL_VALUES]  # type: ignore[valid-type]
    tool_diversity: int = Field(ge=1, le=5)
    paid_subscription: bool
    traditional_study_hours: float = Field(ge=0.0, le=40.0, allow_inf_nan=False)
    perceived_ai_dependency: int = Field(ge=1, le=10)
    institutional_policy: Literal[POLICY_VALUES]  # type: ignore[valid-type]
    anxiety_level_during_exams: int = Field(ge=1, le=10)

    def to_model_record(self) -> dict[str, Any]:
        values = self.model_dump()
        return {
            model_name: values[api_name]
            for api_name, model_name in API_TO_MODEL_FIELD.items()
        }


class RiskProbabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    Low: float = Field(ge=0.0, le=1.0)
    Medium: float = Field(ge=0.0, le=1.0)
    High: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "RiskProbabilities":
        if abs((self.Low + self.Medium + self.High) - 1.0) > 1e-6:
            raise ValueError("classification probabilities must sum to one")
        return self


class ShapFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    feature: str
    label: str
    value: str | float | int | bool
    shap_value: float = Field(allow_inf_nan=False)


class ShapExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    method: str
    explained_output: str
    base_value: float = Field(allow_inf_nan=False)
    output_value: float = Field(allow_inf_nan=False)
    features: list[ShapFeature] = Field(min_length=12, max_length=12)
    warning: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str
    model_version: str
    generated_at: str
    predicted_class: RiskClass
    probabilities: RiskProbabilities
    summary: str
    explanation: ShapExplanation
    recommendations: list[str] = Field(min_length=1)
    disclaimer: str


def _options(values: tuple[Any, ...], labels: dict[Any, str]) -> list[dict[str, Any]]:
    return [{"value": value, "label": labels.get(value, str(value))} for value in values]


def public_feature_schema(metadata_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the frontend schema, keeping deployment limits authoritative.

    Observed medians from the training metadata are included as descriptive hints only;
    they never widen the API's deliberately narrower 0--40 hour limits.
    """

    metadata_schema = metadata_schema or {}
    schema: dict[str, Any] = {
        "major_category": {
            "type": "categorical",
            "required": True,
            "allowed_values": _options(MAJOR_VALUES, VALUE_LABELS["major_category"]),
        },
        "year_of_study": {
            "type": "categorical",
            "required": True,
            "allowed_values": _options(YEAR_VALUES, VALUE_LABELS["year_of_study"]),
        },
        "pre_semester_gpa": {
            "type": "number",
            "required": True,
            "minimum": 0.0,
            "maximum": 4.0,
            "step": 0.01,
        },
        "weekly_genai_hours": {
            "type": "number",
            "required": True,
            "minimum": 0.0,
            "maximum": 40.0,
            "step": 0.5,
        },
        "primary_use_case": {
            "type": "categorical",
            "required": True,
            "allowed_values": _options(USE_CASE_VALUES, VALUE_LABELS["primary_use_case"]),
        },
        "prompt_engineering_skill": {
            "type": "categorical",
            "required": True,
            "allowed_values": _options(SKILL_VALUES, VALUE_LABELS["prompt_engineering_skill"]),
        },
        "tool_diversity": {
            "type": "integer",
            "required": True,
            "minimum": 1,
            "maximum": 5,
            "step": 1,
        },
        "paid_subscription": {
            "type": "boolean",
            "required": True,
            "allowed_values": _options((False, True), VALUE_LABELS["paid_subscription"]),
        },
        "traditional_study_hours": {
            "type": "number",
            "required": True,
            "minimum": 0.0,
            "maximum": 40.0,
            "step": 0.5,
        },
        "perceived_ai_dependency": {
            "type": "integer",
            "required": True,
            "minimum": 1,
            "maximum": 10,
            "step": 1,
        },
        "institutional_policy": {
            "type": "categorical",
            "required": True,
            "allowed_values": _options(POLICY_VALUES, VALUE_LABELS["institutional_policy"]),
        },
        "anxiety_level_during_exams": {
            "type": "integer",
            "required": True,
            "minimum": 1,
            "maximum": 10,
            "step": 1,
        },
    }
    for api_name, model_name in API_TO_MODEL_FIELD.items():
        schema[api_name]["label"] = FEATURE_LABELS[api_name]
        schema[api_name]["source_field"] = model_name
        source = metadata_schema.get(model_name)
        if isinstance(source, dict) and source.get("observed_median") is not None:
            schema[api_name]["observed_median"] = source["observed_median"]
    return schema
