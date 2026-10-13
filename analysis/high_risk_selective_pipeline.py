"""High-risk burnout screening with honest selective classification.

This module complements (rather than overwrites) the original Low/Medium/High
three-class experiment.  The public dataset does not support 85% three-class
holdout accuracy.  Here the operational target is explicitly narrowed to
High versus Non-high, and uncertain predictions are withheld for manual review.

The split is fixed at 60% train / 20% validation / 20% test.  Two asymmetric
decision thresholds are selected on validation data only: the Non-high
threshold is deliberately stricter than the High threshold because an
automated false reassurance is more harmful than sending an uncertain case to
manual review.  Feasible policies must cover at least 70% of validation rows
and have a 95% Wilson lower accuracy bound of at least 85%; among them, the
policy with the highest accepted-set balanced accuracy is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = (
    WORKSPACE_ROOT
    / "2026年7月毕业论文数据集"
    / "数据与思路"
    / "ai_student_impact_dataset (1).csv"
)
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
TABLE_DIR = SCRIPT_DIR / "tables"
FIGURE_DIR = SCRIPT_DIR / "figures"
THESIS_FIGURE_DIR = SCRIPT_DIR.parent / "04_论文插图"

RANDOM_STATE = 42
VALIDATION_RANDOM_STATE = 43
TARGET = "High_Burnout_Risk"
SOURCE_TARGET = "Burnout_Risk_Level"
TARGET_ACCURACY = 0.85
MIN_VALIDATION_COVERAGE = 0.70
LOW_THRESHOLD_MIN = 0.050
LOW_THRESHOLD_MAX = 0.490
HIGH_THRESHOLD_MIN = 0.510
HIGH_THRESHOLD_MAX = 0.950
THRESHOLD_STEP = 0.005

RAW_INPUT_FEATURES = [
    "Major_Category",
    "Year_of_Study",
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Tool_Diversity",
    "Paid_Subscription",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Institutional_Policy",
    "Anxiety_Level_During_Exams",
]
DERIVED_FEATURES = [
    "AI_to_Study_Ratio",
    "Total_Learning_Hours",
    "AI_Hours_Share",
    "AI_minus_Study_Hours",
    "Dependency_x_AI_Hours",
    "Anxiety_per_Study_Hour",
    "Anxiety_x_Dependency",
    "Anxiety_plus_Dependency",
]
NUMERIC_FEATURES = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
    *DERIVED_FEATURES,
]
CATEGORICAL_FEATURES = [
    "Major_Category",
    "Year_of_Study",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Paid_Subscription",
    "Institutional_Policy",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DERIVED_PARENTS = {
    "AI_to_Study_Ratio": [
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
    ],
    "Total_Learning_Hours": [
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
    ],
    "AI_Hours_Share": [
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
    ],
    "AI_minus_Study_Hours": [
        "Weekly_GenAI_Hours",
        "Traditional_Study_Hours",
    ],
    "Dependency_x_AI_Hours": [
        "Perceived_AI_Dependency",
        "Weekly_GenAI_Hours",
    ],
    "Anxiety_per_Study_Hour": [
        "Anxiety_Level_During_Exams",
        "Traditional_Study_Hours",
    ],
    "Anxiety_x_Dependency": [
        "Anxiety_Level_During_Exams",
        "Perceived_AI_Dependency",
    ],
    "Anxiety_plus_Dependency": [
        "Anxiety_Level_During_Exams",
        "Perceived_AI_Dependency",
    ],
}

FEATURE_LABELS = {
    "Major_Category": "专业类别",
    "Year_of_Study": "年级",
    "Pre_Semester_GPA": "学期前GPA",
    "Weekly_GenAI_Hours": "每周生成式AI使用时长",
    "Primary_Use_Case": "AI主要使用场景",
    "Prompt_Engineering_Skill": "提示工程能力",
    "Tool_Diversity": "AI工具多样性",
    "Paid_Subscription": "是否付费订阅",
    "Traditional_Study_Hours": "传统学习时长",
    "Perceived_AI_Dependency": "感知AI依赖",
    "Institutional_Policy": "学校AI政策",
    "Anxiety_Level_During_Exams": "考试焦虑",
    "AI_to_Study_Ratio": "AI与传统学习时长比",
    "Total_Learning_Hours": "总学习时长",
    "AI_Hours_Share": "AI学习时长占比",
    "AI_minus_Study_Hours": "AI与传统学习时长差",
    "Dependency_x_AI_Hours": "AI时长×依赖程度",
    "Anxiety_per_Study_Hour": "单位传统学习时长焦虑",
    "Anxiety_x_Dependency": "考试焦虑×AI依赖",
    "Anxiety_plus_Dependency": "考试焦虑与AI依赖合计",
}


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [json_value(item) for item in list(value)]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    epsilon = 0.25
    ai_hours = result["Weekly_GenAI_Hours"].astype(float)
    study_hours = result["Traditional_Study_Hours"].astype(float)
    dependency = result["Perceived_AI_Dependency"].astype(float)
    anxiety = result["Anxiety_Level_During_Exams"].astype(float)
    total_hours = ai_hours + study_hours
    result["AI_to_Study_Ratio"] = ai_hours / (study_hours + epsilon)
    result["Total_Learning_Hours"] = total_hours
    result["AI_Hours_Share"] = ai_hours / (total_hours + epsilon)
    result["AI_minus_Study_Hours"] = ai_hours - study_hours
    result["Dependency_x_AI_Hours"] = dependency * ai_hours
    result["Anxiety_per_Study_Hour"] = anxiety / (study_hours + epsilon)
    result["Anxiety_x_Dependency"] = anxiety * dependency
    result["Anxiety_plus_Dependency"] = anxiety + dependency
    return result


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def ordinary_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "high_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> list[float | None]:
    """Return a two-sided 95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return [None, None]
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / n
            + z * z / (4.0 * n * n)
        )
        / denominator
    )
    return [float(center - margin), float(center + margin)]


def selective_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    accepted = (probabilities <= low_threshold) | (
        probabilities >= high_threshold
    )
    predictions = (probabilities >= high_threshold).astype(int)
    if not accepted.any():
        return {
            "non_high_probability_threshold": float(low_threshold),
            "high_probability_threshold": float(high_threshold),
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "accepted_n": 0,
            "accuracy": None,
            "accuracy_wilson_95": [None, None],
            "balanced_accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "high_precision": None,
            "high_recall_within_accepted": None,
            "auto_high_recall_all_high": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    accepted_y = y_true.to_numpy()[accepted]
    accepted_predictions = predictions[accepted]
    true_high_total = int(np.sum(y_true.to_numpy() == 1))
    auto_true_high = int(np.sum((accepted_predictions == 1) & (accepted_y == 1)))
    correct = int(np.sum(accepted_y == accepted_predictions))
    return {
        "non_high_probability_threshold": float(low_threshold),
        "high_probability_threshold": float(high_threshold),
        "coverage": float(accepted.mean()),
        "abstention_rate": float(1.0 - accepted.mean()),
        "accepted_n": int(accepted.sum()),
        "accuracy": float(correct / accepted_y.size),
        "accuracy_wilson_95": wilson_interval(correct, int(accepted_y.size)),
        "balanced_accuracy": float(
            balanced_accuracy_score(accepted_y, accepted_predictions)
        ),
        "precision_macro": float(
            precision_score(
                accepted_y, accepted_predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                accepted_y, accepted_predictions, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                accepted_y, accepted_predictions, average="macro", zero_division=0
            )
        ),
        "high_precision": float(
            precision_score(accepted_y, accepted_predictions, zero_division=0)
        ),
        "high_recall_within_accepted": float(
            recall_score(accepted_y, accepted_predictions, zero_division=0)
        ),
        "auto_high_recall_all_high": (
            float(auto_true_high / true_high_total) if true_high_total else None
        ),
        "confusion_matrix": confusion_matrix(
            accepted_y, accepted_predictions, labels=[0, 1]
        ).tolist(),
    }


def choose_threshold(
    y_validation: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    """Choose a cost-sensitive asymmetric reject policy on validation only."""
    y_array = y_validation.to_numpy()
    n_validation = len(y_array)
    positive_total = int(np.sum(y_array == 1))
    low_values = np.round(
        np.arange(
            LOW_THRESHOLD_MIN,
            LOW_THRESHOLD_MAX + THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        3,
    )
    high_values = np.round(
        np.arange(
            HIGH_THRESHOLD_MIN,
            HIGH_THRESHOLD_MAX + THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        3,
    )

    low_counts = {}
    for threshold in low_values:
        low_mask = probabilities <= threshold
        low_counts[float(threshold)] = (
            int(np.sum(low_mask & (y_array == 0))),
            int(np.sum(low_mask & (y_array == 1))),
        )
    high_counts = {}
    for threshold in high_values:
        high_mask = probabilities >= threshold
        high_counts[float(threshold)] = (
            int(np.sum(high_mask & (y_array == 0))),
            int(np.sum(high_mask & (y_array == 1))),
        )

    rows: list[dict[str, Any]] = []
    feasible: list[tuple[tuple[float, float, float], float, float]] = []
    for low_threshold in low_values:
        true_negative, false_negative = low_counts[float(low_threshold)]
        for high_threshold in high_values:
            false_positive, true_positive = high_counts[float(high_threshold)]
            accepted_n = (
                true_negative
                + false_negative
                + false_positive
                + true_positive
            )
            if accepted_n == 0:
                continue
            coverage = accepted_n / n_validation
            correct = true_negative + true_positive
            accuracy = correct / accepted_n
            lower, upper = wilson_interval(correct, accepted_n)
            non_high_recall = (
                true_negative / (true_negative + false_positive)
                if true_negative + false_positive
                else 0.0
            )
            high_recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            balanced_accuracy = (non_high_recall + high_recall) / 2.0
            auto_high_capture = (
                true_positive / positive_total if positive_total else 0.0
            )
            is_feasible = bool(
                coverage >= MIN_VALIDATION_COVERAGE
                and lower is not None
                and lower >= TARGET_ACCURACY
            )
            rows.append(
                {
                    "non_high_probability_threshold": float(low_threshold),
                    "high_probability_threshold": float(high_threshold),
                    "accepted_n": int(accepted_n),
                    "coverage": float(coverage),
                    "accuracy": float(accuracy),
                    "accuracy_wilson_lower_95": lower,
                    "accuracy_wilson_upper_95": upper,
                    "balanced_accuracy": float(balanced_accuracy),
                    "auto_high_recall_all_high": float(auto_high_capture),
                    "feasible": is_feasible,
                }
            )
            if is_feasible:
                objective = (
                    float(balanced_accuracy),
                    float(coverage),
                    float(accuracy),
                )
                feasible.append(
                    (
                        objective,
                        float(low_threshold),
                        float(high_threshold),
                    )
                )
    if not feasible:
        raise RuntimeError(
            "Validation data did not yield an asymmetric threshold pair that "
            "met the declared coverage and Wilson accuracy constraints."
        )
    _, chosen_low, chosen_high = max(feasible, key=lambda item: item[0])
    return chosen_low, chosen_high, pd.DataFrame(rows)


def transformed_group(name: str) -> str:
    suffix = name.split("__", 1)[-1]
    if name.startswith("num__"):
        return suffix
    for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if suffix == feature or suffix.startswith(feature + "_"):
            return feature
    return suffix


def linear_shap_importance(
    pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    transformed_train = np.asarray(preprocessor.transform(X_train), dtype=np.float64)
    transformed_test = np.asarray(preprocessor.transform(X_test), dtype=np.float64)
    background_mean = transformed_train.mean(axis=0)
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
    shap_values = (transformed_test - background_mean) * coefficients
    names = list(preprocessor.get_feature_names_out())
    encoded = pd.DataFrame(
        {
            "transformed_feature": names,
            "source_feature": [transformed_group(name) for name in names],
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "coefficient": coefficients,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    grouped = (
        encoded.groupby("source_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    total = float(grouped["mean_abs_shap"].sum())
    grouped["normalized_importance"] = (
        grouped["mean_abs_shap"] / total if total else 0.0
    )
    grouped["label"] = grouped["source_feature"].map(FEATURE_LABELS).fillna(
        grouped["source_feature"]
    )

    # For a form-facing view, allocate each engineered feature's absolute
    # contribution equally to its raw parent fields. This is a transparent
    # reporting heuristic; the unallocated 20-feature values remain available.
    raw_values = {feature: 0.0 for feature in RAW_INPUT_FEATURES}
    for row in grouped.itertuples(index=False):
        source = str(row.source_feature)
        value = float(row.mean_abs_shap)
        if source in raw_values:
            raw_values[source] += value
        elif source in DERIVED_PARENTS:
            parents = DERIVED_PARENTS[source]
            for parent in parents:
                raw_values[parent] += value / len(parents)
    raw_grouped = pd.DataFrame(
        [
            {
                "source_feature": feature,
                "mean_abs_shap": value,
                "label": FEATURE_LABELS[feature],
            }
            for feature, value in raw_values.items()
        ]
    ).sort_values("mean_abs_shap", ascending=False)
    raw_total = float(raw_grouped["mean_abs_shap"].sum())
    raw_grouped["normalized_importance"] = (
        raw_grouped["mean_abs_shap"] / raw_total if raw_total else 0.0
    )
    return encoded, grouped, raw_grouped


def stability_audit(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Repeat the full validation-only threshold procedure across fixed seeds."""
    rows = []
    for seed in [11, 21, 42, 77, 101]:
        X_development, X_test, y_development, y_test = train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=seed,
            stratify=target,
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_development,
            y_development,
            test_size=0.25,
            random_state=seed + 1,
            stratify=y_development,
        )
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        low_threshold, high_threshold, _ = choose_threshold(
            y_validation,
            validation_probabilities,
        )
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        validation_metrics = selective_metrics(
            y_validation,
            validation_probabilities,
            low_threshold,
            high_threshold,
        )
        test_metrics = selective_metrics(
            y_test,
            test_probabilities,
            low_threshold,
            high_threshold,
        )
        rows.append(
            {
                "split_seed": seed,
                "non_high_probability_threshold": low_threshold,
                "high_probability_threshold": high_threshold,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_wilson_lower_95": validation_metrics[
                    "accuracy_wilson_95"
                ][0],
                "validation_coverage": validation_metrics["coverage"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "test_accuracy": test_metrics["accuracy"],
                "test_wilson_lower_95": test_metrics[
                    "accuracy_wilson_95"
                ][0],
                "test_coverage": test_metrics["coverage"],
                "test_balanced_accuracy": test_metrics[
                    "balanced_accuracy"
                ],
                "test_macro_f1": test_metrics["f1_macro"],
                "test_auto_high_recall_all_high": test_metrics[
                    "auto_high_recall_all_high"
                ],
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "runs": int(len(table)),
        "mean_test_accuracy": float(table["test_accuracy"].mean()),
        "sample_std_test_accuracy": float(table["test_accuracy"].std(ddof=1)),
        "minimum_test_accuracy": float(table["test_accuracy"].min()),
        "maximum_test_accuracy": float(table["test_accuracy"].max()),
        "minimum_test_wilson_lower_95": float(
            table["test_wilson_lower_95"].min()
        ),
        "mean_test_coverage": float(table["test_coverage"].mean()),
        "mean_test_balanced_accuracy": float(
            table["test_balanced_accuracy"].mean()
        ),
        "mean_test_auto_high_recall_all_high": float(
            table["test_auto_high_recall_all_high"].mean()
        ),
    }
    return table, summary


def group_audit(
    source: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    column: str,
) -> pd.DataFrame:
    accepted = (probabilities <= low_threshold) | (
        probabilities >= high_threshold
    )
    predictions = (probabilities >= high_threshold).astype(int)
    rows = []
    source = source.reset_index(drop=True)
    y_array = y_true.reset_index(drop=True).to_numpy()
    for value, indices in source.groupby(column).groups.items():
        position = np.asarray(list(indices), dtype=int)
        group_accepted = accepted[position]
        accepted_position = position[group_accepted]
        accepted_y = y_array[accepted_position]
        accepted_prediction = predictions[accepted_position]
        group_true_high = int(np.sum(y_array[position] == 1))
        group_auto_true_high = int(
            np.sum((accepted_y == 1) & (accepted_prediction == 1))
        )
        row = {
            "group": value,
            "n": int(len(position)),
            "coverage": float(group_accepted.mean()),
            "accepted_n": int(group_accepted.sum()),
            "selective_accuracy": (
                float(accuracy_score(accepted_y, accepted_prediction))
                if accepted_position.size
                else None
            ),
            "balanced_accuracy": (
                float(
                    balanced_accuracy_score(
                        accepted_y,
                        accepted_prediction,
                    )
                )
                if accepted_position.size
                else None
            ),
            "macro_f1": (
                float(
                    f1_score(
                        accepted_y,
                        accepted_prediction,
                        average="macro",
                        zero_division=0,
                    )
                )
                if accepted_position.size
                else None
            ),
            "high_precision": (
                float(
                    precision_score(
                        accepted_y,
                        accepted_prediction,
                        zero_division=0,
                    )
                )
                if accepted_position.size
                else None
            ),
            "high_recall_within_accepted": (
                float(
                    recall_score(
                        accepted_y, accepted_prediction, zero_division=0
                    )
                )
                if accepted_position.size
                else None
            ),
            "auto_high_recall_all_high": (
                float(group_auto_true_high / group_true_high)
                if group_true_high
                else None
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(
    ordinary: dict[str, float],
    selective: dict[str, Any],
    output: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    quality_labels = ["全样本\n准确率", "条件\n准确率", "接受子集\n平衡准确率"]
    quality_values = [
        ordinary["accuracy"],
        selective["accuracy"],
        selective["balanced_accuracy"],
    ]
    workflow_labels = ["自动覆盖率", "人工复核率", "全部High\n自动捕获率"]
    workflow_values = [
        selective["coverage"],
        selective["abstention_rate"],
        selective["auto_high_recall_all_high"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    quality_colors = ["#9CB8E8", "#1769E0", "#44B7B1"]
    workflow_colors = ["#2979E8", "#F0A34A", "#8B6BEA"]
    quality_bars = axes[0].bar(
        quality_labels,
        quality_values,
        color=quality_colors,
        width=0.62,
    )
    axes[0].axhline(
        0.85,
        color="#FB796F",
        linestyle="--",
        linewidth=1.3,
        label="85%条件准确率目标",
    )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("比例")
    axes[0].set_title("分类质量")
    for bar, value in zip(quality_bars, quality_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    workflow_bars = axes[1].bar(
        workflow_labels,
        workflow_values,
        color=workflow_colors,
        width=0.62,
    )
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("比例")
    axes[1].set_title("自动判断与人工复核")
    for bar, value in zip(workflow_bars, workflow_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].legend(loc="lower right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("High vs Non-high 非对称选择性筛查——独立测试集")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in (ARTIFACT_DIR, TABLE_DIR, FIGURE_DIR, THESIS_FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    features = add_derived_features(raw)
    X = features[MODEL_FEATURES].copy()
    y = (raw[SOURCE_TARGET] == "High").astype(int)

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=VALIDATION_RANDOM_STATE,
        stratify=y_development,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
    low_threshold, high_threshold, threshold_table = choose_threshold(
        y_validation, validation_probabilities
    )
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    validation_ordinary = ordinary_metrics(
        y_validation, validation_probabilities
    )
    validation_selective = selective_metrics(
        y_validation,
        validation_probabilities,
        low_threshold,
        high_threshold,
    )
    test_ordinary = ordinary_metrics(y_test, test_probabilities)
    test_selective = selective_metrics(
        y_test,
        test_probabilities,
        low_threshold,
        high_threshold,
    )

    model_path = ARTIFACT_DIR / "selective_high_risk_model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    model_digest = sha256(model_path)
    policy_suffix = (
        f"l{round(low_threshold * 1000):03d}"
        f"-h{round(high_threshold * 1000):03d}"
    )
    model_version = (
        f"asym-selective-logistic-{model_digest[:8]}-{policy_suffix}"
    )

    encoded_shap, grouped_shap, raw_grouped_shap = linear_shap_importance(
        pipeline, X_train, X_test
    )
    encoded_shap.to_csv(
        TABLE_DIR / "selective_shap_encoded.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grouped_shap.to_csv(
        TABLE_DIR / "selective_shap_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_grouped_shap.to_csv(
        TABLE_DIR / "selective_shap_raw_input_aggregation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_table.to_csv(
        TABLE_DIR / "selective_threshold_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {"split": "validation", "mode": "all", **validation_ordinary},
            {"split": "validation", "mode": "selective", **validation_selective},
            {"split": "test", "mode": "all", **test_ordinary},
            {"split": "test", "mode": "selective", **test_selective},
        ]
    ).to_csv(
        TABLE_DIR / "selective_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_source = raw.loc[X_test.index].reset_index(drop=True)
    test_y = y_test.reset_index(drop=True)
    group_audit(
        test_source,
        test_y,
        test_probabilities,
        low_threshold,
        high_threshold,
        "Major_Category",
    ).to_csv(
        TABLE_DIR / "selective_fairness_major.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_audit(
        test_source,
        test_y,
        test_probabilities,
        low_threshold,
        high_threshold,
        "Year_of_Study",
    ).to_csv(
        TABLE_DIR / "selective_fairness_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stability_table, stability_summary = stability_audit(X, y)
    stability_table.to_csv(
        TABLE_DIR / "selective_stability_five_seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_figure(
        test_ordinary,
        test_selective,
        FIGURE_DIR / "selective_high_risk_performance.png",
    )
    shutil.copyfile(
        FIGURE_DIR / "selective_high_risk_performance.png",
        THESIS_FIGURE_DIR / "selective_high_risk_performance.png",
    )

    metrics = {
        "model_version": model_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": {
            "path": str(DATA_PATH),
            "sha256": sha256(DATA_PATH),
            "rows": int(len(raw)),
        },
        "task": {
            "source_target": SOURCE_TARGET,
            "operational_target": TARGET,
            "positive_class": "High",
            "negative_class": "Low or Medium",
            "task_type": "binary selective classification",
            "important_boundary": (
                "This is not the original Low/Medium/High three-class task. "
                "Uncertain cases are withheld for manual review."
            ),
        },
        "split": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "test_random_state": RANDOM_STATE,
            "validation_random_state": VALIDATION_RANDOM_STATE,
            "stratified": True,
        },
        "model": {
            "name": (
                "Logistic Regression + engineered features + "
                "cost-sensitive asymmetric selective classification"
            ),
            "family": "LogisticRegression",
            "parameters": pipeline.named_steps["model"].get_params(),
            "raw_input_features": RAW_INPUT_FEATURES,
            "derived_features": DERIVED_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        "selection": {
            "selected_on": "validation split only",
            "target_selective_accuracy": TARGET_ACCURACY,
            "minimum_validation_coverage": MIN_VALIDATION_COVERAGE,
            "selection_criterion": (
                "among validation policies with coverage at or above 0.70 "
                "and a 95% Wilson lower accuracy bound at or above 0.85, "
                "maximize accepted-set balanced accuracy; break exact ties "
                "by higher coverage and then higher accuracy"
            ),
            "threshold_grid": {
                "non_high_min": LOW_THRESHOLD_MIN,
                "non_high_max": LOW_THRESHOLD_MAX,
                "high_min": HIGH_THRESHOLD_MIN,
                "high_max": HIGH_THRESHOLD_MAX,
                "step": THRESHOLD_STEP,
            },
            "non_high_probability_threshold": low_threshold,
            "high_probability_threshold": high_threshold,
            "decision_rule": (
                f"Non-high when p(High)<={low_threshold:.3f}; High when "
                f"p(High)>={high_threshold:.3f}; otherwise manual review."
            ),
            "cost_assumption": (
                "Automatically reassuring a true High case is treated as "
                "more harmful than referring an uncertain case for review."
            ),
        },
        "validation": {
            "all_predictions": validation_ordinary,
            "selective_predictions": validation_selective,
        },
        "test": {
            "all_predictions": test_ordinary,
            "selective_predictions": test_selective,
        },
        "stability_audit": {
            "protocol": (
                "Five fixed split seeds; each run refits the model and "
                "reselects asymmetric thresholds from validation data by "
                "the same 70% coverage, Wilson, and balanced-accuracy rule."
            ),
            **stability_summary,
            "runs": stability_table.to_dict(orient="records"),
        },
        "comparators": {
            "binary_majority_baseline_accuracy": float((y_test == 0).mean()),
            "legacy_three_class_random_forest_accuracy": 0.5234,
            "legacy_three_class_random_forest_macro_f1": 0.5312617424929382,
        },
        "shap": {
            "method": "analytic LinearSHAP in log-odds space using the training mean background",
            "features": raw_grouped_shap.to_dict(orient="records"),
            "direct_twenty_feature_values": grouped_shap.to_dict(orient="records"),
            "aggregation_note": (
                "One-hot levels are summed to their source field. Each "
                "engineered feature's absolute contribution is divided equally "
                "among its declared raw parent fields for the form-facing view."
            ),
            "caveat": (
                "SHAP explains model association, not causality. Parent "
                "allocation is a reporting heuristic, not a unique SHAP identity."
            ),
        },
        "limitations": [
            "The 85% target is met only for the automatically accepted High/Non-high subset, not all test rows.",
            "The operational target is High versus Non-high, not the original three-class label.",
            "Abstained rows require human review and are excluded from selective accuracy.",
            "Coverage and all-High capture must be reported with conditional accuracy; they are different denominators.",
            "Subgroup performance varies, and some major/year point estimates remain below the overall result.",
            "The public dataset does not document a validated burnout scale or label-generation rule.",
            "This model is for educational research and must not be used as a medical diagnosis.",
        ],
    }
    write_json(ARTIFACT_DIR / "selective_high_risk_metrics.json", metrics)

    report = f"""# 高倦怠风险二分类与非对称选择性预测报告

## 结论

原 Low/Medium/High 三分类随机森林在独立测试集上的 Accuracy 为 52.34%。多模型复核表明，该任务受标签重叠限制，不能通过正常调参达到 85%。

本补充实验将任务明确改为 **High vs Non-high（Low 或 Medium）**，并采用“逻辑回归 + 派生学习行为特征 + 代价敏感非对称拒判”。阈值只按验证集规则选择，独立测试集用于报告固定策略的泛化结果。

## 固定数据划分

- 训练集：{len(X_train):,} 条（60%）
- 验证集：{len(X_validation):,} 条（20%）
- 独立测试集：{len(X_test):,} 条（20%）
- Non-high 自动阈值：`p(High) ≤ {low_threshold:.3f}`
- High 自动阈值：`p(High) ≥ {high_threshold:.3f}`
- 阈值选择准则：验证集自动覆盖率不低于 {MIN_VALIDATION_COVERAGE:.0%}、条件准确率 Wilson 95% 下界不低于 {TARGET_ACCURACY:.0%}，并在可行解中最大化接受子集平衡准确率
- 中间概率区间进入人工复核

## 独立测试结果

- 全样本二分类 Accuracy：{test_ordinary['accuracy']:.2%}
- 全样本二分类 Macro-F1：{test_ordinary['f1_macro']:.2%}
- ROC-AUC：{test_ordinary['roc_auc']:.4f}
- 自动接受子集条件 Accuracy：{test_selective['accuracy']:.2%}（Wilson 95% CI [{test_selective['accuracy_wilson_95'][0]:.2%}, {test_selective['accuracy_wilson_95'][1]:.2%}]）
- 自动覆盖率：{test_selective['coverage']:.2%}
- 人工复核比例：{test_selective['abstention_rate']:.2%}
- 接受子集 Balanced Accuracy：{test_selective['balanced_accuracy']:.2%}
- 接受子集 Macro-F1：{test_selective['f1_macro']:.2%}
- High 精确率：{test_selective['high_precision']:.2%}
- 接受子集内 High 召回率：{test_selective['high_recall_within_accepted']:.2%}
- 全部真实 High 的自动正确捕获率：{test_selective['auto_high_recall_all_high']:.2%}
- 5 个固定随机种子复验条件 Accuracy：{stability_summary['minimum_test_accuracy']:.2%}–{stability_summary['maximum_test_accuracy']:.2%}（均值 {stability_summary['mean_test_accuracy']:.2%}；最低测试 Wilson 下界 {stability_summary['minimum_test_wilson_lower_95']:.2%}）

## 解释边界

**{test_selective['accuracy']:.2%} 不是全样本准确率，也不是原三分类准确率。** 它表示模型只对满足非对称阈值的 {test_selective['coverage']:.2%} 测试样本自动作出 High/Non-high 判断时的正确率；其余 {test_selective['abstention_rate']:.2%} 必须显示“需人工复核”。自动覆盖率也不等于 High 捕获率。

此方案不使用 Student_ID、Post_Semester_GPA、Skill_Retention_Score 或任何目标派生字段。它避免训练集成绩冒充测试成绩，也避免把事后字段用于事前预警。
"""
    (SCRIPT_DIR / "高置信度二分类模型评估报告.md").write_text(
        report, encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "model_version": model_version,
                "non_high_probability_threshold": low_threshold,
                "high_probability_threshold": high_threshold,
                "test_accuracy_all": test_ordinary["accuracy"],
                "test_selective_accuracy": test_selective["accuracy"],
                "test_selective_accuracy_wilson_95": test_selective[
                    "accuracy_wilson_95"
                ],
                "test_coverage": test_selective["coverage"],
                "test_abstention_rate": test_selective["abstention_rate"],
                "test_balanced_accuracy": test_selective[
                    "balanced_accuracy"
                ],
                "test_auto_high_recall_all_high": test_selective[
                    "auto_high_recall_all_high"
                ],
                "roc_auc": test_ordinary["roc_auc"],
                "model_path": str(model_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
