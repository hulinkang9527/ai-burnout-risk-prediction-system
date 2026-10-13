"""AI dependency and academic burnout: reproducible analysis and model build.

This script never modifies the source CSV. It creates audit tables, figures,
fairness diagnostics, SHAP explanations, and a deployable scikit-learn pipeline
under this directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = (
    WORKSPACE_ROOT
    / "2026年7月毕业论文数据集"
    / "数据与思路"
    / "ai_student_impact_dataset (1).csv"
)
TABLE_DIR = SCRIPT_DIR / "tables"
FIGURE_DIR = SCRIPT_DIR / "figures"
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
MPL_DIR = SCRIPT_DIR / ".mplconfig"

for directory in (TABLE_DIR, FIGURE_DIR, ARTIFACT_DIR, MPL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("PYTHONHASHSEED", "42")
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "Burnout_Risk_Level"
DISPLAY_CLASS_ORDER = ["Low", "Medium", "High"]
INPUT_FEATURES = [
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
NUMERIC_FEATURES = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Tool_Diversity",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
]
CATEGORICAL_FEATURES = [
    "Major_Category",
    "Year_of_Study",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Paid_Subscription",
    "Institutional_Policy",
]

FIELD_DEFINITIONS = {
    "Student_ID": ("匿名学生编号；仅作记录标识", "identifier", "unknown"),
    "Major_Category": ("专业大类", "early_input", "screening"),
    "Year_of_Study": ("学习年级/阶段", "early_input", "screening"),
    "Pre_Semester_GPA": ("学期开始前 GPA（0–4）", "early_input", "pre_semester"),
    "Weekly_GenAI_Hours": ("每周生成式 AI 使用时长（小时）", "early_input", "screening"),
    "Primary_Use_Case": ("生成式 AI 的主要使用场景", "early_input", "screening"),
    "Prompt_Engineering_Skill": ("提示工程能力分级", "early_input", "screening"),
    "Tool_Diversity": ("使用的 AI 工具多样性计数（1–5）", "early_input", "screening"),
    "Paid_Subscription": ("是否购买 AI 工具订阅", "early_input", "screening"),
    "Traditional_Study_Hours": ("每周传统学习时长（小时）", "early_input", "screening"),
    "Perceived_AI_Dependency": ("自评 AI 依赖程度（1–10）", "early_input", "screening"),
    "Institutional_Policy": ("所在机构对 AI 使用的政策", "early_input", "screening"),
    "Anxiety_Level_During_Exams": ("考试期间焦虑自评分（1–10）", "early_input", "screening"),
    "Post_Semester_GPA": ("学期结束后 GPA（0–4）", "downstream_outcome", "post_semester"),
    "Skill_Retention_Score": ("学期后技能保持得分（0–100）", "downstream_outcome", "post_semester"),
    "Burnout_Risk_Level": ("学业倦怠风险等级：Low/Medium/High", "target", "label"),
}

EXPECTED_RANGES = {
    "Pre_Semester_GPA": (0.0, 4.0),
    "Weekly_GenAI_Hours": (0.0, 168.0),
    "Tool_Diversity": (1.0, 5.0),
    "Traditional_Study_Hours": (0.0, 168.0),
    "Perceived_AI_Dependency": (1.0, 10.0),
    "Anxiety_Level_During_Exams": (1.0, 10.0),
    "Post_Semester_GPA": (0.0, 4.0),
    "Skill_Retention_Score": (0.0, 100.0),
}


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas values to strict JSON-compatible Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_json_value(item) for item in list(value)]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_value(row) for row in df.to_dict(orient="records")]


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    table = df.loc[:, columns].copy()
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
        else:
            table[column] = table[column].astype(str)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in table.astype(str).values.tolist()]
    return "\n".join([header, divider, *body])


def dependency_level(series: pd.Series) -> pd.Categorical:
    # These are project-specific screening bands, not a validated clinical scale.
    return pd.cut(
        series,
        bins=[0, 3, 5, 10],
        labels=["Low (1-3)", "Moderate (4-5)", "High (6-10)"],
        include_lowest=True,
        ordered=True,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=16,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.10,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.10,
            random_state=RANDOM_STATE,
        ),
    }


def multiclass_auc(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    probabilities = estimator.predict_proba(X)
    return float(
        roc_auc_score(
            y,
            probabilities,
            labels=estimator.classes_,
            multi_class="ovr",
            average="macro",
        )
    )


def audit_data(df: pd.DataFrame) -> dict[str, Any]:
    field_rows = []
    for column in df.columns:
        definition, role, timepoint = FIELD_DEFINITIONS[column]
        field_rows.append(
            {
                "column": column,
                "definition_inferred_from_name_and_values": definition,
                "dtype": str(df[column].dtype),
                "role": role,
                "measurement_time": timepoint,
                "non_null_count": int(df[column].notna().sum()),
                "unique_count": int(df[column].nunique(dropna=True)),
            }
        )
    field_dictionary = pd.DataFrame(field_rows)
    field_dictionary.to_csv(TABLE_DIR / "field_dictionary.csv", index=False, encoding="utf-8-sig")

    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[column].isna().sum()) for column in df.columns],
            "missing_rate": [float(df[column].isna().mean()) for column in df.columns],
        }
    )
    missing.to_csv(TABLE_DIR / "missing_values.csv", index=False, encoding="utf-8-sig")

    numeric_summary = df.select_dtypes(include=[np.number]).describe().T.reset_index()
    numeric_summary = numeric_summary.rename(columns={"index": "column"})
    numeric_summary.to_csv(TABLE_DIR / "numeric_summary.csv", index=False, encoding="utf-8-sig")

    category_rows: list[dict[str, Any]] = []
    distribution_columns = [
        "Major_Category",
        "Year_of_Study",
        "Primary_Use_Case",
        "Prompt_Engineering_Skill",
        "Paid_Subscription",
        "Institutional_Policy",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
        "Burnout_Risk_Level",
    ]
    for column in distribution_columns:
        counts = df[column].value_counts(dropna=False, sort=False)
        for value, count in counts.items():
            category_rows.append(
                {
                    "column": column,
                    "value": str(value),
                    "count": int(count),
                    "percentage": float(count / len(df) * 100),
                }
            )
    category_distributions = pd.DataFrame(category_rows)
    category_distributions.to_csv(
        TABLE_DIR / "category_distributions.csv", index=False, encoding="utf-8-sig"
    )

    outlier_rows = []
    for column in df.select_dtypes(include=[np.number]).columns:
        if column == "Student_ID":
            continue
        values = df[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        expected = EXPECTED_RANGES.get(column)
        expected_violations = 0
        if expected:
            expected_violations = int(((values < expected[0]) | (values > expected[1])).sum())
        outlier_rows.append(
            {
                "column": column,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr_lower_fence": float(iqr_lower),
                "iqr_upper_fence": float(iqr_upper),
                "iqr_outlier_count": int(((values < iqr_lower) | (values > iqr_upper)).sum()),
                "expected_minimum": expected[0] if expected else None,
                "expected_maximum": expected[1] if expected else None,
                "expected_range_violation_count": expected_violations,
                "interpretation": "IQR flags are statistical extremes, not automatically data errors.",
            }
        )
    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(TABLE_DIR / "outlier_audit_iqr.csv", index=False, encoding="utf-8-sig")

    audit = {
        "source_path": str(DATA_PATH),
        "source_sha256": file_sha256(DATA_PATH),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_student_ids": int(df["Student_ID"].duplicated().sum()),
        "student_id_unique": bool(df["Student_ID"].is_unique),
        "student_id_minimum": int(df["Student_ID"].min()),
        "student_id_maximum": int(df["Student_ID"].max()),
        "student_id_contiguous": bool(
            np.array_equal(
                np.sort(df["Student_ID"].to_numpy()),
                np.arange(df["Student_ID"].min(), df["Student_ID"].max() + 1),
            )
        ),
        "expected_range_total_violations": int(outliers["expected_range_violation_count"].sum()),
        "iqr_outlier_total_flags": int(outliers["iqr_outlier_count"].sum()),
        "field_dictionary": records(field_dictionary),
        "missing_values": records(missing),
        "outlier_audit": records(outliers),
        "methodological_warning": (
            "The public dataset does not contain a sampling frame, instrument-validation record, "
            "or data-generation provenance. Sequential IDs and a perfectly complete 50,000-row "
            "table are strong reasons to avoid treating it as a representative clinical sample."
        ),
    }
    write_json(TABLE_DIR / "data_quality_report.json", audit)
    return audit


def build_group_comparison(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    temp = df.copy()
    temp["Dependency_Level_Operational"] = dependency_level(temp["Perceived_AI_Dependency"])
    temp["High_Dependency_Flag"] = (temp["Perceived_AI_Dependency"] >= 6).astype(int)
    temp["High_Burnout_Flag"] = (temp[TARGET] == "High").astype(int)
    temp["Joint_High_Flag"] = (
        (temp["High_Dependency_Flag"] == 1) & (temp["High_Burnout_Flag"] == 1)
    ).astype(int)
    summary = (
        temp.groupby(group_column, observed=False)
        .agg(
            n=("Student_ID", "size"),
            mean_dependency=("Perceived_AI_Dependency", "mean"),
            high_dependency_rate=("High_Dependency_Flag", "mean"),
            high_burnout_rate=("High_Burnout_Flag", "mean"),
            joint_high_rate=("Joint_High_Flag", "mean"),
            mean_weekly_genai_hours=("Weekly_GenAI_Hours", "mean"),
            mean_traditional_study_hours=("Traditional_Study_Hours", "mean"),
            mean_exam_anxiety=("Anxiety_Level_During_Exams", "mean"),
        )
        .reset_index()
    )
    return summary


def fairness_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    group_column: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred,
            group_column: sensitive.to_numpy(),
        }
    )
    rows = []
    for group, subset in frame.groupby(group_column, observed=False):
        actual_high = subset["y_true"] == "High"
        predicted_high = subset["y_pred"] == "High"
        tp = int((actual_high & predicted_high).sum())
        fn = int((actual_high & ~predicted_high).sum())
        fp = int((~actual_high & predicted_high).sum())
        tn = int((~actual_high & ~predicted_high).sum())
        high_recall = tp / (tp + fn) if tp + fn else np.nan
        high_fpr = fp / (fp + tn) if fp + tn else np.nan
        rows.append(
            {
                "attribute": group_column,
                "group": str(group),
                "n": int(len(subset)),
                "accuracy": accuracy_score(subset["y_true"], subset["y_pred"]),
                "precision_macro": precision_score(
                    subset["y_true"],
                    subset["y_pred"],
                    labels=DISPLAY_CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                ),
                "recall_macro": recall_score(
                    subset["y_true"],
                    subset["y_pred"],
                    labels=DISPLAY_CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    subset["y_true"],
                    subset["y_pred"],
                    labels=DISPLAY_CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                ),
                "actual_high_rate": actual_high.mean(),
                "predicted_high_rate": predicted_high.mean(),
                "high_risk_recall_tpr": high_recall,
                "high_risk_false_negative_rate": 1 - high_recall,
                "high_risk_false_positive_rate": high_fpr,
                "high_support": int(actual_high.sum()),
                "tp_high": tp,
                "fn_high": fn,
                "fp_high": fp,
                "tn_high": tn,
            }
        )
    return pd.DataFrame(rows)


def fairness_gaps(table: pd.DataFrame, attribute: str) -> dict[str, Any]:
    metric_columns = [
        "accuracy",
        "f1_macro",
        "predicted_high_rate",
        "high_risk_recall_tpr",
        "high_risk_false_negative_rate",
        "high_risk_false_positive_rate",
    ]
    gaps: dict[str, Any] = {"attribute": attribute, "groups": int(len(table))}
    for metric in metric_columns:
        valid = table.dropna(subset=[metric])
        if valid.empty:
            gaps[f"{metric}_max_minus_min"] = None
            continue
        max_row = valid.loc[valid[metric].idxmax()]
        min_row = valid.loc[valid[metric].idxmin()]
        gaps[f"{metric}_max_minus_min"] = float(max_row[metric] - min_row[metric])
        gaps[f"{metric}_maximum_group"] = str(max_row["group"])
        gaps[f"{metric}_minimum_group"] = str(min_row["group"])
    return gaps


def original_feature_name(transformed_name: str) -> str:
    suffix = transformed_name.split("__", 1)[-1]
    for feature in sorted(INPUT_FEATURES, key=len, reverse=True):
        if suffix == feature or suffix.startswith(feature + "_"):
            return feature
    return suffix


def compute_shap_outputs(
    best_pipeline: Pipeline,
    best_model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    transformed_names = list(preprocessor.get_feature_names_out())
    write_json(ARTIFACT_DIR / "transformed_feature_names.json", transformed_names)

    background_raw = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
    background = np.asarray(preprocessor.transform(background_raw), dtype=np.float64)
    np.save(ARTIFACT_DIR / "shap_background.npy", background)

    # Exact TreeSHAP cost grows with the number and depth of trees. A fixed,
    # reproducible 75-row holdout sample keeps the build tractable while the
    # independent 3,000-row permutation audit provides a second global check.
    explain_raw = X_test.sample(n=min(75, len(X_test)), random_state=RANDOM_STATE)
    explain_matrix = np.asarray(preprocessor.transform(explain_raw), dtype=np.float64)
    method = ""
    try:
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, background)
            explanation = explainer(explain_matrix)
            method = "SHAP LinearExplainer"
        else:
            explainer = shap.TreeExplainer(model)
            explanation = explainer(explain_matrix)
            method = "SHAP TreeExplainer"
    except Exception as tree_error:
        # Model-agnostic SHAP remains genuine Shapley-value approximation and is
        # used only if the model-specific explainer lacks support.
        fallback_background = shap.maskers.Independent(background[:50])
        explainer = shap.PermutationExplainer(
            model.predict_proba,
            fallback_background,
            feature_names=transformed_names,
        )
        explain_matrix = explain_matrix[:120]
        explanation = explainer(
            explain_matrix,
            max_evals=2 * len(transformed_names) + 1,
            silent=True,
        )
        method = f"SHAP PermutationExplainer fallback ({type(tree_error).__name__})"

    values = np.asarray(explanation.values)
    feature_count = len(transformed_names)
    class_names = [str(value) for value in getattr(model, "classes_", [])]

    if values.ndim == 2:
        normalized = values[:, :, np.newaxis]
        if not class_names:
            class_names = ["model_output"]
    elif values.ndim == 3 and values.shape[1] == feature_count:
        normalized = values
    elif values.ndim == 3 and values.shape[2] == feature_count:
        normalized = np.swapaxes(values, 1, 2)
    else:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")

    if len(class_names) != normalized.shape[2]:
        class_names = [f"output_{index}" for index in range(normalized.shape[2])]

    encoded_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "original_feature": [original_feature_name(name) for name in transformed_names],
            "mean_abs_shap": np.mean(np.abs(normalized), axis=(0, 2)),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    encoded_importance.to_csv(
        TABLE_DIR / "shap_importance_encoded.csv", index=False, encoding="utf-8-sig"
    )

    original_importance = (
        encoded_importance.groupby("original_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    total = original_importance["mean_abs_shap"].sum()
    original_importance["normalized_importance"] = (
        original_importance["mean_abs_shap"] / total if total else 0.0
    )
    original_importance.to_csv(
        TABLE_DIR / "shap_feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    class_rows = []
    per_class_encoded = np.mean(np.abs(normalized), axis=0)
    for class_index, class_name in enumerate(class_names):
        for transformed_index, transformed_name in enumerate(transformed_names):
            class_rows.append(
                {
                    "class": class_name,
                    "transformed_feature": transformed_name,
                    "original_feature": original_feature_name(transformed_name),
                    "mean_abs_shap": float(per_class_encoded[transformed_index, class_index]),
                }
            )
    pd.DataFrame(class_rows).to_csv(
        TABLE_DIR / "shap_importance_by_class_encoded.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 6))
    plot_data = original_importance.head(12).sort_values("mean_abs_shap")
    plt.barh(plot_data["original_feature"], plot_data["mean_abs_shap"], color="#3867d6")
    plt.xlabel("Mean absolute SHAP value (aggregated one-hot features)")
    plt.ylabel("Original input feature")
    plt.title(f"Global SHAP Importance — {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_global_importance.png", dpi=220, bbox_inches="tight")
    plt.close()

    status = {
        "status": "success",
        "method": method,
        "best_model": best_model_name,
        "background_rows": int(background.shape[0]),
        "background_role": (
            "Saved for deployment-time explainers. The fitted Random Forest TreeExplainer "
            "uses its tree-path-dependent internal background for this global audit."
        ),
        "explained_rows": int(normalized.shape[0]),
        "transformed_feature_count": feature_count,
        "shap_value_shape_normalized": list(normalized.shape),
        "interpretation_warning": "SHAP explains model behaviour and association; it does not establish causality.",
    }
    write_json(ARTIFACT_DIR / "shap_status.json", status)
    return original_importance, status


def create_figures(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    burnout_counts = df[TARGET].value_counts().reindex(DISPLAY_CLASS_ORDER)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=burnout_counts.index, y=burnout_counts.values, color="#3867d6")
    plt.xlabel("Burnout risk level")
    plt.ylabel("Students")
    plt.title("Burnout Risk Distribution")
    for index, count in enumerate(burnout_counts.values):
        plt.text(index, count + 250, f"{count:,}\n({count / len(df):.1%})", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "burnout_distribution.png", dpi=220, bbox_inches="tight")
    plt.close()

    coupling = pd.crosstab(
        dependency_level(df["Perceived_AI_Dependency"]),
        df[TARGET],
        normalize="index",
    ).reindex(columns=DISPLAY_CLASS_ORDER)
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(coupling, annot=True, fmt=".1%", cmap="YlOrRd", vmin=0, vmax=1)
    plt.xlabel("Burnout risk level")
    plt.ylabel("Operational AI dependency band")
    plt.title("Burnout Distribution within AI Dependency Bands")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "dependency_burnout_coupling.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plot_metrics = metrics.sort_values("f1_macro")
    positions = np.arange(len(plot_metrics))
    width = 0.24
    plt.barh(positions - width, plot_metrics["accuracy"], height=width, label="Accuracy")
    plt.barh(positions, plot_metrics["f1_macro"], height=width, label="Macro F1")
    plt.barh(positions + width, plot_metrics["roc_auc_ovr_macro"], height=width, label="Macro ROC-AUC")
    plt.yticks(positions, plot_metrics["model"])
    plt.xlim(0, 1)
    plt.xlabel("Holdout score")
    plt.title("Burnout Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    started = time.time()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Source CSV not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    expected_columns = set(FIELD_DEFINITIONS)
    if set(df.columns) != expected_columns:
        missing_columns = sorted(expected_columns - set(df.columns))
        extra_columns = sorted(set(df.columns) - expected_columns)
        raise ValueError(f"Unexpected schema. Missing={missing_columns}, extra={extra_columns}")

    audit = audit_data(df)

    analysis_df = df.copy()
    analysis_df["Dependency_Level_Operational"] = dependency_level(
        analysis_df["Perceived_AI_Dependency"]
    )
    dependency_distribution = (
        analysis_df["Dependency_Level_Operational"]
        .value_counts(sort=False, dropna=False)
        .rename_axis("dependency_level")
        .reset_index(name="count")
    )
    dependency_distribution["percentage"] = dependency_distribution["count"] / len(df) * 100
    dependency_distribution.to_csv(
        TABLE_DIR / "dependency_level_distribution.csv", index=False, encoding="utf-8-sig"
    )

    coupling_counts = pd.crosstab(
        analysis_df["Dependency_Level_Operational"],
        analysis_df[TARGET],
        margins=True,
    ).reindex(columns=DISPLAY_CLASS_ORDER + ["All"])
    coupling_counts.to_csv(TABLE_DIR / "dependency_burnout_coupling_counts.csv", encoding="utf-8-sig")
    coupling_row_pct = pd.crosstab(
        analysis_df["Dependency_Level_Operational"],
        analysis_df[TARGET],
        normalize="index",
    ).reindex(columns=DISPLAY_CLASS_ORDER)
    coupling_row_pct.to_csv(
        TABLE_DIR / "dependency_burnout_coupling_row_percent.csv", encoding="utf-8-sig"
    )

    major_comparison = build_group_comparison(df, "Major_Category")
    year_comparison = build_group_comparison(df, "Year_of_Study")
    major_comparison.to_csv(TABLE_DIR / "major_comparison.csv", index=False, encoding="utf-8-sig")
    year_comparison.to_csv(TABLE_DIR / "year_comparison.csv", index=False, encoding="utf-8-sig")

    X = df[INPUT_FEATURES].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = make_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc_ovr_macro": multiclass_auc,
    }

    cv_rows = []
    holdout_rows = []
    fitted_models: dict[str, Pipeline] = {}
    detailed_by_model: dict[str, dict[str, Any]] = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        cv_row: dict[str, Any] = {"model": model_name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            cv_row[f"{metric}_mean"] = float(np.mean(values))
            cv_row[f"{metric}_std"] = float(np.std(values, ddof=1))
        cv_rows.append(cv_row)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        model_classes = list(pipeline.classes_)
        holdout_row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "roc_auc_ovr_macro": roc_auc_score(
                y_test,
                probabilities,
                labels=model_classes,
                average="macro",
                multi_class="ovr",
            ),
            "log_loss": log_loss(y_test, probabilities, labels=model_classes),
        }
        holdout_rows.append(holdout_row)
        fitted_models[model_name] = pipeline
        detailed_by_model[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "classes": model_classes,
        }
        print(f"Completed model: {model_name}", flush=True)

    cv_results = pd.DataFrame(cv_rows).sort_values("f1_macro_mean", ascending=False)
    holdout_results = pd.DataFrame(holdout_rows).sort_values("f1_macro", ascending=False)
    cv_results.to_csv(TABLE_DIR / "metrics_cross_validation.csv", index=False, encoding="utf-8-sig")
    holdout_results.to_csv(TABLE_DIR / "metrics_holdout.csv", index=False, encoding="utf-8-sig")

    # Select only from training-fold performance; the holdout remains an unbiased final check.
    best_model_name = str(cv_results.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    joblib.dump(best_pipeline, ARTIFACT_DIR / "model.joblib", compress=3)
    best_predictions = detailed_by_model[best_model_name]["predictions"]
    best_probabilities = detailed_by_model[best_model_name]["probabilities"]
    best_classes = detailed_by_model[best_model_name]["classes"]

    report_dict = classification_report(
        y_test,
        best_predictions,
        labels=DISPLAY_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "class_or_average"})
    report_df.to_csv(TABLE_DIR / "classification_report_best.csv", index=False, encoding="utf-8-sig")

    matrix = confusion_matrix(y_test, best_predictions, labels=DISPLAY_CLASS_ORDER)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in DISPLAY_CLASS_ORDER],
        columns=[f"predicted_{label}" for label in DISPLAY_CLASS_ORDER],
    )
    matrix_df.to_csv(TABLE_DIR / "confusion_matrix_best.csv", encoding="utf-8-sig")

    y_binary = label_binarize(y_test, classes=best_classes)
    auc_rows = []
    for class_index, class_name in enumerate(best_classes):
        auc_rows.append(
            {
                "class": class_name,
                "roc_auc_ovr": roc_auc_score(
                    y_binary[:, class_index], best_probabilities[:, class_index]
                ),
            }
        )
    per_class_auc = pd.DataFrame(auc_rows)
    per_class_auc.to_csv(TABLE_DIR / "roc_auc_by_class.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Holdout Confusion Matrix — {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix_best.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Majority-class baseline for honest contextualization.
    majority_label = y_train.value_counts().idxmax()
    majority_predictions = np.repeat(majority_label, len(y_test))
    baseline = {
        "strategy": "most_frequent_training_class",
        "predicted_class": str(majority_label),
        "accuracy": accuracy_score(y_test, majority_predictions),
        "f1_macro": f1_score(
            y_test,
            majority_predictions,
            labels=DISPLAY_CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
    }

    fairness_major = fairness_table(
        y_test,
        best_predictions,
        X_test["Major_Category"],
        "Major_Category",
    )
    fairness_year = fairness_table(
        y_test,
        best_predictions,
        X_test["Year_of_Study"],
        "Year_of_Study",
    )
    fairness_major.to_csv(TABLE_DIR / "fairness_major.csv", index=False, encoding="utf-8-sig")
    fairness_year.to_csv(TABLE_DIR / "fairness_year.csv", index=False, encoding="utf-8-sig")
    major_gaps = fairness_gaps(fairness_major, "Major_Category")
    year_gaps = fairness_gaps(fairness_year, "Year_of_Study")
    fairness_payload = {
        "model": best_model_name,
        "evaluation_set": "stratified 20% holdout",
        "available_sensitive_or_group_attributes": ["Major_Category", "Year_of_Study"],
        "unavailable_attributes": {
            "Gender": "No gender field exists in the CSV; no gender fairness claim can be made.",
            "Age": "No age field exists in the CSV; no age fairness claim can be made.",
        },
        "method": (
            "Descriptive subgroup performance audit. High-risk recall is equal-opportunity style; "
            "predicted-high rate is demographic-parity style. No legal or causal fairness conclusion is implied."
        ),
        "by_major": records(fairness_major),
        "by_year": records(fairness_year),
        "gaps": [major_gaps, year_gaps],
        "limitations": [
            "Group metrics may reflect different base rates and do not identify discrimination causes.",
            "Only one random holdout split is used for subgroup reporting.",
            "No intersectional audit is possible for demographics absent from the data.",
        ],
    }
    write_json(ARTIFACT_DIR / "fairness.json", fairness_payload)

    permutation_sample = X_test.sample(n=min(3000, len(X_test)), random_state=RANDOM_STATE)
    permutation_y = y_test.loc[permutation_sample.index]
    permutation = permutation_importance(
        best_pipeline,
        permutation_sample,
        permutation_y,
        scoring="f1_macro",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": INPUT_FEATURES,
            "importance_mean_f1_drop": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean_f1_drop", ascending=False)
    permutation_df.to_csv(
        TABLE_DIR / "permutation_feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Computing SHAP explanations for: {best_model_name}", flush=True)
    shap_importance, shap_status = compute_shap_outputs(
        best_pipeline,
        best_model_name,
        X_train,
        X_test,
    )

    create_figures(df, holdout_results)

    best_cv = cv_results.loc[cv_results["model"] == best_model_name].iloc[0].to_dict()
    best_holdout = holdout_results.loc[holdout_results["model"] == best_model_name].iloc[0].to_dict()
    metrics_payload = {
        "target": TARGET,
        "class_display_order": DISPLAY_CLASS_ORDER,
        "selection_rule": "Highest 5-fold stratified CV macro-F1 on the 80% training partition",
        "selected_model": best_model_name,
        "split": {
            "random_state": RANDOM_STATE,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "stratified_by": TARGET,
            "cross_validation_folds": CV_FOLDS,
        },
        "baseline": baseline,
        "cross_validation": records(cv_results),
        "holdout": records(holdout_results),
        "best_model_cross_validation": best_cv,
        "best_model_holdout": best_holdout,
        "best_model_classification_report": _json_value(report_dict),
        "best_model_confusion_matrix": {
            "labels": DISPLAY_CLASS_ORDER,
            "matrix": matrix.tolist(),
        },
        "best_model_roc_auc_by_class": records(per_class_auc),
    }
    write_json(ARTIFACT_DIR / "metrics.json", metrics_payload)

    joblib.dump(best_pipeline, ARTIFACT_DIR / "model.joblib", compress=3)

    input_schema: dict[str, Any] = {}
    domain_ranges = {
        "Pre_Semester_GPA": (0.0, 4.0),
        "Weekly_GenAI_Hours": (0.0, 168.0),
        "Tool_Diversity": (1, 5),
        "Traditional_Study_Hours": (0.0, 168.0),
        "Perceived_AI_Dependency": (1, 10),
        "Anxiety_Level_During_Exams": (1, 10),
    }
    integer_fields = {
        "Tool_Diversity",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
    }
    for feature in INPUT_FEATURES:
        if feature in NUMERIC_FEATURES:
            minimum, maximum = domain_ranges[feature]
            input_schema[feature] = {
                "type": "integer" if feature in integer_fields else "number",
                "required": True,
                "minimum": minimum,
                "maximum": maximum,
                "observed_minimum": float(df[feature].min()),
                "observed_maximum": float(df[feature].max()),
                "observed_median": float(df[feature].median()),
            }
        elif feature == "Paid_Subscription":
            input_schema[feature] = {
                "type": "boolean",
                "required": True,
                "allowed_values": [False, True],
            }
        else:
            input_schema[feature] = {
                "type": "string",
                "required": True,
                "allowed_values": sorted(str(value) for value in df[feature].unique()),
            }

    metadata = {
        "artifact_name": "AI dependency and academic burnout early-screening pipeline",
        "artifact_file": "model.joblib",
        "model_family": best_model_name,
        "target": TARGET,
        "target_classes_model_order": list(best_pipeline.classes_),
        "target_classes_display_order": DISPLAY_CLASS_ORDER,
        "positive_safety_focus": "High",
        "input_feature_order": INPUT_FEATURES,
        "input_schema": input_schema,
        "dependency_operationalization": {
            "primary_measure": "Perceived_AI_Dependency (self-report, 1-10)",
            "bands": {
                "Low": "1-3",
                "Moderate": "4-5",
                "High": "6-10",
            },
            "high_dependency_rule": "Perceived_AI_Dependency >= 6",
            "warning": (
                "Project-specific analytical bands; they are not a validated clinical diagnosis or cutoff."
            ),
        },
        "burnout_operationalization": {
            "target_field": TARGET,
            "levels": DISPLAY_CLASS_ORDER,
            "warning": (
                "The CSV provides labels but no validated questionnaire, item wording, scoring rubric, "
                "or diagnostic gold standard; output is a dataset-level risk estimate only."
            ),
        },
        "leakage_controls": {
            "excluded_identifier": ["Student_ID"],
            "excluded_post_semester_outcomes": ["Post_Semester_GPA", "Skill_Retention_Score"],
            "reason": (
                "The web system is an early-screening tool. Post-semester GPA and skill retention "
                "would not be available at the prediction time and could leak downstream outcomes."
            ),
            "engineered_redundant_features_not_used": [
                "AI_to_Study_Ratio",
                "Total_Learning_Hours",
                "High_AI_Usage",
            ],
        },
        "data": {
            "source": "https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data",
            "local_file": DATA_PATH.name,
            "sha256": audit["source_sha256"],
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
        },
        "evaluation": {
            "selected_by": metrics_payload["selection_rule"],
            "best_model_holdout": best_holdout,
            "metrics_file": "metrics.json",
            "fairness_file": "fairness.json",
            "shap_status_file": "shap_status.json",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
        "random_state": RANDOM_STATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": (
            "Educational demonstration and early-support triage; never use as a clinical diagnosis, "
            "disciplinary decision, or sole basis for adverse action."
        ),
    }
    write_json(ARTIFACT_DIR / "metadata.json", metadata)

    burnout_distribution = (
        df[TARGET]
        .value_counts()
        .reindex(DISPLAY_CLASS_ORDER)
        .rename_axis("level")
        .reset_index(name="count")
    )
    burnout_distribution["percentage"] = burnout_distribution["count"] / len(df) * 100
    dashboard = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "burnout_distribution": records(burnout_distribution),
        "dependency_distribution": records(dependency_distribution),
        "coupling_row_percent": {
            str(index): {
                str(column): float(coupling_row_pct.loc[index, column])
                for column in DISPLAY_CLASS_ORDER
            }
            for index in coupling_row_pct.index
        },
        "key_numeric_means": {
            "weekly_genai_hours": float(df["Weekly_GenAI_Hours"].mean()),
            "traditional_study_hours": float(df["Traditional_Study_Hours"].mean()),
            "perceived_ai_dependency": float(df["Perceived_AI_Dependency"].mean()),
            "exam_anxiety": float(df["Anxiety_Level_During_Exams"].mean()),
            "pre_semester_gpa": float(df["Pre_Semester_GPA"].mean()),
            "post_semester_gpa": float(df["Post_Semester_GPA"].mean()),
            "skill_retention": float(df["Skill_Retention_Score"].mean()),
        },
        "major_comparison": records(major_comparison),
        "year_comparison": records(year_comparison),
        "model_comparison": records(holdout_results),
        "selected_model": best_model_name,
        "selected_model_holdout": best_holdout,
        "confusion_matrix": metrics_payload["best_model_confusion_matrix"],
        "top_shap_features": records(shap_importance.head(12)),
        "top_permutation_features": records(permutation_df.head(12)),
        "fairness_gaps": fairness_payload["gaps"],
    }
    write_json(ARTIFACT_DIR / "dashboard.json", dashboard)

    runtime_seconds = time.time() - started
    summary = {
        "status": "success",
        "runtime_seconds": runtime_seconds,
        "data_audit": audit,
        "dependency_distribution": records(dependency_distribution),
        "selected_model": best_model_name,
        "best_cv": best_cv,
        "best_holdout": best_holdout,
        "shap": shap_status,
        "fairness_gaps": fairness_payload["gaps"],
        "artifacts_directory": str(ARTIFACT_DIR),
    }
    write_json(SCRIPT_DIR / "analysis_summary.json", summary)

    highest_dependency = dependency_distribution.loc[
        dependency_distribution["dependency_level"] == "High (6-10)"
    ].iloc[0]
    high_burnout_count = int((df[TARGET] == "High").sum())
    high_burnout_rate = high_burnout_count / len(df)
    report = f"""# AI依赖与学业倦怠数据分析及模型评估报告

## 1. 数据与审计结论

本分析使用 Kaggle 数据文件 `{DATA_PATH.name}`，SHA-256 为 `{audit['source_sha256']}`。原始数据共 {len(df):,} 行、{df.shape[1]} 列；缺失值 {audit['total_missing_values']} 个、完全重复行 {audit['duplicate_rows']} 行、重复学生编号 {audit['duplicate_student_ids']} 个。所有预设量表/业务范围的越界值合计 {audit['expected_range_total_violations']} 个。IQR 规则累计标记 {audit['iqr_outlier_total_flags']} 个“字段值—记录”统计极端项；这些值仍在业务允许范围内，不能仅凭 IQR 自动删除。

字段定义、类型、角色和测量时点见 `tables/field_dictionary.csv`；数值统计、缺失和异常值审计分别见 `numeric_summary.csv`、`missing_values.csv` 和 `outlier_audit_iqr.csv`。

必须强调：公开数据未随 CSV 提供抽样框、问卷条目、量表信效度、标签生成方式或外部验证记录；连续编号、整齐边界和完全无缺失也提示应谨慎对待其代表性。因此，本结果只能证明模型能够学习该数据集中的统计规律，不能直接外推到真实高校总体，更不能视为临床诊断效度。

## 2. 可操作化定义

- **AI依赖**：以 `Perceived_AI_Dependency`（1–10）为主指标。为支持仪表盘分层，本项目定义 1–3 为低、4–5 为中、6–10 为高；其中高依赖共有 {int(highest_dependency['count']):,} 人，占 {float(highest_dependency['percentage']):.2f}%。这是项目分析阈值，不是经过验证的临床界值。
- **学业倦怠**：直接使用数据集提供的 `Burnout_Risk_Level`（Low/Medium/High）作为三分类标签。High 共有 {high_burnout_count:,} 人，占 {high_burnout_rate:.2%}。由于缺少量表来源和标签规则，系统输出应表述为“数据集定义的风险等级”。
- **耦合关系**：以依赖分层与倦怠等级的列联表进行描述；具体计数与组内比例见 `dependency_burnout_coupling_counts.csv` 和 `dependency_burnout_coupling_row_percent.csv`。该关系是相关而非因果。

## 3. 目标泄漏控制

Web 端模型只接收 12 个筛查时点可得字段。`Student_ID` 是标识符，已排除；`Post_Semester_GPA` 和 `Skill_Retention_Score` 是学期后结果，若用于早期预测会发生时点泄漏，因此已排除。模板 notebook 中的 `AI_to_Study_Ratio`、`Total_Learning_Hours`、`High_AI_Usage` 与原始输入高度冗余，本部署模型也不使用，以保持 API 口径清楚。

`Perceived_AI_Dependency` 与倦怠概念相关，但它在本系统中是筛查时点的自评输入而不是倦怠标签的直接复制，因此保留。若未来任务改为“预测 AI 依赖”，必须将该字段从预测特征中移除，避免直接预测自身。

## 4. 建模设计

数据以 80%/20% 分层划分训练集与独立测试集（随机种子 {RANDOM_STATE}）。模型候选为逻辑回归、随机森林和直方图梯度提升；数值变量在管线内中位数填补并标准化，类别变量众数填补并独热编码，未知类别可安全忽略。模型只依据训练部分的 {CV_FOLDS} 折分层交叉验证 Macro-F1 选择，测试集不参与选择。

### 独立测试集指标

{markdown_table(holdout_results, ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr_macro', 'log_loss'])}

最终选择 **{best_model_name}**。其训练折 Macro-F1 为 {float(best_cv['f1_macro_mean']):.4f} ± {float(best_cv['f1_macro_std']):.4f}；独立测试集 Accuracy={float(best_holdout['accuracy']):.4f}、Macro-Precision={float(best_holdout['precision_macro']):.4f}、Macro-Recall={float(best_holdout['recall_macro']):.4f}、Macro-F1={float(best_holdout['f1_macro']):.4f}、Macro OvR ROC-AUC={float(best_holdout['roc_auc_ovr_macro']):.4f}。多数类基线 Accuracy={baseline['accuracy']:.4f}、Macro-F1={baseline['f1_macro']:.4f}。

完整逐类指标和混淆矩阵见 `classification_report_best.csv`、`confusion_matrix_best.csv` 与 `figures/confusion_matrix_best.png`。上线时尤其应关注 High 类召回率和假阴性，而不能只看总体准确率。

## 5. 可解释性

已对最终模型执行 **{shap_status['method']}**，解释 {shap_status['explained_rows']} 条固定抽取的测试样本；另保存 {shap_status['background_rows']} 条转换后背景样本供部署时解释接口使用。本次随机森林全局审计采用 TreeExplainer 的 tree-path-dependent 内部背景。全局 SHAP 排名见 `shap_feature_importance.csv` 和 `figures/shap_global_importance.png`；原始 12 字段的置换重要性作为独立核验，见 `permutation_feature_importance.csv`。SHAP 表示模型在当前数据中的预测贡献，不表示因素会因果地导致倦怠。

## 6. 专业、年级与公平性

专业和年级的依赖率、倦怠率、二者共同高风险比例及均值比较见 `major_comparison.csv`、`year_comparison.csv`。公平性审计在同一独立测试集上报告各组 Accuracy、Macro-F1、High 类召回率/假阴性率/假阳性率和预测 High 比例，见 `fairness_major.csv`、`fairness_year.csv` 与 `artifacts/fairness.json`。

数据中**不存在性别和年龄字段**，所以不能严谨开展性别/年龄公平性测试，也不能声称模型对这些群体公平。现有分组差异只是描述性性能审计，不等于法律意义或因果意义上的公平结论。

## 7. 部署交付

`artifacts/model.joblib` 是包含填补、标准化、独热编码和分类器的完整管线；FastAPI 应按 `artifacts/metadata.json` 的 12 字段顺序与校验规则构造单行 DataFrame，然后调用 `predict_proba`。指标、仪表盘数据、公平性、SHAP 背景和转换后字段名分别位于 `metrics.json`、`dashboard.json`、`fairness.json`、`shap_background.npy`、`transformed_feature_names.json`。

系统必须展示“教育支持性筛查，非医学诊断、非惩罚性决策”的用途边界。真实上线前需要使用目标学校的前瞻性数据做外部验证、概率校准、阈值共创、隐私影响评估和持续漂移监控。

---

脚本运行耗时：{runtime_seconds:.1f} 秒。生成时间（UTC）：{metadata['created_at_utc']}。
"""
    (SCRIPT_DIR / "数据分析与模型评估报告.md").write_text(report, encoding="utf-8")

    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
