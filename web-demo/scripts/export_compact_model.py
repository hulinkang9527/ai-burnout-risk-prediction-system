"""Export the canonical selective logistic model for the Sites edge runtime.

This script never retrains a second model. It serializes the fitted sklearn
pipeline and its audited metrics so the web API and page share one source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in list(value)]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def rounded(values: Any, digits: int = 12) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values).ravel()]


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: export_compact_model.py MODEL.joblib METRICS.json "
            "COMPACT_OUTPUT.json SUMMARY_OUTPUT.json"
        )

    model_path = Path(sys.argv[1]).resolve()
    metrics_path = Path(sys.argv[2]).resolve()
    compact_output = Path(sys.argv[3]).resolve()
    summary_output = Path(sys.argv[4]).resolve()

    pipeline = joblib.load(model_path)
    audited = json.loads(metrics_path.read_text(encoding="utf-8"))
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]

    if list(classifier.classes_) != [0, 1]:
        raise RuntimeError("Expected binary classes [0, 1].")

    numeric_transformer = preprocessor.named_transformers_["num"]
    categorical_transformer = preprocessor.named_transformers_["cat"]
    numeric_imputer = numeric_transformer.named_steps["imputer"]
    scaler = numeric_transformer.named_steps["scaler"]
    categorical_imputer = categorical_transformer.named_steps["imputer"]
    encoder = categorical_transformer.named_steps["onehot"]

    numeric_features = list(audited["model"]["numeric_features"])
    categorical_features = list(audited["model"]["categorical_features"])
    expected_width = len(numeric_features) + sum(
        len(categories) for categories in encoder.categories_
    )
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
    if len(coefficients) != expected_width:
        raise RuntimeError(
            f"Coefficient width {len(coefficients)} != exported width {expected_width}."
        )

    test_all = audited["test"]["all_predictions"]
    test_selective = audited["test"]["selective_predictions"]
    validation_selective = audited["validation"]["selective_predictions"]
    non_high_threshold = float(
        audited["selection"]["non_high_probability_threshold"]
    )
    high_threshold = float(audited["selection"]["high_probability_threshold"])
    metrics = {
        "overall_accuracy": float(test_all["accuracy"]),
        "overall_macro_f1": float(test_all["f1_macro"]),
        "roc_auc": float(test_all["roc_auc"]),
        "selective_accuracy": float(test_selective["accuracy"]),
        "selective_macro_f1": float(test_selective["f1_macro"]),
        "selective_balanced_accuracy": float(
            test_selective["balanced_accuracy"]
        ),
        "selective_coverage": float(test_selective["coverage"]),
        "abstention_rate": float(test_selective["abstention_rate"]),
        "accepted_n": int(test_selective["accepted_n"]),
        "selective_wilson_lower_95": float(
            test_selective["accuracy_wilson_95"][0]
        ),
        "selective_wilson_upper_95": float(
            test_selective["accuracy_wilson_95"][1]
        ),
        "high_precision": float(test_selective["high_precision"]),
        "high_recall_within_accepted": float(
            test_selective["high_recall_within_accepted"]
        ),
        "auto_high_recall_all_high": float(
            test_selective["auto_high_recall_all_high"]
        ),
        "validation_selective_accuracy": float(validation_selective["accuracy"]),
        "validation_wilson_lower_95": float(
            validation_selective["accuracy_wilson_95"][0]
        ),
        "legacy_three_class_accuracy": float(
            audited["comparators"]["legacy_three_class_random_forest_accuracy"]
        ),
        "legacy_three_class_macro_f1": float(
            audited["comparators"]["legacy_three_class_random_forest_macro_f1"]
        ),
    }

    categorical_fallback = []
    for value in categorical_imputer.statistics_:
        if isinstance(value, (np.bool_, bool)):
            categorical_fallback.append(bool(value))
        else:
            categorical_fallback.append(str(value))

    categories = []
    for category_values in encoder.categories_:
        categories.append(
            [
                bool(value)
                if isinstance(value, (np.bool_, bool))
                else str(value)
                for value in category_values
            ]
        )

    compact = {
        "modelVersion": audited["model_version"],
        "trainedRows": audited["split"]["train_rows"],
        "validationRows": audited["split"]["validation_rows"],
        "holdoutRows": audited["split"]["test_rows"],
        "task": {
            "positiveClass": "High",
            "negativeClass": "Non-high",
            "reviewClass": "Uncertain",
            "type": "High-versus-non-high selective binary classification",
        },
        "decision": {
            "nonHighProbabilityThreshold": non_high_threshold,
            "highProbabilityThreshold": high_threshold,
            "reviewLowerProbabilityExclusive": non_high_threshold,
            "reviewUpperProbabilityExclusive": high_threshold,
            "rule": audited["selection"]["decision_rule"],
            "selectedOn": "validation only",
            "criterion": audited["selection"]["selection_criterion"],
        },
        "metrics": metrics,
        "rawInputFeatures": audited["model"]["raw_input_features"],
        "derivedFeatures": audited["model"]["derived_features"],
        "numeric": {
            "features": numeric_features,
            "median": rounded(numeric_imputer.statistics_),
            "mean": rounded(scaler.mean_),
            "scale": rounded(scaler.scale_),
        },
        "categorical": {
            "features": categorical_features,
            "fallback": categorical_fallback,
            "categories": categories,
        },
        "linear": {
            "coefficients": rounded(coefficients),
            "intercept": round(float(classifier.intercept_[0]), 12),
            "link": "logit",
        },
        "shap": audited["shap"],
        "stabilityAudit": audited["stability_audit"],
        "safety": {
            "intendedUse": "Educational early-support screening only",
            "disclaimer": (
                "仅用于学习研究和教育支持参考，不构成医学或心理诊断，"
                "也不得作为处分、标签化或其他不利决定的唯一依据。"
            ),
        },
    }

    summary = {
        "modelVersion": audited["model_version"],
        "modelName": "高风险选择性逻辑回归",
        "taskLabel": "High vs 非 High（低置信样本人工复核）",
        "trainedRows": audited["split"]["train_rows"],
        "validationRows": audited["split"]["validation_rows"],
        "holdoutRows": audited["split"]["test_rows"],
        "thresholds": {
            "nonHighProbabilityMaximum": non_high_threshold,
            "highProbabilityMinimum": high_threshold,
        },
        "metrics": metrics,
        "stability": {
            key: value
            for key, value in audited["stability_audit"].items()
            if key != "runs"
        },
        "shapFeatures": audited["shap"]["features"],
        "limitations": audited["limitations"],
    }

    for output, payload, compact_json in [
        (compact_output, compact, True),
        (summary_output, summary, False),
    ]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                json_value(payload),
                ensure_ascii=False,
                separators=(",", ":") if compact_json else None,
                indent=None if compact_json else 2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "model_version": audited["model_version"],
                "compact_output": str(compact_output),
                "compact_bytes": compact_output.stat().st_size,
                "summary_output": str(summary_output),
                **metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
