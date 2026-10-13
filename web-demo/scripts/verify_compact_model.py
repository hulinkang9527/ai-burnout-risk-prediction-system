"""Verify probability parity between sklearn and the edge JSON model."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_FEATURES = [
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


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    epsilon = 0.25
    ai = result["Weekly_GenAI_Hours"].astype(float)
    study = result["Traditional_Study_Hours"].astype(float)
    dependency = result["Perceived_AI_Dependency"].astype(float)
    anxiety = result["Anxiety_Level_During_Exams"].astype(float)
    total = ai + study
    result["AI_to_Study_Ratio"] = ai / (study + epsilon)
    result["Total_Learning_Hours"] = total
    result["AI_Hours_Share"] = ai / (total + epsilon)
    result["AI_minus_Study_Hours"] = ai - study
    result["Dependency_x_AI_Hours"] = dependency * ai
    result["Anxiety_per_Study_Hour"] = anxiety / (study + epsilon)
    result["Anxiety_x_Dependency"] = anxiety * dependency
    result["Anxiety_plus_Dependency"] = anxiety + dependency
    return result


def edge_probability(row: pd.Series, model: dict) -> float:
    values = []
    for index, feature in enumerate(model["numeric"]["features"]):
        value = float(row[feature])
        if not math.isfinite(value):
            value = float(model["numeric"]["median"][index])
        scale = float(model["numeric"]["scale"][index]) or 1.0
        values.append((value - float(model["numeric"]["mean"][index])) / scale)
    for index, feature in enumerate(model["categorical"]["features"]):
        supplied = row[feature]
        if pd.isna(supplied):
            supplied = model["categorical"]["fallback"][index]
        values.extend(
            1.0 if str(supplied) == str(category) else 0.0
            for category in model["categorical"]["categories"][index]
        )
    logit = float(model["linear"]["intercept"]) + float(
        np.dot(
            np.asarray(values, dtype=np.float64),
            np.asarray(model["linear"]["coefficients"], dtype=np.float64),
        )
    )
    return 1.0 / (1.0 + math.exp(-logit))


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: verify_compact_model.py MODEL.joblib COMPACT.json "
            "DATASET.csv SAMPLE_SIZE"
        )
    pipeline = joblib.load(Path(sys.argv[1]).resolve())
    compact = json.loads(Path(sys.argv[2]).resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv(Path(sys.argv[3]).resolve())
    sample_size = int(sys.argv[4])

    _, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=42,
        stratify=frame["Burnout_Risk_Level"],
    )
    sample = test.sample(
        n=min(sample_size, len(test)),
        random_state=20260728,
    )
    engineered = add_derived_features(sample[RAW_FEATURES])
    python_probabilities = pipeline.predict_proba(engineered)[:, 1]
    edge_probabilities = np.asarray(
        [edge_probability(row, compact) for _, row in engineered.iterrows()]
    )
    maximum_error = float(
        np.max(np.abs(python_probabilities - edge_probabilities))
    )
    if maximum_error > 1e-9:
        raise RuntimeError(
            f"Compact model probability error {maximum_error} exceeded 1e-9."
        )

    non_high_threshold = float(
        compact["decision"]["nonHighProbabilityThreshold"]
    )
    high_threshold = float(compact["decision"]["highProbabilityThreshold"])
    python_decisions = np.where(
        python_probabilities >= high_threshold,
        "High",
        np.where(
            python_probabilities <= non_high_threshold,
            "Non-high",
            "Uncertain",
        ),
    )
    edge_decisions = np.where(
        edge_probabilities >= high_threshold,
        "High",
        np.where(
            edge_probabilities <= non_high_threshold,
            "Non-high",
            "Uncertain",
        ),
    )
    mismatches = int(np.sum(python_decisions != edge_decisions))
    if mismatches:
        raise RuntimeError(f"Compact model produced {mismatches} decision mismatches.")

    print(
        json.dumps(
            {
                "checked_rows": int(len(sample)),
                "maximum_probability_error": maximum_error,
                "decision_mismatches": mismatches,
                "model_version": compact["modelVersion"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
