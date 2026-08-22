from __future__ import annotations

import json
import platform
import subprocess
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "experiment.json"
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"
MODELS = RESULTS / "models"

EXPECTED_ROWS = 45211
TARGET = "y"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ensure_dirs() -> None:
    for path in (DATA_DIR, METRICS, FIGURES, PREDICTIONS, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str) -> Path:
    ensure_dirs()
    csv_path = DATA_DIR / "bank-full.csv"
    if csv_path.exists():
        return csv_path

    zip_path = DATA_DIR / "bank+marketing.zip"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        zip_path.write_bytes(response.read())

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith("bank-full.csv")]
        if not members:
            raise FileNotFoundError("Official UCI archive did not contain bank-full.csv")
        with archive.open(members[0]) as src, csv_path.open("wb") as dst:
            dst.write(src.read())
    return csv_path


def load_data(csv_path: Path, excluded_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path, sep=";")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected dataset size: {len(df)} != {EXPECTED_ROWS}")
    if TARGET not in df.columns:
        raise ValueError("Target column 'y' is missing")

    y = (df[TARGET].astype(str).str.lower() == "yes").astype(int)
    drop_cols = [TARGET] + [c for c in excluded_features if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    if X.empty:
        raise ValueError("No features remain after exclusions")
    return X, y


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def fit_models(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict) -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("prep", build_preprocessor(X_train, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=cfg["random_seed"],
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        [
            ("prep", build_preprocessor(X_train, scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=cfg["rf_n_estimators"],
                    min_samples_leaf=cfg["rf_min_samples_leaf"],
                    max_features=cfg["rf_max_features"],
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=cfg["random_seed"],
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    return {"logistic_regression": logistic, "random_forest": random_forest}


def fit_isotonic_calibrator(probabilities: np.ndarray, y: pd.Series) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, y.to_numpy())
    return calibrator


def calibrated_probability(model: Pipeline, calibrator: IsotonicRegression, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    return np.asarray(calibrator.predict(raw), dtype=float)


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "alert_rate": float(pred.mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calibration_ece(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1 else prob <= high)
        if not np.any(mask):
            continue
        confidence = prob[mask].mean()
        accuracy = y[mask].mean()
        ece += (mask.sum() / total) * abs(accuracy - confidence)
    return float(ece)


def utility_if_action(
    prob: np.ndarray,
    benefit_if_success: float,
    action_cost: float,
    missed_opportunity_cost: float,
) -> np.ndarray:
    # Expected value of taking the action relative to doing nothing.
    # Action outcome: success earns benefit; unsuccessful action incurs cost.
    # No-action outcome: positive case incurs missed-opportunity cost.
    action_value = prob * benefit_if_success - action_cost
    no_action_value = -(prob * missed_opportunity_cost)
    return action_value - no_action_value


def choose_threshold(
    y_decision: np.ndarray,
    p_decision: np.ndarray,
    cfg: dict,
    benefit: float,
    action_cost: float,
    missed_cost: float,
) -> tuple[float, pd.DataFrame]:
    thresholds = np.round(
        np.arange(
            cfg["decision_grid_min"],
            cfg["decision_grid_max"] + cfg["decision_grid_step"] / 2,
            cfg["decision_grid_step"],
        ),
        2,
    )
    rows = []
    for threshold in thresholds:
        action = (p_decision >= threshold).astype(int)
        value = utility_if_action(p_decision, benefit, action_cost, missed_cost)
        rows.append(
            {
                "threshold": float(threshold),
                "action_rate": float(action.mean()),
                "expected_incremental_value_per_100": float((value * action).mean() * 100),
                "precision": float(precision_score(y_decision, action, zero_division=0)),
                "recall": float(recall_score(y_decision, action, zero_division=0)),
                "f1": float(f1_score(y_decision, action, zero_division=0)),
                "is_primary": False,
            }
        )
    table = pd.DataFrame(rows)
    best_idx = int(table["expected_incremental_value_per_100"].idxmax())
    table.loc[best_idx, "is_primary"] = True
    return float(table.loc[best_idx, "threshold"]), table


def explain_random_forest(model: Pipeline, X_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = model.named_steps["prep"]
    estimator = model.named_steps["model"]
    transformed = prep.transform(X_sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = prep.get_feature_names_out()
    explainer = shap.TreeExplainer(estimator)
    raw = explainer.shap_values(transformed)
    if isinstance(raw, list):
        values = np.asarray(raw[1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, 1]
    values = np.asarray(values, dtype=float)
    if values.shape[1] != len(feature_names):
        raise ValueError("SHAP feature dimension does not match transformed feature names")

    global_importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return global_importance, pd.DataFrame(values, columns=feature_names)


def save_figures(y_test: np.ndarray, test_prob: np.ndarray, threshold_table: pd.DataFrame, shap_importance: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC={roc_auc_score(y_test, test_prob):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curve.png", dpi=220)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, test_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "calibration_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_table["threshold"], threshold_table["expected_incremental_value_per_100"], marker="o")
    primary = threshold_table.loc[threshold_table["is_primary"], "threshold"]
    if not primary.empty:
        plt.axvline(float(primary.iloc[0]), linestyle="--", label="Primary threshold")
    plt.xlabel("Decision threshold")
    plt.ylabel("Expected incremental value per 100 decisions")
    plt.title("Risk-Aware Decision Utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "decision_utility_curve.png", dpi=220)
    plt.close()

    top = shap_importance.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global Explainability: Top Features")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_global_importance.png", dpi=220)
    plt.close()


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    ensure_dirs()
    csv_path = download_dataset(cfg["dataset_url"])
    X, y = load_data(csv_path, cfg["excluded_features"])

    train_fraction = 1.0 - cfg["test_size"] - cfg["calibration_size"] - cfg["decision_size"]
    if train_fraction <= 0:
        raise ValueError("Training fraction must be positive")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=1.0 - train_fraction,
        stratify=y,
        random_state=cfg["random_seed"],
    )
    remaining = cfg["calibration_size"] + cfg["decision_size"] + cfg["test_size"]
    cal_ratio = cfg["calibration_size"] / remaining
    dec_ratio = cfg["decision_size"] / (cfg["decision_size"] + cfg["test_size"])
    X_cal, X_tmp2, y_cal, y_tmp2 = train_test_split(
        X_tmp, y_tmp,
        test_size=1.0 - cal_ratio,
        stratify=y_tmp,
        random_state=cfg["random_seed"],
    )
    X_dec, X_test, y_dec, y_test = train_test_split(
        X_tmp2, y_tmp2,
        test_size=1.0 - dec_ratio,
        stratify=y_tmp2,
        random_state=cfg["random_seed"],
    )

    models = fit_models(X_train, y_train, cfg)
    calibrators: dict[str, IsotonicRegression] = {}
    for name, model in models.items():
        calibrators[name] = fit_isotonic_calibrator(model.predict_proba(X_cal)[:, 1], y_cal)

    test_results = []
    calibrated_test_probs = {}
    calibrated_decision_probs = {}
    for name, model in models.items():
        p_dec = calibrated_probability(model, calibrators[name], X_dec)
        p_test = calibrated_probability(model, calibrators[name], X_test)
        calibrated_decision_probs[name] = p_dec
        calibrated_test_probs[name] = p_test
        base = binary_metrics(y_test.to_numpy(), p_test, 0.5)
        base["model"] = name
        base["calibration_ece"] = calibration_ece(y_test.to_numpy(), p_test)
        test_results.append(base)

    primary_name = "random_forest"
    primary_model = models[primary_name]
    p_dec = calibrated_decision_probs[primary_name]
    p_test = calibrated_test_probs[primary_name]
    benefits = cfg["utility_scenarios"]["benefit_if_success"]
    action_costs = cfg["utility_scenarios"]["action_cost"]
    missed_costs = cfg["utility_scenarios"]["missed_opportunity_cost"]
    base_benefit = float(benefits[1])
    base_action_cost = float(action_costs[1])
    base_missed_cost = float(missed_costs[1])
    primary_threshold, threshold_table = choose_threshold(
        y_dec.to_numpy(), p_dec, cfg, base_benefit, base_action_cost, base_missed_cost
    )

    primary_pred = (p_test >= primary_threshold).astype(int)
    primary_metrics = binary_metrics(y_test.to_numpy(), p_test, primary_threshold)
    primary_metrics.update(
        {
            "model": primary_name,
            "decision_threshold": primary_threshold,
            "calibration_ece": calibration_ece(y_test.to_numpy(), p_test),
            "expected_incremental_value_per_100": float(
                (utility_if_action(p_test, base_benefit, base_action_cost, base_missed_cost) * primary_pred).mean() * 100
            ),
        }
    )

    capacity = cfg["top_k_rate"]
    top_k = max(1, int(round(len(p_test) * capacity)))
    order = np.argsort(-p_test)
    top_k_pred = np.zeros(len(p_test), dtype=int)
    top_k_pred[order[:top_k]] = 1
    top_k_metrics = {
        "top_k_rate": capacity,
        "top_k_count": top_k,
        "precision": float(precision_score(y_test, top_k_pred, zero_division=0)),
        "capture_rate": float(y_test.to_numpy()[top_k_pred == 1].sum() / y_test.sum()),
        "expected_incremental_value_per_100": float(
            (utility_if_action(p_test, base_benefit, base_action_cost, base_missed_cost) * top_k_pred).mean() * 100
        ),
    }

    scenario_rows = []
    for benefit in benefits:
        for cost in action_costs:
            for missed in missed_costs:
                threshold, _ = choose_threshold(y_dec.to_numpy(), p_dec, cfg, float(benefit), float(cost), float(missed))
                pred = (p_test >= threshold).astype(int)
                scenario_rows.append(
                    {
                        "benefit_if_success": benefit,
                        "action_cost": cost,
                        "missed_opportunity_cost": missed,
                        "decision_threshold": threshold,
                        "action_rate": float(pred.mean()),
                        "precision": float(precision_score(y_test, pred, zero_division=0)),
                        "recall": float(recall_score(y_test, pred, zero_division=0)),
                        "f1": float(f1_score(y_test, pred, zero_division=0)),
                        "expected_incremental_value_per_100": float(
                            (utility_if_action(p_test, float(benefit), float(cost), float(missed)) * pred).mean() * 100
                        ),
                    }
                )

    sample_n = min(cfg["xai_sample_size"], len(X_test))
    X_xai = X_test.iloc[:sample_n].copy()
    shap_importance, shap_values = explain_random_forest(primary_model, X_xai)

    # Local explanations: top five samples by calibrated probability, with strongest contributions.
    local_prob = p_test[:sample_n]
    local_idx = np.argsort(-local_prob)[:5]
    local_rows = []
    for idx in local_idx:
        row = shap_values.iloc[idx]
        strongest = row.abs().sort_values(ascending=False).head(5).index
        for feature in strongest:
            local_rows.append(
                {
                    "sample_index": int(idx),
                    "predicted_probability": float(local_prob[idx]),
                    "feature": feature,
                    "shap_value": float(row[feature]),
                    "actual_outcome": int(y_test.iloc[idx]),
                }
            )

    # Save machine-readable outputs.
    pd.DataFrame(test_results).to_csv(METRICS / "model_comparison.csv", index=False)
    pd.DataFrame([primary_metrics]).to_csv(METRICS / "primary_decision_metrics.csv", index=False)
    threshold_table.to_csv(METRICS / "decision_threshold_sensitivity.csv", index=False)
    pd.DataFrame([top_k_metrics]).to_csv(METRICS / "capacity_policy_metrics.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(METRICS / "utility_scenario_sensitivity.csv", index=False)
    shap_importance.to_csv(METRICS / "shap_global_importance.csv", index=False)
    shap_values.to_csv(METRICS / "shap_values_sample.csv", index=False)
    pd.DataFrame(local_rows).to_csv(METRICS / "local_explanations.csv", index=False)

    split_table = pd.DataFrame(
        {
            "split": ["train", "calibration", "decision_tuning", "test"],
            "rows": [len(X_train), len(X_cal), len(X_dec), len(X_test)],
            "positive_rate": [float(y_train.mean()), float(y_cal.mean()), float(y_dec.mean()), float(y_test.mean())],
        }
    )
    split_table.to_csv(METRICS / "dataset_splits.csv", index=False)

    predictions = X_test.copy()
    predictions["actual_outcome"] = y_test.to_numpy()
    predictions["predicted_probability"] = p_test
    predictions["recommended_action"] = primary_pred
    predictions["expected_incremental_value"] = utility_if_action(
        p_test, base_benefit, base_action_cost, base_missed_cost
    )
    predictions.to_csv(PREDICTIONS / "test_decisions.csv", index=False)

    joblib.dump(primary_model, MODELS / "random_forest.joblib")
    joblib.dump(calibrators[primary_name], MODELS / "isotonic_calibrator.joblib")

    dataset_summary = {
        "dataset": cfg["dataset_name"],
        "source": "UCI Machine Learning Repository",
        "source_url": cfg["dataset_url"],
        "doi": cfg["dataset_doi"],
        "rows": int(len(X)),
        "raw_input_columns": int(len(pd.read_csv(csv_path, sep=";").columns) - 1),
        "modeled_columns": int(X.shape[1]),
        "excluded_features": cfg["excluded_features"],
        "positive_rate": float(y.mean()),
    }
    (METRICS / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

    manifest = {
        "research_title": cfg["research_title"],
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset_summary": dataset_summary,
        "split_sizes": split_table.to_dict(orient="records"),
        "primary_model": primary_name,
        "primary_decision_metrics": primary_metrics,
        "base_utility_scenario": {
            "benefit_if_success": base_benefit,
            "action_cost": base_action_cost,
            "missed_opportunity_cost": base_missed_cost,
        },
        "top_k_policy": top_k_metrics,
        "result_files": [
            "results/metrics/model_comparison.csv",
            "results/metrics/primary_decision_metrics.csv",
            "results/metrics/decision_threshold_sensitivity.csv",
            "results/metrics/capacity_policy_metrics.csv",
            "results/metrics/utility_scenario_sensitivity.csv",
            "results/metrics/shap_global_importance.csv",
            "results/metrics/local_explanations.csv",
            "results/metrics/dataset_splits.csv",
            "results/metrics/dataset_summary.json",
            "results/predictions/test_decisions.csv",
            "results/figures/roc_curve.png",
            "results/figures/calibration_curve.png",
            "results/figures/decision_utility_curve.png",
            "results/figures/shap_global_importance.png",
        ],
    }
    (METRICS / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    save_figures(y_test.to_numpy(), p_test, threshold_table, shap_importance)

    print(json.dumps(primary_metrics, indent=2))
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
