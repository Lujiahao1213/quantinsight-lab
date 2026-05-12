import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = [
    "Daily_Return",
    "MA_5",
    "MA_20",
    "MA_60",
    "Volatility_20",
    "Volume",
    "Close",
]

SUPPORTED_MODELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "knn": "KNN",
}

DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "decision_tree": {"max_depth": 5, "min_samples_split": 2},
    "random_forest": {"n_estimators": 100, "max_depth": 5},
    "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    "knn": {"n_neighbors": 5, "weights": "uniform"},
}


def get_available_features(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def prepare_ml_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    if len(df) < 30:
        raise ValueError("Dataset must have at least 30 rows for ML analysis.")
    if "Close" not in df.columns:
        raise ValueError("Close column is required for ML analysis.")

    features = get_available_features(df)
    if not features:
        raise ValueError("No supported ML feature columns are available.")

    out = df.copy()
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["Next_Close"] = out["Close"].shift(-1)
    out["Target"] = (out["Next_Close"] > out["Close"]).astype(int)
    out = out.iloc[:-1].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=features + ["Target"])

    if len(out) < 30:
        raise ValueError("Not enough valid rows remain after dropping NaN values.")
    if out["Target"].nunique() < 2:
        raise ValueError("ML target needs both upward and downward next-day moves.")

    return out[features], out["Target"], out, features


def split_time_series(X: pd.DataFrame, y: pd.Series):
    split_idx = int(len(X) * 0.8)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Not enough rows to create a train/test split.")
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def build_model(model_name: str, params: dict, train_size: int):
    if model_name == "logistic_regression":
        C = float(params.get("C", DEFAULT_PARAMS[model_name]["C"]))
        max_iter = int(params.get("max_iter", DEFAULT_PARAMS[model_name]["max_iter"]))
        if C <= 0 or max_iter <= 0:
            raise ValueError("Logistic Regression C and max_iter must be positive.")
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=C, max_iter=max_iter)),
            ]
        )

    if model_name == "decision_tree":
        max_depth = int(params.get("max_depth", DEFAULT_PARAMS[model_name]["max_depth"]))
        min_samples_split = int(
            params.get("min_samples_split", DEFAULT_PARAMS[model_name]["min_samples_split"])
        )
        if max_depth <= 0 or min_samples_split < 2:
            raise ValueError("Decision Tree max_depth must be positive and min_samples_split must be at least 2.")
        return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

    if model_name == "random_forest":
        n_estimators = int(params.get("n_estimators", DEFAULT_PARAMS[model_name]["n_estimators"]))
        max_depth = int(params.get("max_depth", DEFAULT_PARAMS[model_name]["max_depth"]))
        if n_estimators <= 0 or max_depth <= 0:
            raise ValueError("Random Forest n_estimators and max_depth must be positive.")
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    if model_name == "svm":
        C = float(params.get("C", DEFAULT_PARAMS[model_name]["C"]))
        kernel = params.get("kernel", DEFAULT_PARAMS[model_name]["kernel"])
        gamma = params.get("gamma", DEFAULT_PARAMS[model_name]["gamma"])
        if C <= 0:
            raise ValueError("SVM C must be positive.")
        if kernel not in {"linear", "rbf", "poly", "sigmoid"}:
            raise ValueError("SVM kernel must be linear, rbf, poly, or sigmoid.")
        if gamma not in {"scale", "auto"}:
            raise ValueError("SVM gamma must be scale or auto.")
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=C, kernel=kernel, gamma=gamma)),
            ]
        )

    if model_name == "knn":
        n_neighbors = int(params.get("n_neighbors", DEFAULT_PARAMS[model_name]["n_neighbors"]))
        weights = params.get("weights", DEFAULT_PARAMS[model_name]["weights"])
        if n_neighbors <= 0:
            raise ValueError("KNN n_neighbors must be positive.")
        if n_neighbors >= train_size:
            raise ValueError("KNN n_neighbors must be smaller than the training sample size.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("KNN weights must be uniform or distance.")
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
            ]
        )

    raise ValueError("Selected model is not supported.")


def _inner_model(model):
    return model.named_steps["model"] if hasattr(model, "named_steps") else model


def create_confusion_matrix_chart(matrix) -> str:
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Down", "Up"],
        y=["Down", "Up"],
        title="Confusion Matrix",
    )
    fig.update_layout(template="plotly_dark")
    return fig.to_html(full_html=False)


def create_metrics_chart(metrics: dict) -> str:
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    fig = px.bar(
        x=[name.replace("_", " ").title() for name in metric_names],
        y=[metrics[name] for name in metric_names],
        title="Model Metrics",
        labels={"x": "Metric", "y": "Score"},
        range_y=[0, 1],
    )
    fig.update_layout(template="plotly_dark")
    return fig.to_html(full_html=False)


def create_feature_importance_chart(model, features: list[str]) -> str | None:
    estimator = _inner_model(model)
    values = None

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])

    if values is None:
        return None

    fig = px.bar(
        x=features,
        y=values,
        title="Feature Importance",
        labels={"x": "Feature", "y": "Importance"},
    )
    fig.update_layout(template="plotly_dark")
    return fig.to_html(full_html=False)


def _direction_counts(values) -> dict:
    series = pd.Series(values)
    return {
        "down": int((series == 0).sum()),
        "up": int((series == 1).sum()),
    }


def _direction_label(value) -> str:
    return "Up" if int(value) == 1 else "Down"


def run_ml_analysis(df: pd.DataFrame, model_name: str, params: dict) -> dict:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Selected model is not supported.")

    X, y, prepared_df, features = prepare_ml_dataset(df)
    X_train, X_test, y_train, y_test = split_time_series(X, y)
    if y_train.nunique() < 2:
        raise ValueError("Training data needs both upward and downward next-day moves.")
    model = build_model(model_name, params, len(X_train))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    prediction_counts = _direction_counts(predictions)
    test_distribution = _direction_counts(y_test)
    warnings = []

    if len(set(predictions)) == 1:
        warnings.append(
            "The model predicted only one class. This may indicate class imbalance or weak predictive power."
        )
    if y_test.nunique() == 1:
        warnings.append(
            "The test set contains only one class. Evaluation metrics may be misleading."
        )

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    preview_df = prepared_df.loc[X_test.index].copy()
    preview_df["Actual Direction"] = [_direction_label(value) for value in y_test.values]
    preview_df["Predicted Direction"] = [_direction_label(value) for value in predictions]
    if "Date" in preview_df.columns:
        preview_df["Date"] = preview_df["Date"].astype(str)

    preview_cols = [
        col
        for col in ["Date", "Close", "Actual Direction", "Predicted Direction"]
        if col in preview_df.columns
    ]

    return {
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "target_distribution": _direction_counts(y),
        "train_distribution": _direction_counts(y_train),
        "test_distribution": test_distribution,
        "prediction_distribution": prediction_counts,
        "warnings": warnings,
        "charts": {
            "confusion_matrix": create_confusion_matrix_chart(matrix),
            "metrics": create_metrics_chart(metrics),
            "feature_importance": create_feature_importance_chart(model, features),
        },
        "features": features,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "preview_rows": preview_df[preview_cols].tail(20).to_dict(orient="records"),
        "table_columns": preview_cols,
    }
