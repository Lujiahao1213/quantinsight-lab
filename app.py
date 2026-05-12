import math
import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, session, url_for
import pandas as pd
import plotly.express as px
from werkzeug.utils import secure_filename

from config import Config
from modules.backtester import run_backtest
from modules.buy_score import compute_buy_score
from modules.data_agent import DataAgent
from modules.data_cleaner import clean_financial_data
from modules.data_loader import load_file
from modules.indicators import add_basic_indicators
from modules.ml_engine import DEFAULT_PARAMS, SUPPORTED_MODELS, get_available_features, run_ml_analysis
from modules.ml_predictor import run_rf_direction_predictor
from modules.strategies import apply_strategy
from modules.visualizer import (
    create_close_price_chart,
    create_drawdown_chart,
    create_moving_average_chart,
    create_price_with_signals_chart,
    create_return_histogram,
    create_strategy_comparison_equity_chart,
    create_strategy_comparison_metrics_chart,
    create_strategy_equity_chart,
    create_strategy_vs_market_chart,
    create_volume_chart,
)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
APP_BOOT_ID = uuid4().hex


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


def _display_metric(value, percent=False):
    safe = _safe_float(value)
    if safe is None:
        return "N/A"
    return f"{safe:.4f}" if not percent else f"{safe:.2%}"


def _safe_metric_text(value):
    safe = _safe_float(value)
    return "N/A" if safe is None else f"{safe:.4f}"


def _ml_comparison_chart_from_rows(rows: list) -> str:
    """Rebuild Plotly bar HTML from compact comparison rows (session-safe)."""
    if not rows:
        return "<div>No valid model comparison data available.</div>"

    def _parse_score(value):
        if value in (None, "N/A", ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    chart_rows = []
    for row in rows:
        model = row.get("model", "Unknown")
        chart_rows.extend(
            [
                {
                    "Model": model,
                    "Metric": "Accuracy",
                    "Score": _parse_score(row.get("accuracy")),
                },
                {
                    "Model": model,
                    "Metric": "Precision",
                    "Score": _parse_score(row.get("precision")),
                },
                {
                    "Model": model,
                    "Metric": "Recall",
                    "Score": _parse_score(row.get("recall")),
                },
                {
                    "Model": model,
                    "Metric": "F1-score",
                    "Score": _parse_score(row.get("f1_score")),
                },
            ]
        )
    compare_df = pd.DataFrame(chart_rows)
    if compare_df.empty:
        return "<div>No valid model comparison data available.</div>"
    comparison_chart = px.bar(
        compare_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Comparison Metrics",
        range_y=[0, 1],
    )
    comparison_chart.update_layout(template="plotly_dark")
    return comparison_chart.to_html(full_html=False)


def _build_decision_narrative(buy_signal: dict, rf_predictor: dict) -> str:
    """Short summary combining rule-based score and RF outlook (display only)."""
    chunks: list[str] = []

    if buy_signal.get("ok"):
        score = buy_signal.get("buy_score")
        rec = buy_signal.get("recommendation", "Neutral")
        neg_txt = " ".join(str(x) for x in (buy_signal.get("negative_signals") or [])).lower()
        pos_txt = " ".join(str(x) for x in (buy_signal.get("positive_signals") or [])).lower()

        tail = ""
        if "below ma60" in neg_txt or "ma60" in neg_txt:
            tail = " with price still wrestling below MA60"
        elif "bullish crossover" in pos_txt or "dif > dea" in pos_txt:
            tail = " while MACD leans constructive"

        chunks.append(f"Rules register {rec} at score {score}{tail}.")

    if rf_predictor.get("ok"):
        pu = rf_predictor.get("probability_up")
        if pu is not None:
            if pu >= 0.58:
                chunks.append(
                    f"RF research view: ~{pu * 100:.0f}% Up probability over five sessions "
                    "(ordinary upside odds, not a limit-up call)."
                )
            elif pu <= 0.42:
                chunks.append(
                    f"RF research view: Up probability near {pu * 100:.0f}% for the same horizon."
                )
            else:
                chunks.append("RF research view: roughly balanced on the five-session horizon.")

    text = " ".join(chunks).strip()
    if not text:
        return (
            "Extend the uploaded OHLCV window so both the rule stack and RF module can populate "
            "this summary."
        )
    if len(text) > 300:
        return text[:297].rstrip() + "…"
    return text


def _to_int(form, key, default):
    try:
        return int(form.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _to_float(form, key, default):
    try:
        return float(form.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _strict_int(form, key, default, label):
    if key not in form:
        return int(default)
    value = form.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{label} must be a positive integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid integer.") from exc


def _strict_float(form, key, default, label):
    if key not in form:
        return float(default)
    value = form.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{label} must be a valid number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return parsed


def _strategy_params_from_form(form):
    params = {
        "short_window": _strict_int(form, "short_window", 5, "MA short window"),
        "long_window": _strict_int(form, "long_window", 20, "MA long window"),
        "rsi_period": _strict_int(form, "rsi_period", 14, "RSI period"),
        "oversold_threshold": _strict_float(
            form, "oversold_threshold", 30, "RSI oversold threshold"
        ),
        "overbought_threshold": _strict_float(
            form, "overbought_threshold", 70, "RSI overbought threshold"
        ),
        "fast_period": _strict_int(form, "fast_period", 12, "MACD fast period"),
        "slow_period": _strict_int(form, "slow_period", 26, "MACD slow period"),
        "signal_period": _strict_int(form, "signal_period", 9, "MACD signal period"),
        "initial_capital": _strict_float(form, "initial_capital", 10000, "Initial capital"),
        "transaction_cost": _strict_float(
            form, "transaction_cost", 0.001, "Transaction cost"
        ),
    }

    if params["short_window"] <= 0:
        raise ValueError("MA short window must be a positive integer.")
    if params["long_window"] <= 0:
        raise ValueError("MA long window must be a positive integer.")
    if params["short_window"] >= params["long_window"]:
        raise ValueError("MA short window must be smaller than MA long window.")

    if params["rsi_period"] <= 0:
        raise ValueError("RSI period must be a positive integer.")
    if not 0 <= params["oversold_threshold"] <= 100:
        raise ValueError("RSI oversold threshold must be between 0 and 100.")
    if not 0 <= params["overbought_threshold"] <= 100:
        raise ValueError("RSI overbought threshold must be between 0 and 100.")
    if params["oversold_threshold"] >= params["overbought_threshold"]:
        raise ValueError("RSI oversold threshold must be lower than overbought threshold.")

    if params["fast_period"] <= 0:
        raise ValueError("MACD fast period must be a positive integer.")
    if params["slow_period"] <= 0:
        raise ValueError("MACD slow period must be a positive integer.")
    if params["signal_period"] <= 0:
        raise ValueError("MACD signal period must be a positive integer.")
    if params["fast_period"] >= params["slow_period"]:
        raise ValueError("MACD fast period must be smaller than slow period.")

    if params["initial_capital"] <= 0:
        raise ValueError("Initial capital must be greater than 0.")
    if params["transaction_cost"] < 0 or params["transaction_cost"] > 0.1:
        raise ValueError("Transaction cost must be between 0 and 0.1.")

    return params


def _ml_float(form, key, default, label):
    value = form.get(key, default)
    if value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid number.") from exc


def _ml_int(form, key, default, label):
    value = form.get(key, default)
    if value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid integer.") from exc


def _ml_params_from_form(form, model_name):
    defaults = DEFAULT_PARAMS[model_name]

    if model_name == "logistic_regression":
        return {
            "C": _ml_float(form, "lr_C", defaults["C"], "Logistic Regression C"),
            "max_iter": _ml_int(
                form, "lr_max_iter", defaults["max_iter"], "Logistic Regression max_iter"
            ),
        }
    if model_name == "decision_tree":
        return {
            "max_depth": _ml_int(
                form, "dt_max_depth", defaults["max_depth"], "Decision Tree max_depth"
            ),
            "min_samples_split": _ml_int(
                form,
                "dt_min_samples_split",
                defaults["min_samples_split"],
                "Decision Tree min_samples_split",
            ),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": _ml_int(
                form,
                "rf_n_estimators",
                defaults["n_estimators"],
                "Random Forest n_estimators",
            ),
            "max_depth": _ml_int(
                form, "rf_max_depth", defaults["max_depth"], "Random Forest max_depth"
            ),
        }
    if model_name == "svm":
        return {
            "C": _ml_float(form, "svm_C", defaults["C"], "SVM C"),
            "kernel": form.get("svm_kernel", defaults["kernel"]),
            "gamma": form.get("svm_gamma", defaults["gamma"]),
        }
    if model_name == "knn":
        return {
            "n_neighbors": _ml_int(
                form, "knn_n_neighbors", defaults["n_neighbors"], "KNN n_neighbors"
            ),
            "weights": form.get("knn_weights", defaults["weights"]),
        }

    return {}


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    root = Path(__file__).resolve().parent
    upload_folder = app.config["UPLOAD_FOLDER"]
    generated_folder = app.config["GENERATED_FOLDER"]
    if not Path(upload_folder).is_absolute():
        upload_folder = str(root / upload_folder)
    if not Path(generated_folder).is_absolute():
        generated_folder = str(root / generated_folder)
    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["GENERATED_FOLDER"] = generated_folder

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["GENERATED_FOLDER"], exist_ok=True)

    strategy_processed_dir = os.path.join(app.config["UPLOAD_FOLDER"], "processed")
    os.makedirs(strategy_processed_dir, exist_ok=True)

    # This application intentionally has no user accounts or persistent history.
    # Processed analysis data is treated as temporary runtime/session state.
    # Restarting the app should reset active analysis context.
    try:
        for entry in os.listdir(strategy_processed_dir):
            if entry == ".gitkeep":
                continue
            entry_path = os.path.join(strategy_processed_dir, entry)
            if os.path.isfile(entry_path):
                os.remove(entry_path)
    except FileNotFoundError:
        os.makedirs(strategy_processed_dir, exist_ok=True)

    @app.before_request
    def _runtime_session_guard():
        if session.get("app_boot_id") != APP_BOOT_ID:
            session.clear()
            session["app_boot_id"] = APP_BOOT_ID

    @app.route("/", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html")

        if "file" not in request.files:
            flash("No file part found in request.", "error")
            return redirect(url_for("upload"))

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":
            flash("Please choose a file before submitting.", "error")
            return redirect(url_for("upload"))

        if not _allowed_file(uploaded_file.filename):
            flash("Unsupported file type. Please upload CSV, XLSX, or XLS.", "error")
            return redirect(url_for("upload"))

        try:
            safe_name = secure_filename(uploaded_file.filename)
            source_name = f"{uuid4().hex}_{safe_name}"
            source_path = os.path.join(app.config["UPLOAD_FOLDER"], source_name)
            uploaded_file.save(source_path)

            raw_df = load_file(source_path)

            agent = DataAgent()
            mapped_df, mapping_report, missing_columns = agent.map_to_ohlcv(raw_df)
            if missing_columns:
                missing_text = ", ".join(missing_columns)
                detected_cols = ", ".join([str(col) for col in raw_df.columns])
                raise ValueError(
                    "Missing required columns after auto-mapping: "
                    f"{missing_text}. Detected columns: {detected_cols}"
                )

            cleaned_df, quality_report = clean_financial_data(mapped_df)
            enriched_df = add_basic_indicators(cleaned_df)

            processed_name = f"processed_{uuid4().hex}.csv"
            processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_name)
            enriched_df.to_csv(processed_path, index=False)
            latest_cleaned_path = os.path.join(strategy_processed_dir, "latest_cleaned.csv")
            enriched_df.to_csv(latest_cleaned_path, index=False)

            session["app_boot_id"] = APP_BOOT_ID
            session["processed_file"] = processed_name
            session["dataset_id"] = processed_name
            session.pop("strategy_result", None)
            session.pop("ml_result", None)
            session.pop("ml_train_params", None)
            session.pop("ml_rf_workflow", None)
            session.pop("ml_last_selected_model", None)
            session.pop("strategy_comparison_result", None)
            session.pop("ml_comparison_result", None)
            session["mapping_report"] = mapping_report
            session["quality_report"] = quality_report
            session["source_file_name"] = uploaded_file.filename

            return redirect(url_for("dashboard"))
        except Exception as exc:
            flash(f"Upload failed: {exc}", "error")
            return redirect(url_for("upload"))

    @app.route("/dashboard")
    def dashboard():
        processed_name = session.get("processed_file")
        if not processed_name:
            flash("Please upload a file first.", "error")
            return redirect(url_for("upload"))

        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_name)
        if not os.path.exists(processed_path):
            flash("Processed dataset not found. Please upload again.", "error")
            return redirect(url_for("upload"))

        try:
            df = load_file(processed_path)
            df["Date"] = df["Date"].astype(str)

            summary = {
                "total_rows": int(len(df)),
                "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
                "avg_close": _safe_float(df["Close"].mean()),
                "max_close": _safe_float(df["Close"].max()),
                "min_close": _safe_float(df["Close"].min()),
                "avg_volume": _safe_float(df["Volume"].mean()),
            }

            chart_df = df.copy()
            chart_df["Date"] = chart_df["Date"].astype("datetime64[ns]")

            charts = {
                "close": create_close_price_chart(chart_df),
                "volume": create_volume_chart(chart_df),
                "returns": create_return_histogram(chart_df),
                "ma": create_moving_average_chart(chart_df),
            }

            buy_signal = compute_buy_score(chart_df)
            try:
                rf_predictor = run_rf_direction_predictor(chart_df)
            except Exception as exc:
                rf_predictor = {
                    "ok": False,
                    "error": str(exc),
                    "prediction_latest": None,
                    "probability_up": None,
                    "probability_down": None,
                    "strong_rise_probability": None,
                    "sharp_drop_risk": None,
                    "test_accuracy": None,
                    "strong_rise_accuracy": None,
                    "sharp_drop_accuracy": None,
                    "feature_importance": [],
                    "n_train_rows": 0,
                    "n_test_rows": 0,
                    "recommendation": None,
                }
            decision_narrative = _build_decision_narrative(buy_signal, rf_predictor)

            preview_rows = df.head(10).to_dict(orient="records")
            table_columns = list(df.columns)

            return render_template(
                "dashboard.html",
                summary=summary,
                quality_report=session.get("quality_report", {}),
                mapping_report=session.get("mapping_report", {}),
                charts=charts,
                buy_signal=buy_signal,
                rf_predictor=rf_predictor,
                decision_narrative=decision_narrative,
                preview_rows=preview_rows,
                table_columns=table_columns,
                source_file_name=session.get("source_file_name", "Unknown"),
            )
        except Exception as exc:
            flash(f"Could not render dashboard: {exc}", "error")
            return redirect(url_for("upload"))

    @app.route("/strategy", methods=["GET", "POST"])
    def strategy_lab():
        processed_name = session.get("processed_file")
        if not processed_name:
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_name)
        if not os.path.exists(processed_path):
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        try:
            df = load_file(processed_path)
            df["Date"] = df["Date"].astype("datetime64[ns]")
        except Exception:
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        try:
            selected_strategy = request.form.get("strategy_name", "ma_crossover")
            params = _strategy_params_from_form(request.form)

            strategy_df = apply_strategy(df, selected_strategy, params)
            bt_df, metrics = run_backtest(
                strategy_df,
                initial_capital=params["initial_capital"],
                transaction_cost=params["transaction_cost"],
            )

            charts = {
                "signals": create_price_with_signals_chart(bt_df),
                "equity": create_strategy_equity_chart(bt_df),
                "drawdown": create_drawdown_chart(bt_df),
                "comparison": create_strategy_vs_market_chart(bt_df),
            }

            preview_cols = [
                "Date",
                "Close",
                "Signal",
                "Market_Return",
                "Strategy_Return",
                "Market_Equity",
                "Strategy_Equity",
                "Drawdown",
            ]
            preview_df = bt_df[preview_cols].copy().tail(20)
            preview_df["Date"] = preview_df["Date"].astype(str)
            session["strategy_result"] = {
                "selected_strategy": selected_strategy,
                "metrics": metrics,
                "dataset_id": session.get("dataset_id"),
            }

            return render_template(
                "strategy.html",
                selected_strategy=selected_strategy,
                params=params,
                metrics=metrics,
                charts=charts,
                preview_rows=preview_df.to_dict(orient="records"),
                table_columns=preview_cols,
            )
        except Exception as exc:
            flash(f"Strategy backtest failed: {exc}", "error")
            return redirect(url_for("strategy_lab"))

    @app.route("/strategy-comparison")
    def strategy_comparison():
        processed_name = session.get("processed_file")
        if not processed_name:
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_name)
        if not os.path.exists(processed_path):
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        try:
            df = load_file(processed_path)
            df["Date"] = df["Date"].astype("datetime64[ns]")

            default_setups = {
                "Moving Average": {
                    "strategy_name": "ma_crossover",
                    "params": {"short_window": 5, "long_window": 20},
                },
                "RSI": {
                    "strategy_name": "rsi",
                    "params": {
                        "rsi_period": 14,
                        "oversold_threshold": 30,
                        "overbought_threshold": 70,
                    },
                },
                "MACD": {
                    "strategy_name": "macd",
                    "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                },
            }

            comparison_rows = []
            equity_frame = pd.DataFrame({"Date": df["Date"]})

            for label, setup in default_setups.items():
                strategy_df = apply_strategy(df, setup["strategy_name"], setup["params"])
                bt_df, metrics = run_backtest(strategy_df)

                metric_total_return = _safe_float(metrics.get("total_return"))
                metric_annualized_return = _safe_float(metrics.get("annualized_return"))
                metric_annualized_vol = _safe_float(metrics.get("annualized_volatility"))
                metric_sharpe = _safe_float(metrics.get("sharpe_ratio"))
                metric_mdd = _safe_float(metrics.get("max_drawdown"))
                metric_win_rate = _safe_float(metrics.get("win_rate"))
                metric_trades = metrics.get("number_of_trades", 0)

                comparison_rows.append(
                    {
                        "strategy": label,
                        "total_return": _display_metric(metric_total_return, percent=True),
                        "annualized_return": _display_metric(
                            metric_annualized_return, percent=True
                        ),
                        "annualized_volatility": _display_metric(
                            metric_annualized_vol, percent=True
                        ),
                        "sharpe_ratio": _display_metric(metric_sharpe),
                        "max_drawdown": _display_metric(metric_mdd, percent=True),
                        "win_rate": _display_metric(metric_win_rate, percent=True),
                        "number_of_trades": int(metric_trades),
                        "raw_total_return": metric_total_return,
                        "raw_sharpe_ratio": metric_sharpe,
                        "raw_max_drawdown": metric_mdd,
                    }
                )

                equity_frame[label if label != "Moving Average" else "MA"] = bt_df[
                    "Strategy_Equity"
                ].values

            metric_chart_df = pd.DataFrame(
                [
                    {
                        "Strategy": row["strategy"],
                        "Total Return": row["raw_total_return"]
                        if row["raw_total_return"] is not None
                        else 0,
                        "Sharpe Ratio": row["raw_sharpe_ratio"]
                        if row["raw_sharpe_ratio"] is not None
                        else 0,
                        "Max Drawdown": row["raw_max_drawdown"]
                        if row["raw_max_drawdown"] is not None
                        else 0,
                    }
                    for row in comparison_rows
                ]
            )

            valid_sharpe = [r for r in comparison_rows if r["raw_sharpe_ratio"] is not None]
            valid_return = [r for r in comparison_rows if r["raw_total_return"] is not None]
            valid_drawdown = [r for r in comparison_rows if r["raw_max_drawdown"] is not None]

            best_summary = {
                "best_sharpe": max(valid_sharpe, key=lambda r: r["raw_sharpe_ratio"])["strategy"]
                if valid_sharpe
                else "N/A",
                "best_return": max(valid_return, key=lambda r: r["raw_total_return"])["strategy"]
                if valid_return
                else "N/A",
                "lowest_drawdown": max(
                    valid_drawdown, key=lambda r: r["raw_max_drawdown"]
                )["strategy"]
                if valid_drawdown
                else "N/A",
            }

            warning_message = (
                "Dataset has fewer than 30 rows. Strategy comparison may be unreliable."
                if len(df) < 30
                else None
            )
            compact_rows = [
                {
                    "strategy": row["strategy"],
                    "total_return": row["total_return"],
                    "annualized_return": row["annualized_return"],
                    "annualized_volatility": row["annualized_volatility"],
                    "sharpe_ratio": row["sharpe_ratio"],
                    "max_drawdown": row["max_drawdown"],
                    "win_rate": row["win_rate"],
                    "number_of_trades": row["number_of_trades"],
                }
                for row in comparison_rows
            ]
            session["strategy_comparison_result"] = {
                "dataset_id": session.get("dataset_id"),
                "rows": compact_rows,
                "best_summary": best_summary,
                "warning_message": warning_message,
            }

            return render_template(
                "strategy_comparison.html",
                comparison_rows=comparison_rows,
                best_summary=best_summary,
                warning_message=warning_message,
                equity_chart=create_strategy_comparison_equity_chart(equity_frame),
                metrics_chart=create_strategy_comparison_metrics_chart(metric_chart_df),
            )
        except Exception as exc:
            flash(f"Strategy comparison failed: {exc}", "error")
            return redirect(url_for("strategy_lab"))

    @app.route("/ml-analysis", methods=["GET", "POST"])
    def ml_analysis():
        processed_name = session.get("processed_file")
        if not processed_name:
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_name)
        if not os.path.exists(processed_path):
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        try:
            df = load_file(processed_path)
        except Exception:
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        available_features = get_available_features(df)

        selected_model = session.get("ml_last_selected_model", "logistic_regression")
        if request.method == "GET":
            cand = request.args.get("model")
            if cand and cand in SUPPORTED_MODELS:
                selected_model = cand
        if selected_model not in SUPPORTED_MODELS:
            selected_model = "logistic_regression"

        result = None
        comparison_result = None
        rf_predictor = None

        ds_id = session.get("dataset_id")
        cached_rf = session.get("ml_rf_workflow")
        if isinstance(cached_rf, dict) and cached_rf.get("dataset_id") == ds_id:
            rf_predictor = cached_rf.get("data")

        if request.method == "POST":
            ml_action = request.form.get("ml_action", "").strip()
            try:
                if ml_action == "rf_predictor":
                    try:
                        rf_predictor = run_rf_direction_predictor(df)
                    except Exception as exc:
                        rf_predictor = {
                            "ok": False,
                            "error": str(exc),
                            "prediction_latest": None,
                            "probability_up": None,
                            "probability_down": None,
                            "strong_rise_probability": None,
                            "sharp_drop_risk": None,
                            "test_accuracy": None,
                            "strong_rise_accuracy": None,
                            "sharp_drop_accuracy": None,
                            "feature_importance": [],
                            "n_train_rows": 0,
                            "n_test_rows": 0,
                            "recommendation": None,
                        }
                    session["ml_rf_workflow"] = {"dataset_id": ds_id, "data": rf_predictor}
                    flash("Direction predictor refreshed.")

                elif ml_action == "compare_all":
                    model_rows = []
                    chart_rows = []
                    warnings = []

                    for model_id, model_label in SUPPORTED_MODELS.items():
                        params_for_model = dict(DEFAULT_PARAMS[model_id])
                        try:
                            model_result = run_ml_analysis(df, model_id, params_for_model)
                        except Exception as exc:
                            if model_id == "knn":
                                fallback_params = dict(params_for_model)
                                fallback_params["n_neighbors"] = 1
                                try:
                                    model_result = run_ml_analysis(
                                        df, model_id, fallback_params
                                    )
                                except Exception:
                                    model_rows.append(
                                        {
                                            "model": model_label,
                                            "status": "Failed",
                                            "accuracy": "N/A",
                                            "precision": "N/A",
                                            "recall": "N/A",
                                            "f1_score": "N/A",
                                            "raw_accuracy": None,
                                            "raw_f1_score": None,
                                        }
                                    )
                                    warnings.append(f"{model_label} failed: {exc}")
                                    continue
                            else:
                                model_rows.append(
                                    {
                                        "model": model_label,
                                        "status": "Failed",
                                        "accuracy": "N/A",
                                        "precision": "N/A",
                                        "recall": "N/A",
                                        "f1_score": "N/A",
                                        "raw_accuracy": None,
                                        "raw_f1_score": None,
                                    }
                                )
                                warnings.append(f"{model_label} failed: {exc}")
                                continue

                        metrics = model_result["metrics"]
                        row = {
                            "model": model_label,
                            "status": "OK",
                            "accuracy": _safe_metric_text(metrics.get("accuracy")),
                            "precision": _safe_metric_text(metrics.get("precision")),
                            "recall": _safe_metric_text(metrics.get("recall")),
                            "f1_score": _safe_metric_text(metrics.get("f1_score")),
                            "raw_accuracy": _safe_float(metrics.get("accuracy")),
                            "raw_f1_score": _safe_float(metrics.get("f1_score")),
                        }
                        model_rows.append(row)
                        chart_rows.extend(
                            [
                                {
                                    "Model": model_label,
                                    "Metric": "Accuracy",
                                    "Score": row["raw_accuracy"] or 0,
                                },
                                {
                                    "Model": model_label,
                                    "Metric": "Precision",
                                    "Score": _safe_float(metrics.get("precision")) or 0,
                                },
                                {
                                    "Model": model_label,
                                    "Metric": "Recall",
                                    "Score": _safe_float(metrics.get("recall")) or 0,
                                },
                                {
                                    "Model": model_label,
                                    "Metric": "F1-score",
                                    "Score": row["raw_f1_score"] or 0,
                                },
                            ]
                        )

                    valid_by_f1 = [r for r in model_rows if r["raw_f1_score"] is not None]
                    valid_by_acc = [r for r in model_rows if r["raw_accuracy"] is not None]
                    best_model_summary = {
                        "best_f1": max(valid_by_f1, key=lambda r: r["raw_f1_score"])["model"]
                        if valid_by_f1
                        else "N/A",
                        "best_accuracy": max(
                            valid_by_acc, key=lambda r: r["raw_accuracy"]
                        )["model"]
                        if valid_by_acc
                        else "N/A",
                    }

                    compare_df = pd.DataFrame(chart_rows)
                    if compare_df.empty:
                        comparison_chart_html = "<div>No valid model comparison data available.</div>"
                    else:
                        comparison_chart = px.bar(
                            compare_df,
                            x="Model",
                            y="Score",
                            color="Metric",
                            barmode="group",
                            title="Model Comparison Metrics",
                            range_y=[0, 1],
                        )
                        comparison_chart.update_layout(template="plotly_dark")
                        comparison_chart_html = comparison_chart.to_html(full_html=False)

                    comparison_result = {
                        "rows": model_rows,
                        "warnings": warnings,
                        "best_model_summary": best_model_summary,
                        "chart": comparison_chart_html,
                        "chart_rows": chart_rows,
                        "small_dataset_warning": len(df) < 30,
                    }
                    compact_model_rows = [
                        {
                            "model": row["model"],
                            "status": row["status"],
                            "accuracy": row["accuracy"],
                            "precision": row["precision"],
                            "recall": row["recall"],
                            "f1_score": row["f1_score"],
                        }
                        for row in model_rows
                    ]
                    session["ml_comparison_result"] = {
                        "dataset_id": ds_id,
                        "rows": compact_model_rows,
                        "warnings": warnings,
                        "best_model_summary": best_model_summary,
                        "small_dataset_warning": len(df) < 30,
                    }
                    flash("Model comparison complete.")

                elif ml_action == "train_single":
                    selected_model = request.form.get("model_name", "logistic_regression")
                    if selected_model not in SUPPORTED_MODELS:
                        flash("Selected model is not supported.", "error")
                        return redirect(url_for("ml_analysis"))
                    params = dict(DEFAULT_PARAMS[selected_model])
                    try:
                        params = _ml_params_from_form(request.form, selected_model)
                    except ValueError as exc:
                        flash(str(exc), "error")
                        return redirect(url_for("ml_analysis"))
                    result = run_ml_analysis(df, selected_model, params)
                    session["ml_last_selected_model"] = selected_model
                    session["ml_train_params"] = params
                    session["ml_result"] = {
                        "selected_model": SUPPORTED_MODELS[selected_model],
                        "metrics": result["metrics"],
                        "train_size": result["train_size"],
                        "test_size": result["test_size"],
                        "target_distribution": result["target_distribution"],
                        "warnings": result["warnings"],
                        "dataset_id": ds_id,
                    }
                    flash("Single model training complete.")

                else:
                    flash("Unknown ML action.", "error")

            except Exception as exc:
                flash(f"ML analysis failed: {exc}", "error")
                return redirect(url_for("ml_analysis"))

        if result is None:
            cached = session.get("ml_result")
            params_used = session.get("ml_train_params")
            sm = session.get("ml_last_selected_model")
            if (
                isinstance(cached, dict)
                and cached.get("dataset_id") == ds_id
                and sm in SUPPORTED_MODELS
                and isinstance(params_used, dict)
            ):
                try:
                    result = run_ml_analysis(df, sm, params_used)
                except Exception:
                    result = None

        if comparison_result is None:
            cached_cmp = session.get("ml_comparison_result")
            if isinstance(cached_cmp, dict) and cached_cmp.get("dataset_id") == ds_id:
                comparison_result = {
                    "rows": cached_cmp.get("rows", []),
                    "warnings": cached_cmp.get("warnings", []),
                    "best_model_summary": cached_cmp.get("best_model_summary", {}),
                    "small_dataset_warning": cached_cmp.get("small_dataset_warning", False),
                    "chart": _ml_comparison_chart_from_rows(cached_cmp.get("rows") or []),
                }

        params = dict(DEFAULT_PARAMS[selected_model])

        return render_template(
            "ml_analysis.html",
            available_features=available_features,
            available_models=SUPPORTED_MODELS,
            default_params=DEFAULT_PARAMS,
            selected_model=selected_model,
            params=params,
            result=result,
            comparison_result=comparison_result,
            rf_predictor=rf_predictor,
            source_file_name=session.get("source_file_name", "Unknown"),
        )

    @app.route("/report")
    def report():
        processed_name = session.get("processed_file")
        if not processed_name:
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_name)
        if not os.path.exists(processed_path):
            flash("Please upload a dataset first.", "error")
            return redirect(url_for("upload"))

        try:
            df = load_file(processed_path)
            df["Date"] = df["Date"].astype(str)

            overview = {
                "source_file_name": session.get("source_file_name", "Unknown"),
                "total_rows": int(len(df)),
                "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
                "avg_close": _safe_float(df["Close"].mean()),
                "max_close": _safe_float(df["Close"].max()),
                "min_close": _safe_float(df["Close"].min()),
                "avg_volume": _safe_float(df["Volume"].mean()),
                "avg_daily_return": _safe_float(df["Daily_Return"].dropna().mean())
                if "Daily_Return" in df.columns
                else None,
                "volatility": _safe_float(df["Daily_Return"].dropna().std())
                if "Daily_Return" in df.columns
                else None,
            }

            chart_df = df.copy()
            chart_df["Date"] = chart_df["Date"].astype("datetime64[ns]")
            charts = {
                "close": create_close_price_chart(chart_df),
                "volume": create_volume_chart(chart_df),
                "ma": create_moving_average_chart(chart_df),
            }

            strategy_result = session.get("strategy_result")
            if strategy_result and strategy_result.get("dataset_id") != session.get("dataset_id"):
                strategy_result = None

            ml_result = session.get("ml_result")
            if ml_result and ml_result.get("dataset_id") != session.get("dataset_id"):
                ml_result = None

            strategy_comparison_result = session.get("strategy_comparison_result")
            strategy_comparison_chart = None
            if (
                strategy_comparison_result
                and strategy_comparison_result.get("dataset_id") == session.get("dataset_id")
            ):
                # Strategy comparison report section uses compact summary table.
                # Equity chart data is intentionally not persisted in session to keep
                # cookie size within browser limits.
                strategy_comparison_chart = None
            else:
                strategy_comparison_result = None

            ml_comparison_result = session.get("ml_comparison_result")
            ml_comparison_chart = None
            if (
                ml_comparison_result
                and ml_comparison_result.get("dataset_id") == session.get("dataset_id")
            ):
                rows = ml_comparison_result.get("rows") or []
                if rows:
                    ml_comparison_chart = _ml_comparison_chart_from_rows(rows)
            else:
                ml_comparison_result = None

            return render_template(
                "report.html",
                overview=overview,
                quality_report=session.get("quality_report", {}),
                charts=charts,
                strategy_result=strategy_result,
                ml_result=ml_result,
                strategy_comparison_result=strategy_comparison_result,
                strategy_comparison_chart=strategy_comparison_chart,
                ml_comparison_result=ml_comparison_result,
                ml_comparison_chart=ml_comparison_chart,
            )
        except Exception as exc:
            flash(f"Could not render report: {exc}", "error")
            return redirect(url_for("upload"))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
