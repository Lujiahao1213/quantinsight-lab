import pandas as pd
import numpy as np


def _quality_level(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def clean_financial_data(df: pd.DataFrame):
    required_cols = ["Date", "Close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {', '.join(missing)}")

    out = df.copy()
    raw_rows = len(out)
    checklist = []
    penalties = []

    standard_required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_standard = [col for col in standard_required if col not in out.columns]

    invalid_date_count = int(pd.to_datetime(out["Date"], errors="coerce").isna().sum())
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    close_numeric = pd.to_numeric(out["Close"], errors="coerce")
    non_positive_close_count = int((close_numeric <= 0).fillna(False).sum())

    volume_missing_initially = "Volume" not in out.columns

    # Reconstruct optional OHLCV columns when absent so downstream charts/summary
    # continue to work for simplified CSV inputs (e.g. Date + Close only).
    if "Open" not in out.columns:
        out["Open"] = out["Close"]
    if "High" not in out.columns:
        out["High"] = out["Close"]
    if "Low" not in out.columns:
        out["Low"] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0

    for col in ["Open", "High", "Low", "Close"]:
        out[col] = (
            out[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["Volume"] = (
        out["Volume"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=["Date", "Close"])
    out = out.sort_values("Date", ascending=True)

    duplicate_dates_removed = int(out.duplicated(subset=["Date"]).sum())
    out = out.drop_duplicates(subset=["Date"], keep="last")

    out = out.ffill().bfill()

    if out.empty:
        raise ValueError("No valid rows remain after cleaning.")

    final_missing_values = int(out.isna().sum().sum())
    total_cells = int(out.shape[0] * out.shape[1]) if out.shape[0] and out.shape[1] else 1
    missing_ratio = final_missing_values / total_cells

    zero_volume_ratio = float((out["Volume"] <= 0).mean()) if "Volume" in out.columns else 1.0

    score = 100

    missing_cols_penalty = min(30 * len(missing_standard), 60)
    if missing_cols_penalty > 0:
        score -= missing_cols_penalty
        penalties.append(f"Missing standard columns: -{missing_cols_penalty}")
        checklist.append(
            f"Missing standard columns detected: {', '.join(missing_standard)}"
        )
    else:
        checklist.append("Required OHLCV columns detected")

    missing_values_penalty = min(20, int(round(missing_ratio * 20)))
    if missing_values_penalty > 0:
        score -= missing_values_penalty
        penalties.append(f"Missing values after cleaning: -{missing_values_penalty}")
        checklist.append("Missing values remain after cleaning")
    else:
        checklist.append("Missing values filled")

    duplicate_ratio = duplicate_dates_removed / max(raw_rows, 1)
    duplicate_penalty = min(10, int(round(duplicate_ratio * 10)))
    if duplicate_penalty > 0:
        score -= duplicate_penalty
        penalties.append(f"Duplicate dates removed: -{duplicate_penalty}")
        checklist.append(f"Duplicate dates removed: {duplicate_dates_removed}")
    else:
        checklist.append("No duplicate dates found")

    if len(out) < 30:
        score -= 20
        penalties.append("Row count < 30: -20")
        checklist.append("Dataset has fewer than 30 rows")
    elif len(out) < 100:
        score -= 10
        penalties.append("Row count < 100: -10")
        checklist.append("Dataset has fewer than 100 rows")
    else:
        checklist.append("Dataset has at least 100 rows")

    if non_positive_close_count > 0:
        score -= 20
        penalties.append("Non-positive Close detected: -20")
        checklist.append(f"Non-positive Close values detected: {non_positive_close_count}")
    else:
        checklist.append("Close prices are positive")

    if invalid_date_count > 0:
        score -= 15
        penalties.append("Invalid Date values detected: -15")
        checklist.append(f"Invalid Date values detected: {invalid_date_count}")
    else:
        checklist.append("Date column converted successfully")

    if volume_missing_initially or zero_volume_ratio >= 0.8:
        score -= 10
        penalties.append("Volume missing or mostly zero: -10")
        checklist.append("Volume missing or mostly zero")
    else:
        checklist.append("Volume data quality is acceptable")

    score = max(0, min(100, int(score)))
    quality_level = _quality_level(score)

    report = {
        "row_count": int(len(out)),
        "column_count": int(out.shape[1]),
        "date_range": {
            "start": out["Date"].min().strftime("%Y-%m-%d"),
            "end": out["Date"].max().strftime("%Y-%m-%d"),
        },
        "missing_values": final_missing_values,
        "duplicate_rows_removed": duplicate_dates_removed,
        "quality_score": score,
        "quality_level": quality_level,
        "quality_checklist": checklist,
        "quality_penalties": penalties,
        "status": "cleaned",
    }

    return out, report
