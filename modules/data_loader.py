from pathlib import Path
from typing import Optional

import pandas as pd

_YF_HEADER_COLS = {"Price", "Close", "High", "Low", "Open", "Volume"}


def _looks_like_yfinance_multilevel_csv(path: Path, encoding: str) -> bool:
    """True if the file matches yfinance MultiIndex CSV (header + Ticker row + Date row)."""
    try:
        with path.open("r", encoding=encoding, errors="strict") as handle:
            header_line = handle.readline().lstrip("\ufeff").strip()
            ticker_line = handle.readline().strip()
    except UnicodeDecodeError:
        return False

    header_parts = [p.strip() for p in header_line.split(",")]
    if len(header_parts) < 2:
        return False
    if set(header_parts) != _YF_HEADER_COLS:
        return False
    ticker_parts = [p.strip() for p in ticker_line.strip().split(",")]
    return bool(ticker_parts) and ticker_parts[0] == "Ticker"


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load a yfinance-style MultiIndex OHLCV CSV:
    row 1 headers, row 2 tickers, row 3 date labels, data from row 4.
    """
    path = Path(file_path)
    decode_errors: list[str] = []
    last_exc: Optional[Exception] = None

    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding, skiprows=[1, 2])
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")
            last_exc = exc
            continue
        except Exception as exc:
            raise ValueError(f"Failed to read stock CSV file: {exc}") from exc

        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])

        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        ordered = ["Date", "Close", "High", "Low", "Open", "Volume"]
        missing = [c for c in ordered if c not in df.columns]
        if missing:
            raise ValueError(
                f"Stock CSV is missing expected columns after parse: {', '.join(missing)}"
            )

        return df[ordered].reset_index(drop=True)

    error_text = "; ".join(decode_errors)
    raise ValueError(
        "Could not decode stock CSV using supported encodings. "
        f"Details: {error_text}"
    ) from last_exc


def _post_process_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.dropna(axis=1, how="all")
    empty_name_cols = [col for col in cleaned.columns if str(col).strip() == ""]
    if empty_name_cols:
        cleaned = cleaned.drop(columns=empty_name_cols)
    return cleaned


def _read_csv_robust(path: Path, encoding: str) -> pd.DataFrame:
    # First attempt automatic delimiter inference.
    df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    df = _post_process_columns(df)
    if len(df.columns) > 1:
        return df

    # Fallback delimiters for non-standard CSV-like files.
    for sep in ("\t", ";", "|", r"\s+"):
        trial = pd.read_csv(path, encoding=encoding, sep=sep, engine="python")
        trial = _post_process_columns(trial)
        if len(trial.columns) > 1:
            return trial

    return df


def load_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        csv_errors = []
        for encoding in ("utf-8", "utf-8-sig", "gbk", "latin1"):
            try:
                if _looks_like_yfinance_multilevel_csv(path, encoding):
                    return load_stock_data(file_path)
                return _read_csv_robust(path, encoding)
            except UnicodeDecodeError as exc:
                csv_errors.append(f"{encoding}: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to read CSV file: {exc}") from exc
        error_text = "; ".join(csv_errors)
        raise ValueError(
            f"Could not decode CSV file using supported encodings. Details: {error_text}"
        )

    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file: {exc}") from exc

    raise ValueError(
        f"Unsupported file type '{suffix}'. Please upload CSV, XLSX, or XLS files."
    )
