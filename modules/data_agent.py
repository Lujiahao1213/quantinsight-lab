import re
from difflib import SequenceMatcher, get_close_matches

import pandas as pd


class DataAgent:
    # Only require the minimum needed to process a price series.
    REQUIRED_COLUMNS = ["Date", "Close"]

    ALIAS_MAP = {
        "Date": [
            "date",
            "time",
            "datetime",
            "timestamp",
            "tradingdate",
            "tradedate",
            "日期",
        ],
        "Close": [
            "close",
            "closelast",
            "last",
            "lastprice",
            "lastsale",
            "adjclose",
            "adjustedclose",
            "closingprice",
            "price",
            "收盘价",
        ],
        "Open": ["open", "openprice", "openingprice", "开盘价"],
        "High": ["high", "highest", "highprice", "最高价"],
        "Low": ["low", "lowest", "lowprice", "最低价"],
        "Volume": ["volume", "vol", "tradingvolume", "turnovervolume", "成交量"],
    }
    FUZZY_THRESHOLD = 0.85

    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"[\/_\-\.\(\)\s]+", "", text)
        return text

    def map_to_ohlcv(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Uploaded file is empty.")

        normalized_to_original = {}
        for col in df.columns:
            norm = self._normalize(col)
            normalized_to_original.setdefault(norm, []).append(col)

        rename_dict = {}
        mapping_report = {}
        used_sources = set()

        for target_col, aliases in self.ALIAS_MAP.items():
            resolved_source = None
            method = None
            confidence = None

            if target_col in df.columns and target_col not in used_sources:
                resolved_source = target_col
                method = "original"
                confidence = 1.0

            alias_set = {self._normalize(a) for a in aliases}
            alias_set.add(self._normalize(target_col))

            if not resolved_source:
                for alias_norm in alias_set:
                    if alias_norm in normalized_to_original:
                        candidates = normalized_to_original[alias_norm]
                        for candidate in candidates:
                            if candidate not in used_sources:
                                resolved_source = candidate
                                method = "exact_alias"
                                confidence = 1.0
                                break
                        if resolved_source:
                            break

            if not resolved_source:
                available_norms = [
                    norm
                    for norm, originals in normalized_to_original.items()
                    if any(orig not in used_sources for orig in originals)
                ]
                best_norm = None
                best_score = 0.0
                for alias_norm in alias_set:
                    matches = get_close_matches(
                        alias_norm,
                        available_norms,
                        n=1,
                        cutoff=self.FUZZY_THRESHOLD,
                    )
                    if matches:
                        candidate_norm = matches[0]
                        score = SequenceMatcher(
                            None, alias_norm, candidate_norm
                        ).ratio()
                        if score > best_score:
                            best_score = score
                            best_norm = candidate_norm
                if best_norm:
                    candidates = normalized_to_original[best_norm]
                    for candidate in candidates:
                        if candidate not in used_sources:
                            resolved_source = candidate
                            method = "fuzzy"
                            confidence = round(float(best_score), 3)
                            break

            if resolved_source:
                rename_dict[resolved_source] = target_col
                used_sources.add(resolved_source)
                mapping_report[target_col] = {
                    "original_column": resolved_source,
                    "mapped_standard_column": target_col,
                    "method": method,
                    "confidence": confidence,
                }
            else:
                mapping_report[target_col] = {
                    "original_column": None,
                    "mapped_standard_column": target_col,
                    "method": None,
                    "confidence": None,
                }

        mapped_df = df.rename(columns=rename_dict).copy()
        missing_required = [
            col for col in self.REQUIRED_COLUMNS if col not in mapped_df.columns
        ]

        return mapped_df, mapping_report, missing_required
