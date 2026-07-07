"""Reusable model pipeline components for the MP response prediction web app.

The class is intentionally kept in a standalone module so the joblib model can be
loaded safely by Streamlit/Flask without redefining classes inside the app file.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MODEL_FEATURES = [
    "Age",
    "Calcification",
    "Curve_Washout",
    "Curve_Plateau",
    "ADC",
    "Tumor_size_cm",
    "ER_percent",
    "PR_percent",
    "HER2_score",
    "Ki67_percent",
]


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert heterogeneous Chinese clinical/MRI/IHC columns into numeric features."""

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names if feature_names is not None else MODEL_FEATURES

    def fit(self, X, y=None):
        self.input_columns_ = list(X.columns)
        return self

    def _find_col(self, candidates: list[str]) -> Optional[str]:
        lower_map = {str(c).strip().lower(): c for c in self.input_columns_}
        for c in candidates:
            if c in self.input_columns_:
                return c
            key = str(c).strip().lower()
            if key in lower_map:
                return lower_map[key]
        return None

    @staticmethod
    def _clean_percent(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace("％", "%").lower()
        if s in {"", "nan", "none"}:
            return np.nan
        negative_tokens = ["(-)", "（-）", "阴", "negative", "neg", "—", "-"]
        if any(tok in s for tok in negative_tokens) and not re.search(r"\d", s):
            return 0.0
        nums = re.findall(r"\d+\.?\d*", s)
        if not nums:
            return np.nan
        # For ranges such as 5-10%, use the upper bound because it is usually the clinically conservative value.
        val = float(nums[-1])
        # Values like 0.7 in the source sheet represent 70% rather than 0.7%.
        if 0 < val <= 1 and "%" not in s:
            val *= 100.0
        return float(np.clip(val, 0, 100))

    @staticmethod
    def _clean_her2(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower().replace("＋", "+")
        if "3+" in s:
            return 3.0
        if "2+" in s:
            return 2.0
        if "1+" in s:
            return 1.0
        if _has_any(s, ["0", "阴", "negative", "neg", "-"]):
            return 0.0
        nums = re.findall(r"\d+\.?\d*", s)
        if nums:
            return float(np.clip(float(nums[-1]), 0, 3))
        return np.nan

    @staticmethod
    def _to_float(x):
        if pd.isna(x):
            return np.nan
        nums = re.findall(r"\d+\.?\d*", str(x))
        if not nums:
            return np.nan
        return float(nums[0])

    @staticmethod
    def _parse_size_cm(x):
        if pd.isna(x):
            return np.nan
        nums = [float(v) for v in re.findall(r"\d+\.?\d*", str(x))]
        if not nums:
            return np.nan
        return float(max(nums))

    @staticmethod
    def _calcification_binary(x):
        if pd.isna(x):
            return 0.0
        s = str(x).strip().lower()
        if s in {"", "nan", "none", "无", "未见", "否", "0", "(-)", "（-）"}:
            return 0.0
        return 1.0

    @staticmethod
    def _curve_washout(x):
        s = "" if pd.isna(x) else str(x).strip().lower()
        return 1.0 if _has_any(s, ["流出", "washout", "type iii", "iii"]) else 0.0

    @staticmethod
    def _curve_plateau(x):
        s = "" if pd.isna(x) else str(x).strip().lower()
        return 1.0 if _has_any(s, ["平台", "plateau", "type ii", "ii"]) else 0.0

    def transform(self, X):
        X = X.copy()
        res = pd.DataFrame(index=X.index)

        age_col = self._find_col(["年龄", "Age", "age"])
        calc_col = self._find_col(["钙化", "Calc", "calcification"])
        curve_col = self._find_col(["曲线", "Curve", "DCE curve"])
        adc_col = self._find_col(["ADC值", "ADC", "adc"])
        size_col = self._find_col(["cm", "大小", "Size", "Tumor size", "tumor_size"])
        er_col = self._find_col(["ER", "ER%", "ER_percent"])
        pr_col = self._find_col(["PR", "PR%", "PR_percent"])
        her2_col = self._find_col(["HER2", "HER-2", "HER2_score"])
        ki67_col = self._find_col(["Ki-67", "Ki67", "Ki67_percent"])

        res["Age"] = pd.to_numeric(X[age_col], errors="coerce") if age_col else np.nan
        res["Calcification"] = X[calc_col].apply(self._calcification_binary) if calc_col else 0.0
        res["Curve_Washout"] = X[curve_col].apply(self._curve_washout) if curve_col else 0.0
        res["Curve_Plateau"] = X[curve_col].apply(self._curve_plateau) if curve_col else 0.0
        res["ADC"] = X[adc_col].apply(self._to_float) if adc_col else np.nan
        res["Tumor_size_cm"] = X[size_col].apply(self._parse_size_cm) if size_col else np.nan
        res["ER_percent"] = X[er_col].apply(self._clean_percent) if er_col else np.nan
        res["PR_percent"] = X[pr_col].apply(self._clean_percent) if pr_col else np.nan
        res["HER2_score"] = X[her2_col].apply(self._clean_her2) if her2_col else np.nan
        res["Ki67_percent"] = X[ki67_col].apply(self._clean_percent) if ki67_col else np.nan

        for col in self.feature_names:
            if col not in res.columns:
                res[col] = np.nan
        return res[self.feature_names]
