import logging
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from lightgbm import Booster

from core.exchange import Exchange
from core.paths import get_root_dir
from core.time_utils import format_date


def get_importance_file_path(day: date, target_exchange: Exchange) -> Path:
    return (
            get_root_dir() /
            "src/models/prediction/feature_importances" /
            f"{target_exchange.name}-importances@{format_date(day)}.csv"
    )


def save_feature_importances_to_file(booster: Booster, day: date, target_exchange: Exchange) -> None:
    """Save booster feature importances to file located at get_importance_file_path() location"""
    df: pd.DataFrame = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance": booster.feature_importance(importance_type="gain")
    })
    df.sort_values("importance", ascending=False, inplace=True)
    df.to_csv(get_importance_file_path(day=day, target_exchange=target_exchange), index=False)


class FeatureFilter:

    def __init__(self, allowed_features: Optional[List[str]] = None):
        self.allowed_features: Optional[List[str]] = allowed_features

    @classmethod
    def from_importance(cls, file: Path, use_first: int = 25) -> "FeatureFilter":
        logging.info("Loading feature importance file %s", file)
        df_importance: pd.DataFrame = pd.read_csv(file).sort_values(by="importance", ascending=False)
        return cls(
            allowed_features=df_importance["feature"].head(use_first).tolist(),
        )

    def is_allowed(self, feature_name: str) -> bool:
        if self.allowed_features is None:
            return True
        if feature_name in self.allowed_features:
            return True
        return False
