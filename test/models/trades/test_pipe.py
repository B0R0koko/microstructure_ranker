from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import polars as pl

from core.columns import SYMBOL, TRADE_TIME, PRICE
from core.currency_pair import CurrencyPair
from core.paths import FEATURE_DIR, BINANCE_SPOT_TRADES
from core.time_utils import Bounds, TimeOffset
from models.trades.features.features_27_11 import compute_features


# This file is to doublecheck if the target and features are attached correctly in the pipeline

def compute_target(bounds: Bounds, currency_pair: CurrencyPair, time_offset: TimeOffset) -> float:
    offset_bounds: Bounds = bounds.create_offset_bounds(time_offset=time_offset)

    df_currency_pair: pl.DataFrame = pl.scan_parquet(BINANCE_SPOT_TRADES, hive_partitioning=True).filter(
        (pl.col("date").is_between(offset_bounds.day0, offset_bounds.day1)) &
        (pl.col(TRADE_TIME).is_between(offset_bounds.start_inclusive, offset_bounds.end_exclusive)) &
        (pl.col(SYMBOL) == currency_pair.name)
    ).collect()

    df_currency_pair = df_currency_pair.sort(by=TRADE_TIME, descending=False)

    return df_currency_pair.select(
        np.log(pl.col(PRICE).last() / pl.col(PRICE).first())
    ).item()


def manually_compute_features(bounds: Bounds, currency_pair: CurrencyPair) -> Dict[str, float]:
    df_currency_pair: pl.DataFrame = pl.scan_parquet(BINANCE_SPOT_TRADES, hive_partitioning=True).filter(
        (pl.col("date").is_between(bounds.day0, bounds.day1)) &
        (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive)) &
        (pl.col(SYMBOL) == currency_pair.name)
    ).collect()

    features: Dict[str, float] = compute_features(
        df_currency_pair=df_currency_pair, currency_pair=currency_pair, bounds=bounds
    )
    features["log_return"] = compute_target(
        bounds=bounds, currency_pair=currency_pair, time_offset=TimeOffset.HOUR
    )
    return features


def load_compute_cross_section(currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
    df = pd.read_parquet(FEATURE_DIR.joinpath("features_19-02-2025.parquet"))
    df_filtered = df[
        (df["cross_section_end_time"] == bounds.end_exclusive) & (df["currency_pair"] == currency_pair.binance_name)
        ].copy()

    return df_filtered.to_dict(orient="records")[0]


def check_features_are_the_same(pipeline_features: Dict[str, Any], manual_features: Dict[str, Any]) -> bool:
    return all(pipeline_features[key] == manual_features[key] for key in manual_features.keys())


def test_if_dataframe_assembled_correctly():
    """Check if the way we collect data and attach target is correct and there is no mistakes"""
    start_time: datetime = datetime(2024, 11, 1, 0, 0, 0)
    end_time: datetime = datetime(2024, 11, 1, 12, 0, 0)
    bounds: Bounds = Bounds(start_time, end_time)
    currency_pair: CurrencyPair = CurrencyPair(base="BTC", term="USDT")

    manual_features: Dict[str, float] = manually_compute_features(bounds=bounds, currency_pair=currency_pair)
    pipeline_features: Dict[str, Any] = load_compute_cross_section(currency_pair=currency_pair, bounds=bounds)

    check_features_are_the_same(pipeline_features=pipeline_features, manual_features=manual_features)
