import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from core.columns import TRADE_TIME, PRICE
from core.data_type import SamplingType
from core.paths import FEATURE_DIR
from core.time_utils import format_date


def write_feature(values: np.ndarray, day: date, sampling_type: SamplingType, subpath: Path) -> None:
    """Writes sampled np.ndarray with feature to {FEATURE_DIR}/HFT/20250101/MS500/asset_return/ADA"""
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / subpath
    os.makedirs(path.parent, exist_ok=True)
    np.save(path, values)


def aggregate_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades by TRADE_TIME"""
    df_trades: pl.DataFrame = (
        df_ticks
        .group_by(TRADE_TIME, maintain_order=True)
        .agg(
            price_first=pl.col(PRICE).first(),  # if someone placed a trade with price impact, then price_first
            price_last=pl.col(PRICE).last(),  # and price_last will differ
            # Amount spent in quote asset for the trade
            quote_abs=pl.col("quote_abs").sum(),
            quote_sign=pl.col("quote_sign").sum(),
            quantity_sign=pl.col("quantity_sign").sum(),
            # Amount of base asset transacted
            quantity_abs=pl.col("quantity").sum(),
            num_ticks=pl.col("price").count(),  # number of ticks for each trade
        )
    )
    df_trades = df_trades.with_columns(is_long=pl.col("quantity_sign") >= 0)
    return df_trades
