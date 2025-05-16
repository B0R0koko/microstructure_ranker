import os
from datetime import timedelta, date
from functools import partial
from multiprocessing import freeze_support, RLock, Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from core.columns import SYMBOL, TRADE_TIME, QUANTITY, PRICE
from core.currency import CurrencyPair
from core.data_type import DataType, SamplingType
from core.paths import HIVE_TRADES, FEATURE_DIR
from core.time_utils import Bounds, format_date, get_seconds_slug

WINDOWS: List[timedelta] = [
    timedelta(milliseconds=500),
    timedelta(seconds=1),
    timedelta(seconds=2),
    timedelta(seconds=5),
    timedelta(seconds=10),
    timedelta(seconds=30),
    timedelta(minutes=1),
    timedelta(minutes=2),
    timedelta(minutes=5),
]


def aggregate_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    df_trades: pl.DataFrame = (
        df_ticks
        .group_by(TRADE_TIME, maintain_order=True)
        .agg(
            price_first=pl.col("price").first(),  # if someone placed a trade with price impact, then price_first
            price_last=pl.col("price").last(),  # and price_last will differ
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


def preprocess_ticks(df_ticks: pl.DataFrame) -> pl.DataFrame:
    df_ticks = df_ticks.sort(by=TRADE_TIME, descending=False)
    df_ticks = df_ticks.with_columns(
        quote_abs=pl.col(PRICE) * pl.col(QUANTITY),
        side=1 - 2 * pl.col("is_buyer_maker")  # -1 if SELL, 1 if BUY
    )
    df_ticks = df_ticks.with_columns(
        quote_sign=pl.col("quote_abs") * pl.col("side"),
        quantity_sign=pl.col("quantity") * pl.col("side")
    )
    return df_ticks


def write_features_daily(
        sampled_features: pl.DataFrame,
        write_cols: Dict[DataType, str],
        currency_pair: CurrencyPair,
        sampling_type: SamplingType,
        window_slug: str
) -> None:
    for (day, _, _), features_daily in (
            sampled_features.group_by_dynamic(
                index_column=TRADE_TIME, every=timedelta(days=1),
                period=timedelta(days=1), include_boundaries=True
            )
    ):
        for data_type, col in write_cols.items():
            values: np.ndarray = features_daily[col].to_numpy()
            assert len(values) == sampling_type.get_valid_size(), "Invalid shape for features"
            write_feature(
                values=features_daily[col].to_numpy(),
                data_type=data_type,
                sampling_type=sampling_type,
                day=day,
                name=f"{data_type.value}-{currency_pair.name}-{window_slug}",
            )


def write_feature(values: np.ndarray, data_type: DataType, day: date, sampling_type: SamplingType, name: str) -> None:
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / data_type.value / name
    os.makedirs(path.parent, exist_ok=True)
    np.save(path, values)


def feature_pipeline(
        df_ticks: pl.DataFrame,
        bounds: Bounds,
        currency_pair: CurrencyPair,
        pbar,
        sampling_type: SamplingType = SamplingType.MS500,
) -> None:
    df_ticks = preprocess_ticks(df_ticks=df_ticks)
    df_trades: pl.DataFrame = aggregate_into_trades(df_ticks=df_ticks)

    assert df_trades[TRADE_TIME].is_sorted(descending=False), "Data must be in ascending order by TRADE_TIME"

    # Compute slippages
    df_trades = df_trades.with_columns(
        quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
    )
    df_trades = df_trades.with_columns(
        quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign()
    )

    df_trades = df_trades.with_columns(
        price_last_lag=pl.col("price_last").shift(1),
        trade_time_lag=pl.col("trade_time").shift(1),
        price_first_neg_lag=pl.col("price_first").shift(-1),
        trade_time_neg_lag=pl.col("trade_time").shift(-1)
    )

    for window in pbar:
        index: pd.DatetimeIndex = pd.date_range(
            bounds.start_inclusive, bounds.end_exclusive, freq=timedelta(milliseconds=500)
        )
        df_index: pl.DataFrame = pl.DataFrame({"sampled_time": index})

        sampled_features: pl.DataFrame = (
            df_trades
            .group_by_dynamic(
                index_column=TRADE_TIME,
                every=sampling_type.value,
                period=window,
                closed="right",
                label="right",
            )
            # Compute features sampled for this WINDOW at 500ms frequency
            .agg(
                asset_return=(pl.col("price_first_neg_lag").last() / pl.col("price_last_lag").first() - 1) * 1e4,
                asset_hold_time=(
                        (pl.col("trade_time_neg_lag").last() - pl.col(
                            "trade_time_lag").first()).dt.total_nanoseconds() / 1e9
                ),
                flow_imbalance=pl.col("quote_sign").sum() / pl.col("quote_abs").sum(),
                slippage_imbalance=pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum(),
                powerlaw_alpha=1 + pl.len() / (pl.col("quote_abs") / pl.col("quote_abs").min()).log().sum(),
                share_of_long_trades=pl.col("is_long").sum() / pl.len(),
            )
            # Trim ends with empty features
            .filter(
                pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive)
            )
        )
        # left join to desired time index to make sure that dimensions are correct
        df_features: pl.DataFrame = df_index.join(
            sampled_features, left_on="sampled_time", right_on="trade_time", how="left"
        )
        # Sample daily to write to numpy array files
        write_features_daily(
            sampled_features=df_features,
            write_cols={
                DataType.ASSET_RETURN: "asset_return",
                DataType.ASSET_HOLD_TIME: "asset_hold_time",
                DataType.FLOW_IMBALANCE: "flow_imbalance",
                DataType.SLIPPAGE_IMBALANCE: "slippage_imbalance",
                DataType.POWERLAW_ALPHA: "powerlaw_alpha",
                DataType.SHARE_OF_LONG_TRADES: "share_of_long_trades",
            },
            currency_pair=currency_pair,
            sampling_type=sampling_type,
            window_slug=get_seconds_slug(td=window),
        )


class HFTFeatureWriter:
    """Define HFT feature writer that will sample MS500 features using Polars group_by_dynamic"""

    def __init__(self, bounds: Bounds):
        self._hive = pl.scan_parquet(HIVE_TRADES, hive_partitioning=True)
        self.bounds: Bounds = bounds

    def load_data_for_currency(self, bounds: Bounds, currency_pair: CurrencyPair):
        """Load data for currency from HiveDataset"""
        return (
            self._hive
            .filter(
                (pl.col(SYMBOL) == currency_pair.name) &
                (pl.col("date").is_between(bounds.day0, bounds.day1)) &
                (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
            )
            .collect()
            .sort(by=TRADE_TIME)
        )

    def generate_features_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds, position: int) -> None:
        pbar = tqdm(WINDOWS, position=2 + position, desc=f"Reading file...", leave=False)
        # Expand bounds such that ends of interval are computed as well
        expanded_bounds: Bounds = bounds.expand_bounds(
            lb_timedelta=timedelta(minutes=5), rb_timedelta=timedelta(minutes=5)
        )
        # Collect data with expanded bounds such that the ends of features are computed
        df_ticks: pl.DataFrame = self.load_data_for_currency(bounds=expanded_bounds, currency_pair=currency_pair)
        pbar.set_description(desc=f"{currency_pair.name}@{str(bounds)}")
        feature_pipeline(df_ticks=df_ticks, bounds=bounds, currency_pair=currency_pair, pbar=pbar)

    def run_in_multiprocessing_pool(self, currency_pairs: List[CurrencyPair], cpu_count: int = 10) -> None:
        """Run daily feature writer using all cpu cores, susceptible to RAM limit"""
        # We want to parallelize over (CurrencyPair, day) to avoid cases when all workers are finished
        # and there is only one currency pair left that is run in the single process
        freeze_support()  # for Windows support
        tqdm.set_lock(RLock())  # for managing output contention

        with Pool(processes=cpu_count, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),),
                  maxtasksperchild=1) as pool:

            promises: List[AsyncResult] = []
            i: int = 0

            for sub_bounds in self.bounds.iter_days():
                for currency_pair in currency_pairs:
                    promises.append(
                        pool.apply_async(
                            partial(
                                self.generate_features_for_currency_pair, bounds=sub_bounds,
                                currency_pair=currency_pair, position=i % cpu_count
                            )
                        )
                    )
                    i += 1

            for p in tqdm(promises, desc="Overall progress", position=0):
                p.get()


if __name__ == "__main__":
    bounds = Bounds.for_days(date(2024, 11, 1), date(2024, 11, 20))

    HFTFeatureWriter(bounds=bounds).run_in_multiprocessing_pool(
        currency_pairs=[
            CurrencyPair.from_string("BTC-USDT"),
            CurrencyPair.from_string("ETH-USDT"),
            CurrencyPair.from_string("ADA-USDT"),
        ]
    )
