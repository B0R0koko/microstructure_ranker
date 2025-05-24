import gc
import logging
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

from core.columns import SYMBOL, TRADE_TIME, PRICE, QUANTITY, SAMPLED_TIME
from core.currency import Currency, get_target_currencies
from core.currency_pair import CurrencyPair
from core.data_type import SamplingType, Feature
from core.exchange import Exchange
from core.paths import FEATURE_DIR
from core.time_utils import Bounds, format_date
from feature_writer.feature_exprs import compute_asset_hold_time, compute_flow_imbalance, compute_return, \
    compute_slippage_imbalance, compute_powerlaw_alpha, compute_share_of_long_trades, compute_sigma, compute_return_adj, \
    compute_close_price
from models.prediction.features import statistic_name

SAMPLING_WINDOWS: List[timedelta] = [
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


def write_feature(values: np.ndarray, day: date, sampling_type: SamplingType, subpath: Path) -> None:
    """Writes sampled np.ndarray with feature to {FEATURE_DIR}/HFT/20250101/MS500/asset_return/ADA"""
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / subpath
    os.makedirs(path.parent, exist_ok=True)
    np.save(path, values)


class HFTFeatureWriter:
    """Define HFT feature writer that will sample MS500 features using Polars group_by_dynamic"""

    def __init__(
            self, bounds: Bounds, exchange: Exchange, sampling_type: SamplingType = SamplingType.MS500
    ):
        logging.info("FeatureWriter started for %s", exchange.name)
        self._hive = pl.scan_parquet(exchange.get_hive_location(), hive_partitioning=True)
        self.exchange: Exchange = exchange
        self.bounds: Bounds = bounds
        self.sampling_type: SamplingType = sampling_type

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

    def preprocess_data(self, df_ticks: pl.DataFrame) -> pl.DataFrame:
        """Aggregate ticks into trades and add additional columns here that will be used to compute features"""
        df_ticks = df_ticks.sort(by=TRADE_TIME, descending=False)
        df_ticks = df_ticks.with_columns(
            quote_abs=pl.col(PRICE) * pl.col(QUANTITY),
            side=1 - 2 * pl.col("is_buyer_maker"),
        )
        df_ticks = df_ticks.with_columns(
            quote_sign=pl.col("quote_abs") * pl.col("side"),
            quantity_sign=pl.col("quantity") * pl.col("side")
        )
        # Aggregate into trades
        df_trades: pl.DataFrame = aggregate_into_trades(df_ticks=df_ticks)

        assert df_trades[TRADE_TIME].is_sorted(descending=False), "Data must be in ascending order by TRADE_TIME"

        # Compute slippages
        df_trades = df_trades.with_columns(
            quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
        )
        df_trades = df_trades.with_columns(
            quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign(),
            # Add lags of price_last and trade_time
            price_last_prev=pl.col("price_last").shift(1),
            trade_time_prev=pl.col("trade_time").shift(1)
        )
        return df_trades

    def write_features_daily(
            self,
            sampled_features: pl.DataFrame,
            currency_pair: CurrencyPair,
            features: List[Feature],
            window_td: timedelta,
    ) -> None:
        """Write sampled features to local filesystem"""
        # Resample feature by each day and save as np.ndarray to filesystem
        for (day,), daily_features in sampled_features.group_by_dynamic(
                index_column=SAMPLED_TIME, every=timedelta(days=1), period=timedelta(days=1),
        ):
            # Write time index
            write_feature(
                values=daily_features[SAMPLED_TIME].to_numpy(), day=day, sampling_type=self.sampling_type,
                subpath=Path("time")
            )

            for feature in features:
                values: np.ndarray = daily_features[feature.value].to_numpy()
                assert len(values) == self.sampling_type.get_valid_size(), "Invalid shape for features"
                # Save feature to local filesystem
                name: str = statistic_name(
                    feature=feature, exchange=self.exchange, currency_pair=currency_pair, window=window_td
                )
                write_feature(
                    values=values, sampling_type=self.sampling_type, day=day,
                    subpath=Path(feature.value) / self.exchange.name / name
                )

    def compute_features(self, df_trades: pl.DataFrame, bounds: Bounds, currency_pair: CurrencyPair, position: int):
        """Define how features are computed here"""
        pbar = tqdm(
            SAMPLING_WINDOWS, desc=f"Computing features for {currency_pair.name}@{str(bounds)}...",
            position=2 + position,
            leave=False
        )

        for window in pbar:

            features: Dict[Feature, pl.Expr] = {
                Feature.ASSET_RETURN: compute_return(),
                Feature.ASSET_HOLD_TIME: compute_asset_hold_time(),
                Feature.SIGMA: compute_sigma(),
                Feature.SHARE_OF_LONG_TRADES: compute_flow_imbalance(),
                Feature.POWERLAW_ALPHA: compute_slippage_imbalance(),
                Feature.SLIPPAGE_IMBALANCE: compute_powerlaw_alpha(),
                Feature.FLOW_IMBALANCE: compute_share_of_long_trades()
            }

            date_index: pd.DatetimeIndex = pd.date_range(
                bounds.start_inclusive, bounds.end_exclusive, freq=timedelta(milliseconds=500), inclusive="left"
            )
            df_index: pl.DataFrame = pl.DataFrame({SAMPLED_TIME: date_index})

            if window == timedelta(milliseconds=500):
                features[Feature.CLOSE_PRICE] = compute_close_price()

            sampled_features: pl.DataFrame = (
                df_trades
                .group_by_dynamic(
                    index_column=TRADE_TIME, every=self.sampling_type.value, period=window, closed="right",
                    label="right",
                )
                # Compute features sampled for this WINDOW at 500ms frequency
                .agg(features.values())
                .filter(pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
                .with_columns(
                    compute_return_adj(window=window)
                )
            )

            # left join to desired time index to make sure that dimensions are correct
            sampled_features = (
                df_index
                .join(sampled_features, left_on=SAMPLED_TIME, right_on=TRADE_TIME, how="left")
                .with_columns(
                    # Post process some features
                    # 1. fill missing values in asset_return with 0
                    asset_return=pl.col("asset_return").fill_null(0),
                    asset_return_adj=compute_return_adj(window=window)
                )
            )

            if Feature.CLOSE_PRICE in features:
                sampled_features = sampled_features.with_columns(
                    close_price=pl.col("close_price").forward_fill()
                )

            # Save features to daily structure
            features_to_save: List[Feature] = list(features.keys()) + [Feature.ASSET_RETURN_ADJ]

            self.write_features_daily(
                sampled_features=sampled_features,
                currency_pair=currency_pair,
                features=features_to_save,
                window_td=window
            )

    def generate_features_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds, position: int) -> None:
        # Expand bounds such that ends of interval are computed as well
        expanded_bounds: Bounds = bounds.expand_bounds(
            lb_timedelta=timedelta(minutes=5), rb_timedelta=timedelta(minutes=5)
        )
        read_bar = tqdm(
            total=1,
            desc=f"Reading data for {currency_pair.name}@{str(bounds)}...",
            position=2 + position,
            leave=False
        )
        # Collect data with expanded bounds such that the ends of features are computed
        df_ticks: pl.DataFrame = self.load_data_for_currency(bounds=expanded_bounds, currency_pair=currency_pair)
        read_bar.update(1)

        df_trades: pl.DataFrame = self.preprocess_data(df_ticks=df_ticks)

        del df_ticks  # free RAM from df_trades
        gc.collect()

        self.compute_features(df_trades=df_trades, bounds=bounds, currency_pair=currency_pair, position=position)

        del df_trades
        gc.collect()

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

            for day in self.bounds.date_range():
                for currency_pair in currency_pairs:
                    promises.append(
                        pool.apply_async(
                            partial(
                                self.generate_features_for_currency_pair, bounds=Bounds.for_day(day),
                                currency_pair=currency_pair, position=i % cpu_count
                            )
                        )
                    )
                    i += 1

            for p in tqdm(promises, desc="Overall progress", position=0):
                p.get()


if __name__ == "__main__":
    bounds = Bounds.for_days(date(2024, 1, 1), date(2024, 2, 1))
    exchange: Exchange = Exchange.BINANCE_USDM

    writer = HFTFeatureWriter(exchange=exchange, bounds=bounds)

    writer.run_in_multiprocessing_pool(
        currency_pairs=[CurrencyPair(base=currency, term=Currency.USDT) for currency in get_target_currencies()],
        cpu_count=5
    )
