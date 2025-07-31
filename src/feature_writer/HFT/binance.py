import logging
from datetime import date
from functools import partial
from multiprocessing import freeze_support, RLock, Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.columns import SYMBOL, TRADE_TIME, PRICE, IS_BUYER_MAKER, QUANTITY, SAMPLED_TIME
from core.currency import get_target_currencies, Currency
from core.currency_pair import CurrencyPair
from core.data_type import Feature, SamplingType
from core.exchange import Exchange
from core.time_utils import Bounds
from feature_writer.HFT.feature_exprs import *
from feature_writer.utils import write_feature, aggregate_into_trades
from models.prediction.features.utils import statistic_name

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


class SampledFeatureWriter:

    def __init__(self, bounds: Bounds, exchange: Exchange):
        hive_location: Path = exchange.get_hive_location()
        logging.info("FeatureWriter started for %s", exchange.name)
        logging.info("Scanning Hive location %s", hive_location)
        self._hive = pl.scan_parquet(hive_location, hive_partitioning=True)

        self.bounds: Bounds = bounds
        self.exchange: Exchange = exchange
        self.sampling_type: SamplingType = SamplingType.MS500

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

    @staticmethod
    def side_expr() -> pl.Expr:
        """
        Overwrite the way we compute side sign. For Binance we do it with IS_BUYER_MAKER field
        for OKX we use simply use Side Literal string
        """
        return 1 - 2 * pl.col(IS_BUYER_MAKER)

    def preprocess_data_for_currency(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.sort(by=TRADE_TIME, descending=False)
        df = df.with_columns(
            quote_abs=pl.col(PRICE) * pl.col(QUANTITY),
            side=self.side_expr(),
        )
        df = df.with_columns(
            quote_sign=pl.col("quote_abs") * pl.col("side"),
            quantity_sign=pl.col(QUANTITY) * pl.col("side")
        )
        # Aggregate into trades
        df_trades: pl.DataFrame = aggregate_into_trades(df_ticks=df)

        assert df_trades[TRADE_TIME].is_sorted(descending=False), "Data must be in ascending order by TRADE_TIME"

        # Compute slippages
        df_trades = df_trades.with_columns(
            quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
        )
        df_trades = df_trades.with_columns(
            quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign(),
            # Add lags of price_last and trade_time
            price_last_prev=pl.col("price_last").shift(1),
            trade_time_prev=pl.col(TRADE_TIME).shift(1)
        )
        return df_trades

    def write_features_daily(
            self, sampled_features: pl.DataFrame, currency_pair: CurrencyPair, features: List[Feature],
            window_td: timedelta,
    ) -> None:
        """Write sampled features to local filesystem"""
        # Resample feature by each day and save as np.ndarray to filesystem
        for (day,), daily_features in sampled_features.group_by_dynamic(
                index_column=SAMPLED_TIME, every=timedelta(days=1), period=timedelta(days=1),
        ):
            # Write time index
            write_feature(
                values=daily_features[SAMPLED_TIME].to_numpy(),
                day=day,
                sampling_type=self.sampling_type,
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

    def write_features_for_currency(self, chunk_bounds: Bounds, currency_pair: CurrencyPair, position: int):
        pbar = tqdm(
            SAMPLING_WINDOWS, desc="Reading file...",
            position=2 + position,
            leave=False
        )

        df: pl.DataFrame = self.load_data_for_currency(bounds=chunk_bounds, currency_pair=currency_pair)
        df = self.preprocess_data_for_currency(df=df)

        pbar.set_description(f"Computing features for {currency_pair.name}@{str(chunk_bounds)}")

        for window in pbar:

            features: Dict[Feature, pl.Expr] = {
                Feature.ASSET_RETURN: compute_return(),
                Feature.ASSET_HOLD_TIME: compute_asset_hold_time(),
                Feature.SHARE_OF_LONG_TRADES: compute_flow_imbalance(),
                Feature.POWERLAW_ALPHA: compute_slippage_imbalance(),
                Feature.SLIPPAGE_IMBALANCE: compute_powerlaw_alpha(),
                Feature.FLOW_IMBALANCE: compute_share_of_long_trades()
            }

            date_index: pd.DatetimeIndex = pd.date_range(
                chunk_bounds.start_inclusive, chunk_bounds.end_exclusive,
                freq=timedelta(milliseconds=500), inclusive="left"
            )
            df_index: pl.DataFrame = pl.DataFrame({SAMPLED_TIME: date_index})

            if window == timedelta(milliseconds=500):
                features[Feature.CLOSE_PRICE] = compute_close_price()

            sampled_features: pl.DataFrame = (
                df
                .group_by_dynamic(
                    index_column=TRADE_TIME, every=self.sampling_type.value, period=window, closed="right",
                    label="right",
                )
                # Compute features sampled for this WINDOW at 500ms frequency
                .agg(features.values())
                .filter(pl.col(TRADE_TIME).is_between(chunk_bounds.start_inclusive, chunk_bounds.end_exclusive))
                .with_columns(
                    compute_return_adj(window=window)
                )
            )

            # left join to desired time index to make sure that dimensions are correct
            sampled_features = (
                df_index
                .join(sampled_features, left_on=SAMPLED_TIME, right_on=TRADE_TIME, how="left")
                .with_columns(
                    # Post-process some features
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

    def run_in_multiprocessing_pool(self, currency_pairs: List[CurrencyPair], cpu_count: int = 10) -> None:
        """Run daily feature writer using all cpu cores, susceptible to RAM limit"""
        # We want to parallelize over (CurrencyPair, day) to avoid cases when all workers are finished,
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
                                self.write_features_for_currency, chunk_bounds=Bounds.for_day(day),
                                currency_pair=currency_pair, position=i % cpu_count
                            )
                        )
                    )
                    i += 1

            for p in tqdm(promises, desc="Overall progress", position=0):
                p.get()


def run_main():
    # Run SampledFeatureWriter from here,
    # set PYTHONPATH to src folder and run from terminal such that Process progress bar is displayed correctly
    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 25)
    )
    writer = SampledFeatureWriter(bounds=bounds, exchange=Exchange.BINANCE_USDM)
    currency_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency.name, term=Currency.USDT.name) for currency in get_target_currencies()
    ]
    writer.run_in_multiprocessing_pool(currency_pairs=currency_pairs, cpu_count=3)


if __name__ == "__main__":
    run_main()
