import gc
from datetime import timedelta, date, datetime
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Dict, Any, List

import polars as pl
from tqdm import tqdm

from core.columns import TRADE_TIME, SYMBOL, PRICE
from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset
from models.trades.features.features_27_11 import compute_features

EXCLUDED_SYMBOLS: set[str] = {"BTC-USDT", "ETH-USDT"}
USE_COLS: List[str] = ["price", "quantity", "trade_time", "is_buyer_maker"]


def compute_features_for_currency_pair(
        currency_pair: CurrencyPair, df_currency_pair: pl.DataFrame, bounds: Bounds
) -> Dict[str, Any]:
    """
    Given the data from df_currency_pair, call compute_features function on it
    which returns a mapping of feature names to their corresponding values
    """
    features: Dict[str, Any] = compute_features(
        df_currency_pair=df_currency_pair, currency_pair=currency_pair, bounds=bounds
    )
    return features


class TradesPipeline:
    """Define first feature pipeline here. Make sure to implement all methods from abstract parent class"""

    def __init__(self, hive_dir: Path):
        self._hive = pl.scan_parquet(source=hive_dir, hive_partitioning=True)

    def get_currency_pairs_for_cross_section(self, bounds: Bounds) -> List[CurrencyPair]:
        """
        Returns a list of CurrencyPair for which there is data stored in self.hive_dir: Path for the
        given time interval
        """
        # Extract dates of start_time and end_time
        unique_symbols: set[str] = set(
            self._hive
            .filter(
                pl.col("date").is_between(bounds.day0, bounds.day1) &
                pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive)
            )
            .select(SYMBOL).unique().collect()[SYMBOL].to_list()
        )

        return [CurrencyPair.from_string(symbol=symbol) for symbol in set(unique_symbols) - EXCLUDED_SYMBOLS]

    def load_data_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds) -> pl.DataFrame:
        """Load data for a given CurrencyPair with specific time interval [start_time, end_time + return_timedelta)"""
        df_currency_pair: pl.LazyFrame = self._hive.filter(
            (pl.col(SYMBOL) == currency_pair.name) &
            # Load data by filtering by both hive folder structure and columns inside each parquet file
            (pl.col("date").is_between(bounds.day0, bounds.day1)) &
            (pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
        )

        return df_currency_pair.select(USE_COLS).collect()

    def attach_target_for_currency_pair(
            self, currency_pair: CurrencyPair, bounds: Bounds, prediction_offset: timedelta
    ) -> float:
        """Attach target log_return column that we aim to predict"""
        effective_end_time: datetime = bounds.end_exclusive + prediction_offset

        currency_pair_log_return: float = (
            self._hive
            .filter(
                (pl.col(SYMBOL) == currency_pair.name) &
                (pl.col("date").is_between(bounds.day0, bounds.day1)) &
                (pl.col(TRADE_TIME).is_between(bounds.end_exclusive, effective_end_time))
            )
            .select((pl.col(PRICE).last() / pl.col(PRICE).first()).log())
            .collect()
            .item()
        )

        return currency_pair_log_return

    def load_cross_section(self, bounds: Bounds) -> pl.DataFrame:
        """
        This function runs self.compute_features_for_currency_pair for each of the currency_pair available within
        a given range of time defined by passed in start_time and end_time. Returns pl.DataFrame with all features
        attached
        """
        currency_pairs: List[CurrencyPair] = self.get_currency_pairs_for_cross_section(bounds=bounds)
        cross_section_features: List[Dict[str, Any]] = []

        for currency_pair in currency_pairs:
            # Load and collect pl.DataFrame for current CurrencyPair, read to RAM no avoid calling collect multiple times
            df_currency_pair: pl.DataFrame = self.load_data_for_currency_pair(
                currency_pair=currency_pair, bounds=bounds
            )
            # Compute features using loaded pl.DataFrame
            currency_pair_features: Dict[str, Any] = compute_features_for_currency_pair(
                currency_pair=currency_pair, df_currency_pair=df_currency_pair, bounds=bounds
            )
            currency_pair_features["log_return"] = self.attach_target_for_currency_pair(
                currency_pair=currency_pair,
                bounds=bounds,
                prediction_offset=TimeOffset.HOUR.value
            )
            # Delete collected data from ram to perhaps free up some ram as we get a lot of MemoryErrors
            del df_currency_pair
            gc.collect()

            cross_section_features.append(currency_pair_features)

        df_cross_section: pl.DataFrame = pl.DataFrame(cross_section_features)
        df_cross_section = df_cross_section.with_columns(
            pl.lit(value=bounds.start_inclusive).alias("cross_section_start_time"),
            pl.lit(value=bounds.end_exclusive).alias("cross_section_end_time"),
        )
        return df_cross_section

    # Parallelize this function to be able to run at least using 10 processes
    def load_multiple_cross_sections(self, cross_section_bounds: List[Bounds]) -> pl.DataFrame:
        dfs: List[pl.DataFrame] = []

        with (
            tqdm(total=len(cross_section_bounds), desc="Computing cross-sections with multiprocessing: ") as pbar,
            Pool(processes=5) as pool,
        ):
            promises: List[AsyncResult] = []

            for bounds in cross_section_bounds:
                promise: AsyncResult = pool.apply_async(
                    partial(self.load_cross_section, bounds=bounds)
                )
                promises.append(promise)

            for promise in promises:
                df_cross_section: pl.DataFrame = promise.get()  # fetch output of self.load_cross_section from Future
                dfs.append(df_cross_section)

                pbar.update(1)

        return pl.concat(dfs)


def _test_main():
    hive_dir: Path = Path("D:/data/transformed/trades")

    start_date: date = date(2024, 11, 1)
    end_date: date = date(2024, 11, 30)
    bounds: Bounds = Bounds.for_days(start_date, end_date)

    step: timedelta = timedelta(hours=1)
    interval: timedelta = timedelta(hours=4)

    cross_section_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=step, interval=interval)

    pipeline: TradesPipeline = TradesPipeline(hive_dir=hive_dir)
    df_features: pl.DataFrame = pipeline.load_multiple_cross_sections(cross_section_bounds=cross_section_bounds)

    df_features.to_pandas().to_parquet("D:/data/features/features_10-02-2025.parquet", index=False)


if __name__ == "__main__":
    _test_main()
