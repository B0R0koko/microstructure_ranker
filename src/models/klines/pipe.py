from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import polars as pl
from tqdm import tqdm

from core.columns import CLOSE_TIME, CLOSE_PRICE, OPEN_PRICE
from core.currency import CurrencyPair
from core.feature_pipeline import EXCLUDED_SYMBOLS
from core.time_utils import Bounds
from models.klines.features.features_29_02 import compute_features


class KlinePipeline:
    """Define first feature pipeline here. Make sure to implement all methods from abstract parent class"""

    def __init__(self, hive_dir: Path):
        self._hive: pl.LazyFrame = pl.scan_parquet(hive_dir, hive_partitioning=True)

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
                pl.col(CLOSE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive)
            )
            .select("symbol").unique().collect()["symbol"].to_list()
        )

        return [
            CurrencyPair.from_string(symbol=symbol) for symbol in unique_symbols if symbol not in EXCLUDED_SYMBOLS
        ]

    def load_data_for_currency_pair(self, currency_pair: CurrencyPair, bounds: Bounds) -> pl.DataFrame:
        """Load data for a given CurrencyPair with specific time interval [start_time, end_time + return_timedelta)"""

        df_currency_pair: pl.LazyFrame = self._hive.filter(
            (pl.col("symbol") == currency_pair.name) &
            # Load data by filtering by both hive folder structure and columns inside each parquet file
            (pl.col("date").is_between(bounds.day0, bounds.day1)) &
            (pl.col(CLOSE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive))
        )

        return df_currency_pair.collect()

    def attach_target_for_currency_pair(
            self, currency_pair: CurrencyPair, bounds: Bounds, prediction_offset: timedelta
    ) -> float:
        """Attach target log_return column that we aim to predict"""
        effective_end_time: datetime = bounds.end_exclusive + prediction_offset

        currency_pair_log_return: float = (
            self._hive
            .filter(
                (pl.col("symbol") == currency_pair.name) &
                (pl.col("date").is_between(bounds.day0, bounds.day1)) &
                (pl.col(CLOSE_TIME).is_between(bounds.end_exclusive, effective_end_time))
            )
            .select((pl.col(CLOSE_PRICE).last() / pl.col(OPEN_PRICE).first()).log())
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

        for currency_pair in tqdm(currency_pairs):
            # Load and collect pl.DataFrame for current CurrencyPair, read to RAM no avoid calling collect multiple times
            df_currency_pair: pl.DataFrame = self.load_data_for_currency_pair(
                currency_pair=currency_pair, bounds=bounds
            )
            computed_features: Dict[str, Any] = self.compute_features_for_currency_pair(
                currency_pair=currency_pair, df_currency_pair=df_currency_pair, bounds=bounds
            )
            computed_log_return: float = self.attach_target_for_currency_pair(
                currency_pair=currency_pair, bounds=bounds, prediction_offset=timedelta(hours=1)
            )
            computed_features.update({"log_return": computed_log_return})
            cross_section_features.append(computed_features)

        return pl.DataFrame(cross_section_features)

    def compute_features_for_currency_pair(
            self, currency_pair: CurrencyPair, df_currency_pair: pl.DataFrame, bounds: Bounds
    ) -> Dict[str, Any]:
        """
        Given the data from df_currency_pair, call compute_features function on it
        which returns a mapping of feature names to their corresponding values
        """
        features: Dict[str, Any] = compute_features(
            df_currency_pair=df_currency_pair, currency_pair=currency_pair, bounds=bounds
        )
        return features


if __name__ == "__main__":
    hive_dir: Path = Path("D:/data/transformed/klines/1m")
    start_time: datetime = datetime(2024, 11, 1, 0, 0, 0)
    end_time: datetime = datetime(2024, 11, 2, 0, 0, 0)

    step: timedelta = timedelta(hours=1)
    interval: timedelta = timedelta(hours=4)

    bounds: Bounds = Bounds(start_inclusive=start_time, end_exclusive=end_time)
    cross_section_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=step, interval=interval)

    pipeline: KlinePipeline = KlinePipeline(hive_dir=hive_dir)
    df_features: pl.DataFrame = pipeline.load_cross_section(bounds=cross_section_bounds[0])

    print(df_features.head())
