import gc
import json
import os
from datetime import timedelta, date, datetime
from functools import partial
from multiprocessing import Pool, freeze_support, RLock
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl
from tqdm import tqdm

from core.columns import TRADE_TIME, SYMBOL, PRICE
from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset
from models.trades.features.features_27_11 import compute_features

USE_COLS: List[str] = ["price", "quantity", "trade_time", "is_buyer_maker"]
PROGRESS_FILE: Path = Path("src/core/progress.json")


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


class ProgressTracker:

    def __init__(self, progress_path: Path, task_map: Dict[str, Dict[str, Any]]):
        self.progress_path: Path = progress_path
        self.task_map: Dict[str, Dict[str, Any]] = task_map

    @classmethod
    def from_warm_start(cls, progress_path: Path) -> "ProgressTracker":
        assert os.path.exists(progress_path), "Progress path does not exist"
        with open(progress_path, "r") as file:
            task_map: Dict[str, Dict[str, Any]] = json.load(file)
        return cls(progress_path=progress_path, task_map=task_map)

    def iterate(self) -> Tuple[str, Dict[str, Any]]:
        task_map_copy = self.task_map.copy()
        for task_id, kwargs in task_map_copy.items():
            yield task_id, kwargs

    def complete(self, task_id: str) -> None:
        del self.task_map[task_id]
        gc.collect()

        with open(self.progress_path, "w") as file:
            json.dump(self.task_map, file)  # type:ignore


class TradesPipeline:
    """Define first feature pipeline here. Make sure to implement all methods from abstract parent class"""

    def __init__(
            self, hive_dir: Path, output_features_path: Path, warmup_start: bool = False, num_processes: int = 5
    ):
        self._hive = pl.scan_parquet(source=hive_dir, hive_partitioning=True)
        self.output_features_path: Path = output_features_path
        self.warmup_start: bool = warmup_start
        self.num_processes: int = num_processes

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

        return [CurrencyPair.from_string(symbol=symbol) for symbol in set(unique_symbols)]

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

    def load_cross_section(self, bounds: Bounds, task_id: str, position: int) -> Tuple[str, pl.DataFrame]:
        """
        This function runs self.compute_features_for_currency_pair for each of the currency_pair available within
        a given range of time defined by passed in start_time and end_time. Returns pl.DataFrame with all features
        attached as well as task_id
        """
        currency_pairs: List[CurrencyPair] = self.get_currency_pairs_for_cross_section(bounds=bounds)
        cross_section_features: List[Dict[str, Any]] = []

        pbar = tqdm(
            total=len(currency_pairs[:10]),
            desc=f"{task_id}: {str(bounds)}",
            position=2 + position,
            leave=False
        )

        for currency_pair in currency_pairs[:10]:
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
            cross_section_features.append(currency_pair_features)

            # Delete collected data from ram to perhaps free up some ram as we get a lot of MemoryErrors
            del df_currency_pair
            gc.collect()

            pbar.update(1)

        df_cross_section: pl.DataFrame = pl.DataFrame(cross_section_features)
        df_cross_section = df_cross_section.with_columns(
            pl.lit(value=bounds.start_inclusive).alias("cross_section_start_time"),
            pl.lit(value=bounds.end_exclusive).alias("cross_section_end_time"),
        )
        return task_id, df_cross_section

    def write_features(self, df: pl.DataFrame) -> None:
        """Output features to parquet file in append mode"""
        df.to_pandas().to_parquet(
            self.output_features_path, engine="fastparquet",
            append=self.output_features_path.exists()  # if the file exists we will append to it
        )

    def get_process_tracker(self, cross_section_bounds: List[Bounds]) -> ProgressTracker:
        """Returns an instance of ProgressTracker with task_map"""
        if self.warmup_start:
            print("Warmup start")
            return ProgressTracker.from_warm_start(progress_path=PROGRESS_FILE)

        # Task map must be json serializable
        task_map: Dict[str, Dict[str, Any]] = {
            # Task id and kwargs
            f"task_{i}": {
                "start_inclusive": str(bounds.start_inclusive),
                "end_exclusive": str(bounds.end_exclusive),
                "task_id": f"task_{i}"
            }
            for i, bounds in enumerate(cross_section_bounds)
        }

        return ProgressTracker(progress_path=PROGRESS_FILE, task_map=task_map)

    # Parallelize this function to be able to run at least using multiple processes
    # Added progress bars with the help of
    # https://github.com/tqdm/tqdm?tab=readme-ov-file#nested-progress-bars
    def load_multiple_cross_sections(self, cross_section_bounds: List[Bounds]) -> None:

        freeze_support()  # for Windows support
        tqdm.set_lock(RLock())  # for managing output contention

        progress_tracker: ProgressTracker = self.get_process_tracker(cross_section_bounds=cross_section_bounds)
        position: int = 0

        with Pool(processes=self.num_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
            promises: List[AsyncResult] = []

            for task_id, kwargs in progress_tracker.iterate():
                bounds: Bounds = Bounds.from_datetime_str(
                    start_inclusive=kwargs["start_inclusive"], end_exclusive=kwargs["end_exclusive"]
                )
                promise: AsyncResult = pool.apply_async(
                    partial(
                        self.load_cross_section, bounds=bounds, task_id=task_id, position=position % self.num_processes
                    ),
                )
                position += 1
                promises.append(promise)

            for promise in tqdm(promises, desc="Overall progress", position=0):
                task_id, df_cross_section = promise.get()  # fetch output of self.load_cross_section from Future
                self.write_features(df=df_cross_section)

                del df_cross_section
                gc.collect()
                # Update ProgressTracker
                progress_tracker.complete(task_id=task_id)


def _test_main():
    hive_dir: Path = Path("D:/data/transformed/trades")
    output_features_path: Path = Path("D:/data/features/features_19-02-2025.parquet")

    start_date: date = date(2024, 11, 8)
    end_date: date = date(2024, 11, 30)
    bounds: Bounds = Bounds.for_days(start_date, end_date)

    step: timedelta = timedelta(hours=1)
    interval: timedelta = timedelta(hours=12)

    cross_section_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=step, interval=interval)

    pipeline: TradesPipeline = TradesPipeline(
        hive_dir=hive_dir, output_features_path=output_features_path, num_processes=20, warmup_start=False
    )
    pipeline.load_multiple_cross_sections(cross_section_bounds=cross_section_bounds)


if __name__ == "__main__":
    _test_main()
