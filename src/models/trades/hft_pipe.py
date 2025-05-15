import logging
from datetime import date, timedelta, datetime
from functools import partial
from multiprocessing import Pool, freeze_support, RLock
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Dict, List, Any

import polars as pl
from tqdm import tqdm

from core.columns import TRADE_TIME, SYMBOL, PRICE
from core.currency import CurrencyPair
from core.paths import HIVE_TRADES, FEATURE_DIR
from core.time_utils import Bounds, start_of_the_day
from models.trades.features.high_frequency_features import compute_features


def init_worker(tqdm_lock: RLock):
    """Pool initializer: hook up tqdm’s lock."""
    tqdm.set_lock(tqdm_lock)


def generate_intervals(bounds: Bounds, interval: timedelta, step: timedelta):
    lb: datetime = bounds.start_inclusive
    while True:
        rb: datetime = lb + interval
        yield lb, rb
        lb += step
        if rb >= bounds.end_exclusive:
            break


def idx_before(times: pl.Series, ts: datetime) -> int:
    idx = times.search_sorted(ts)
    return max(idx - 1, 0)


def idx_after(times: pl.Series, ts: datetime) -> int:
    idx = times.search_sorted(ts)
    return min(idx, len(times) - 1)


class HFTFeaturePipeline:
    """
    Pipeline that writes each worker’s output to its own shard file,
    then merges shards at the end to avoid append-related errors.
    """

    def __init__(
            self,
            interval: timedelta,
            step: timedelta,
            prediction_step: timedelta,
            output_features_path: Path,
            num_processes: int = 10
    ):
        self._hive = pl.scan_parquet(HIVE_TRADES, hive_partitioning=True)
        self.interval = interval
        self.step = step
        self.prediction_step = prediction_step
        self.output_features_path = output_features_path
        self.num_processes = num_processes

        # Directory for intermediate shard files
        self.shard_dir = output_features_path.with_name(output_features_path.stem + "_shards")
        self.shard_dir.mkdir(exist_ok=True, parents=True)

    def load_data_for_currency_pair(self, bounds: Bounds, currency_pair: CurrencyPair) -> pl.DataFrame:
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

    def write_shard(self, df: pl.DataFrame, shard_name: str) -> None:
        """
        Write this worker’s features to its own shard file.
        """
        out_path = self.shard_dir / f"{shard_name}.parquet"
        # Use Polars native writer:
        df.write_parquet(out_path)

    def create_cross_sections_for_currency_pair(
            self, bounds: Bounds, currency_pair: CurrencyPair, position: int
    ) -> None:
        """Compute multiple intervals for a given currency pair over bounds"""
        expanded_bounds: Bounds = bounds.expand_bounds(rb_timedelta=timedelta(minutes=10))
        df_ticks = self.load_data_for_currency_pair(expanded_bounds, currency_pair)

        times = df_ticks[TRADE_TIME]
        prices = df_ticks[PRICE]
        features: List[Dict[str, Any]] = []

        total = ((bounds.end_exclusive - bounds.start_inclusive - self.interval) // self.step) + 1
        pbar = tqdm(
            total=total,
            desc=f"{currency_pair.binance_name}@[{bounds.start_inclusive}: {bounds.end_exclusive})",
            position=2 + position,
            leave=False
        )

        for lb, rb in generate_intervals(bounds, self.interval, self.step):
            df_int = df_ticks.filter(pl.col(TRADE_TIME).is_between(lb, rb))
            comp = compute_features(df_int, currency_pair, Bounds(lb, rb))

            b_idx = idx_before(times, rb)
            a_idx = idx_after(times, rb + self.prediction_step)
            p0, p1 = prices[b_idx], prices[a_idx]
            return_pips: float = (p1 / p0 - 1) * 1e4
            t0, t1 = times[b_idx], times[a_idx]
            hold = (t1 - t0).total_seconds()

            features.append({
                **comp,
                "return_pips": return_pips,
                "hold_time_ms": hold,
                "start_time": lb,
                "end_time": rb,
                "starting_trade": t0,
                "ending_trade": t1,
                "return_pips_adjusted": return_pips / (hold / self.prediction_step.total_seconds()),
            })
            pbar.update(1)

        shard_tag: str = f"{currency_pair.name}@{bounds.start_inclusive.strftime("%y-%m-%d_%H-%M-%S")}"
        self.write_shard(df=pl.DataFrame(features), shard_name=shard_tag)

    def cleanup(self) -> None:
        # 2) Merge all shards into the final parquet
        shard_files = sorted(self.shard_dir.glob("*.parquet"))

        if not shard_files:
            raise RuntimeError("No shard files found to merge!")

        logging.info("Merging %d shard files...", len(shard_files))
        df_final: pl.DataFrame = pl.concat([pl.read_parquet(p) for p in shard_files])
        df_final.write_parquet(self.output_features_path)
        logging.info("Final features written to %s", self.output_features_path)

        # clean up shards:
        for p in shard_files:
            p.unlink()
        self.shard_dir.rmdir()

    def create_cross_sections(self, bounds: Bounds, currency_pairs: List[CurrencyPair]) -> None:
        freeze_support()  # Windows
        tqdm.set_lock(RLock())  # for interleaved bars

        with Pool(
                processes=self.num_processes,
                initializer=init_worker,
                initargs=(tqdm.get_lock(),),
                maxtasksperchild=1
        ) as pool:

            promises: List[AsyncResult] = []
            i = 0

            for sub in bounds.iter_hours():
                for cp in currency_pairs:
                    promises.append(
                        pool.apply_async(
                            partial(
                                self.create_cross_sections_for_currency_pair,
                                bounds=sub,
                                currency_pair=cp,
                                position=i % self.num_processes
                            )
                        )
                    )
                    i += 1

            for p in tqdm(promises, desc="Overall progress", position=0):
                p.get()

        self.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bounds = Bounds(
        start_of_the_day(date(2024, 1, 1)),
        datetime(2024, 1, 10)
    )
    out = FEATURE_DIR / "features_2025-05-15@RETURN_4S.parquet"

    pipe = HFTFeaturePipeline(
        interval=timedelta(minutes=5),
        step=timedelta(seconds=1),
        prediction_step=timedelta(seconds=1),
        output_features_path=out,
        num_processes=15
    )
    pipe.create_cross_sections(
        bounds=bounds,
        currency_pairs=[
            CurrencyPair.from_string("BTC-USDT"),
            CurrencyPair.from_string("ETH-USDT"),
            CurrencyPair.from_string("ADA-USDT")
        ],
    )
