from datetime import date
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from core.columns import TRADE_TIME
from core.paths import OKX_SPOT_HIVE_TRADES, OKX_SPOT_RAW_TRADES
from core.time_utils import Bounds

_RAW_COLS: List[str] = [
    # instrument_name, trade_id, side, size, price, created_time
    "symbol", "trade_id", "side", "quantity", "price", "trade_time"
]

_USE_COLS: List[str] = ["trade_time", "symbol", "quantity", "price", "side"]


class OKXSpotTrades2Hive:

    def __init__(self, raw_data_path: Path, bounds: Bounds):
        self.raw_data_path: Path = raw_data_path
        self.bounds: Bounds = bounds

    @staticmethod
    def preprocess_batched_data(df_batch: pd.DataFrame) -> pd.DataFrame:
        """Attach new columns and convert dtypes here before saving to hive structure"""
        df_batch[TRADE_TIME] = pd.to_datetime(df_batch[TRADE_TIME], unit="ms")
        # Create date column from TRADE_TIME
        df_batch["date"] = df_batch[TRADE_TIME].dt.date
        return df_batch

    @staticmethod
    def save_batched_data_to_hive(df_batch: pd.DataFrame) -> None:
        df_batch.to_parquet(
            OKX_SPOT_HIVE_TRADES,
            engine="pyarrow",
            compression="gzip",
            partition_cols=["date", "symbol"],
            existing_data_behavior="overwrite_or_ignore",
        )

    def unzip_and_save_to_hive(self, subpath: Path) -> None:
        path: Path = self.raw_data_path / subpath

        csv_reader = pd.read_csv(
            # OKX sends zipped files with some localisation, that's why we have to use cp1252 encoding to read
            # csv files without errors
            path, chunksize=1_000_000, header=None, skiprows=1, encoding="cp1252", names=_RAW_COLS, usecols=_USE_COLS,
        )

        for batch_id, df_batch in enumerate(csv_reader):
            df_batch = self.preprocess_batched_data(df_batch=df_batch)
            self.save_batched_data_to_hive(df_batch=df_batch)

    def run_multiprocessing(self, processes: int = 10) -> None:
        with Pool(processes=processes) as pool:
            promises: List[AsyncResult] = []

            for day in self.bounds.date_range():
                promise: AsyncResult = pool.apply_async(
                    partial(
                        self.unzip_and_save_to_hive,
                        subpath=Path(f"trades@{str(day)}.zip")
                    ),
                )
                promises.append(promise)

            for promise in tqdm(promises, desc="Saving zipped csv files to HiveDataset: "):
                promise.get()


def run_main():
    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 1)
    )
    pipe = OKXSpotTrades2Hive(
        bounds=bounds,
        raw_data_path=OKX_SPOT_RAW_TRADES,
    )
    pipe.run_multiprocessing()


if __name__ == "__main__":
    run_main()
