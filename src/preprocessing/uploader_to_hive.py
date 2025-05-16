import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List

import pandas as pd
import polars as pl
from tqdm import tqdm

from core.currency_pair import CurrencyPair


def check_if_zipped_file_is_csv(file_path: str) -> bool:
    """Checks if zipped file stored is a csv"""
    return file_path.endswith(".csv")


class Uploader2Hive(ABC):

    def __init__(
            self,
            zipped_data_dir: Path,
            output_dir: Path,
            column_names: List[str],
            include_columns: List[str]
    ):
        self.zipped_data_dir: Path = zipped_data_dir
        self.output_dir: Path = output_dir
        self.column_names: List[str] = column_names
        self.include_columns: List[str] = include_columns

        self.currency_pairs: List[CurrencyPair] = self._parse_collected_currency_pairs()

    @abstractmethod
    def preprocess_batched_data(self, df: pd.DataFrame, currency_pair: CurrencyPair, file_date: date) -> pl.DataFrame:
        """Preprocess data read from batched csv reader"""

    @abstractmethod
    def save_batch_to_hive(self, df: pl.DataFrame) -> None:
        """Save batched data to Hive structure. Define the partitioning here"""

    def _parse_collected_currency_pairs(self) -> List[CurrencyPair]:
        """Folders are named in the following pattern: BASE-TERM"""
        currency_pairs: List[CurrencyPair] = []
        for folder in os.listdir(self.zipped_data_dir):
            currency_pairs.append(CurrencyPair.from_string(symbol=folder))
        return currency_pairs

    def save_currency_pair_to_hive(self, currency_pair: CurrencyPair) -> None:
        # Get the name of directory where zipped csv files for a given CurrencyPair are stored
        currency_pair_folder: Path = self.zipped_data_dir.joinpath(currency_pair.name)
        # Iterate over each zip file store in currency_pair_folder
        for file in os.listdir(currency_pair_folder):
            zip_file: Path = (
                self.zipped_data_dir
                .joinpath(currency_pair.name)
                .joinpath(file)
            )
            parsed_date: date = (
                datetime.strptime(re.search(r"-(\d{4}-\d{2})\.zip$", file).group(1), "%Y-%m")
                .date()
            )
            self.save_to_pyarrow_hive(zipped_csv_file_path=zip_file, currency_pair=currency_pair, file_date=parsed_date)

    def save_to_pyarrow_hive(
            self, zipped_csv_file_path: Path, currency_pair: CurrencyPair, file_date: date
    ) -> None:
        """
        Preprocess and partition data by date using PyArrow hive partitioning, optimized for large datasets.
        Specify partition to be able to query parquet files using pl.scan_parquet with filter queries
        """
        # Define CSV read options

        csv_reader = pd.read_csv(
            zipped_csv_file_path,
            chunksize=1_000_000,
            header=None,
            names=self.column_names,
            usecols=self.include_columns,
        )

        for batch_id, batch in enumerate(csv_reader):
            # Example processing: Display the first few rows
            self.preprocess_batched_data(df=batch, currency_pair=currency_pair, file_date=file_date)
            self.save_batch_to_hive(df=batch)

    def run(self) -> None:
        """
        Data should be organized like this, then we will iterate over each CurrencyPair
        D:/DATA/ZIPPED_DATA
        ├───ADA-USDT
        ├───BNB-BTC
        ├───BTC-USDT
        ├───ETH-BTC
        ├───LTC-BTC
        └───NEO-BTC
        """

        for currency_pair in self.currency_pairs:
            currency_pair: CurrencyPair
            self.save_currency_pair_to_hive(currency_pair=currency_pair)

    def run_multiprocessing(self, processes: int = 10) -> None:
        with (
            tqdm(total=len(self.currency_pairs), desc="Saving zipped csv files to HiveDataset: ") as pbar,
            Pool(processes=processes) as pool,
        ):
            promises: List[AsyncResult] = []

            for currency_pair in self.currency_pairs:
                promise: AsyncResult = pool.apply_async(
                    partial(self.save_currency_pair_to_hive, currency_pair=currency_pair),
                )
                promises.append(promise)

            for promise in promises:
                promise.get()
                pbar.update(1)
