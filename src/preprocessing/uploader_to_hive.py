import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl
from pyarrow import RecordBatch, csv

from core.columns import *
from core.currency import CurrencyPair


def check_if_zipped_file_is_csv(file_path: str) -> bool:
    """Checks if zipped file stored is a csv"""
    return file_path.endswith(".csv")


class Uploader2Hive(ABC):

    def __init__(
            self,
            zipped_data_dir: Path,
            temp_dir: Path,
            output_dir: Path,
            column_names: List[str],
            include_columns: List[str]
    ):
        self.zipped_data_dir: Path = zipped_data_dir
        self.temp_dir: Path = temp_dir
        self.output_dir: Path = output_dir
        self.column_names: List[str] = column_names
        self.include_columns: List[str] = include_columns

        self.currency_pairs: List[CurrencyPair] = self._parse_collected_currency_pairs()

    @abstractmethod
    def preprocess_batched_data(self, df: pl.DataFrame, currency_pair: CurrencyPair) -> pl.DataFrame:
        """Preprocess data read from batched csv reader"""

    @abstractmethod
    def save_to_hive(self, df: pl.DataFrame) -> None:
        """Save batched data to Hive structure. Define the partitioning here"""

    def _parse_collected_currency_pairs(self) -> List[CurrencyPair]:
        """Folders are named in the following pattern: BASE-TERM"""
        currency_pairs: List[CurrencyPair] = []
        for folder in os.listdir(self.zipped_data_dir):
            base, term = folder.split("-")
            currency_pairs.append(CurrencyPair(base=base, term=term))
        return currency_pairs

    def _save_to_pyarrow_hive(
            self, csv_file_path: Path, currency_pair: CurrencyPair
    ) -> None:
        """
        Preprocess and partition data by date using PyArrow hive partitioning, optimized for large datasets.
        Specify partition to be able to query parquet files using pl.scan_parquet with filter queries
        """
        # Define CSV read options
        MB: int = 1024 ** 2

        read_options: csv.ReadOptions = csv.ReadOptions(
            column_names=self.column_names,
            block_size=MB * 128,  # batch_size in megabytes
            use_threads=None
        )
        # subset leave_cols from new_columns
        convert_options: csv.ConvertOptions = csv.ConvertOptions(include_columns=self.include_columns)

        # Use a context manager to ensure the CSV reader is closed
        with csv.open_csv(csv_file_path, read_options=read_options, convert_options=convert_options) as csv_reader:
            batch: RecordBatch

            for batch in csv_reader:
                df: pl.DataFrame = pl.from_arrow(data=batch)
                df = self.preprocess_batched_data(df=df, currency_pair=currency_pair)
                self.save_to_hive(df=df)

    def preprocess_zip_file(self, currency_pair: CurrencyPair, zip_file: str) -> None:
        zip_path: Path = (
            self.zipped_data_dir
            .joinpath(currency_pair.name)
            .joinpath(zip_file)
        )
        # unzip data stored in the zipped archive
        with zipfile.ZipFile(zip_path, mode="r") as zip_ref:
            csv_file: str = zip_ref.namelist()[0]  # name of csv file once zip file is unzipped

            if not check_if_zipped_file_is_csv(file_path=csv_file):
                raise ValueError("Unzipped file is not a csv file")

            zip_ref.extract(member=csv_file, path=self.temp_dir)  # extract zipped csv file to self.temp_dir
            csv_file_path: Path = self.temp_dir.joinpath(csv_file)

            self._save_to_pyarrow_hive(csv_file_path=csv_file_path, currency_pair=currency_pair)

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
            currency_pair_folder: Path = self.zipped_data_dir.joinpath(currency_pair.name)
            # Iterate over each zip file store in currency_pair_folder
            for file in os.listdir(currency_pair_folder):
                self.preprocess_zip_file(
                    currency_pair=currency_pair,
                    zip_file=file
                )
