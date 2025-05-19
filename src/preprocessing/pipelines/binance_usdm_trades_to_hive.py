from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

from core.columns import BINANCE_TRADE_USDM_COLS, PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER, SYMBOL
from core.currency_pair import CurrencyPair
from preprocessing.uploader_to_hive import Uploader2Hive

_INCLUDE_COLUMNS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]


class BinanceUSDM2Hive(Uploader2Hive):

    def __init__(self, zipped_data_dir: Path, output_dir: Path):
        super().__init__(
            zipped_data_dir=zipped_data_dir,
            output_dir=output_dir,
            column_names=BINANCE_TRADE_USDM_COLS,
            include_columns=_INCLUDE_COLUMNS
        )

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
            skiprows=1,
            names=self.column_names,
            usecols=self.include_columns,
        )

        for batch_id, batch in enumerate(csv_reader):
            # Example processing: Display the first few rows
            self.preprocess_batched_data(df=batch, currency_pair=currency_pair, file_date=file_date)
            self.save_batch_to_hive(df=batch)

    def preprocess_batched_data(self, df: pd.DataFrame, currency_pair: CurrencyPair, file_date: date) -> pd.DataFrame:
        # Binance updated their data streams to microseconds => if needed we will have to append 3 zeros at the end
        df[TRADE_TIME] = pd.to_datetime(df[TRADE_TIME], unit="ms", errors="coerce")
        df[SYMBOL] = currency_pair.name
        # Create date column from TRADE_TIME
        df["date"] = df[TRADE_TIME].dt.date
        return df

    def save_batch_to_hive(self, df: pd.DataFrame) -> None:
        df.to_parquet(
            self.output_dir,
            engine="pyarrow",
            compression="gzip",
            partition_cols=["date", "symbol"],
            existing_data_behavior="overwrite_or_ignore",
        )
