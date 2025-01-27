from pathlib import Path

import polars as pl

from core.columns import *
from core.currency import CurrencyPair
from preprocessing.uploader_to_hive import Uploader2Hive

_INCLUDE_COLUMNS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]


class Trades2HiveUploader(Uploader2Hive):

    def __init__(self, zipped_data_dir: Path, temp_dir: Path, output_dir: Path):
        super().__init__(
            zipped_data_dir=zipped_data_dir,
            temp_dir=temp_dir,
            output_dir=output_dir,
            column_names=BINANCE_TRADE_COLS,
            include_columns=_INCLUDE_COLUMNS
        )

    def preprocess_batched_data(self, df: pl.DataFrame, currency_pair: CurrencyPair) -> pl.DataFrame:
        df = df.with_columns(
            pl.lit(currency_pair.name).alias(SYMBOL),
            pl.col(TRADE_TIME).cast(pl.Datetime(time_unit="ms")).alias(TRADE_TIME)
        )
        # Create date column from TRADE_TIME
        df = df.with_columns(pl.col(TRADE_TIME).dt.date().alias("date"))
        return df

    def save_to_hive(self, df: pl.DataFrame) -> None:
        df.write_parquet(
            self.output_dir,
            use_pyarrow=True,
            pyarrow_options={
                "partition_cols": ["date", SYMBOL],  # define set of filters here
                "existing_data_behavior": "overwrite_or_ignore",
            }
        )
