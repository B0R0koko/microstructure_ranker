from pathlib import Path

import polars as pl

from core.columns import *
from core.currency import CurrencyPair
from preprocessing.uploader_to_hive import Uploader2Hive

_INCLUDE_COLUMNS: List[str] = [
    OPEN_TIME,
    OPEN_PRICE,
    HIGH_PRICE,
    LOW_PRICE,
    CLOSE_PRICE,
    VOLUME,
    QUOTE_ASSET_VOLUME,
    TAKER_BUY_QUOTE_ASSET_VOLUME,
    CLOSE_TIME,
    NUM_TRADES
]


class Klines2HiveUploader(Uploader2Hive):

    def __init__(self, zipped_data_dir: Path, temp_dir: Path, output_dir: Path):
        super().__init__(
            zipped_data_dir=zipped_data_dir,
            temp_dir=temp_dir,
            output_dir=output_dir,
            column_names=BINANCE_KLINES_COLS,
            include_columns=_INCLUDE_COLUMNS
        )

    def preprocess_batched_data(self, df: pl.DataFrame, currency_pair: CurrencyPair) -> pl.DataFrame:
        df = df.with_columns(
            pl.lit(currency_pair.name).alias(SYMBOL),
            pl.col(OPEN_TIME).cast(pl.Datetime(time_unit="ms")).alias(OPEN_TIME),
            pl.col(CLOSE_TIME).cast(pl.Datetime(time_unit="ms")).alias(CLOSE_TIME),
        )
        # Create date column from TRADE_TIME
        df = df.with_columns(pl.col(OPEN_TIME).dt.date().alias("date"))
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
