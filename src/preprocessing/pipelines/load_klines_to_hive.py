from pathlib import Path

import pandas as pd
from preprocessing.uploader_to_hive import Uploader2Hive

from core.columns import *
from core.currency import CurrencyPair

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

    def __init__(self, zipped_data_dir: Path, output_dir: Path):
        super().__init__(
            zipped_data_dir=zipped_data_dir,
            output_dir=output_dir,
            column_names=BINANCE_KLINES_COLS,
            include_columns=_INCLUDE_COLUMNS
        )

    def preprocess_batched_data(self, df: pd.DataFrame, currency_pair: CurrencyPair) -> pd.DataFrame:
        df[SYMBOL] = currency_pair.name

        df[OPEN_TIME] = pd.to_datetime(df[OPEN_TIME], unit="ms")
        df[CLOSE_TIME] = pd.to_datetime(df[CLOSE_TIME], unit="ms")
        # Create date column from TRADE_TIME
        df["date"] = df[CLOSE_TIME].dt.date

        return df

    def save_batch_to_hive(self, df: pd.DataFrame) -> None:
        df.to_parquet(
            self.output_dir,
            engine="pyarrow",
            compression="lz4",
            partition_cols=["date", "symbol"],
            existing_data_behavior="overwrite_or_ignore"
        )
