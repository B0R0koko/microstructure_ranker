from pathlib import Path

import pandas as pd

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

    def preprocess_batched_data(self, df: pd.DataFrame, currency_pair: CurrencyPair) -> pd.DataFrame:
        df[TRADE_TIME] = pd.to_datetime(df[TRADE_TIME], unit="ms", errors="coerce")
        df[SYMBOL] = currency_pair.name
        # Create date column from TRADE_TIME
        df["date"] = df[TRADE_TIME].dt.date
        return df

    def save_to_hive(self, df: pd.DataFrame) -> None:
        df.to_parquet(
            self.output_dir,
            engine='pyarrow',
            compression="lz4",
            partition_cols=["date", "symbol"],
            existing_data_behavior="delete_matching"
        )
