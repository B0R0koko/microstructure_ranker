import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import polars as pl
from polars.io.csv import BatchedCsvReader

from core.columns import *
from core.currency import CurrencyPair

_BINANCE_LEAVE_COLS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]


def preprocess_csv_file_in_batches(
        csv_file_path: Path,
        output_dir: Path,
        file_name_to_ext: str,
        new_columns: Optional[Sequence[str]] = None,
        leave_cols: Optional[Sequence[str]] = None,
        batch_rows_loaded: int = int(1e6)
) -> None:
    """
    Reads csv file using polars with batch read enabled, such that it is not loaded in one go to the memory
    """
    reader: BatchedCsvReader = pl.read_csv_batched(
        source=csv_file_path,
        has_header=False,
        new_columns=new_columns,
        low_memory=True,
        batch_size=batch_rows_loaded,
    )

    os.makedirs(output_dir, exist_ok=True)
    file_path: Path = output_dir.joinpath(f"{file_name_to_ext}.parquet")

    batch: List[pl.DataFrame] | None
    # Stop when chunk is None
    while (batch := reader.next_batches(1)) is not None:
        df: pl.DataFrame = batch[0]
        # Append to parquet file each batched df with selected leave_cols
        df.select(leave_cols).to_pandas().to_parquet(
            file_path,
            compression="gzip",
            engine="fastparquet",
            append=os.path.exists(file_path),
            index=False
        )


def preprocess_csv_file_partitioned_by_trade_time(
        csv_file_path: Path,
        hive_dir: Path,
        currency_pair: CurrencyPair,
        new_columns: Optional[Sequence[str]] = None,
        leave_cols: Optional[Sequence[str]] = None,
) -> None:
    """Partition loaded data from csv file into chunks of size one day using pyarrow backend"""

    df: pl.LazyFrame = pl.scan_csv(source=csv_file_path, new_columns=new_columns)
    # Cast TRADE_TIME column to datetime and then extract date into a separate column
    df = df.select(leave_cols)
    df = df.with_columns(
        pl.lit(currency_pair.name).alias(SYMBOL),
        pl.col(TRADE_TIME).cast(pl.Datetime(time_unit="ms")).alias(TRADE_TIME)
    )
    df = df.with_columns(pl.col(TRADE_TIME).dt.date().alias(f"{TRADE_TIME}_date"))

    # Collect dataframe in batches and dump them into parquet hive
    slice_offset: int = 0
    step: int = int(1e6)

    while True:
        df_sliced: pl.DataFrame = df.slice(offset=slice_offset, length=step).collect()

        if df_sliced.is_empty():
            break

        df_sliced.write_parquet(
            hive_dir,
            use_pyarrow=True,
            pyarrow_options={
                "partition_cols": [SYMBOL, f"{TRADE_TIME}_date"],
            }
        )
        slice_offset += step


def check_if_zipped_file_is_csv(file_path: str) -> bool:
    """Checks if zipped file stored is a csv"""
    return file_path.endswith(".csv")


class ParquetTransformer:
    ZIPPED_DATA_DIR: Path = Path("D:/data/zipped_data")
    PARQUET_DATA_DIR: Path = Path("D:/data/parquet_data")

    def __init__(self):
        ...

    def preprocess_zip_file(self, currency_pair: CurrencyPair, zip_file: str) -> None:
        zip_path: Path = (
            self.ZIPPED_DATA_DIR
            .joinpath(currency_pair.name)
            .joinpath(zip_file)
        )
        # unzip data stored in the zipped archive
        with tempfile.TemporaryDirectory() as tmp_dir, zipfile.ZipFile(zip_path, mode="r") as zip_ref:
            zipped_file: str = zip_ref.namelist()[0]  # name of csv file once zip file is unzipped

            if not check_if_zipped_file_is_csv(file_path=zipped_file):
                raise ValueError("Unzipped file is not a csv file")

            zip_ref.extract(member=zipped_file, path=tmp_dir)

            csv_file_path: Path = Path(tmp_dir).joinpath(zipped_file)

            preprocess_csv_file_partitioned_by_trade_time(
                csv_file_path=csv_file_path,
                hive_dir=self.PARQUET_DATA_DIR,
                currency_pair=currency_pair,
                new_columns=BINANCE_TRADE_COLS,
                leave_cols=_BINANCE_LEAVE_COLS,
            )


if __name__ == "__main__":
    pair: CurrencyPair = CurrencyPair(base="BTC", term="USDT")
    ParquetTransformer().preprocess_zip_file(
        currency_pair=pair,
        zip_file="BTCUSDT-trades-2024-09.zip"
    )
