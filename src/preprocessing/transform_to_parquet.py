import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence, List

import polars as pl
from polars.io.csv import BatchedCsvReader

from core.currency import CurrencyPair


def preprocess_csv_file_in_batches(
        file_path: Path,
        output_path: Path,
        new_columns: Optional[Sequence[str]] = None,
        batch_rows_loaded: int = int(1e6)
) -> None:
    """Reads csv file using polars with batch read enabled, such that it is not loaded in one go to the memory"""

    reader: BatchedCsvReader = pl.read_csv_batched(
        source=file_path,
        has_header=False,
        new_columns=new_columns,
        low_memory=True,
        batch_size=batch_rows_loaded,
    )

    batch: List[pl.DataFrame] | None
    # Stop when chunk is None
    while (batch := reader.next_batches(1)) is not None:
        df: pl.DataFrame = batch[0]
        df.write_parquet(output_path, compression="gzip")


def check_if_zipped_file_is_csv(file_path: str) -> bool:
    """Checks if zipped file stored is a csv"""
    return file_path.endswith(".csv")


class ParquetTransformer:
    ZIPPED_DATA_DIR: Path = Path("D:/data/zipped_data")
    PARQUET_DATA_DIR: Path = Path("D:/data/parquet_data")

    def __init__(self):
        ...

    def load_zip_file(self, currency_pair: CurrencyPair, file_name: str) -> None:
        zip_path: Path = (
            self.ZIPPED_DATA_DIR
            .joinpath(currency_pair.name)
            .joinpath(file_name)
        )
        # unzip data stored in the zipped archive
        with tempfile.TemporaryDirectory() as tmp_dir, zipfile.ZipFile(zip_path, mode="r") as zip_ref:
            zipped_file: str = zip_ref.namelist()[0]  # name of csv file once zip file is unzipped

            if not check_if_zipped_file_is_csv(file_path=zipped_file):
                raise ValueError("Unzipped file is not a csv file")

            zipped_file_no_ext: str = zipped_file.split(".")[0]
            zip_ref.extract(member=zipped_file, path=tmp_dir)

            # Create directory for parquet files for CurrencyPair
            output_dir: Path = self.PARQUET_DATA_DIR.joinpath(currency_pair.name)
            os.makedirs(output_dir, exist_ok=True)

            file_path: Path = Path(tmp_dir).joinpath(zipped_file)

            preprocess_csv_file_in_batches(
                file_path=file_path,
                output_path=output_dir.joinpath(f"{zipped_file_no_ext}.parquet"),
                new_columns=["price", "qty", "time", "isBuyerMaker"],
            )


if __name__ == "__main__":
    currency_pair: CurrencyPair = CurrencyPair(base="BTC", term="USDT")
    ParquetTransformer().load_zip_file(
        currency_pair=currency_pair,
        file_name="BTCUSDT-trades-2024-10.zip"
    )
