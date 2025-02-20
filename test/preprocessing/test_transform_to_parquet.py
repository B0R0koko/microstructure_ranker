from pathlib import Path

import pandas as pd
import polars as pl

from analysis.core import SYMBOL
from analysis.core.currency import CurrencyPair
from analysis.preprocessing import Trades2HiveUploader


def test_preprocess_zip_file() -> None:
    """
    This test makes sure that the way HiveDataset is created matches the result produced by simply unpacking csv file
    and reading it with pandas, we simply compare shapes of two dataframes
    """
    zipped_data_dir: Path = Path("D:/data/zipped_data/trades")
    output_dir: Path = Path("D:/data/test")
    currency_pair: CurrencyPair = CurrencyPair(base="ADA", term="USDT")
    zip_file_name: str = "ADAUSDT-trades-2024-11.zip"

    # Create hive structure in the test folder
    uploader: Trades2HiveUploader = Trades2HiveUploader(
        zipped_data_dir=zipped_data_dir, output_dir=output_dir
    )
    uploader.save_to_pyarrow_hive(
        zipped_csv_file_path=(
            zipped_data_dir
            .joinpath(currency_pair.name)
            .joinpath(zip_file_name)
        ),
        currency_pair=currency_pair,
    )

    # Collect number of rows lazily
    hive: pl.LazyFrame = pl.scan_parquet(source=output_dir)

    hive_size: int = (
        hive
        .filter(
            (pl.col(SYMBOL) == currency_pair.name)
        )
        .select(pl.len()).collect().item()
    )

    # unzip data using pandas by simply unpacking csv file and then reading csv with pandas
    zip_file_path: Path = (
        zipped_data_dir
        .joinpath(currency_pair.name)
        .joinpath(zip_file_name)
    )

    df_pandas: pd.DataFrame = pd.read_csv(zip_file_path, compression="zip", header=None)

    assert df_pandas.shape[0] == hive_size, "Polars HiveDataset and Pandas.DataFrame have different shapes"
