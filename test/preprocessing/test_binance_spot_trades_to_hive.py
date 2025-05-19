import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
from scrapy.utils.log import configure_logging

from core.columns import SYMBOL
from core.currency import Currency
from core.currency_pair import CurrencyPair
from preprocessing.pipelines.binance_usdm_trades_to_hive import BinanceUSDM2Hive


def test_binance_spot_trades_to_hive() -> None:
    """
    This test makes sure that the way HiveDataset is created matches the result produced by simply unpacking csv file
    and reading it with pandas, we simply compare shapes of two dataframes
    """
    configure_logging()
    zipped_data_dir: Path = Path("D:/data/zipped_data/USDM")
    currency_pair: CurrencyPair = CurrencyPair(base=Currency.ADA, term=Currency.USDT)
    zip_file_name: str = "ADAUSDT-trades-2024-03.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create hive structure in the test folder
        output_dir: Path = Path(temp_dir)
        uploader: BinanceUSDM2Hive = BinanceUSDM2Hive(zipped_data_dir=zipped_data_dir, output_dir=output_dir)

        uploader.save_to_pyarrow_hive(
            zipped_csv_file_path=(
                zipped_data_dir
                .joinpath(currency_pair.name)
                .joinpath(zip_file_name)
            ),
            currency_pair=currency_pair,
            file_date=date(2024, 3, 1)
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

        df_pandas: pd.DataFrame = pd.read_csv(zip_file_path, compression="zip", header=None, skiprows=1)

        assert df_pandas.shape[0] == hive_size, "Polars HiveDataset and Pandas.DataFrame have different shapes"
