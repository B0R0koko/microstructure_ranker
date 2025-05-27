from datetime import date
from pathlib import Path

import pandas as pd

from core.columns import *
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_USDM_RAW_TRADES, BINANCE_USDM_HIVE_TRADES
from core.time_utils import Bounds
from preprocessing.pipelines.binance_spot_trades_to_hive import BinanceSpotTrades2Hive

_USE_COLS: List[str] = [PRICE, QUANTITY, TRADE_TIME, IS_BUYER_MAKER]


class BinanceUSDMTrades2Hive(BinanceSpotTrades2Hive):

    def __init__(self, bounds: Bounds, raw_data_dir: Path, output_dir: Path):
        super().__init__(
            bounds=bounds,
            raw_data_dir=raw_data_dir,
            output_dir=output_dir
        )

    def preprocess_batched_data(self, df_batch: pd.DataFrame, currency_pair: CurrencyPair, day: date) -> pd.DataFrame:
        """Attach new columns and convert dtypes here before saving to hive structure"""
        # BINANCE_USDM TRADE_TIME is still written in ms
        df_batch[TRADE_TIME] = pd.to_datetime(df_batch[TRADE_TIME], unit="ms")
        # Create date column from TRADE_TIME
        df_batch["date"] = day
        # Create symbol column
        df_batch["symbol"] = currency_pair.name

        return df_batch

    def unzip_and_save_to_hive(self, currency_pair: CurrencyPair, day: date) -> None:
        # BINANCE_SPOT data doesn't contain header while USDM data has the header => we will need to skip it
        csv_reader = pd.read_csv(
            self.raw_data_dir / currency_pair.name / f"trades@{str(day)}.zip",
            chunksize=1_000_000,
            header=None,
            skiprows=1,
            names=BINANCE_TRADE_USDM_COLS,
            usecols=_USE_COLS,
        )

        for batch_id, df_batch in enumerate(csv_reader):
            df_batch = self.preprocess_batched_data(df_batch=df_batch, currency_pair=currency_pair, day=day)
            self.save_batched_data_to_hive(df_batch=df_batch)


def run_main():
    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 25)
    )
    pipe = BinanceUSDMTrades2Hive(
        bounds=bounds,
        raw_data_dir=BINANCE_USDM_RAW_TRADES,
        output_dir=BINANCE_USDM_HIVE_TRADES
    )
    pipe.run_multiprocessing()


if __name__ == "__main__":
    run_main()
