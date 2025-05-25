from datetime import date
from pathlib import Path

from core.paths import BINANCE_USDM_HIVE_TRADES
from core.time_utils import Bounds
from preprocessing.pipelines.binance_trades_to_hive import BinanceSpotTrades2Hive


class BinanceUSDMTrades2Hive(BinanceSpotTrades2Hive):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir: Path = BINANCE_USDM_HIVE_TRADES


def run_main():
    bounds: Bounds = Bounds.for_days(
        date(2025, 5, 1), date(2025, 5, 24)
    )
    pipe = BinanceUSDMTrades2Hive(bounds=bounds)
    pipe.run_multiprocessing()


if __name__ == "__main__":
    run_main()
