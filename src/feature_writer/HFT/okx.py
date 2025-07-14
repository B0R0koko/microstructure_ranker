from datetime import date
from typing import List

import polars as pl

from core.currency import Currency, get_target_currencies
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.time_utils import Bounds
from feature_writer.HFT.binance import SampledFeatureWriter


class OKXFeatureWriter(SampledFeatureWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def side_expr() -> pl.Expr:
        """
        Overwrite the way we compute side sign.
        """
        return pl.when(pl.col("side") == "SELL").then(-1).otherwise(1)


def run_main():
    # Run OKXFeatureWriter from here,
    # set PYTHONPATH to src folder and run from terminal such that Process progress bar is displayed correctly
    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 1)
    )
    writer = OKXFeatureWriter(bounds=bounds, exchange=Exchange.OKX_SPOT)
    currency_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency.name, term=Currency.USDT.name) for currency in get_target_currencies()
    ]
    writer.run_in_multiprocessing_pool(currency_pairs=currency_pairs)


if __name__ == "__main__":
    run_main()
