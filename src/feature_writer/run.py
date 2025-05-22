from datetime import date

import polars as pl

from core.columns import SYMBOL
from core.currency import Currency
from core.currency_pair import CurrencyPair
from core.paths import USDM_TRADES
from core.time_utils import Bounds


def run_test():
    bounds: Bounds = Bounds.for_days(
        date(2024, 1, 1), date(2024, 1, 5)
    )
    hive = pl.scan_parquet(USDM_TRADES, hive_partitioning=True)
    df = (
        hive
        .filter(
            (pl.col(SYMBOL) == CurrencyPair(Currency.SUI, Currency.USDT).name) &
            (pl.col("date").is_between(bounds.day0, bounds.day1))
        )
        .collect()
    )

    print(df)


if __name__ == "__main__":
    run_test()
