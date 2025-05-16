from datetime import date

import polars as pl

from core.currency import CurrencyPair
from core.paths import FEATURE_DIR
from core.time_utils import Bounds


def main():
    hive: pl.LazyFrame = pl.scan_parquet(FEATURE_DIR)
    bounds: Bounds = Bounds.for_days(
        date(2024, 1, 1), date(2024, 1, 3)
    )

    data = hive.filter(
        (pl.col("currency_pair") == CurrencyPair.from_string("BTC-USDT").name) &
        (pl.col("date").is_between(bounds.day0, bounds.day1)) &
        (pl.col("window") == "500MS")
    ).collect()

    print(data)


if __name__ == '__main__':
    main()
