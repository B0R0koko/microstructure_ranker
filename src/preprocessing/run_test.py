from datetime import date

import polars as pl

from core.columns import SYMBOL
from core.currency import CurrencyPair
from core.time_utils import Bounds


def main():
    df: pl.LazyFrame = pl.scan_parquet("D:/data/transformed/trades")
    currency_pair: CurrencyPair = CurrencyPair(base="ADA", term="USDT")

    start_date: date = date(2024, 11, 1)
    end_date: date = date(2024, 11, 30)
    bounds: Bounds = Bounds.for_days(start_date, end_date)

    df = df.filter(
        (pl.col("date").is_between(bounds.day0, bounds.day1)) &
        (pl.col(SYMBOL) == currency_pair.name)
    )

    print(df.select(pl.len()).collect())


if __name__ == '__main__':
    main()
