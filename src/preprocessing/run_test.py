from datetime import datetime

import polars as pl

from core.columns import OPEN_TIME
from core.currency import CurrencyPair


def main():
    df: pl.LazyFrame = pl.scan_parquet("D:/data/transformed/klines/1m")
    currency_pair: CurrencyPair = CurrencyPair(base="ADA", term="USDT")
    start_time: datetime = datetime(2024, 11, 20, 10)
    end_time: datetime = datetime(2024, 11, 20, 11)

    df = df.filter(
        (pl.col("date").is_between(lower_bound=start_time.date(), upper_bound=end_time.date())) &
        (pl.col(OPEN_TIME).is_between(lower_bound=start_time, upper_bound=end_time))
    )

    print(df.sort(by=OPEN_TIME).head().collect())


if __name__ == '__main__':
    main()
