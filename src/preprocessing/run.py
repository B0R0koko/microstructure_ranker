from datetime import datetime

import polars as pl

from core.columns import TRADE_TIME
from core.currency import CurrencyPair


def main():
    df: pl.LazyFrame = pl.scan_parquet("D:/data/transformed_data", low_memory=True)
    pair: CurrencyPair = CurrencyPair(base="BTC", term="USDT")

    df = df.filter(
        (pl.col("symbol") == pair.name) &
        (pl.col("date") >= datetime(2024, 9, 20)) &
        (pl.col("date") < datetime(2024, 10, 1))
    )

    df = df.sort(by=TRADE_TIME, descending=False)

    print(df.collect())


if __name__ == "__main__":
    main()
