from datetime import datetime

import polars as pl

from core.currency import CurrencyPair


def main():
    df: pl.LazyFrame = pl.scan_parquet("D:/data/transformed_data")
    currency_pair: CurrencyPair = CurrencyPair(base="ADA", term="USDT")

    df = df.filter(
        (pl.col("symbol") == currency_pair.name) &
        (pl.col("date") >= datetime(2024, 9, 20)) &
        (pl.col("date") < datetime(2024, 10, 20))
    )

    print(df.collect())


if __name__ == "__main__":
    main()
