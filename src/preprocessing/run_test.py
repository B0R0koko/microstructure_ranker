from datetime import date
from pathlib import Path

import polars as pl

from core.time_utils import Bounds


def main():
    hive_dir: Path = Path("D:/data/transformed/trades/")
    start_date: date = date(2024, 11, 1)
    end_date: date = date(2024, 11, 2)
    bounds: Bounds = Bounds.for_days(start_date, end_date)

    # df = pd.read_parquet(
    #     hive_dir,
    #     engine="pyarrow",
    #     filters=[("date", ">=", "2024-11-01"), ("symbol", "=", "ADA-USDT"), ("date", "<=", "2024-11-02")]
    # )
    # print(df)

    df = pl.scan_parquet(source=hive_dir).filter(
        (pl.col("date").is_between(bounds.day0, bounds.day1)) &
        (pl.col("symbol") == "ADA-USDT")
    )

    print(df.collect())


if __name__ == '__main__':
    main()
