from datetime import date

import polars as pl

from core.paths import FEATURE_DIR
from core.time_utils import Bounds
from feature_writer.utils import to_wide_format


def main():
    hive: pl.LazyFrame = pl.scan_parquet(FEATURE_DIR)
    bounds: Bounds = Bounds.for_days(
        date(2024, 1, 1), date(2024, 1, 4)
    )

    data = hive.filter(
        (pl.col("date").is_between(bounds.day0, bounds.day1))
    ).collect()

    print(to_wide_format(data))


if __name__ == '__main__':
    main()
