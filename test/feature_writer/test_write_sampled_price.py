from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

from core.columns import TRADE_TIME, SYMBOL, SAMPLED_TIME, PRICE, OPEN_PRICE, CLOSE_PRICE
from core.currency import Currency
from core.currency_pair import CurrencyPair
from core.exchange import Exchange
from core.paths import BINANCE_SPOT_TRADES
from core.time_utils import Bounds
from models.prediction.features import read_sampled_open_price, read_sampled_close_price


def test_write_sampled_price():
    """Load tick data from SPOT_TRADES hive and compare sampled price through Polars with result written to features"""
    bounds: Bounds = Bounds.for_days(date(2024, 1, 1), date(2024, 1, 2))
    exchange: Exchange = Exchange.BINANCE_SPOT
    # Expand bounds such that start and end of the daily interval will not have missing values
    expanded_bounds: Bounds = bounds.expand_bounds(lb_timedelta=timedelta(minutes=5), rb_timedelta=timedelta(minutes=5))
    currency_pair: CurrencyPair = CurrencyPair(base=Currency.ADA, term=Currency.USDT)

    df_ticks: pl.DataFrame = (
        pl.scan_parquet(BINANCE_SPOT_TRADES, hive_partitioning=True)
        .filter(
            (pl.col("date").is_between(expanded_bounds.day0, expanded_bounds.day1)) &
            (pl.col(SYMBOL) == currency_pair.name) &
            (pl.col(TRADE_TIME).is_between(expanded_bounds.start_inclusive, expanded_bounds.end_exclusive))
        )
        .collect()
        .sort(by=TRADE_TIME, descending=False)
    )

    date_index: pd.DatetimeIndex = pd.date_range(
        bounds.start_inclusive, bounds.end_exclusive, freq=timedelta(milliseconds=500), inclusive="left"
    )
    df_index: pl.DataFrame = pl.DataFrame({SAMPLED_TIME: date_index})

    df: pd.DataFrame = (
        df_ticks
        .sort(by=TRADE_TIME, descending=False)
        .group_by_dynamic(
            index_column=TRADE_TIME,
            period=timedelta(milliseconds=500),
            every=timedelta(milliseconds=500),
            closed="right",
            label="right"
        )
        .agg(
            open_price=pl.col(PRICE).first(),
            close_price=pl.col(PRICE).last()
        )
        .filter(
            pl.col(TRADE_TIME).is_between(bounds.start_inclusive, bounds.end_exclusive)
        )
        .join(df_index, left_on=TRADE_TIME, right_on=SAMPLED_TIME, how="right")
        .to_pandas()
    )

    # Read sampled prices written to filesystem
    open_prices: np.ndarray = read_sampled_open_price(
        bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=timedelta(milliseconds=500)
    )
    close_prices: np.ndarray = read_sampled_close_price(
        bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=timedelta(milliseconds=500)
    )

    df["open_price_expected"] = open_prices
    df["close_price_expected"] = close_prices

    df = df.fillna(-1)

    assert all(df[OPEN_PRICE] == df["open_price_expected"]), "Sampled open prices do not match"  # type:ignore
    assert all(df[CLOSE_PRICE] == df["close_price_expected"]), "Sampled close prices do not match"  # type:ignore
