from datetime import timedelta
from typing import Dict, Any, List

import polars as pl

from core.currency import CurrencyPair
from core.time_utils import Bounds, get_seconds_postfix


def compute_slippage(df_trades: pl.DataFrame) -> pl.DataFrame:
    """
    Compute slippage as the difference between the actual amount of quote asset spent and the amount that could
    have been spent had all been executed at price_first
    """
    df_trades = df_trades.with_columns(
        quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
    )
    df_trades = df_trades.with_columns(
        quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign()
    )
    return df_trades


def aggregate_ticks_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades using ns timestamps"""
    df_trades: pl.DataFrame = (
        df_ticks
        .group_by("trade_time", maintain_order=True)
        .agg(
            price_first=pl.col("price").first(),  # if someone placed a trade with price impact, then price_first
            price_last=pl.col("price").last(),  # and price_last will differ
            # Amount spent in quote asset for the trade
            quote_abs=pl.col("quote_abs").sum(),
            quote_sign=pl.col("quote_sign").sum(),
            # Amount of base asset transacted
            quantity_abs=pl.col("quantity").sum(),
            quantity_sign=pl.col("quantity_sign").sum(),
            num_ticks=pl.col("price").count(),  # number of ticks for each trade
        )
    )
    # Create boolean indicating if the trade was long or short
    df_trades = df_trades.with_columns(
        (pl.col("quantity_sign") >= 0).alias("is_long")
    )
    return df_trades


def compute_volume_imbalance(
        df_trades: pl.DataFrame, bounds: Bounds, time_offsets: List[timedelta]
) -> Dict[str, float]:
    """Returns a dict of volume imbalance features computed using different offsets from end_date"""

    return {
        f"volume_imbalance_{get_seconds_postfix(offset)}": (
            df_trades
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset, bounds.end_exclusive))
            .select(pl.col("quote_sign").sum() / pl.col("quote_abs").sum())
            .item()
        )
        for offset in time_offsets
    }


def compute_slippage_features(
        df_trades: pl.DataFrame, bounds: Bounds, time_offsets: List[timedelta]
) -> Dict[str, float]:
    """Compute slippage features based on quote_slippage_abs and quote_slippage_sign fields"""

    return {
        f"slippage_imbalance_{get_seconds_postfix(offset)}": (
            df_trades
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset, bounds.end_exclusive)
            )
            .select(pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum())
            .item()
        )
        for offset in time_offsets
    }


def compute_share_of_longs(
        df_trades: pl.DataFrame, bounds: Bounds, time_offsets: List[timedelta]
) -> Dict[str, float]:
    """Compute share of longs in overall number of trades"""
    return {
        f"share_of_long_trades_{get_seconds_postfix(offset)}": (
            df_trades
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset, bounds.end_exclusive))
            .select(pl.col("is_long").sum() / pl.len())
            .item()
        ) for offset in time_offsets
    }


def compute_log_return_features(
        df_trades: pl.DataFrame, bounds: Bounds, time_offsets: List[timedelta]
) -> Dict[str, float]:
    """Compute log returns for different intervals before the prediction timestamp with different time_offsets"""
    # Overall log_returns over intervals
    overall_log_returns: Dict[str, float] = {
        f"log_return_{get_seconds_postfix(offset)}": (
            df_trades
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset, bounds.end_exclusive))
            .select((pl.col("price_last").last() / pl.col("price_first").first()).log())
            .item()
        )
        for offset in time_offsets
    }
    return overall_log_returns


def compute_alpha_powerlaw(
        df_trades: pl.DataFrame, bounds: Bounds, time_offsets: List[timedelta]
) -> Dict[str, float]:
    """Compute alpha using MLE estimate for alpha in Powerlaw"""
    return {
        f"mle_alpha_powerlaw_{get_seconds_postfix(offset)}": (
            df_trades
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset, bounds.end_exclusive))
            # 1 + N / (sum log(q / q_min))
            .select(
                1 + pl.len() / (pl.col("quote_abs") / pl.col("quote_abs").min()).log().sum()
            )
            .item()
        )
        for offset in time_offsets
    }


def compute_features(df_ticks: pl.DataFrame, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
    """Computes features for a give currency pair"""
    # Compute quote_volumes
    df_ticks = df_ticks.with_columns(
        quote_abs=(pl.col("price") * pl.col("quantity")),  # absolute value of quote transacted
        # When is_buyer_maker is True => someone came in and matched existing BID order => SELL
        # When is_buyer_maker is False => buyer came on and matched existing ASK order => BUY
        side_sign=1 - 2 * pl.col("is_buyer_maker")  # -1 if SELL, 1 if BUY
    )
    df_ticks = df_ticks.with_columns(
        quantity_sign=pl.col("side_sign") * pl.col("quantity"),
        quote_sign=pl.col("side_sign") * pl.col("quote_abs")
    )
    # Aggregate ticks executed at the same ns timestamp into trades
    df_trades = aggregate_ticks_into_trades(df_ticks=df_ticks)
    df_trades = compute_slippage(df_trades=df_trades)

    assert df_trades["trade_time"].is_sorted(), "Data must be sorted in ascending order by trade_time"

    desired_offsets: List[timedelta] = [
        timedelta(milliseconds=500),
        timedelta(seconds=1),
        timedelta(seconds=2),
        timedelta(seconds=5),
        timedelta(seconds=10),
        timedelta(seconds=30),
        timedelta(minutes=1),
        timedelta(minutes=2),
        timedelta(minutes=5),
    ]

    return {
        "currency_pair": currency_pair.binance_name,
        **compute_volume_imbalance(df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets),
        **compute_slippage_features(df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets),
        **compute_log_return_features(df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets),
        **compute_share_of_longs(df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets),
        **compute_alpha_powerlaw(df_trades=df_trades, bounds=bounds, time_offsets=desired_offsets)
    }
