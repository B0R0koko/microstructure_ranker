from typing import Dict, Any, List

import polars as pl

from core.currency_pair import CurrencyPair
from core.time_utils import TimeOffset, Bounds


def compute_slippage(df_currency_pair: pl.DataFrame) -> pl.DataFrame:
    """
    Compute slippage as the difference between the actual amount of quote asset spent and the amount that could
    have been spent had all been executed at price_first
    """
    df_currency_pair = df_currency_pair.with_columns(
        quote_slippage_abs=(pl.col("quote_abs") - pl.col("price_first") * pl.col("quantity_abs")).abs()
    )
    df_currency_pair = df_currency_pair.with_columns(
        quote_slippage_sign=pl.col("quote_slippage_abs") * pl.col("quantity_sign").sign()
    )
    return df_currency_pair


def aggregate_ticks_into_trades(df_currency_pair: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades using ns timestamps"""
    df_currency_pair: pl.DataFrame = (
        df_currency_pair
        .group_by("trade_time")
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
    df_currency_pair = df_currency_pair.with_columns(
        (pl.col("quantity_sign") >= 0).alias("is_long")
    )
    return df_currency_pair


def compute_volume_imbalance(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Returns a dict of volume imbalance features computed using different offsets from end_date"""

    return {
        f"volume_imbalance_{offset.name}": (
            df_currency_pair
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
            .select(pl.col("quote_sign").sum() / pl.col("quote_abs").sum())
            .item()
        )
        for offset in time_offsets
    }


def compute_slippage_features(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Compute slippage features based on quote_slippage_abs and quote_slippage_sign fields"""

    return {
        f"slippage_imbalance_{offset.name}": (
            df_currency_pair
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            .select(pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum())
            .item()
        )
        for offset in time_offsets
    }


def compute_share_of_longs(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Compute share of longs in overall number of trades"""
    return {
        f"share_of_long_trades_{offset.name}": (
            df_currency_pair
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
            .select(pl.col("is_long").sum() / pl.len())
            .item()
        ) for offset in time_offsets
    }


def compute_log_return_features(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Compute log returns for different intervals before the prediction timestamp with different time_offsets"""
    # Overall log_returns over intervals
    overall_log_returns: Dict[str, float] = {
        f"log_return_{offset.name}": (
            df_currency_pair
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
            .select((pl.col("price_last").last() / pl.col("price_first").first()).log())
            .item()
        )
        for offset in time_offsets
    }
    return overall_log_returns


def compute_alpha_powerlaw(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Compute alpha using MLE estimate for alpha in Powerlaw"""
    return {
        f"mle_alpha_powerlaw_{offset.name}": (
            df_currency_pair
            .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
            # 1 + N / (sum log(q / q_min))
            .select(
                1 + pl.len() / (pl.col("quote_abs") / pl.col("quote_abs").min()).log().sum()
            )
            .item()
        )
        for offset in time_offsets
    }


def compute_features(df_currency_pair: pl.DataFrame, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
    """Computes features for a give currency pair"""
    # Compute quote_volumes
    df_currency_pair = df_currency_pair.with_columns(
        quote_abs=(pl.col("price") * pl.col("quantity")),  # absolute value of quote transacted
        # When is_buyer_maker is True => someone came in and matched existing BID order => SELL
        # When is_buyer_maker is False => buyer came on and matched existing ASK order => BUY
        side_sign=1 - 2 * pl.col("is_buyer_maker")  # -1 if SELL, 1 if BUY
    )
    df_currency_pair = df_currency_pair.with_columns(
        quantity_sign=pl.col("side_sign") * pl.col("quantity"),
        quote_sign=pl.col("side_sign") * pl.col("quote_abs")
    )
    # Aggregate ticks executed at the same ns timestamp into trades
    df_currency_pair = aggregate_ticks_into_trades(df_currency_pair=df_currency_pair)
    df_currency_pair = compute_slippage(df_currency_pair=df_currency_pair)

    df_hourly: pl.DataFrame = (
        df_currency_pair
        .set_sorted(column="trade_time")
        .group_by_dynamic(
            index_column="trade_time", every="1h", period="1h", closed="left", label="left"
        )
        .agg(
            (pl.col("price_last").last() / pl.col("price_first").first()).log().alias("log_return"),
        )
    )

    print(df_hourly)

    desired_offsets: List[TimeOffset] = [
        TimeOffset.MINUTE,
        TimeOffset.FIVE_MINUTES,
        TimeOffset.FIFTEEN_MINUTES,
        TimeOffset.HALF_HOUR,
        TimeOffset.HOUR,
        TimeOffset.TWO_HOURS,
        TimeOffset.FOUR_HOURS,
        TimeOffset.TWELVE_HOURS,
        TimeOffset.DAY,
        TimeOffset.THREE_DAYS,
        TimeOffset.WEEK
    ]
    # Start computing features using desired offsets
    volume_imbalance_features: Dict[str, float] = compute_volume_imbalance(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )
    slippage_features: Dict[str, float] = compute_slippage_features(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )
    log_return_features: Dict[str, float] = compute_log_return_features(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )
    num_trades_features: Dict[str, float] = compute_share_of_longs(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )
    powerlaw_features: Dict[str, float] = compute_alpha_powerlaw(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )

    return {
        "currency_pair": currency_pair.binance_name,
        **log_return_features,
        **num_trades_features,
        **volume_imbalance_features,
        **slippage_features,
        **powerlaw_features
    }
