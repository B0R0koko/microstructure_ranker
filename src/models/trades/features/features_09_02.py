from typing import Dict, Any, List

import polars as pl

from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset


def compute_slippage(df_trades: pl.LazyFrame) -> pl.LazyFrame:
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


def aggregate_ticks_into_trades(df_currency_pair: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate ticks into trades using ns timestamps"""
    df_trades: pl.LazyFrame = (
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
        )
    )
    # Create boolean indicating if the trade was long or short
    df_trades = df_trades.with_columns(
        (pl.col("quantity_sign") >= 0).alias("is_long")
    )
    return df_trades


def select_features(df_trades: pl.LazyFrame, bounds: Bounds, offset: TimeOffset) -> Dict[str, float]:
    features_fetched: Dict[str, List[float]] = (
        df_trades
        .filter(pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
        .select(
            (
                (pl.col("quote_sign").sum() / pl.col("quote_abs").sum())
                .alias(f"volume_imbalance_{offset.name}")
            ),
            (
                (pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum())
                .alias(f"slippage_imbalance_{offset.name}")
            ),
            (
                (pl.col("is_long").sum() / pl.len())
                .alias(f"share_of_long_trades_{offset.name}")
            ),
            (
                (pl.col("price_last").last() / pl.col("price_first").first()).log()
                .alias(f"log_return_{offset.name}")
            ),
            (
                (1 + pl.len() / (pl.col("quote_abs") / pl.col("quote_abs").min()).log().sum())
                .alias(f"mle_alpha_powerlaw_{offset.name}")
            )
        )
        .collect(engine="gpu")
        .to_dict(as_series=False)
    )

    return {key: val[0] for key, val in features_fetched.items()}


def compute_features(df_currency_pair: pl.LazyFrame, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
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
    df_trades = aggregate_ticks_into_trades(df_currency_pair=df_currency_pair)
    df_trades = compute_slippage(df_trades=df_trades)

    desired_offsets: List[TimeOffset] = [
        TimeOffset.MINUTE,
        TimeOffset.FIVE_MINUTES,
        TimeOffset.HALF_HOUR,
        TimeOffset.HOUR,
        TimeOffset.TWO_HOURS,
        TimeOffset.FOUR_HOURS,
        TimeOffset.TWELVE_HOURS
    ]

    all_features: Dict[str, Any] = {"currency_pair": currency_pair.binance_name}

    for offset in desired_offsets:
        features: Dict[str, float] = select_features(df_trades=df_trades, bounds=bounds, offset=offset)
        all_features.update(features)

    return all_features
