from typing import Dict, Any, List

import polars as pl

from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset

SMALLER_OFFSETS: List[TimeOffset] = [
    TimeOffset.FIVE_SECONDS, TimeOffset.MINUTE, TimeOffset.FIVE_MINUTES, TimeOffset.HALF_HOUR
]

HOURLY_OFFSETS: List[TimeOffset] = [
    TimeOffset.HOUR,
    TimeOffset.TWO_HOURS,
    TimeOffset.FOUR_HOURS,
    TimeOffset.EIGHT_HOURS,
    TimeOffset.TWELVE_HOURS,
    TimeOffset.DAY,
    TimeOffset.THREE_DAYS,
    TimeOffset.WEEK
]


def aggregate_ticks_into_trades(df_ticks: pl.DataFrame) -> pl.DataFrame:
    """Aggregate ticks into trades using ns timestamps"""
    df_currency_pair: pl.DataFrame = (
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
    df_currency_pair = df_currency_pair.with_columns(
        (pl.col("quantity_sign") >= 0).alias("is_long")
    )
    return df_currency_pair


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


def resample_to_hourly(df_trades: pl.DataFrame) -> pl.DataFrame:
    # create hourly candles to capture dynamics of hourly pct_returns over different intervals
    df_hourly: pl.DataFrame = (
        df_trades
        .group_by_dynamic(
            index_column="trade_time",
            every="1h",
            period="1h",
            closed="left",
            label="left"
        )
        .agg(
            pct_return=pl.col("price_last").last() / pl.col("price_first").first() - 1,
            quote_abs_volume=pl.col("quote_abs").sum(),
            long_quote_abs_volume=(pl.col("quote_abs") * pl.col("is_long")).sum(),
            quote_slippage_abs=pl.col("quote_slippage_abs").sum()
        )
    )
    return df_hourly


def compute_pct_return_features(df_trades: pl.DataFrame, df_hourly: pl.DataFrame, bounds: Bounds) -> Dict[str, float]:
    """
    Computes two types of pct_return features:
    1. overall_pct_return_{offset.name} - overall pct_returns over some interval
    2. pct_return_zscore_{offset.name}_14d - mean hourly pct_return over interval scaled by long run 14d std of
    long_returns
    """
    overall_pct_returns: Dict[str, float] = {
        f"overall_pct_return_{offset.name}": (
            df_trades
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            .select(pl.col("price_last").last() / pl.col("price_first").first() - 1)
            .item()
        )
        for offset in SMALLER_OFFSETS + HOURLY_OFFSETS  # use all time_offsets for overall interval pct_returns
    }

    pct_return_std_14d: float = df_hourly["pct_return"].std()  # compute long run std of hourly pct_returns to scale by

    # pct_return_zscore_7d_14d - average hourly pct_return over the last 7 days scaled by 14d std of hourly pct_return
    # This allows to capture how pct_returns evolved over time relative to their long-term behaviour
    pct_return_zscores: Dict[str, float] = {
        f"hourly_pct_return_zscore_{offset.name}_14D": (
            df_hourly
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            .select(pl.col("pct_return").mean() / pct_return_std_14d).item()
        )
        for offset in HOURLY_OFFSETS  # use hourly offsets
    }

    return overall_pct_returns | pct_return_zscores


def compute_volume_features(df_trades: pl.DataFrame, df_hourly: pl.DataFrame, bounds: Bounds) -> Dict[str, float]:
    # Compute flow_imbalances over different intervals
    flow_imbalances: Dict[str, float] = {
        f"flow_imbalance_{offset.name}": (
            df_trades
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            .select(pl.col("quote_sign").sum() / pl.col("quote_abs").sum())
            .item()
        )
        for offset in SMALLER_OFFSETS + HOURLY_OFFSETS
    }

    # Compute features that capture hourly volume dynamics relative to historic long-term behaviour
    volume_features: Dict[str, float] = {}
    # Get long run dynamics of quote_abs_volume
    quote_abs_volume_mean_14d: float = df_hourly["quote_abs_volume"].mean()
    quote_abs_volume_std_14d: float = df_hourly["quote_abs_volume"].std()
    # Get long-term dynamics of long_quote_abs_volume
    long_quote_abs_volume_mean_14d: float = df_hourly["long_quote_abs_volume"].mean()
    long_quote_abs_volume_std_14d: float = df_hourly["long_quote_abs_volume"].std()

    for offset in HOURLY_OFFSETS:
        df_interval: pl.DataFrame = df_hourly.filter(
            pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
        )
        # this feature describes hourly quote_abs dynamics relative to past 14d hourly volumes
        volume_features[f"hourly_quote_abs_volume_zscore_{offset.name}_14D"] = (
            df_interval
            .select(
                (pl.col("quote_abs_volume").mean() - quote_abs_volume_mean_14d) / quote_abs_volume_std_14d
            )
            .item()
        )
        # describes hourly quote_abs volume of longs relative to long volumes over the last 14d
        volume_features[f"hourly_long_quote_abs_volume_zscore_{offset.name}_14D"] = (
            df_interval
            .select(
                (pl.col("long_quote_abs_volume").mean() - long_quote_abs_volume_mean_14d) /
                long_quote_abs_volume_std_14d
            )
            .item()
        )

    return flow_imbalances | volume_features


def compute_slippage_features(df_trades: pl.DataFrame, df_hourly: pl.DataFrame, bounds: Bounds) -> Dict[str, float]:
    slippage_imbalances: Dict[str, float] = {
        f"slippage_imbalance_{offset.name}": (
            df_trades
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            .select(pl.col("quote_slippage_sign").sum() / pl.col("quote_slippage_abs").sum())
            .item()
        )
        for offset in SMALLER_OFFSETS + HOURLY_OFFSETS
    }

    # Describe dynamics of hourly slippage_imbalances relative to long-run mean and std
    quote_slippage_abs_mean_14d: float = df_hourly["quote_slippage_abs"].mean()
    quote_slippage_abs_std_14d: float = df_hourly["quote_slippage_abs"].std()

    # total slippage for the last 2 weeks
    quote_slippage_abs_sum_14d: float = (
        df_hourly
        .select(pl.col("quote_slippage_abs").sum())
        .item()
    )

    slippage_features: Dict[str, float] = {}

    for offset in HOURLY_OFFSETS:
        df_interval: pl.DataFrame = df_hourly.filter(
            pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
        )
        slippage_features[f"hourly_slippage_zscore_{offset.name}_14D"] = (
            df_interval
            .select(
                # scale average hourly absolute quote_slippage by its long run hourly mean and std to
                # see how it is relative to its long run dynamics
                (pl.col("quote_slippage_abs").mean() - quote_slippage_abs_mean_14d) / quote_slippage_abs_std_14d
            )
            .item()
        )
        # Compute share of slippages for current interval in 14d overall slippages
        slippage_features[f"share_quote_slippage_abs_{offset.name}_14D"] = (
            df_interval
            .select(
                pl.col("quote_slippage_abs").sum() / quote_slippage_abs_sum_14d
            )
            .item()
        )

    return slippage_imbalances | slippage_features


def compute_alpha_powerlaw(df_trades: pl.DataFrame, bounds: Bounds) -> Dict[str, float]:
    """Compute alpha using MLE estimate for alpha in Powerlaw"""
    return {
        f"mle_alpha_powerlaw_{offset.name}": (
            df_trades
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            # 1 + N / (sum pct(q / q_min))
            .select(
                1 + pl.len() / (pl.col("quote_abs") / pl.col("quote_abs").min()).log().sum()
            )
            .item()
        )
        for offset in HOURLY_OFFSETS
    }


def compute_share_of_longs(df_trades: pl.DataFrame, bounds: Bounds) -> Dict[str, float]:
    """Compute share of longs in overall number of trades"""
    return {
        f"share_of_long_trades_{offset.name}": (
            df_trades
            .filter(
                pl.col("trade_time").is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive)
            )
            .select(pl.col("is_long").sum() / pl.len())
            .item()
        ) for offset in HOURLY_OFFSETS
    }


def compute_features(df_ticks: pl.DataFrame, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
    """Computes features for a give currency pair"""
    # Compute quote_volumes
    df_ticks = df_ticks.sort(by="trade_time", descending=False)
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
    df_trades: pl.DataFrame = aggregate_ticks_into_trades(df_ticks=df_ticks)
    df_trades = compute_slippage(df_trades=df_trades)  # attach slippage columns

    assert df_trades["trade_time"].is_sorted(), "Data must be sorted in ascending order by trade_time"

    # Create hourly sampled data to compute not only overall values of features over intervals but also their
    # dynamics within each interval
    df_hourly: pl.DataFrame = resample_to_hourly(df_trades=df_trades)

    # Compute features
    return {
        "currency_pair": currency_pair.binance_name,
        **compute_pct_return_features(df_trades=df_trades, df_hourly=df_hourly, bounds=bounds),
        **compute_volume_features(df_trades=df_trades, df_hourly=df_hourly, bounds=bounds),
        **compute_share_of_longs(df_trades=df_trades, bounds=bounds),
        **compute_alpha_powerlaw(df_trades=df_trades, bounds=bounds),
        **compute_slippage_features(df_trades=df_trades, df_hourly=df_hourly, bounds=bounds),
    }
