from typing import List, Dict, Any

import polars as pl

from core.columns import CLOSE_TIME, TAKER_BUY_QUOTE_ASSET_VOLUME, QUOTE_ASSET_VOLUME, CLOSE_PRICE, OPEN_PRICE
from core.currency import CurrencyPair
from core.time_utils import Bounds, TimeOffset


def compute_volume_imbalance(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Returns a dict of volume imbalance features computed using different offsets from end_date"""

    return {
        f"volume_imbalance_{offset.name}": (
            df_currency_pair
            .filter(pl.col(CLOSE_TIME).is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
            # TAKER_BUY_BASE_ASSET_VOLUME - (QUOTE_ASSET_VOLUME - TAKER_BUY_BASE_ASSET_VOLUME) =
            # 2 * TAKER_BUY_BASE_ASSET_VOLUME - QUOTE_ASSET_VOLUME -> signed volume
            .select(
                (2 * pl.col(TAKER_BUY_QUOTE_ASSET_VOLUME) - pl.col(QUOTE_ASSET_VOLUME)).sum() /
                pl.col(QUOTE_ASSET_VOLUME).sum()
            )
            .item()
        )
        for offset in time_offsets
    }


def compute_log_return_features(
        df_currency_pair: pl.DataFrame, bounds: Bounds, time_offsets: List[TimeOffset]
) -> Dict[str, float]:
    """Compute log returns for different intervals before the prediction timestamp with different time_offsets"""
    return {
        f"log_return_{offset.name}": (
            df_currency_pair
            .filter(pl.col(CLOSE_TIME).is_between(bounds.end_exclusive - offset.value, bounds.end_exclusive))
            .select((pl.col(CLOSE_PRICE).last() / pl.col(OPEN_PRICE).first()).log())
            .item()
        )
        for offset in time_offsets
    }


def compute_features(df_currency_pair: pl.DataFrame, currency_pair: CurrencyPair, bounds: Bounds) -> Dict[str, Any]:
    desired_offsets: List[TimeOffset] = [
        TimeOffset.MINUTE,
        TimeOffset.FIVE_MINUTES,
        TimeOffset.HALF_HOUR,
        TimeOffset.HOUR,
        TimeOffset.TWO_HOURS,
        TimeOffset.FOUR_HOURS
    ]

    volume_imbalance_features: Dict[str, float] = compute_volume_imbalance(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )
    log_return_features: Dict[str, float] = compute_log_return_features(
        df_currency_pair=df_currency_pair, bounds=bounds, time_offsets=desired_offsets
    )

    return {
        "currency_pair": currency_pair.binance_name,
        **volume_imbalance_features,
        **log_return_features,
    }
