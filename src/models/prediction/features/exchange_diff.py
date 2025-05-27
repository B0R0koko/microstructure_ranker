from datetime import timedelta, date
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from core.currency import Currency
from core.currency_pair import CurrencyPair
from core.data_type import Feature
from core.exchange import Exchange, ExchangeSet
from core.time_utils import Bounds, get_seconds_slug
from ml_base.features import FeatureFilter
from models.prediction.features.utils import multi_day_ts, read_scalar, statistic_name


def _get_exchange_diff_name(
        target_exchange: Exchange,
        other_exchange: Exchange,
        window: timedelta,
        prefix: str = "SELF"
) -> str:
    """Returns feature names like this SELF-exchange_diff-BINANCE_SPOT-OKX_SPOT-5S"""
    return (f"{prefix}-{Feature.EXCHANGE_DIFF.value}-{target_exchange.name}-{other_exchange.name}"
            f"-{get_seconds_slug(td=window)}")


def read_exchange_diff(
        bounds: Bounds,
        target_exchange: Exchange,
        other_exchange: Exchange,
        currency_pair: CurrencyPair,
        window: timedelta
) -> np.ndarray:
    """Returns the difference in returns between target_exchange and other_exchange"""
    # Read first target_exchange returns
    target_returns_name: str = statistic_name(
        feature=Feature.ASSET_RETURN, exchange=target_exchange, currency_pair=currency_pair, window=window
    )
    target_returns: np.ndarray = multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_RETURN.value) / target_exchange.name / target_returns_name,
        )
    )
    # Read returns for another exchange
    other_returns_name: str = statistic_name(
        feature=Feature.ASSET_RETURN, exchange=other_exchange, currency_pair=currency_pair, window=window
    )
    other_returns: np.ndarray = multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_RETURN.value) / other_exchange.name / other_returns_name
        )
    )
    assert target_returns.shape == other_returns.shape

    return target_returns - other_returns


def add_exchange_diffs(
        ts_by_name: Dict[str, np.ndarray],
        bounds: Bounds,
        currency: Currency,
        exchange_set: ExchangeSet,
        feature_filter: FeatureFilter,
        window: timedelta,
        prefix: str = "SELF",
) -> None:
    for feature_exchange in exchange_set.feature_exchanges:
        feature_name: str = _get_exchange_diff_name(
            target_exchange=exchange_set.target_exchange,
            other_exchange=feature_exchange,
            window=window,
            prefix=prefix,
        )
        # Check if feature is allowed by feature importance file
        if feature_filter.is_allowed(feature_name=feature_name):
            ts_by_name[feature_name] = read_exchange_diff(
                bounds=bounds,
                target_exchange=exchange_set.target_exchange,
                other_exchange=feature_exchange,
                currency_pair=CurrencyPair(base=currency.name, term=Currency.USDT.name),
                window=window
            )

def run_test() -> None:
    """Check if the data is not identical on different exchanges"""
    bounds: Bounds = Bounds.for_day(day=date(2025, 5, 1))
    exchange_diff: np.ndarray = read_exchange_diff(
        bounds=bounds,
        target_exchange=Exchange.BINANCE_SPOT,
        other_exchange=Exchange.BINANCE_USDM,
        currency_pair=CurrencyPair.from_string("BTC-USDT"),
        window=timedelta(seconds=5),
    )
    print(pd.Series(exchange_diff).describe())


if __name__ == "__main__":
    run_test()

