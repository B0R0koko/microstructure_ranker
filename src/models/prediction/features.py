from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List

import numpy as np
from scrapy.utils.log import configure_logging

from core.currency_pair import CurrencyPair
from core.data_type import Feature, SamplingType
from core.paths import FEATURE_DIR
from core.time_utils import format_date, Bounds, get_seconds_slug


def read_scalar(
        day: date, subpath: Path, sampling_type: SamplingType = SamplingType.MS500
) -> np.ndarray:
    """Read np.ndarray from file"""
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / subpath
    return np.load(str(path) + ".npy")


def multi_day_ts(bounds: Bounds, get_day_ts: Callable[[date], np.ndarray]) -> np.ndarray:
    """Merge multiple numpy np.ndarray into a single array"""
    tss: List[np.ndarray] = [get_day_ts(day) for day in bounds.date_range()]
    return np.concatenate((*tss,))


def statistic_name(feature: Feature, currency_pair: CurrencyPair, window: timedelta) -> str:
    return "-".join([
        feature.value,
        currency_pair.name,
        get_seconds_slug(window),
    ])


def read_returns(bounds: Bounds, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """Reads features like this D:/data/features/HFT/20240102/MS500/asset_hold_time/asset_return-ADA-USDT-0.5S"""
    stat_name: str = statistic_name(feature=Feature.ASSET_RETURN, currency_pair=currency_pair, window=window)
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_RETURN.value) / stat_name,
        )
    )


def read_hold_time(bounds: Bounds, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """Reads features like this D:/data/features/HFT/20240102/MS500/asset_hold_time/asset_return-ADA-USDT-0.5S"""
    hold_time_name: str = statistic_name(
        feature=Feature.ASSET_HOLD_TIME, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_HOLD_TIME.value) / hold_time_name
        )
    )


def read_slippage_imbalance(bounds: Bounds, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/slippage_imbalance/slippage_imbalance-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(feature=Feature.SLIPPAGE_IMBALANCE, currency_pair=currency_pair, window=window)
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.SLIPPAGE_IMBALANCE.value) / stat_name,
        )
    )


def read_flow_imbalance(bounds: Bounds, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/flow_imbalance/flow_imbalance-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(feature=Feature.FLOW_IMBALANCE, currency_pair=currency_pair, window=window)
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.FLOW_IMBALANCE.value) / stat_name,
        )
    )


def read_powerlaw_alpha(bounds: Bounds, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/powerlaw_alpha/powerlaw_alpha-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(feature=Feature.POWERLAW_ALPHA, currency_pair=currency_pair, window=window)
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.POWERLAW_ALPHA.value) / stat_name,
        )
    )


def read_share_long_trades(bounds: Bounds, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/share_of_long_trades/share_of_long_trades-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(
        feature=Feature.SHARE_OF_LONG_TRADES, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.SHARE_OF_LONG_TRADES.value) / stat_name,
        )
    )


def read_index(bounds: Bounds) -> np.ndarray:
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path("time")
        )
    )


def shift(arr: np.ndarray, n: int) -> np.ndarray:
    """
    This method shifts all values in a given array by a given number of steps.
    If n=3, then element #0 becomes element #3, and the first three elements become NaN.
    If n=-3, then the first three elements are lost, element #3 becomes element #0,
    and the last three elements become NaN.
    """
    e = np.empty_like(arr)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = arr[:-n]
    else:
        e[n:] = np.nan
        e[:n] = arr[-n:]
    return e


def run_test():
    bounds: Bounds = Bounds.for_days(date(2024, 1, 1), date(2024, 1, 5))
    returns: np.ndarray = read_returns(
        bounds=bounds, currency_pair=CurrencyPair.from_string("BTC-USDT"), window=timedelta(milliseconds=500)
    )

    print(returns.shape)
    print(returns)


if __name__ == "__main__":
    configure_logging()
    run_test()
