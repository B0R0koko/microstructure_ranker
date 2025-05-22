from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List

import numpy as np
from scrapy.utils.log import configure_logging

from core.currency_pair import CurrencyPair
from core.data_type import Feature, SamplingType
from core.exchange import Exchange
from core.paths import FEATURE_DIR
from core.time_utils import format_date, Bounds, get_seconds_slug


def read_scalar(
        day: date, subpath: Path, sampling_type: SamplingType = SamplingType.MS500
) -> np.ndarray:
    """Read np.ndarray from file"""
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / subpath
    values: np.ndarray = np.load(str(path) + ".npy")

    assert len(values) == sampling_type.get_valid_size(), \
        f"Array size {len(values)} didn't match expected {sampling_type.get_valid_size()}"
    return values


def multi_day_ts(bounds: Bounds, get_day_ts: Callable[[date], np.ndarray]) -> np.ndarray:
    """Merge multiple numpy np.ndarray into a single array"""
    tss: List[np.ndarray] = [get_day_ts(day) for day in bounds.date_range()]
    return np.concatenate((*tss,))


def statistic_name(feature: Feature, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta) -> str:
    return f"{feature.value}-{currency_pair.name}-{get_seconds_slug(window)}@{exchange.name}"


def read_returns(bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """Reads features like this D:/data/features/HFT/20240102/MS500/asset_return/asset_return-ADA-USDT-0.5S"""
    stat_name: str = statistic_name(
        feature=Feature.ASSET_RETURN, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_RETURN.value) / exchange.name / stat_name,
        )
    )


def read_returns_adj(bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """Reads features like this D:/data/features/HFT/20240102/MS500/asset_return_adj/asset_return_adj-ADA-USDT-0.5S"""
    stat_name: str = statistic_name(
        feature=Feature.ASSET_RETURN_ADJ, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_RETURN_ADJ.value) / exchange.name / stat_name,
        )
    )


def read_hold_time(bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta) -> np.ndarray:
    """Reads features like this D:/data/features/HFT/20240102/MS500/asset_hold_time/asset_return-ADA-USDT-0.5S"""
    hold_time_name: str = statistic_name(
        feature=Feature.ASSET_HOLD_TIME, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.ASSET_HOLD_TIME.value) / exchange.name / hold_time_name
        )
    )


def read_slippage_imbalance(
        bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta
) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/slippage_imbalance/slippage_imbalance-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(
        feature=Feature.SLIPPAGE_IMBALANCE, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.SLIPPAGE_IMBALANCE.value) / exchange.name / stat_name,
        )
    )


def read_flow_imbalance(
        bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta
) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/flow_imbalance/flow_imbalance-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(
        feature=Feature.FLOW_IMBALANCE, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.FLOW_IMBALANCE.value) / exchange.name / stat_name,
        )
    )


def read_powerlaw_alpha(
        bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta
) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/powerlaw_alpha/powerlaw_alpha-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(
        feature=Feature.POWERLAW_ALPHA, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.POWERLAW_ALPHA.value) / exchange.name / stat_name,
        )
    )


def read_share_long_trades(
        bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta
) -> np.ndarray:
    """
    Reads features like this D:/data/features/HFT/20240102/MS500/share_of_long_trades/share_of_long_trades-ADA-USDT-0.5S
    """
    stat_name: str = statistic_name(
        feature=Feature.SHARE_OF_LONG_TRADES, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.SHARE_OF_LONG_TRADES.value) / exchange.name / stat_name,
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
        bounds=bounds,
        exchange=Exchange.BINANCE_USDM,
        currency_pair=CurrencyPair.from_string("BTC-USDT"),
        window=timedelta(milliseconds=500),
    )
    print(returns.shape)
    print(returns)


if __name__ == "__main__":
    configure_logging()
    run_test()
