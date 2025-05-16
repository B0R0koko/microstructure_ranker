import logging
from datetime import date
from pathlib import Path
from typing import Callable, List

import numpy as np
from scrapy.utils.log import configure_logging

from core.currency import CurrencyPair
from core.data_type import Feature, SamplingType
from core.paths import FEATURE_DIR
from core.time_utils import format_date, Bounds


def read_scalar(
        day: date, data_type: Feature, subpath: Path, sampling_type: SamplingType = SamplingType.MS500
) -> np.ndarray:
    """Read np.ndarray from file"""
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / data_type.name / subpath
    logging.info("Reading %s from %s", data_type.name, str(path))
    return np.load(str(path) + ".npy")


def multi_day_ts(bounds: Bounds, get_day_ts: Callable[[date], np.ndarray]) -> np.ndarray:
    """Merge multiple numpy np.ndarray into a single array"""
    tss: List[np.ndarray] = [get_day_ts(day) for day in bounds.date_range()]
    return np.concatenate((*tss,))


def return_statistic_name(currency_pair: CurrencyPair, window_seconds: float) -> str:
    return "-".join([Feature.ASSET_RETURN.value, currency_pair.name, f"{str(float(window_seconds))}S"])


def read_returns(bounds: Bounds, currency_pair: CurrencyPair, window_seconds: float) -> np.ndarray:
    r"""Reads features like this D:\data\features\HFT\20240102\MS500\asset_hold_time\asset_return-ADA-USDT-0.5S"""
    return_name: str = return_statistic_name(currency_pair=currency_pair, window_seconds=window_seconds)
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            data_type=Feature.ASSET_RETURN,
            subpath=Path(return_name)
        )
    )


def run_test():
    bounds: Bounds = Bounds.for_days(date(2024, 1, 1), date(2024, 1, 5))
    returns: np.ndarray = read_returns(
        bounds=bounds, currency_pair=CurrencyPair.from_string("BTC-USDT"), window_seconds=1
    )

    print(returns.shape)
    print(returns)


if __name__ == "__main__":
    configure_logging()
    run_test()
