import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List

import numpy as np

from core.currency_pair import CurrencyPair
from core.data_type import SamplingType, Feature
from core.exchange import Exchange
from core.paths import FEATURE_DIR
from core.time_utils import format_date, Bounds, get_seconds_slug


def read_scalar(
        day: date, subpath: Path, sampling_type: SamplingType = SamplingType.MS500
) -> np.ndarray:
    """
    If the file exists and its size is valid for the specified SamplingType, then read the data from file and return it
    otherwise return empty np.ndarray with nan values
    """
    path: Path = FEATURE_DIR / "HFT" / format_date(day=day) / sampling_type.name / subpath
    path = path.with_suffix(suffix=".npy")
    expected_shape = (sampling_type.get_valid_size(),)

    data: np.ndarray = np.full(shape=expected_shape, fill_value=np.nan, dtype=np.float64)

    if not path.exists():
        logging.info("Path %s does not exist", path)
        return data

    values: np.ndarray = np.load(path)
    if values.shape == expected_shape:
        return values

    logging.info("Wrong shape of %s. Ignoring the whole day.", path)

    return data


def multi_day_ts(bounds: Bounds, get_day_ts: Callable[[date], np.ndarray]) -> np.ndarray:
    """Merge multiple numpy np.ndarray into a single array"""
    tss: List[np.ndarray] = [get_day_ts(day) for day in bounds.date_range()]
    return np.concatenate((*tss,))


def statistic_name(feature: Feature, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta) -> str:
    return f"{feature.value}-{currency_pair.name}-{get_seconds_slug(window)}@{exchange.name}"


def get_feature_name(feature: Feature, exchange: Exchange, window: timedelta, prefix: str = "SELF") -> str:
    """Returns feature names that are used during training and the names that are stored in feature importance files"""
    return f"{prefix}-{feature.value}-{get_seconds_slug(window)}@{exchange.name}"


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
