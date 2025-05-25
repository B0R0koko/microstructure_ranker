from datetime import timedelta
from pathlib import Path

import numpy as np

from core.currency_pair import CurrencyPair
from core.data_type import Feature
from core.exchange import Exchange
from core.time_utils import Bounds
from models.prediction.features.utils import statistic_name, multi_day_ts, read_scalar


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
