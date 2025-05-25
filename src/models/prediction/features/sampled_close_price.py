from datetime import timedelta
from pathlib import Path

import numpy as np

from core.currency_pair import CurrencyPair
from core.data_type import Feature
from core.exchange import Exchange
from core.time_utils import Bounds
from models.prediction.features.utils import statistic_name, multi_day_ts, read_scalar


def read_sampled_close_price(
        bounds: Bounds, exchange: Exchange, currency_pair: CurrencyPair, window: timedelta
) -> np.ndarray:
    stat_name: str = statistic_name(
        feature=Feature.CLOSE_PRICE, exchange=exchange, currency_pair=currency_pair, window=window
    )
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path(Feature.CLOSE_PRICE.value) / exchange.name / stat_name,
        )
    )
