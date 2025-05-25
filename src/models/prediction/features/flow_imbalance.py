from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np

from core.currency import Currency
from core.currency_pair import CurrencyPair
from core.data_type import Feature
from core.exchange import Exchange, ExchangeSet
from core.time_utils import Bounds
from ml_base.features import FeatureFilter
from models.prediction.features.utils import statistic_name, multi_day_ts, read_scalar, get_feature_name


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


def add_flow_imbalances(
        ts_by_name: Dict[str, np.ndarray],
        bounds: Bounds,
        currency: Currency,
        exchange_set: ExchangeSet,
        feature_filter: FeatureFilter,
        window: timedelta,
        prefix: str = "SELF",
) -> None:
    for exchange in exchange_set.all_exchanges():
        feature_name: str = get_feature_name(
            feature=Feature.FLOW_IMBALANCE, exchange=exchange, window=window, prefix=prefix
        )
        # Check if feature is allowed by feature importance file
        if feature_filter.is_allowed(feature_name=feature_name):
            ts_by_name[feature_name] = read_flow_imbalance(
                bounds=bounds,
                exchange=exchange,
                currency_pair=CurrencyPair(base=currency.name, term=Currency.USDT.name),
                window=window
            )
