from datetime import timedelta
from enum import Enum
from typing import Dict


class Feature(Enum):
    ASSET_RETURN = "asset_return"
    ASSET_RETURN_ADJ = "asset_return_adj"
    ASSET_HOLD_TIME = "asset_hold_time"
    SIGMA = "sigma"
    FLOW_IMBALANCE = "flow_imbalance"
    SLIPPAGE_IMBALANCE = "slippage_imbalance"
    POWERLAW_ALPHA = "powerlaw_alpha"
    SHARE_OF_LONG_TRADES = "share_of_long_trades"
    CLOSE_PRICE = "close_price"


class SamplingType(Enum):
    MS500 = timedelta(milliseconds=500)

    def get_valid_size(self) -> int:
        """Returns expected number of observations for SamplingType within a day"""
        sizes: Dict[SamplingType, int] = {
            SamplingType.MS500: 2 * 3600 * 24
        }
        return sizes[self]
