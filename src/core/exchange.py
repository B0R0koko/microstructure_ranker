from enum import Enum
from pathlib import Path
from typing import Dict

from core.paths import SPOT_TRADES, USDM_TRADES


class Exchange(Enum):
    BINANCE_SPOT = "binance_spot"
    BINANCE_USDM = "binance_usdm"

    def get_hive_location(self) -> Path:
        hives: Dict[Exchange, Path] = {
            Exchange.BINANCE_SPOT: SPOT_TRADES,
            Exchange.BINANCE_USDM: USDM_TRADES
        }
        return hives[self]
