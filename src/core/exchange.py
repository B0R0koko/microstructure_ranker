from enum import Enum
from pathlib import Path
from typing import Dict

from core.paths import BINANCE_SPOT_HIVE_TRADES, BINANCE_USDM_HIVE_TRADES, OKX_SPOT_HIVE_TRADES


class Exchange(Enum):
    BINANCE_SPOT = "binance_spot"
    BINANCE_USDM = "binance_usdm"
    OKX_SPOT = "okx_spot"

    def get_hive_location(self) -> Path:
        hives: Dict[Exchange, Path] = {
            Exchange.BINANCE_SPOT: BINANCE_SPOT_HIVE_TRADES,
            Exchange.BINANCE_USDM: BINANCE_USDM_HIVE_TRADES,
            Exchange.OKX_SPOT: OKX_SPOT_HIVE_TRADES,
        }
        return hives[self]
