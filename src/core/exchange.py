from enum import Enum
from pathlib import Path
from typing import Dict, List

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

    @staticmethod
    def parse_from_lower(exchange_str: str) -> "Exchange":
        return Exchange[exchange_str.upper()]


class ExchangeSet:

    def __init__(self, target_exchange: Exchange, feature_exchanges: List[Exchange]):
        self.target_exchange: Exchange = target_exchange
        self.feature_exchanges: List[Exchange] = feature_exchanges

    def all_exchanges(self) -> List[Exchange]:
        """Return all exchanges including the target one"""
        return [self.target_exchange] + self.feature_exchanges

    @classmethod
    def for_exchange(cls, target_exchange: Exchange) -> "ExchangeSet":
        feature_exchanges: Dict[Exchange, List[Exchange]] = {
            Exchange.BINANCE_SPOT: [Exchange.BINANCE_USDM, Exchange.OKX_SPOT],
            Exchange.BINANCE_USDM: [Exchange.BINANCE_SPOT, Exchange.OKX_SPOT],
            Exchange.OKX_SPOT: [Exchange.BINANCE_SPOT, Exchange.BINANCE_USDM],
        }
        assert target_exchange in feature_exchanges
        return cls(target_exchange=target_exchange, feature_exchanges=feature_exchanges[target_exchange])
