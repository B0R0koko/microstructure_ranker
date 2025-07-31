from dataclasses import dataclass
from datetime import datetime

from core.currency_pair import CurrencyPair
from core.exchange import Exchange


@dataclass
class PumpEvent:
    currency_pair: CurrencyPair
    time: datetime
    exchange: Exchange

    def __str__(self) -> str:
        return f"{self.currency_pair.name}@{self.exchange.name}-{self.time.date()}"
