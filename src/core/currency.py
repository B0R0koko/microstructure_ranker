import functools
from enum import Enum


@functools.total_ordering
class Currency(Enum):
    BTC = 1
    ETH = 2
    XRP = 3
    BNB = 4
    SOL = 5
    DOGE = 6
    ADA = 7
    TRX = 8
    SUI = 9
    LINK = 10
    AVAX = 11
    XLM = 12
    SHIB = 13
    BCH = 14
    HBAR = 15
    USDT = 19
    BUSD = 20
    TUSD = 21
    USDC = 22
    FDUSD = 23

    def __lt__(self, other):
        return str(self) < str(other)

    def is_stable_coin(self) -> bool:
        return self in (Currency.BUSD, Currency.USDT, Currency.TUSD, Currency.FDUSD, Currency.USDC)
