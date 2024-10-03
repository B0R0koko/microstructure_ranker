from dataclasses import dataclass


@dataclass
class CurrencyPair:
    BASE: str
    TERM: str

    def __str__(self) -> str:
        return f"{self.BASE}{self.TERM}"
