from dataclasses import dataclass


@dataclass
class CurrencyPair:
    base: str
    term: str

    def __str__(self) -> str:
        return f"{self.base}{self.term}"

    @property
    def name(self) -> str:
        return f"{self.base}{self.term}"
