from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency import Currency, get_target_currencies
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_USDM_RAW_L1
from core.time_utils import Bounds
from historic_md.parsers.binance.BinanceSpotTradesParser import BinanceSpotTradesParser
from historic_md.parsers.settings import SETTINGS


class BinanceUSDML1Parser(BinanceSpotTradesParser):
    name: str = "binance_usdm_l1_parser"

    def __init__(
            self,
            currency_pairs: List[CurrencyPair],
            bounds: Bounds,
            output_dir: Path,
    ):
        super().__init__(
            currency_pairs=currency_pairs,
            bounds=bounds,
            output_dir=output_dir
        )

    @staticmethod
    def get_prefix(currency_pair: CurrencyPair) -> str:
        return f"data/futures/um/daily/bookTicker/{currency_pair.binance_name}/"


def main():
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2025, 5, 1),
        end_exclusive=date(2025, 5, 25)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    # Only collect data for USDT paired currency_pairs
    currency_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency.name, term=Currency.USDT.name) for currency in get_target_currencies()
    ]

    process.crawl(
        BinanceUSDML1Parser,
        bounds=bounds,
        currency_pairs=currency_pairs,
        output_dir=BINANCE_USDM_RAW_L1
    )
    process.start()


if __name__ == "__main__":
    main()
