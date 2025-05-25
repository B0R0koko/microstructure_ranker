from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency import Currency
from core.currency_pair import CurrencyPair
from core.paths import BINANCE_USDM_RAW_TRADES
from core.time_utils import Bounds
from historic_md.parsers.binance.BinanceSpotTradesParser import BinanceSpotTradesParser
from historic_md.parsers.settings import SETTINGS


class BinanceUSDMTradesParser(BinanceSpotTradesParser):
    name: str = "binance_usdm_trades_parser"

    def __init__(
            self,
            bounds: Bounds,
            currency_pairs: List[CurrencyPair],
            output_dir: Path,
    ):
        super().__init__(
            currency_pairs=currency_pairs,
            bounds=bounds,
            output_dir=output_dir
        )

    def get_prefix(self, currency_pair: CurrencyPair) -> str:
        return f"data/futures/um/daily/trades/{currency_pair.binance_name}/"


def main():
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2025, 5, 1),
        end_exclusive=date(2025, 5, 25)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    # Only collect data for USDT paired currency_pairs
    currency_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency.name, term="USDT") for currency in (Currency.BTC,)
    ]

    process.crawl(
        BinanceUSDMTradesParser,
        bounds=bounds,
        currency_pairs=currency_pairs,
        output_dir=BINANCE_USDM_RAW_TRADES
    )
    process.start()


if __name__ == "__main__":
    main()
