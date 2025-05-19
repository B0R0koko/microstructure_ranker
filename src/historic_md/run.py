from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency import get_target_currencies, Currency
from core.currency_pair import CurrencyPair
from core.parser_enums import CollectMode
from core.time_utils import Bounds
from data_collection.parsers.binance.FuturesTradeParser import BinanceFuturesTradeParser
from data_collection.settings import SETTINGS


def main():
    data_dir: Path = Path("D:/data/zipped_data/USDM")
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2024, 1, 1),
        end_exclusive=date(2025, 5, 14)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    # Only collect data for USDT paired currency_pairs
    usdt_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency, term=Currency.USDT) for currency in get_target_currencies()
    ]

    process.crawl(
        BinanceFuturesTradeParser,
        bounds=bounds,
        currency_pairs=usdt_pairs,
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
