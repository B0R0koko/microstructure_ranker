from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency import get_target_currencies, Currency
from core.currency_pair import CurrencyPair
from core.time_utils import Bounds
from historic_md.parsers.okx.OKXParser import OKXTradeParser
from historic_md.settings import SETTINGS


def main():
    data_dir: Path = Path("D:/data/zipped_data/OKX_SPOT")
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2024, 1, 1),
        end_exclusive=date(2024, 2, 1)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    # Only collect data for USDT paired currency_pairs
    usdt_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency, term=Currency.USDT) for currency in get_target_currencies()
    ]

    process.crawl(
        OKXTradeParser,
        bounds=bounds,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
