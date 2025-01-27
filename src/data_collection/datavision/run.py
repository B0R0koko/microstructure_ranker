from datetime import date
from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.currency import collect_all_currency_pairs, CurrencyPair
from core.parser_enums import CollectMode, KlineInterval
from core.time_utils import Bounds
from data_collection.datavision.parsers.klines import KlineParser
from data_collection.datavision.settings import SETTINGS


def main():
    data_dir: Path = Path("D:/data/zipped_data/1m")
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2024, 11, 1),
        end_exclusive=date(2024, 12, 1)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    # Only collect data for USDT paired currency_pairs
    currency_pairs: List[CurrencyPair] = collect_all_currency_pairs()
    usdt_pairs: List[CurrencyPair] = [
        currency_pair for currency_pair in currency_pairs if currency_pair.term == "USDT"
    ]

    process.crawl(
        KlineParser,
        bounds=bounds,
        currency_pairs=usdt_pairs,
        collect_mode=CollectMode.MONTHLY,
        kline_interval=KlineInterval.MINUTE,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
