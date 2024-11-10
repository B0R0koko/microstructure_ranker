from pathlib import Path
from typing import List

from scrapy.crawler import CrawlerProcess

from core.collect_mode import CollectMode
from core.currency import CurrencyPair
from data_collection.datavision.crawler import TradesCrawler
from data_collection.datavision.settings import SETTINGS

CURRENCY_PAIRS: List[CurrencyPair] = [
    CurrencyPair(base="BTC", term="USDT")
]


def main():
    data_dir: Path = Path("D:/data/zipped_data")

    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    process.crawl(
        TradesCrawler,
        currency_pairs=CURRENCY_PAIRS,
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
