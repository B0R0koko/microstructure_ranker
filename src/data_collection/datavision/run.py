from pathlib import Path

from scrapy.crawler import CrawlerProcess

from data_collection.core.collect_mode import CollectMode
from data_collection.core.currency import collect_all_currency_pairs
from data_collection.datavision.crawler import TradesCrawler
from data_collection.datavision.settings import SETTINGS


def main():
    data_dir: Path = Path("D:/data")

    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    process.crawl(
        TradesCrawler,
        currency_pairs=collect_all_currency_pairs(),
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
