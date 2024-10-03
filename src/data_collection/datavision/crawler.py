import re
from typing import List

import scrapy
from scrapy.http import Request
from scrapy.http.response import Response

from data_collection.core.currency import CurrencyPair

BASE_URL: str = (
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/trades"
)


def get_currency_url(currency_pair: CurrencyPair) -> str:
    datavision_url: str = f"{BASE_URL}/{str(currency_pair)}"
    return datavision_url


class TickerCrawler(scrapy.Spider):
    name = "ticker_crawler"

    def __init__(self, currency_pairs: List[CurrencyPair], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currency_pairs: List[CurrencyPair] = currency_pairs

    def start_requests(self) -> scrapy.Request:
        for currency_pair in self.currency_pairs:
            yield Request(
                url=get_currency_url(currency_pair),
                callback=self.parse_currency_pair,  # type:ignore
            )

    def parse_currency_pair(self, response: Response):
        hrefs: List[str] = re.findall(pattern=r"<Key>(.*?)</Key>", string=response.text)
        hrefs = [href for href in hrefs if "CHECKSUM" not in href]

        ...


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    process: CrawlerProcess = CrawlerProcess(
        settings=dict(
            USER_AGENT="Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
        )
    )
    process.crawl(TickerCrawler, currency_pairs=[CurrencyPair("BTC", "USDT")])
    process.start()
