from typing import List

import scrapy
from scrapy.http import Request

from core.currency_pair import CurrencyPair
from core.time_utils import Bounds


class OKXParser(scrapy.Spider):
    """Parser of historic market data for OKX exchange"""

    def __init__(self, bounds: Bounds, currency_pairs: List[CurrencyPair], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currency_pairs: List[CurrencyPair] = currency_pairs
        self.bounds: Bounds = bounds

    def get_currency_url(self, currency_pair: CurrencyPair) -> str:
        return "https://www.okx.com/cdn/okex/traderecords/trades/monthly/202409/allspot-trades-2024-09-02.zip"

    def start_requests(self) -> scrapy.Request:
        for currency_pair in self.currency_pairs:
            yield Request(
                url=self.get_currency_url(currency_pair),
                callback=self.parse_currency_pair,  # type:ignore
                meta={"currency_pair": currency_pair, "href_container": []},  # mutable object
            )
