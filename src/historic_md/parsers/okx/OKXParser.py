import os
from datetime import date
from pathlib import Path

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import Request, Response

from core.paths import OKX_SPOT_RAW_TRADES
from core.time_utils import Bounds
from historic_md.parsers.settings import SETTINGS


class OKXTradeParser(scrapy.Spider):
    """
    Parser of historic market data for OKX exchange. OKX stores data in big zip files that contain all CurrencyPairs
    OKX uses 16:00 UTC+0 which corresponds to Hong Kong 24:00 local time
    """
    name = "okx_spot_trade_parser"

    def __init__(self, bounds: Bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds: Bounds = bounds
        self.output_dir: Path = OKX_SPOT_RAW_TRADES

    def get_zip_file_url(self, day: date) -> str:
        return f"https://www.okx.com/cdn/okex/traderecords/trades/monthly/{day.strftime("%Y%m")}/" \
               f"allspot-trades-{str(day)}.zip"

    def start_requests(self):
        for day in self.bounds.date_range():
            yield Request(
                url=self.get_zip_file_url(day=day),
                callback=self.parse_zip_file,
                meta={"day": day}
            )

    def parse_zip_file(self, response: Response) -> None:
        day: date = response.meta.get("day")
        path: Path = self.output_dir / f"trades@{str(day)}.zip"

        os.makedirs(path.parent, exist_ok=True)

        with open(path, "wb") as file:
            file.write(response.body)


def run_main():
    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 7),
        date(2025, 4, 8)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)

    process.crawl(
        OKXTradeParser,
        bounds=bounds,
    )
    process.start()


if __name__ == "__main__":
    run_main()
