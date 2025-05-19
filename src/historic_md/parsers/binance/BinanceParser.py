import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import scrapy
from scrapy import Request
from scrapy.http import Response

from core.currency_pair import CurrencyPair
from core.parser_enums import CollectMode
from core.time_utils import Bounds, start_of_the_day

BINANCE_S3: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_DATAVISION: str = "https://data.binance.vision"


def filter_hrefs_by_bounds(
        hrefs: List[str], bounds: Bounds, collect_mode: CollectMode
) -> Tuple[List[str], List[date]]:
    """Takes bounds as input and returns a list of hrefs that matches passed in Bounds"""

    filtered_hrefs: List[str] = []
    href_dates: List[date] = []

    pattern_str: str = (
        r"\d{4}-\d{2}" if collect_mode == CollectMode.MONTHLY else
        r"\d{4}-\d{2}-\d{2}"
    )

    for href in hrefs:
        # Find date in href string and parse it to datetime
        href_date_string: str = re.search(pattern=pattern_str, string=href)[0]
        href_date: datetime = start_of_the_day(
            day=datetime.strptime(
                href_date_string,
                # collect_mode is monthly parse yyyy-mm, if daily parse yyyy-mm-dd
                "%Y-%m" if collect_mode == CollectMode.MONTHLY else "%Y-%m-%d"
            ).date()
        )
        if bounds.start_inclusive <= href_date <= bounds.end_exclusive:
            filtered_hrefs.append(href)
            href_dates.append(href_date)

    return filtered_hrefs, href_dates


def get_zip_file_url(href: str) -> str:
    """Returns a formatted url string which leads to a zip file with trades data"""
    return f"{BINANCE_DATAVISION}/{href}"


class BinanceParser(ABC, scrapy.Spider):

    def __init__(
            self,
            currency_pairs: List[CurrencyPair],
            bounds: Bounds,
            collect_mode: CollectMode,
            output_dir: Path,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.currency_pairs: List[CurrencyPair] = currency_pairs
        self.bounds: Bounds = bounds
        self.collect_mode: CollectMode = collect_mode
        self.output_dir: Path = output_dir

    @abstractmethod
    def get_currency_url(self, currency_pair: CurrencyPair, marker: Optional[str] = None) -> str:
        """Returns a string url for a given CurrencyPair"""

    def start_requests(self) -> scrapy.Request:
        """This method is run first when the Spider starts"""
        for currency_pair in self.currency_pairs:
            yield Request(
                url=self.get_currency_url(currency_pair),
                callback=self.parse_currency_pair,  # type:ignore
                meta={"currency_pair": currency_pair, "href_container": []},  # mutable object
            )

    def parse_currency_pair(self, response: Response):
        """Parse hrefs with zip files from currency_pair page"""

        currency_pair: Optional[CurrencyPair] = response.meta.get("currency_pair")
        href_container: List[str] = response.meta.get("href_container")

        assert currency_pair, "Currency pair must be supplied in scrapy.http.Response.meta"
        assert href_container is not None, "Href container must be supplied in scrapy.http.Response.meta"

        hrefs: List[str] = re.findall(pattern=r"<Key>(.*?)</Key>", string=response.text)
        hrefs: List[str] = [href for href in hrefs if "CHECKSUM" not in href]
        href_container.extend(hrefs)

        # if len is 500, then we need to send another request with marker param which is the last entry in hrefs
        if len(hrefs) == 500:
            yield scrapy.Request(
                url=self.get_currency_url(currency_pair=currency_pair, marker=hrefs[-1]),
                callback=self.parse_currency_pair,  # call itself one more time
                meta={"currency_pair": currency_pair, "href_container": href_container},
            )
        # Once we have collected all hrefs into response.meta.href_container we loop over it and send requests that
        # collect zip files
        # Filter hrefs by dates that we want to collect data for
        filtered_hrefs, _ = filter_hrefs_by_bounds(
            hrefs=href_container, bounds=self.bounds, collect_mode=self.collect_mode
        )

        for href in filtered_hrefs:
            yield scrapy.Request(
                url=get_zip_file_url(href=href),
                callback=self.parse_response_from_zip_endpoint,
                meta={"href": href, "currency_pair": currency_pair},
            )

    @staticmethod
    def parse_response_from_zip_endpoint(response: Response) -> Dict[str, Any]:
        return {
            "response": response,
            "href": response.meta.get("href"),
            "currency_pair": response.meta.get("currency_pair"),
        }
