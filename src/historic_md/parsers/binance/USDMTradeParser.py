from datetime import date
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlencode

from scrapy.crawler import CrawlerProcess

from core.currency import Currency, get_target_currencies
from core.currency_pair import CurrencyPair
from core.parser_enums import CollectMode
from core.time_utils import Bounds
from historic_md.parsers.binance.BinanceParser import BinanceParser, BINANCE_S3
from historic_md.settings import SETTINGS


class BinanceUSDMTradesParser(BinanceParser):
    name: str = "trades_parser"

    def __init__(
            self,
            currency_pairs: List[CurrencyPair],
            bounds: Bounds,
            collect_mode: CollectMode,
            output_dir: Path,
    ):
        super().__init__(
            currency_pairs=currency_pairs,
            bounds=bounds,
            collect_mode=collect_mode,
            output_dir=output_dir
        )

    def get_currency_url(self, currency_pair: CurrencyPair, marker: Optional[str] = None) -> str:
        params: Dict[str, str] = {
            "delimiter": "/",
            "prefix": f"data/futures/um/{self.collect_mode.lower()}/trades/{currency_pair.binance_name}/",
        }

        if marker is not None:
            params["marker"] = marker

        datavision_url: str = f"{BINANCE_S3}?{urlencode(params)}"
        return datavision_url


def main():
    data_dir: Path = Path("D:/data/zipped_data/trades/USDM")
    bounds: Bounds = Bounds.for_days(
        start_inclusive=date(2024, 1, 1),
        end_exclusive=date(2024, 2, 1)
    )
    process: CrawlerProcess = CrawlerProcess(settings=SETTINGS)
    # Only collect data for USDT paired currency_pairs
    currency_pairs: List[CurrencyPair] = [
        CurrencyPair(base=currency, term=Currency.USDT) for currency in get_target_currencies()
    ]

    process.crawl(
        BinanceUSDMTradesParser,
        bounds=bounds,
        currency_pairs=currency_pairs,
        collect_mode=CollectMode.MONTHLY,
        output_dir=data_dir
    )
    process.start()


if __name__ == "__main__":
    main()
