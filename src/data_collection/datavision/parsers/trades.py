from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlencode

from core.currency import CurrencyPair
from core.parser_enums import CollectMode, KlineInterval
from core.time_utils import Bounds
from data_collection.datavision.parser import DataParser, BINANCE_S3


class TradeParser(DataParser):
    name: str = "trades_parser"

    def __init__(
            self,
            currency_pairs: List[CurrencyPair],
            bounds: Bounds,
            collect_mode: CollectMode,
            output_dir: Path,
            kline_interval: KlineInterval
    ):
        super().__init__(
            currency_pairs=currency_pairs,
            bounds=bounds,
            collect_mode=collect_mode,
            output_dir=output_dir
        )
        self.kline_interval: KlineInterval = kline_interval

    def get_currency_url(self, currency_pair: CurrencyPair, marker: Optional[str] = None) -> str:
        params: Dict[str, str] = {
            "delimiter": "/",
            "prefix": f"data/spot/{self.collect_mode.lower()}/trades/{currency_pair.binance_name}/",
        }

        if marker is not None:
            params["marker"] = marker

        datavision_url: str = f"{BINANCE_S3}?{urlencode(params)}"
        return datavision_url
