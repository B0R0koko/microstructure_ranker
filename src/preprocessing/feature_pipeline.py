from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

import polars as pl
from tqdm import tqdm

from core.currency import CurrencyPair


class FeaturePipeline(ABC):

    def __init__(self, hive_dir: Path):
        self.hive_dir: Path = hive_dir

    def _get_currency_pairs_cross_section(self, start_time: datetime, end_time: datetime) -> List[CurrencyPair]:
        """
        Returns a list of CurrencyPair for which there is data stored in self.hive_dir: Path for the
        given time interval
        """
        df_hive: pl.LazyFrame = pl.scan_parquet(self.hive_dir)
        unique_symbols: List[str] = (
            df_hive.filter(
                pl.col("date").is_between(lower_bound=start_time, upper_bound=end_time),
            )
            .select("symbol").unique().collect()["symbol"].to_list()
        )
        return [CurrencyPair.from_string(symbol=symbol) for symbol in unique_symbols]

    def load_cross_section(self, start_time: datetime, end_time: datetime) -> pl.DataFrame:
        currency_pairs: List[CurrencyPair] = self._get_currency_pairs_cross_section(
            start_time=start_time, end_time=end_time
        )
        pbar = tqdm(currency_pairs)

        dfs: List[pl.DataFrame] = []

        for currency_pair in pbar:
            pbar.set_description(desc=f"Computing features for {currency_pair.name}")
            df_currency_pair: pl.DataFrame = self.compute_features_for_currency_pair(currency_pair=currency_pair)
            dfs.append(df_currency_pair)

        return pl.concat(dfs)

    @abstractmethod
    def compute_features_for_currency_pair(self, currency_pair: CurrencyPair) -> pl.DataFrame:
        """
        Define logic of how we load data for a passed in currency_pair as well as how we compute features. This method
        must return pl.DataFrame with all compute features. Later this dataframe will be attached to df_cross_sections:
        pl.DataFrame
        """


if __name__ == "__main__":
    hive_dir: Path = Path("D:/data/transformed_data")
    start_date: datetime = datetime(2024, 9, 1)
    end_date: datetime = datetime(2024, 10, 1)

    pipeline: FeaturePipeline = FeaturePipeline(hive_dir=hive_dir)
    pipeline.load_cross_section(start_time=start_date, end_time=end_date)
