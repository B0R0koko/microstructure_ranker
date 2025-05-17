import logging
from datetime import timedelta, date
from typing import List, Dict

import numpy as np
import pandas as pd
from scrapy.utils.log import configure_logging

from core.currency import Currency
from core.currency_pair import CurrencyPair
from core.data_type import SamplingType
from core.feature_set import FeatureSet
from core.time_utils import Bounds, get_seconds_slug
from feature_writer.HFTFeatureWriter import SAMPLING_WINDOWS
from ml_base.enums import DatasetType
from ml_base.sample import SampleParams, Sample, concat_samples
from models.prediction.columns import COL_CURRENCY_INDEX, COL_OUTPUT
from models.prediction.features import read_returns, read_slippage_imbalance, read_flow_imbalance, read_powerlaw_alpha, \
    read_share_long_trades, read_hold_time, shift, read_index


def get_target_currencies() -> List[Currency]:
    return [currency for currency in Currency if not currency.is_stable_coin()]


class BuildDataset:

    def __init__(
            self,
            bounds: Bounds,
            target_currencies: List[Currency],
            sample_params: SampleParams,
            forecast_step: timedelta
    ):
        self.bounds: Bounds = bounds
        self.target_currencies: List[Currency] = target_currencies
        self.sample_params: SampleParams = sample_params
        self.forecast_step: timedelta = forecast_step

    def read_common_features(self) -> Dict[str, np.ndarray]:
        """Read features for feature currencies like BTC and ETH"""
        btc_features: Dict[str, np.ndarray] = self.read_currency_specific_features(
            currency=Currency.BTC, prefix="BTC"
        )
        eth_features: Dict[str, np.ndarray] = self.read_currency_specific_features(
            currency=Currency.ETH, prefix="ETH"
        )
        return btc_features | eth_features

    def read_currency_specific_features(self, currency: Currency, prefix: str = "SELF") -> Dict[str, np.ndarray]:
        logging.info("Reading currency specific features for %s", currency.name)
        features: Dict[str, np.ndarray] = {}
        currency_pair: CurrencyPair = CurrencyPair(base=currency, term=Currency.USDT)

        for window in SAMPLING_WINDOWS:
            # Read sampled features for each SAMPLING_WINDOW
            features[f"{prefix}-asset_return-{get_seconds_slug(window)}"] = read_returns(
                bounds=self.bounds, currency_pair=currency_pair, window=window
            )
            features[f"{prefix}-hold_time-{get_seconds_slug(window)}"] = read_hold_time(
                bounds=self.bounds, currency_pair=currency_pair, window=window
            )
            features[f"{prefix}-slippage_imbalance-{get_seconds_slug(window)}"] = read_slippage_imbalance(
                bounds=self.bounds, currency_pair=currency_pair, window=window
            )
            features[f"{prefix}-flow_imbalance_{get_seconds_slug(window)}"] = read_flow_imbalance(
                bounds=self.bounds, currency_pair=currency_pair, window=window
            )
            features[f"{prefix}-powerlaw_alpha-{get_seconds_slug(window)}"] = read_powerlaw_alpha(
                bounds=self.bounds, currency_pair=currency_pair, window=window
            )
            features[f"{prefix}-share_of_long_trades-{get_seconds_slug(window)}"] = read_share_long_trades(
                bounds=self.bounds, currency_pair=currency_pair, window=window
            )

        return features

    def read_output(self, currency: Currency) -> np.ndarray:
        """Reads output return in pips with forecast step"""
        logging.info("Reading output for %s", currency.name)
        currency_pair: CurrencyPair = CurrencyPair(base=currency, term=Currency.USDT)
        returns: np.ndarray = read_returns(
            bounds=self.bounds, currency_pair=currency_pair, window=self.forecast_step
        )
        hold_time_sec: np.ndarray = read_hold_time(
            bounds=self.bounds, currency_pair=currency_pair, window=self.forecast_step
        )
        returns_adj: np.ndarray = returns / hold_time_sec
        shifted_returns: np.ndarray = shift(
            returns_adj, n=-int(self.forecast_step / SamplingType.MS500.value)
        )
        return shifted_returns

    def read_categorical_features(self, currency: Currency, col_size: int) -> Dict[str, np.ndarray]:
        """Read categorical features for the currency"""
        logging.info("Reading categorical features for %s", currency.name)
        # Add categorical features
        categorical_features: Dict[str, np.ndarray] = {
            COL_CURRENCY_INDEX: np.full(col_size, fill_value=currency.value, dtype=np.int32),
        }
        return categorical_features

    def split_and_wrap_into_samples(self, df: pd.DataFrame, feature_set: FeatureSet) -> Sample:
        """
        Split data into TRAIN/VALIDATION/TEST using SampleParams and wrap DataFrames into Sample object.
        Do all preprocessing here as well like standardizations and clipping for each DataFrame then wrap into
        Sample
        """
        datasets: Dict[DatasetType, pd.DataFrame] = self.sample_params.split_by_indecies(df=df)
        # Do preprocessing here if needed
        sample: Sample = Sample.from_pandas_datasets(datasets=datasets, feature_set=feature_set)
        logging.info("Created %s", list(sample.dataset_types))
        return sample

    def create_for_currency(
            self, currency: Currency, features_common: Dict[str, np.ndarray], index: np.ndarray
    ) -> Sample:
        logging.info("\n\nCreating sample for %s", currency.name)
        features_currency_specific: Dict[str, np.ndarray] = self.read_currency_specific_features(currency=currency)
        features_categorical: Dict[str, np.ndarray] = self.read_categorical_features(
            currency=currency, col_size=len(index)
        )
        output: np.ndarray = self.read_output(currency=currency)
        # Merge all dictionaries together and create DataFrame from it
        regressors: Dict[str, np.ndarray] = features_currency_specific | features_common | features_categorical
        features: Dict[str, np.ndarray] = regressors | {COL_OUTPUT: output}
        # Create FeatureSet
        feature_set: FeatureSet = FeatureSet(
            regressors=list(regressors.keys()),
            categorical=list(features_categorical.keys()),
            target=COL_OUTPUT
        )

        df: pd.DataFrame = pd.DataFrame(features)
        df = df.dropna(subset=["output"])  # remove observations with missing targets
        return self.split_and_wrap_into_samples(df=df, feature_set=feature_set)

    def create_dataset(self) -> Sample:
        # Read BTC/ETH features as sort of like to capture overall market movements
        features_common: Dict[str, np.ndarray] = self.read_common_features()
        date_index: np.ndarray = read_index(bounds=self.bounds)
        samples: List[Sample] = []

        for currency in self.target_currencies:
            currency_sample: Sample = self.create_for_currency(
                currency=currency, features_common=features_common, index=date_index
            )
            samples.append(currency_sample)

        return concat_samples(samples=samples, require_lgb=True)


def run_test():
    configure_logging()
    bounds: Bounds = Bounds.for_days(
        date(2024, 1, 5), date(2024, 1, 12)
    )
    build = BuildDataset(
        bounds=bounds,
        target_currencies=get_target_currencies(),
        sample_params=SampleParams(
            train_share=.7, validation_share=.15, test_share=.15
        ),
        forecast_step=timedelta(seconds=2)
    )
    build.create_dataset()


if __name__ == "__main__":
    run_test()
