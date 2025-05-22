import logging
from datetime import timedelta, date
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.currency import Currency, get_target_currencies
from core.currency_pair import CurrencyPair
from core.data_type import SamplingType
from core.exchange import Exchange
from core.feature_set import FeatureSet
from core.time_utils import Bounds, get_seconds_slug
from core.utils import configure_logging
from feature_writer.HFTFeatureWriter import SAMPLING_WINDOWS
from ml_base.enums import DatasetType
from ml_base.sample import SampleParams, Sample, concat_samples, MLDataset
from models.prediction.columns import COL_CURRENCY_INDEX, COL_OUTPUT
from models.prediction.features import read_slippage_imbalance, read_flow_imbalance, read_powerlaw_alpha, \
    read_share_long_trades, shift, read_index, read_returns


class BuildDataset:

    def __init__(
            self,
            exchange: Exchange,
            target_currencies: List[Currency],
            forecast_step: timedelta
    ):
        self.exchange: Exchange = exchange
        self.target_currencies: List[Currency] = target_currencies
        self.forecast_step: timedelta = forecast_step

    def read_common_features(self, bounds: Bounds) -> Dict[str, np.ndarray]:
        """Read features for feature currencies like BTC and ETH"""
        btc_features: Dict[str, np.ndarray] = self.read_currency_specific_features(
            bounds=bounds, currency=Currency.BTC, prefix="BTC"
        )
        eth_features: Dict[str, np.ndarray] = self.read_currency_specific_features(
            bounds=bounds, currency=Currency.ETH, prefix="ETH"
        )
        return btc_features | eth_features

    def read_currency_specific_features(self, bounds: Bounds, currency: Currency, prefix: str = "SELF") -> Dict[
        str, np.ndarray]:
        logging.info("Reading currency specific features for %s", currency.name)
        features: Dict[str, np.ndarray] = {}
        currency_pair: CurrencyPair = CurrencyPair(base=currency, term=Currency.USDT)

        for window in SAMPLING_WINDOWS:
            for exchange in (Exchange.BINANCE_SPOT, Exchange.BINANCE_USDM):
                # Read sampled features for each SAMPLING_WINDOW
                features[f"{prefix}-asset_return-{get_seconds_slug(window)}@{exchange.name}"] = read_returns(
                    bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=window
                )
                features[
                    f"{prefix}-slippage_imbalance-{get_seconds_slug(window)}@{exchange.name}"] = read_slippage_imbalance(
                    bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=window
                )
                features[f"{prefix}-flow_imbalance_{get_seconds_slug(window)}@{exchange.name}"] = read_flow_imbalance(
                    bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=window
                )
                features[f"{prefix}-powerlaw_alpha-{get_seconds_slug(window)}@{exchange.name}"] = read_powerlaw_alpha(
                    bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=window
                )
                features[
                    f"{prefix}-share_of_long_trades-{get_seconds_slug(window)}@{exchange.name}"] = read_share_long_trades(
                    bounds=bounds, exchange=exchange, currency_pair=currency_pair, window=window
                )

        return features

    def read_output(self, bounds: Bounds, currency: Currency) -> np.ndarray:
        """Reads output return in pips with forecast step"""
        logging.info("Reading output for %s", currency.name)
        currency_pair: CurrencyPair = CurrencyPair(base=currency, term=Currency.USDT)
        returns: np.ndarray = read_returns(
            bounds=bounds, exchange=self.exchange, currency_pair=currency_pair, window=self.forecast_step
        )
        shifted_returns: np.ndarray = shift(returns, n=-int(self.forecast_step / SamplingType.MS500.value))
        return shifted_returns

    def read_categorical_features(self, currency: Currency, col_size: int) -> Dict[str, np.ndarray]:
        """Read categorical features for the currency"""
        logging.info("Reading categorical features for %s", currency.name)
        # Add categorical features
        categorical_features: Dict[str, np.ndarray] = {
            COL_CURRENCY_INDEX: np.full(col_size, fill_value=currency.value, dtype=np.int32),
        }
        return categorical_features

    def split_and_wrap_into_samples(
            self, df: pd.DataFrame, feature_set: FeatureSet, sample_params: SampleParams, currency: Currency
    ) -> Optional[Sample]:
        """
        Split data into TRAIN/VALIDATION/TEST using SampleParams and wrap DataFrames into Sample object.
        Do all preprocessing here as well like standardizations and clipping for each DataFrame then wrap into
        Sample
        """
        datasets: Dict[DatasetType, pd.DataFrame] = sample_params.split_by_indecies(df=df)
        # if all pd.DataFrame are empty return None
        if all(dataset.empty for dataset in datasets.values()):
            return None

        # Do preprocessing here if needed
        sample: Sample = Sample.from_pandas_datasets(datasets=datasets, name=currency.name, feature_set=feature_set)
        # Output logs with statistics
        sample.describe()

        logging.info("Created %s", list(sample.dataset_types))
        return sample

    def create_dataframe_for_currency(
            self, bounds: Bounds, currency: Currency, features_common: Dict[str, np.ndarray], index: np.ndarray
    ) -> Tuple[pd.DataFrame, FeatureSet]:
        """Read features from local filesystem and output as pd.DataFrame"""
        logging.info(
            "\n------------------------------------\nReading data for %s\n------------------------------------",
            currency.name,
        )
        features_currency_specific: Dict[str, np.ndarray] = self.read_currency_specific_features(
            bounds=bounds, currency=currency
        )
        features_categorical: Dict[str, np.ndarray] = self.read_categorical_features(
            currency=currency, col_size=len(index)
        )
        output: np.ndarray = self.read_output(bounds=bounds, currency=currency)
        # Merge all dictionaries together and create DataFrame from it
        regressors: Dict[str, np.ndarray] = features_currency_specific | features_categorical
        features: Dict[str, np.ndarray] = regressors | {COL_OUTPUT: output}
        # Create FeatureSet
        feature_set: FeatureSet = FeatureSet(
            regressors=list(regressors.keys()),
            categorical=list(features_categorical.keys()),
            target=COL_OUTPUT
        )
        df: pd.DataFrame = pd.DataFrame(features)
        df = df.dropna(subset=[COL_OUTPUT], axis=0)  # remove observations with missing targets
        return df, feature_set

    def create_sample(self, bounds: Bounds, sample_params: SampleParams) -> Sample:
        """
        Collect data for each currency and wrap each Currency data into its own Sample and then merge all of them into
        a single concatenated Sample
        """
        # Read BTC/ETH features as sort of like to capture overall market movements
        features_common: Dict[str, np.ndarray] = self.read_common_features(bounds=bounds)
        date_index: np.ndarray = read_index(bounds=bounds)
        samples: List[Sample] = []

        for currency in self.target_currencies:

            feature_set: FeatureSet
            df_currency: pd.DataFrame

            df_currency, feature_set = self.create_dataframe_for_currency(
                bounds=bounds, currency=currency, features_common=features_common, index=date_index
            )
            # Wrap collected pd.DataFrame into sample
            sample_currency: Optional[Sample] = self.split_and_wrap_into_samples(
                df=df_currency, feature_set=feature_set, sample_params=sample_params, currency=currency
            )
            if sample_currency is None:
                logging.info("Skipping currency %s because sample is empty", currency.name)
                continue

            samples.append(sample_currency)

        return concat_samples(samples=samples, require_lgb=True)

    def create_dataset(self, bounds: Bounds, ds_type: DatasetType) -> MLDataset:
        """Collect data for each currency and combine all of it into a single MLDataset of DatasetType"""
        separator: str = "-" * 33
        logging.info(
            "\n%s\nCreating dataset %s for %s\n%s", separator, ds_type.name, bounds, separator
        )
        # Read BTC/ETH features as sort of like to capture overall market movements
        features_common: Dict[str, np.ndarray] = self.read_common_features(bounds=bounds)
        date_index: np.ndarray = read_index(bounds=bounds)

        merged_dataset = None

        for i, currency in enumerate(self.target_currencies):
            feature_set: FeatureSet
            df_currency: pd.DataFrame
            df, feature_set = self.create_dataframe_for_currency(
                bounds=bounds, currency=currency, features_common=features_common, index=date_index
            )
            dataset: MLDataset = MLDataset(
                ds_type=ds_type,
                data=df[feature_set.regressors],
                label=df[feature_set.target],
                feature_set=feature_set,
                name=currency.name,
                eval_data=df[feature_set.eval_fields] if feature_set.eval_fields else None
            )
            # Output statistics for collected MLDataset
            dataset.describe()

            if dataset.is_empty():
                logging.info("Skipping currency %s because dataset is empty", currency.name)
                continue

            if merged_dataset is None:
                merged_dataset = dataset
            else:
                merged_dataset.add_dataset(dataset=dataset)

        return merged_dataset


def run_test():
    configure_logging()
    bounds: Bounds = Bounds.for_days(
        date(2024, 1, 1), date(2024, 1, 5)
    )
    build = BuildDataset(
        exchange=Exchange.BINANCE_SPOT,
        target_currencies=get_target_currencies(),
        forecast_step=timedelta(seconds=5),
    )
    build.create_dataset(bounds=bounds, ds_type=DatasetType.TRAIN)


if __name__ == "__main__":
    run_test()
