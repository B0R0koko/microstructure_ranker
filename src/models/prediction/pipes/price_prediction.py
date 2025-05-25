import gc
import logging
import os
from datetime import timedelta, date
from typing import List, Dict, Any

import lightgbm as lgb
import numpy as np
from lightgbm import Booster
from scrapy.utils.log import configure_logging
from sklearn.metrics import r2_score
from tqdm import tqdm

from core.currency import Currency
from core.exchange import Exchange
from core.time_utils import Bounds
from ml_base.enums import DatasetType
from ml_base.features import FeatureFilter, save_feature_importances_to_file
from ml_base.sample import MLDataset
from models.prediction.build_sample import BuildDataset
from models.prediction.columns import COL_CURRENCY_INDEX

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "mse",
    "max_depth": 5,
    "n_estimators": 150,
    "num_threads": os.cpu_count(),
    "verbose": -1
}


class PrimaryPricePrediction:

    def __init__(
            self,
            target_exchange: Exchange,
            target_currencies: List[Currency],
            feature_filter: FeatureFilter,
            forecast_steps: timedelta,
            train_bounds: Bounds,
            test_bounds: Bounds,
    ):
        self.target_exchange: Exchange = target_exchange
        self.target_currencies: List[Currency] = target_currencies
        self.feature_filter: FeatureFilter = feature_filter
        self.forecast_steps: timedelta = forecast_steps
        self.train_bounds: Bounds = train_bounds
        self.test_bounds: Bounds = test_bounds

    def get_dataset_builder(self) -> BuildDataset:
        builder: BuildDataset = BuildDataset(
            target_exchange=self.target_exchange,
            target_currencies=self.target_currencies,
            feature_filter=self.feature_filter,
            forecast_step=self.forecast_steps,
        )
        return builder

    def train_model(self, dataset: MLDataset) -> Booster:
        """Define the logic how the model is trained, this might be different in different pipelines"""
        # Construct lgb.Datasets for training and validation, we might gain in reduction of training time
        train: lgb.Dataset = dataset.to_lgb_dataset()
        # Use XGBoost api as it is more versatile
        booster: Booster = lgb.train(params=_BASE_PARAMS, train_set=train, valid_names=["train"])
        return booster

    def fit_model_partially(self, chunk_interval: timedelta) -> Booster:
        """
        Fit model partially with different leaf weights such that in the end we have model with leaves outputs
        equally weighted across different time intervals
        """
        builder: BuildDataset = self.get_dataset_builder()
        # Use self.train_bounds for training the model
        sub_bounds: List[Bounds] = self.train_bounds.generate_overlapping_bounds(
            step=chunk_interval, interval=chunk_interval
        )
        dataset: MLDataset = builder.create_dataset(bounds=sub_bounds[0], ds_type=DatasetType.TRAIN)
        booster: Booster = self.train_model(dataset=dataset)

        del dataset
        gc.collect()

        for i, sub_bound in tqdm(enumerate(sub_bounds[1:], start=2), desc="Partial fit the model"):
            dataset = builder.create_dataset(bounds=sub_bound, ds_type=DatasetType.TRAIN)
            booster = booster.refit(data=dataset.data, label=dataset.label, decay_rate=1 - 1 / i)

            # Collect dataset once we used it for training
            del dataset
            gc.collect()

        return booster

    def evaluate_model(self, booster: Booster, bounds: Bounds, dataset: MLDataset) -> None:
        """Log different statistics for each currency"""
        y_pred: np.ndarray = booster.predict(dataset.data)  # type:ignore
        logging.info("\n\nPrice prediction model performance for %s\n\n", str(bounds))

        for currency in self.target_currencies:
            mask = dataset.data[COL_CURRENCY_INDEX] == currency.value

            if sum(mask) == 0:
                logging.info("Skipping currency %s, there is no data", currency.name)
                continue

            logging.info(
                "R2 for %s: %s", currency.name, r2_score(y_pred=y_pred[mask], y_true=dataset.label[mask])
            )

    def build_model_pipeline(self) -> Booster:
        logging.info("Starting build_model_pipeline")
        builder: BuildDataset = self.get_dataset_builder()

        booster: Booster = self.fit_model_partially(chunk_interval=timedelta(days=3))

        sub_test_bounds: List[Bounds] = self.test_bounds.generate_overlapping_bounds(
            step=timedelta(days=3), interval=timedelta(days=3)
        )

        for i, sub_bound in enumerate(sub_test_bounds):
            dataset: MLDataset = builder.create_dataset(bounds=sub_bound, ds_type=DatasetType.TEST)
            self.evaluate_model(booster=booster, bounds=sub_bound, dataset=dataset)

            del dataset
            gc.collect()

        return booster

    def feature_selection_pipeline(self) -> None:
        """Trains the model and writes feature_importances to local filesystem"""
        booster: Booster = self.fit_model_partially(chunk_interval=timedelta(days=3))
        save_feature_importances_to_file(
            booster=booster, day=self.train_bounds.day1, target_exchange=self.target_exchange
        )


def main():
    configure_logging()
    train_bounds: Bounds = Bounds.for_days(date(2025, 5, 1), date(2025, 5, 20))
    test_bounds: Bounds = Bounds.for_days(date(2025, 5, 20), date(2025, 5, 24))

    feature_filter: FeatureFilter = FeatureFilter()

    pipe = PrimaryPricePrediction(
        target_exchange=Exchange.BINANCE_SPOT,
        target_currencies=[Currency.BTC, Currency.ETH, Currency.TRX, Currency.ADA, Currency.SOL],
        feature_filter=feature_filter,
        forecast_steps=timedelta(seconds=5),
        train_bounds=train_bounds,
        test_bounds=test_bounds,
    )

    pipe.feature_selection_pipeline()


if __name__ == "__main__":
    main()
