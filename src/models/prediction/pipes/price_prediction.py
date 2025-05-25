import gc
import logging
import os
from datetime import timedelta, date
from typing import List, Dict, Any

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster
from scrapy.utils.log import configure_logging
from tqdm import tqdm

from core.currency import Currency, get_target_currencies
from core.exchange import Exchange
from core.time_utils import Bounds
from ml_base.enums import DatasetType
from ml_base.features import FeatureFilter, save_feature_importances_to_file, get_importance_file_path
from ml_base.metrics import compute_metrics
from ml_base.sample import MLDataset
from models.prediction.build_sample import BuildDataset

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

    def build_model_pipeline(self) -> Booster:
        """Once we ran feature_selection pipeline and narrowed down on the set of features, run this pipeline"""
        logging.info("Running <build_model_pipeline>")
        builder: BuildDataset = self.get_dataset_builder()
        train_dataset: MLDataset = builder.create_dataset(bounds=self.train_bounds, ds_type=DatasetType.TRAIN)
        booster: Booster = self.train_model(dataset=train_dataset)

        del train_dataset
        gc.collect()
        # After training the model evaluate the model and log statistics to stdout
        test_dataset: MLDataset = builder.create_dataset(bounds=self.test_bounds, ds_type=DatasetType.TEST)
        df_metrics: pd.DataFrame = compute_metrics(
            booster=booster, dataset=test_dataset, target_currencies=self.target_currencies
        )
        logging.info("TEST metrics\n%s", df_metrics)
        return booster

    def feature_selection_pipeline(self, chunk_interval: timedelta) -> None:
        """Trains the model and writes feature_importances to local filesystem"""
        booster: Booster = self.fit_model_partially(chunk_interval=chunk_interval)
        save_feature_importances_to_file(
            booster=booster, day=self.train_bounds.day1, target_exchange=self.target_exchange
        )


def main():
    configure_logging()
    train_bounds: Bounds = Bounds.for_days(date(2025, 4, 1), date(2025, 5, 5))
    test_bounds: Bounds = Bounds.for_days(date(2025, 5, 5), date(2025, 5, 24))

    feature_filter: FeatureFilter = FeatureFilter.from_importance(
        get_importance_file_path(day=date(2025, 5, 19), target_exchange=Exchange.BINANCE_SPOT)
    )

    pipe = PrimaryPricePrediction(
        target_exchange=Exchange.BINANCE_SPOT,
        target_currencies=get_target_currencies(),
        feature_filter=feature_filter,
        forecast_steps=timedelta(seconds=5),
        train_bounds=train_bounds,
        test_bounds=test_bounds,
    )

    pipe.build_model_pipeline()


if __name__ == "__main__":
    main()
