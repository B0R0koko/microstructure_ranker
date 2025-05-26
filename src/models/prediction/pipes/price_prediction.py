import gc
import logging
import os
from datetime import timedelta, date
from typing import List, Dict, Any

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster, early_stopping
from scrapy.utils.log import configure_logging
from tqdm import tqdm

from core.currency import Currency, get_target_currencies
from core.exchange import Exchange
from core.time_utils import Bounds
from ml_base.enums import DatasetType
from ml_base.features import FeatureFilter, save_feature_importances_to_file, get_importance_file_path
from ml_base.metrics import compute_metrics, log_lgbm_iteration_to_stdout
from ml_base.sample import MLDataset, SampleParams, Sample
from models.prediction.build_sample import BuildDataset

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "mse",
    "max_depth": 5,
    "learning_rate": 0.02,
    "n_estimators": 120,
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
    ):
        self.target_exchange: Exchange = target_exchange
        self.target_currencies: List[Currency] = target_currencies
        self.feature_filter: FeatureFilter = feature_filter
        self.forecast_steps: timedelta = forecast_steps

    def get_dataset_builder(self) -> BuildDataset:
        builder: BuildDataset = BuildDataset(
            target_exchange=self.target_exchange,
            target_currencies=self.target_currencies,
            feature_filter=self.feature_filter,
            forecast_step=self.forecast_steps,
        )
        return builder

    def fit_model_partially(self, bounds: Bounds, load_interval: timedelta) -> Booster:
        """
        Fit model partially with different leaf weights such that in the end we have model with leaves outputs
        equally weighted across different time intervals
        """
        builder: BuildDataset = self.get_dataset_builder()
        # Use self.train_bounds for training the model
        # At first iteration use early stopping, then update using MLDatasets without SampleParams
        sub_bounds: List[Bounds] = bounds.generate_overlapping_bounds(step=load_interval, interval=load_interval)
        sample: Sample = builder.create_sample(
            bounds=sub_bounds[0],
            sample_params=SampleParams(train_share=.8, validation_share=.2),
        )

        booster: Booster = self.train_model_sample(sample=sample)

        del sample
        gc.collect()

        for i, sub_bound in tqdm(enumerate(sub_bounds[1:], start=2), desc="Partial fit the model"):
            dataset: MLDataset = builder.create_dataset(bounds=sub_bound, ds_type=DatasetType.TRAIN)
            booster = booster.refit(data=dataset.data, label=dataset.label, decay_rate=1 - 1 / i)

            # Collect dataset once we used it for training
            del dataset
            gc.collect()

        return booster

    def train_model_sample(self, sample: Sample) -> Booster:
        logging.info("Training the model with early stopping")
        train: lgb.Dataset = sample.get_lgb_dataset(ds_type=DatasetType.TRAIN)
        validation: lgb.Dataset = sample.get_lgb_dataset(ds_type=DatasetType.VALIDATION)

        booster: Booster = lgb.train(
            params=_BASE_PARAMS,
            train_set=train,
            valid_sets=[train, validation],
            valid_names=["train", "validation"],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_lgbm_iteration_to_stdout
            ]
        )

        return booster

    def build_model_pipeline(self, bounds: Bounds, sample_params: SampleParams) -> Booster:
        """
        Once we ran feature_selection pipeline and narrowed down on the set of features, run this pipeline.
        """
        logging.info("Running <build_model_pipeline>")

        builder: BuildDataset = self.get_dataset_builder()
        sample: Sample = builder.create_sample(bounds=bounds, sample_params=sample_params)
        booster: Booster = self.train_model_sample(sample=sample)

        df_metrics: pd.DataFrame = compute_metrics(
            booster=booster,
            dataset=sample.get_dataset(ds_type=DatasetType.TEST),
            target_currencies=self.target_currencies
        )

        logging.info("TEST metrics\n%s", df_metrics)
        return booster

    def feature_selection_pipeline(
            self, bounds: Bounds, load_interval: timedelta, day: date
    ) -> None:
        """Trains the model and writes feature_importances to local filesystem"""
        booster: Booster = self.fit_model_partially(bounds=bounds, load_interval=load_interval)
        save_feature_importances_to_file(
            booster=booster,
            day=day,
            target_exchange=self.target_exchange
        )


def main():
    configure_logging()

    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 25)
    )
    feature_filter: FeatureFilter = FeatureFilter.from_importance(
        file=get_importance_file_path(day=date(2025, 5, 25), target_exchange=Exchange.BINANCE_SPOT)
    )

    pipe = PrimaryPricePrediction(
        target_exchange=Exchange.BINANCE_SPOT,
        target_currencies=get_target_currencies(),
        feature_filter=feature_filter,
        forecast_steps=timedelta(seconds=15),
    )

    # pipe.feature_selection_pipeline(
    #     bounds=bounds,
    #     load_interval=timedelta(days=3),
    #     day=date(2025, 5, 25)
    # )

    pipe.build_model_pipeline(
        bounds=bounds,
        sample_params=SampleParams(train_share=0.6, validation_share=0.15, test_share=0.25),
    )


if __name__ == "__main__":
    main()
