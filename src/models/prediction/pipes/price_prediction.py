import logging
import os
from datetime import timedelta, date
from typing import List, Dict, Any

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster, early_stopping
from scrapy.utils.log import configure_logging
from sklearn.metrics import r2_score

from core.currency import Currency
from core.time_utils import Bounds
from ml_base.enums import DatasetType
from ml_base.sample import Sample, SampleParams
from models.prediction.build_dataset import BuildDataset, get_target_currencies
from models.prediction.columns import COL_CURRENCY_INDEX

_BASE_PARAMS: Dict[str, Any] = {
    "objective": "mse",
    "max_depth": 5,
    "n_estimators": 100,
    "num_threads": os.cpu_count()
}


class PrimaryPricePrediction:

    def __init__(
            self, bounds: Bounds,
            sample_params: SampleParams,
            target_currencies: List[Currency],
            forecast_steps: timedelta,
    ):
        self.bounds: Bounds = bounds
        self.sample_params: SampleParams = sample_params
        self.target_currencies: List[Currency] = target_currencies
        self.forecast_steps: timedelta = forecast_steps

    def create_sample(self) -> Sample:
        builder: BuildDataset = BuildDataset(
            bounds=self.bounds,
            target_currencies=self.target_currencies,
            sample_params=self.sample_params,
            forecast_step=self.forecast_steps,
        )
        return builder.create_dataset()

    def train_model(self, sample: Sample) -> Booster:
        """Define the logic how the model is trained, this might be different in different pipelines"""
        # Construct lgb.Datasets for training and validation, we might gain in reduction of training time
        train: lgb.Dataset = sample.get_lgb_dataset(ds_type=DatasetType.TRAIN)
        validation: lgb.Dataset = sample.get_lgb_dataset(ds_type=DatasetType.VALIDATION)

        # Use XGBoost api as it is more versatile
        booster: Booster = lgb.train(
            params=_BASE_PARAMS,
            train_set=train,
            valid_sets=[train, validation],
            valid_names=["train", "validation"],
            callbacks=[
                early_stopping(stopping_rounds=50, min_delta=.01, verbose=False),
            ]
        )
        return booster

    def build_best_model(self) -> Booster:
        sample: Sample = self.create_sample()
        booster: Booster = self.train_model(sample=sample)

        data: pd.DataFrame = sample.get_data(ds_type=DatasetType.TEST)
        label: pd.Series = sample.get_label(ds_type=DatasetType.TEST)

        for currency in self.target_currencies:
            mask = data[COL_CURRENCY_INDEX] == currency.value
            y_pred: np.ndarray = booster.predict(data[mask], num_iteration=booster.best_iteration)  # type:ignore
            logging.info("Currency %s R2 = %s", currency.name, r2_score(y_pred=y_pred, y_true=label[mask]))

        return booster


def main():
    configure_logging()
    pipe = PrimaryPricePrediction(
        bounds=Bounds.for_days(date(2024, 1, 5), date(2024, 1, 10)),
        sample_params=SampleParams(
            train_share=.7, validation_share=.15, test_share=.15,
        ),
        target_currencies=get_target_currencies(),
        forecast_steps=timedelta(seconds=1)
    )
    pipe.build_best_model()


if __name__ == "__main__":
    main()
