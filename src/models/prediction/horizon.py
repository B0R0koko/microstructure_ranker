import logging
import os
from datetime import timedelta, date
from pathlib import Path
from typing import List

import pandas as pd
from lightgbm import Booster

from core.currency import Currency, get_target_currencies
from core.exchange import Exchange
from core.paths import get_root_dir
from core.time_utils import Bounds, get_seconds_slug, format_date
from core.utils import configure_logging
from ml_base.enums import DatasetType
from ml_base.features import FeatureFilter, get_importance_file_path
from ml_base.metrics import compute_metrics
from ml_base.sample import SampleParams, MLDataset
from models.prediction.build_sample import BuildDataset
from models.prediction.pipes.price_prediction import PrimaryPricePrediction

"""Analysis dedicated to how predictability of returns decays as prediction horizon increases"""

HORIZONS: List[timedelta] = [
    timedelta(seconds=1),
    timedelta(seconds=2),
    timedelta(seconds=3),
    timedelta(seconds=4),
    timedelta(seconds=5),
    timedelta(seconds=10),
    timedelta(seconds=15),
    timedelta(seconds=20),
    timedelta(seconds=30),
    timedelta(minutes=1),
    timedelta(minutes=5),
]


def run_feature_selection(
        bounds: Bounds,
        target_exchange: Exchange,
        currency_subset: List[Currency],
        forecast_step: timedelta,
        ref_day: date
) -> None:
    """Writes feature importance file to local filesystem"""
    pipe = PrimaryPricePrediction(
        target_exchange=target_exchange,
        target_currencies=currency_subset,
        feature_filter=FeatureFilter.all(),  # all features are allowed
        forecast_steps=forecast_step,
    )
    pipe.feature_selection_pipeline(
        bounds=bounds, sample_params=SampleParams(train_share=.9, validation_share=.1), day=ref_day
    )


def run_build_best_model(
        bounds: Bounds,
        target_exchange: Exchange,
        target_currencies: List[Currency],
        forecast_step: timedelta,
        ref_day: date
) -> Booster:
    pipe = PrimaryPricePrediction(
        target_exchange=target_exchange,
        target_currencies=target_currencies,
        # Now load importance file and select top 25 features from it
        feature_filter=FeatureFilter.from_importance(
            file=get_importance_file_path(
                day=ref_day, target_exchange=target_exchange, forecast_step=forecast_step
            ),
            use_first=25
        ),
        forecast_steps=forecast_step,
    )
    # The model will be saved to boosters folder
    booster: Booster = pipe.build_model_pipeline(
        bounds=bounds,
        # use 90% for training and 10% for early stopping
        sample_params=SampleParams(train_share=.9, validation_share=.1),
        day=ref_day
    )

    return booster


def get_statitics_path(target_exchange: Exchange, forecast_step: timedelta, day: date) -> Path:
    return (
            get_root_dir() /
            "src/models/prediction/artifacts/statistics" /
            target_exchange.name /
            f"{target_exchange.name}-statistics-{get_seconds_slug(td=forecast_step)}@{format_date(day)}.csv"
    )


def save_stats(
        df_stats: pd.DataFrame, target_exchange: Exchange, forecast_step: timedelta, day: date
) -> None:
    path: Path = get_statitics_path(target_exchange=target_exchange, forecast_step=forecast_step, day=day)
    logging.info(
        "Saving statistics for %s@%s to %s",
        target_exchange.name,
        format_date(day=day),
        path
    )
    os.makedirs(path.parent, exist_ok=True)
    df_stats.to_csv(path, index=False)


def run_evaluate_model(
        booster: Booster,
        test_bounds: Bounds,
        target_exchange: Exchange,
        target_currencies: List[Currency],
        forecast_step: timedelta,
        ref_day: date
) -> None:
    """Create TEST MLDataset and evaluate trained model and output dataframes to files"""

    dataset: MLDataset = (
        BuildDataset(
            target_exchange=target_exchange,
            target_currencies=target_currencies,
            feature_filter=FeatureFilter.from_importance(
                file=get_importance_file_path(
                    day=ref_day, target_exchange=target_exchange, forecast_step=forecast_step
                ),
                use_first=25
            ),
            forecast_step=forecast_step,
        )
        .create_dataset(bounds=test_bounds, ds_type=DatasetType.TEST)
    )

    df_stats: pd.DataFrame = compute_metrics(
        booster=booster, dataset=dataset, target_currencies=target_currencies
    )
    logging.info(
        "\n\nEvaluation results for %s, forecast step %s\n%s",
        target_exchange.name,
        get_seconds_slug(td=forecast_step),
        df_stats
    )

    save_stats(
        df_stats=df_stats,
        target_exchange=target_exchange,
        forecast_step=forecast_step,
        day=ref_day
    )


def train_multiple_models_with_different_horizons(
        build_bounds: Bounds,
        selection_bounds: Bounds,
        test_bounds: Bounds,
        target_exchange: Exchange,
        target_currencies: List[Currency],
        ref_day: date
):
    """Run feature selection and build best model for each prediction horizon from HORIZONS for a given Exchange"""
    for forecast_step in HORIZONS:
        logging.info(
            "\n\nRunning for %s@%s. Forecast step - %s",
            target_exchange.name,
            format_date(day=ref_day),
            get_seconds_slug(td=forecast_step)
        )
        # Run feature selection and save feature importances to feature importances folder
        run_feature_selection(
            bounds=selection_bounds,
            target_exchange=target_exchange,
            # We only can use 5 features, otherwise it won't fit into the RAM
            currency_subset=[Currency.BTC, Currency.ETH, Currency.ADA, Currency.SOL, Currency.BNB],
            forecast_step=forecast_step,
            ref_day=ref_day
        )
        # Then run build_best_model pipeline
        booster: Booster = run_build_best_model(
            bounds=build_bounds,
            target_exchange=target_exchange,
            target_currencies=target_currencies,
            forecast_step=forecast_step,
            ref_day=ref_day
        )
        # Evaluate models
        run_evaluate_model(
            booster=booster,
            test_bounds=test_bounds,
            target_exchange=target_exchange,
            target_currencies=target_currencies,
            forecast_step=forecast_step,
            ref_day=ref_day
        )


def main():
    configure_logging()
    # Bounds for feature selection task
    selection_bounds: Bounds = Bounds.for_days(
        date(2025, 5, 1), date(2025, 5, 10),
    )
    # Bounds for training the model
    build_bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 10),
    )
    # Bounds for evaluation
    test_bounds: Bounds = Bounds.for_days(
        date(2025, 5, 10), date(2025, 5, 25)
    )

    target_exchange: Exchange = Exchange.BINANCE_USDM
    ref_day: date = date(2025, 5, 25)

    train_multiple_models_with_different_horizons(
        build_bounds=build_bounds,
        selection_bounds=selection_bounds,
        test_bounds=test_bounds,
        target_exchange=target_exchange,
        target_currencies=get_target_currencies(),
        ref_day=ref_day
    )


if __name__ == "__main__":
    main()
