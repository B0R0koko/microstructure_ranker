import logging
from datetime import timedelta, date
from typing import List

from core.currency import Currency, get_target_currencies
from core.exchange import Exchange
from core.time_utils import Bounds, get_seconds_slug, format_date
from core.utils import configure_logging
from ml_base.features import FeatureFilter, get_importance_file_path
from ml_base.sample import SampleParams
from models.prediction.pipes.price_prediction import PrimaryPricePrediction

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
        target_currencies: List[Currency],
        forecast_step: timedelta,
        ref_day: date
) -> None:
    """Writes feature importance file to local filesystem"""
    pipe = PrimaryPricePrediction(
        target_exchange=target_exchange,
        target_currencies=target_currencies,
        feature_filter=FeatureFilter.all(),  # all features are allowed
        forecast_steps=forecast_step,
    )
    pipe.feature_selection_pipeline(
        bounds=bounds,
        load_interval=timedelta(days=3),  # use 3 days worth of data to fit the model, it"s sort of like batch size
        day=ref_day
    )


def run_build_best_model(
        bounds: Bounds,
        target_exchange: Exchange,
        target_currencies: List[Currency],
        forecast_step: timedelta,
        ref_day: date
) -> None:
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
    pipe.build_model_pipeline(
        bounds=bounds,
        # use 90% for training and 10% for early stopping
        sample_params=SampleParams(train_share=.9, validation_share=.1),
        day=ref_day
    )


def train_multiple_models_with_different_horizons(
        bounds: Bounds, target_exchange: Exchange, target_currencies: List[Currency], ref_day: date
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
            bounds=bounds,
            target_exchange=target_exchange,
            target_currencies=target_currencies,
            forecast_step=forecast_step,
            ref_day=ref_day
        )
        # Then run build_best_model pipeline
        run_build_best_model(
            bounds=bounds,
            target_exchange=target_exchange,
            target_currencies=target_currencies,
            forecast_step=forecast_step,
            ref_day=ref_day
        )


def main():
    configure_logging()
    bounds: Bounds = Bounds.for_days(
        date(2025, 4, 1), date(2025, 5, 10),
    )
    target_exchange: Exchange = Exchange.BINANCE_SPOT
    ref_day: date = date(2025, 5, 25)

    train_multiple_models_with_different_horizons(
        bounds=bounds,
        target_exchange=target_exchange,
        target_currencies=get_target_currencies(),
        ref_day=ref_day
    )


if __name__ == "__main__":
    main()
