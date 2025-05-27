import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from lightgbm import Booster
from lightgbm.callback import CallbackEnv
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score

from core.currency import Currency
from ml_base.sample import MLDataset
from models.prediction.columns import COL_CURRENCY_INDEX


def log_lgbm_iteration_to_stdout(env: CallbackEnv) -> None:
    """LightGBM callback to log metrics from eval_sets with eval_names to mlflow"""
    for _, metric_name, value, _ in env.evaluation_result_list:  # type:ignore
        logging.info(
            "Training model at iteration #%s %s = %s",
            env.iteration,
            metric_name,
            value
        )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, currency: str) -> Dict[str, float]:
    y_pred_binary: np.ndarray = (y_pred > 0).astype(int)
    y_true_binary: np.ndarray = (y_true > 0).astype(int)

    return {
        "currency": currency,
        "R2": r2_score(y_true=y_true, y_pred=y_pred),
        "MAE": mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "F1": f1_score(y_pred=y_pred_binary, y_true=y_true_binary),
        "Accuracy": accuracy_score(y_pred=y_pred_binary, y_true=y_true_binary)
    }


def compute_metrics(booster: Booster, dataset: MLDataset, target_currencies: List[Currency]) -> pd.DataFrame:
    """Computes a table with regression and classification metrics"""
    statistics: List[Dict[str, Any]] = []

    y_pred: np.ndarray = booster.predict(dataset.data, num_iteration=booster.best_iteration)
    y_true: np.ndarray = dataset.label.to_numpy()

    # Compute statistics for all currencies
    statistics.append(
        _metrics(y_true=y_true, y_pred=y_pred, currency="ALL")
    )

    for currency in target_currencies:
        logging.info("Computing metrics for %s", currency.name)
        mask = dataset.data[COL_CURRENCY_INDEX] == currency.value

        if mask.sum() == 0:
            logging.info("Skipping metrics for %s", currency.name)
            continue

        statistics.append(
            _metrics(y_true=y_true[mask], y_pred=y_pred[mask], currency=currency.name)
        )

    return pd.DataFrame(statistics)
