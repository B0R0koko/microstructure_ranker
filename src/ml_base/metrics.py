from typing import List, Dict, Any

import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score

from core.currency import Currency
from ml_base.sample import MLDataset
from models.prediction.columns import COL_CURRENCY_INDEX


def compute_metrics(booster: Booster, dataset: MLDataset, target_currencies: List[Currency]) -> pd.DataFrame:
    statistics: List[Dict[str, Any]] = []

    for currency in target_currencies:
        mask = dataset.data[COL_CURRENCY_INDEX] == currency.value
        y_pred: np.ndarray = booster.predict(dataset.data[mask], num_iteration=booster.best_iteration)
        y_true: np.ndarray = dataset.label[mask]

        y_pred_binary: np.ndarray = (y_pred > 0).astype(int)  # evaluate model as binary classification
        label_binary: np.ndarray = (y_true > 0).astype(int)

        statistics.append({
            "currency": currency.name,
            "R2": r2_score(y_true=dataset.label[mask], y_pred=y_pred),
            "MAE": mean_absolute_error(y_true=dataset.label[mask], y_pred=y_pred),
            "F1": f1_score(y_pred=y_pred_binary, y_true=label_binary),
            "Accuracy": accuracy_score(y_pred=y_pred_binary, y_true=label_binary)
        })

    return pd.DataFrame(statistics)
