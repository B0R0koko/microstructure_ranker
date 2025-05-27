import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from lightgbm import Booster

from core.exchange import Exchange
from core.paths import get_root_dir
from core.time_utils import get_seconds_slug, format_date


def get_booster_path(day: date, target_exchange: Exchange, forecast_step: timedelta) -> Path:
    """Returns path for a trained model"""
    return (
            get_root_dir() /
            "src/models/prediction/artifacts/boosters" /
            target_exchange.name /
            f"{target_exchange.name}-{get_seconds_slug(td=forecast_step)}@{format_date(day=day)}.txt"
    )


def save_model(booster: Booster, out_file: Path, num_iteration: Optional[int] = None) -> None:
    """Save model to .txt file"""
    logging.info("Saving model to %s", out_file)
    os.makedirs(out_file.parent, exist_ok=True)
    booster.save_model(out_file, num_iteration=num_iteration)
    logging.info("Saving model of size %s KB to %s", round(os.path.getsize(out_file) / 1024, 3), out_file)
