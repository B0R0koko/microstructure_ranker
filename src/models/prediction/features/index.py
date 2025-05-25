from pathlib import Path

import numpy as np

from core.time_utils import Bounds
from models.prediction.features.utils import multi_day_ts, read_scalar


def read_index(bounds: Bounds) -> np.ndarray:
    return multi_day_ts(
        bounds=bounds,
        get_day_ts=lambda day: read_scalar(
            day=day,
            subpath=Path("time")
        )
    )
