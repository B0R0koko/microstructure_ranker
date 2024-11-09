from abc import ABC, abstractmethod

import pandas as pd


class FeaturePipeline(ABC):

    @abstractmethod
    def load_cross_section(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """Loads cross-section from Data folder"""
