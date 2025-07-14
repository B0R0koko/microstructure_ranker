import gc
import logging
from typing import Optional, Dict, List

import lightgbm as lgb
import pandas as pd

from core.feature_set import FeatureSet
from ml_base.enums import DatasetType


class MLDataset:

    def __init__(
            self,
            ds_type: DatasetType,
            data: pd.DataFrame,
            label: pd.Series,
            feature_set: FeatureSet,
            name: str,
            eval_data: Optional[pd.DataFrame] = None,
    ):
        self.ds_type: DatasetType = ds_type
        self.data: pd.DataFrame = data
        self.label: pd.Series = label
        self.feature_set: FeatureSet = feature_set
        self.name: str = name
        # eval_data is optional, used for evaluation only
        self.eval_data: Optional[pd.DataFrame] = eval_data

    def to_lgb_dataset(self, reference: Optional[lgb.Dataset] = None) -> lgb.Dataset:
        """Returns lightgbm Dataset for training purposes, maybe it will result in training speed increase"""
        return lgb.Dataset(
            data=self.data,
            label=self.label,
            categorical_feature=self.feature_set.categorical,
            reference=reference,
            free_raw_data=True
        ).construct()

    def is_empty(self) -> bool:
        return self.data.empty

    def describe(self) -> None:
        logging.info("Dataset %s shape %s\n\n", self.name, self.data.shape)

    def add_dataset(self, dataset: "MLDataset") -> None:
        """concatenates all data inplace and gc.collect() dataset passed to the function"""
        assert self.feature_set == dataset.feature_set, "FeatureSet must be the same for merged MLDatasets"
        self.data = pd.concat([self.data, dataset.data])
        self.label = pd.concat([self.label, dataset.label])

        if self.eval_data is not None and dataset.eval_data is not None:
            self.eval_data = pd.concat([self.eval_data, dataset.eval_data])

        del dataset
        gc.collect()


class Sample:

    def __init__(self, datasets: Dict[DatasetType, MLDataset], name: str):
        self.datasets: Dict[DatasetType, MLDataset] = datasets
        self.name: str = name

        self._lgb_datasets: Dict[DatasetType, lgb.Dataset] = {}

    @classmethod
    def empty(cls) -> "Sample":
        return cls(datasets={}, name="EMPTY")

    def add_sample(self, other: "Sample") -> None:
        """Add sample inplace"""
        logging.info("Merging samples")
        for ds_type, dataset in other.datasets.items():
            if ds_type not in self.datasets:
                self.datasets[ds_type] = dataset
            else:
                self.datasets[ds_type].add_dataset(dataset=dataset)

        del other
        gc.collect()

    def construct(self) -> None:
        """Call this method once the Sample is finalized to create lightgbm Datasets"""
        train: lgb.Dataset = self.datasets[DatasetType.TRAIN].to_lgb_dataset()
        self._lgb_datasets: Dict[DatasetType, lgb.Dataset] = {DatasetType.TRAIN: train}

        for ds_type, dataset in self.datasets.items():
            if ds_type != DatasetType.TRAIN:
                logging.info("Constructing %s", ds_type.name)
                self._lgb_datasets[ds_type] = dataset.to_lgb_dataset(reference=train)

    @property
    def dataset_types(self) -> List[DatasetType]:
        return list(self.datasets.keys())

    def describe(self) -> None:
        for ds_type, dataset in self.datasets.items():
            logging.info("Shape for %s@%s %s\n", self.name, ds_type.name, dataset.data.shape)

    def get_dataset(self, ds_type: DatasetType) -> MLDataset:
        assert ds_type in self.dataset_types, f"{ds_type} not in {self.dataset_types}"
        return self.datasets[ds_type]

    def get_lgb_dataset(self, ds_type: DatasetType) -> lgb.Dataset:
        assert ds_type in self.dataset_types, f"{ds_type} not in {self.dataset_types}"
        return self._lgb_datasets[ds_type]

    def get_data(self, ds_type: DatasetType) -> pd.DataFrame:
        return self.datasets[ds_type].data

    def get_label(self, ds_type: DatasetType) -> pd.Series:
        return self.datasets[ds_type].label

    def get_eval_data(self, ds_type: DatasetType) -> Optional[pd.DataFrame]:
        return self.datasets[ds_type].eval_data

    @classmethod
    def from_pandas_datasets(
            cls, datasets: Dict[DatasetType, pd.DataFrame], feature_set: FeatureSet, name: str
    ) -> "Sample":
        ml_datasets: Dict[DatasetType, MLDataset] = {
            # Create MLDataset from passed in inferred or hard-coded FeatureSet
            ds_type: MLDataset(
                ds_type=ds_type,
                data=df[feature_set.regressors],
                label=df[feature_set.target],
                feature_set=feature_set,
                eval_data=df[feature_set.eval_fields] if feature_set.eval_fields else None,
                name="merged_dataset"
            )
            for ds_type, df in datasets.items()
            if not df.empty
        }
        return cls(datasets=ml_datasets, name=name)


class SampleParams:

    def __init__(
            self,
            train_share: float,
            validation_share: float = 0,
            test_share: float = 0
    ):
        assert round(train_share + validation_share + test_share) == 1.0, "All shares must add up to 1"

        self.train_share: float = train_share
        self.validation_share: float = validation_share
        self.test_share: float = test_share

    def split_by_indecies(self, df: pd.DataFrame) -> Dict[DatasetType, pd.DataFrame]:
        """Generate indecies to split dataframe into TRAIN/VALIDATION/TEST sets"""
        train_idx: int = int(len(df) * self.train_share)
        val_idx: int = int(len(df) * (self.train_share + self.validation_share))
        df_train, df_val, df_test = df.iloc[:train_idx], df.iloc[train_idx:val_idx], df.iloc[val_idx:]

        return {
            ds_type: data
            for ds_type, data in (
                (DatasetType.TRAIN, df_train), (DatasetType.VALIDATION, df_val), (DatasetType.TEST, df_test),
            )
            # Filter out all empty splits
            if not data.empty
        }
