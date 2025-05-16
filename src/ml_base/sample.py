import logging
from typing import Optional, Dict, List

import lightgbm as lgb
import pandas as pd

from core.feature_set import FeatureSet
from ml_base.enums import DatasetType


class MLDataset:

    def __init__(
            self, ds_type: DatasetType, data: pd.DataFrame, label: pd.Series,
            feature_set: FeatureSet, eval_data: Optional[pd.DataFrame] = None,
    ):
        self.ds_type: DatasetType = ds_type
        self.data: pd.DataFrame = data
        self.label: pd.Series = label
        self.feature_set: FeatureSet = feature_set
        # eval_data is optional, used for evaluation only
        self.eval_data: Optional[pd.DataFrame] = eval_data

    def to_lgb_dataset(self, reference: Optional[lgb.Dataset] = None) -> lgb.Dataset:
        """Returns lightgbm Dataset for training purposes, maybe it will result in training speed increase"""
        return lgb.Dataset(
            data=self.data,
            label=self.label,
            categorical_feature=self.feature_set.categorical,
            reference=reference,
            free_raw_data=False
        ).construct()


class Sample:

    def __init__(self, datasets: Dict[DatasetType, MLDataset], require_lgb: bool = False):
        self.datasets: Dict[DatasetType, MLDataset] = datasets

        if require_lgb:
            # Create reference to train dataset
            train: lgb.Dataset = self.datasets[DatasetType.TRAIN].to_lgb_dataset()
            self.lgb_datasets: Dict[DatasetType, lgb.Dataset] = {DatasetType.TRAIN: train}

            for ds_type, dataset in self.datasets.items():
                if ds_type != DatasetType.TRAIN:
                    logging.info("Constructing %s", ds_type.name)
                    self.lgb_datasets[ds_type] = dataset.to_lgb_dataset(reference=train)

    @property
    def dataset_types(self) -> List[DatasetType]:
        return list(self.datasets.keys())

    def get_dataset(self, ds_type: DatasetType) -> MLDataset:
        assert ds_type in self.dataset_types, f"{ds_type} not in {self.dataset_types}"
        return self.datasets[ds_type]

    def get_lgb_dataset(self, ds_type: DatasetType) -> lgb.Dataset:
        assert ds_type in self.dataset_types, f"{ds_type} not in {self.dataset_types}"
        return self.lgb_datasets[ds_type]

    def get_data(self, ds_type: DatasetType) -> pd.DataFrame:
        return self.datasets[ds_type].data

    def get_label(self, ds_type: DatasetType) -> pd.Series:
        return self.datasets[ds_type].label

    def get_eval_data(self, ds_type: DatasetType) -> Optional[pd.DataFrame]:
        return self.datasets[ds_type].eval_data

    @classmethod
    def from_pandas_datasets(cls, datasets: Dict[DatasetType, pd.DataFrame], feature_set: FeatureSet) -> "Sample":
        ml_datasets: Dict[DatasetType, MLDataset] = {
            # Create MLDataset from passed in inferred or hard-coded FeatureSet
            ds_type: MLDataset(
                ds_type=ds_type,
                data=df[feature_set.regressors],
                label=df[feature_set.target],
                feature_set=feature_set,
                eval_data=df[feature_set.eval_fields] if feature_set.eval_fields else None
            )
            for ds_type, df in datasets.items()
            if not df.empty
        }
        return cls(datasets=ml_datasets, require_lgb=False)


def concat_datasets(dss: List[MLDataset], ds_type: DatasetType) -> MLDataset:
    logging.info("Concatenating %s", ds_type.name)

    merged_data: pd.DataFrame = pd.concat([ds.data for ds in dss])
    merged_label: pd.Series = pd.concat([ds.label for ds in dss])
    list_eval_data: List[pd.DataFrame] = [ds.eval_data for ds in dss if ds.eval_data is not None]

    return MLDataset(
        ds_type=ds_type,
        data=merged_data,
        label=merged_label,
        feature_set=dss[0].feature_set,
        eval_data=pd.concat(list_eval_data) if len(list_eval_data) > 0 else None
    )


def concat_samples(samples: List[Sample], require_lgb: bool = True) -> Sample:
    """Concatenate multiple samples preserving train, validation and test sets"""
    logging.info("Concatenating datasets")
    datasets: Dict[DatasetType, MLDataset] = {}

    for ds_type in samples[0].dataset_types:
        dss: List[MLDataset] = [sample.get_dataset(ds_type=ds_type) for sample in samples]
        dataset: MLDataset = concat_datasets(dss, ds_type=ds_type)
        datasets[ds_type] = dataset

    return Sample(datasets=datasets, require_lgb=require_lgb)


class SampleParams:

    def __init__(self, train_share: float, validation_share: float = 0, test_share: float = 0):
        assert train_share + validation_share + test_share == 1, "All shares must add up to 1"
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
