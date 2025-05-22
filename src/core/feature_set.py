from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeatureSet:
    regressors: List[str]
    categorical: List[str]
    target: str
    eval_fields: Optional[List[str]] = None

    def __add__(self, other):
        """Define merging of two feature sets"""
        assert self.target == other.target, "Can only concat feature sets that share the same target"
        return FeatureSet(
            regressors=list(set(self.regressors + other.regressors)),
            categorical=list(set(self.categorical + other.categorical)),
            target=self.target
        )

    def add_regressors(
            self,
            new_regressors: List[str],
            new_categorical_regressors: Optional[List[str]] = None,
    ):
        """
        Add new regressors, you can also pass in new categorical regressors to specify which ones out
        of regressors are categorical
        """
        regressors: List[str] = list(set(self.regressors) | set(new_regressors))
        categorical: List[str] = (
            list(set(self.categorical) | set(new_regressors)) if new_categorical_regressors else self.categorical
        )
        return FeatureSet(regressors=regressors, categorical=categorical, target=self.target)

    def remove_regressors(self, regressors: List[str]):
        """
        Removes regressors from the already initialized FeatureSet and returns a new FeatureSet instance without
        regressors passed in
        """
        new_regressors: List[str] = list(set(self.regressors) - set(regressors))
        new_categorical_regressors: List[str] = list(set(self.categorical) - set(regressors))

        return FeatureSet(regressors=new_regressors, categorical=new_categorical_regressors, target=self.target)

    def __eq__(self, other) -> bool:
        return (
                self.regressors == other.regressors and
                self.categorical == other.categorical and
                self.target == other.target
        )
