import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy

from abc_dataset import Dataset
from classification_dataset import ClassificationDataset
from regression_dataset import RegressionDataset

sys.path.append(os.getcwd() + "/src/")


class PreProcessingTechnique(ABC):
    """An ABC class for Preprocessing techniques that is describing\
            the generic behavior of various preprocessing tools.
    """

    def __call__(self, dataset: Dataset) -> Dataset:
        """An instance of the processing tool is called with a dataset as\
            anargument.


        Args:
            dataset (Dataset): dataset for the preprocessing

        Returns:
            Dataset: preprocessed dataset
        """
        dataset_for_processing = deepcopy(dataset)
        if dataset_for_processing.lazy is True:
            dataset_for_processing.data = dataset_for_processing._if_lazy(
                dataset_for_processing.data
            )
            dataset_for_processing.lazy = False
        new_data = self._transformation(dataset_for_processing)
        new_dataset = deepcopy(dataset)
        new_dataset.data, new_dataset.labels = new_data, dataset.labels
        return new_dataset

    @abstractmethod
    def _transformation(
        self, old_data: ClassificationDataset | RegressionDataset
    ) -> list:
        """_transformation applies a preprocessing transformation to the\
            dataset of the type ClassificationDataset or RegressionDataset.

        Args:
            old_data (Dataset): a dataset of the type ClassificationDataset or
            RegressionDataset,\
                that needs to be preprocessed

        Raises:
            NotImplementedError: given method is not implemented\
                in the ABC but in the individual preprocessing techniques

        Returns:
            list: a list that consists of the preprocessed data entries
        """
        raise NotImplementedError
