import os
import sys

from abc_dataset import Dataset

sys.path.append(os.getcwd() + "/src/")


class RegressionDataset(Dataset):
    """RegressionDataset represents a dataset for regression tasks.
    It inherits from the Dataset class and extends its
    functionality to handle regression-specific data loading and
    preprocessing.

    Args:
        data_path (str): The root path to the dataset.
        data_type (str): Type of data, e.g., 'image' or 'audio'.
        labels (bool): Indicates whether the dataset has labels.
        csv_path (str, optional): Path to the CSV file containing
            names of data points and corresponding labels.
            (required if labels=True). Defaults to None.
        lazy (bool, optional): Lazy loading of data. Defaults to False.
    """

    def __init__(
        self,
        data_path: str,
        data_type: str,
        labels: bool,
        csv_path: str = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(
            data_path=data_path,
            data_type=data_type,
            labels=labels,
            csv_path=csv_path,
            lazy=lazy,
        )

        if self._labels_bool is True and csv_path is None:
            raise NameError(
                "Labels is set to True but no label_path is given.\
            Give label_path."
            )

        self.data = (
            super()._create_data_from_folder(data_path)
            if csv_path is None
            else super()._create_data_from_csv()
        )
        self.labels = (
            None if csv_path is None else super()._create_labels_from_csv()
        )
