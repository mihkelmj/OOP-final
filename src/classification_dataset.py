import os
import sys

from abc_dataset import Dataset
from custom_errors import NoCSVPathError

sys.path.append(os.getcwd() + "/src/")


class ClassificationDataset(Dataset):
    """ClassificationDataset manager deals with two different formats of data:\
        with the labels stored in a separate file outside of the root folder
        in a .csv file. In the case of hierarchical structure the data should\
        be stored accordingly,\
        with each subfolder containing the data for a single class. In the\
        given configuration, you don't need a file with labels, since the\
        labels are already encoded in the folder hierarchy (via the folder\
         names).

    """

    def __init__(
        self,
        data_path: str,
        data_type: str,
        labels: bool,
        dataset_format: str,
        lazy: bool = False,
        csv_path: str = None,
    ) -> None:
        if dataset_format not in ["csv", "hierarchical"]:
            raise NameError(
                "data format has to be one of 'csv' or 'hierarchical'"
            )
        if dataset_format == "csv" and csv_path is None:
            raise NoCSVPathError(csv_path)

        super().__init__(data_path, data_type, labels, csv_path, lazy)
        self._dataset_format = dataset_format

        self.data, self.labels = self._create_data_object()

    def _create_data_object(self) -> None:
        """_create_data_object() is a private helper method that initialises\
        the 'data' and 'labels' lists based on the rutine established
        by dataset_format ('csv' or 'hierarchical').
        """
        if self._dataset_format == "hierarchical":
            data, labels = self._create_data_from_hierarchical()
            return data, labels
        elif self._dataset_format == "csv":
            data = super()._create_data_from_csv()
            labels = super()._create_labels_from_csv()
            return data, labels

    def _create_data_from_hierarchical(self) -> list:
        """_create_data_from_hierarchical() is a private helper function\
            that deals with file retrieval from a hierarchical data structure.\


        Returns:
            List: a list consisting of the data entries that are retireved\
            from the files
        """
        # assign data directory
        directory = self._root
        labels = []
        data = []

        # this loops through every class directory and adds its labels
        for class_name in os.listdir(directory):
            if os.path.isdir(
                os.path.join(directory, class_name)
            ):  # performs the loop only on directories
                # reasigning the path to the class_folder
                class_folder_path = os.path.join(directory, str(class_name))
                # create the numpy arrays out of all
                # pictures in the class_folder
                class_data = super()._create_data_from_folder(
                    class_folder_path
                )
                # add the image (numpy array) info to the
                # dataset for the given class
                data = data + class_data
                # if the user set the labels_bool to True,
                # eject the corresponding labels
                if self._labels_bool is True:
                    number_of_files = len(
                        [
                            file
                            for file in os.listdir(class_folder_path)
                            if file != ".DS_Store"
                        ]
                    )
                    # num_of_entries = len(os.listdir(class_folder_path))
                    labels.extend([class_name] * number_of_files)
                else:
                    labels = None
        return data, labels
