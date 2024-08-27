import os
import random
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.getcwd() + "/src/")


class AbstractDataset(ABC):
    """An abstract class for all Datasets that sets the structure how\
        they should be implemented.

       Attributes:
        _root (str): Path to the dataset.
        data (Any): Placeholder for data.
        labels (Any): Placeholder for labels.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize the abstract dataset

        Args:
            data_path (str): Path to the dataset.
        """
        self._root = data_path
        self.data = None
        self.labels = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> np.array | Tuple[np.array, np.array]:
        """Get an item from the dataset based on the index."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        raise NotImplementedError


class Dataset(AbstractDataset):
    """A class representing datasets with optional labels.

    Attributes:
        _data_type (str): Type of data (e.g., "image", "audio").
        _labels_bool (bool): Indicates if the dataset has labels.
        lazy (bool): Lazy loading flag.
        _csv_path (str): Path to the CSV file containing labels.
    """

    def __init__(
        self,
        data_path: str,
        data_type: str,
        labels: bool = False,
        csv_path: str = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(data_path=data_path)
        self._data_type = data_type
        self._labels_bool = labels
        self.lazy = lazy

        self._csv_path = csv_path

    def __repr__(self) -> str:
        """__repr__() defines the string representation of the given object.

        use case:
            dataset = Dataset()
            print(dataset)
        """
        return (
            f"{self.__class__.__name__}:(\n"
            f"Data type: {self._data_type},\n"
            f"Lazy loading type: {self.lazy},\n"
            f"Has labels: {self._labels_bool},\n"
            f"Number of datapoints: {len(self)}\n"
            ")"
        )

    def __len__(self) -> int:
        """__len__() defines the len() function output for the given object.

        use case:
            dataset = Dataset()
            len(dataset)
        """
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, index: int) -> tuple:
        """__getitem__() defines the iteratable output of the given object.

        Args:
            index (int): index of the dataset (n = dataset length - 1) that
            the user wants

        Returns:

            tuple: outputs tuple of the form (data[index], label) if labels is\
            set to True\n
            during the initialisation of the dataset, (data[index]) if labels\
            is set to False.

        """
        if self._labels_bool is False:
            return deepcopy(self._if_lazy(self.data[index]))
        else:
            return (
                deepcopy(self._if_lazy(self.data[index])),
                deepcopy(self.labels[index]),
            )

    def _if_lazy(
        self, data: Tuple[str, List] | str
    ) -> Tuple[np.array, List] | np.array:
        """A private method to handle lazy loading. It can be used with a full\
             dataset or just a single datapoint. If attribute lazy is set to\
             true the method determines if it needs to read in a full dataset\
             or not and calls a helper function to read in data.

        Args:
            data (Tuple[str, List]| str): A set of filepaths and labels or\
             just a filepath

        Returns:
            Tuple[np.array, List] | np.array: Loaded data.
        """
        if self.lazy is True:
            if isinstance(data, list):
                return [self._read_data_point(filename) for filename in data]
            else:
                return self._read_data_point(data)
        else:
            return data

    def _read_data_point(self, filename: str = "") -> np.array:
        """A helper method which determines if the data to be read in\
            is audio data or image data.

        Args:
            filename (str, optional): Path to the datapoint. Defaults to "".

        Returns:
            np.array: Datapoint which is loaded to memory.
        """
        file = filename
        if self._data_type == "image":
            return self._read_image(filename=file)
        elif self._data_type == "audio":
            return self._read_sound(filename=file)
        else:
            print("data type {data_type} is not supported.")
            raise NotImplementedError

    def _read_image(self, filename: str = "") -> np.array:
        """Reads image from an specified file and returns it as an array\
            of pixel values.

        Args:
            filename (str, optional): Path to the file. Defaults to "".

        Returns:
            np.array: Picture data as an array
        """

        # Open the image file
        image = Image.open(filename).convert("RGB")
        # Convert the image data to a numpy array
        image_data = np.array(image)
        return image_data

    def _read_sound(self, filename: str = "") -> np.array:
        """Reads audio from an specified file and returns it as an array.

        Args:
            filename (str, optional): Path to the file. Defaults to "".

        Returns:
            np.array: Audio data as an array.

        """
        sound_time_series, sampling_rate = librosa.load(filename)
        sound_data = (sound_time_series, sampling_rate)
        return sound_data

    def _create_data_from_folder(self, folder: str) -> np.array:
        """Reads datapoints from a folder and loads their respective
        datapoints as arrays. Supported datatypes are: ".jpg", ".jpeg",
        ".png", ".gif", ".webp", ".wav", ".mp3".

        Args:
            folder (str): Path to the folder from where to read in data.


        args:
            folder (str): folder name
        Returns:

            np.array: a list of all of the datapoints

        """
        file_list = os.listdir(folder)
        filepaths = [
            os.path.join(folder, name)
            for name in file_list
            if name != ".DS_Store"
        ]
        if self.lazy is False:
            data = []
            for file in filepaths:
                if any(
                    ext.lower() in file.lower()
                    for ext in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".webp",
                        ".wav",
                        ".mp3",
                    ]
                ):
                    data.append(self._read_data_point(file))

            return data
        else:
            return filepaths

    def _create_data_from_csv(self) -> np.array:
        """
        Reads datapoints from a csv file and loads their respective pictures\
            audio data
        as np.arrays.

        Returns:
            np.ndarray: Image/audio data as an np.ndarray
        """
        df = pd.read_csv(self._csv_path)
        filenames = df.iloc[:, 0].tolist()
        filepaths = [os.path.join(self._root, str(name)) for name in filenames]
        if self.lazy is False:
            data = [self._read_data_point(filename) for filename in filepaths]
            return data
        else:
            return filepaths

    def _create_labels_from_csv(self) -> np.array:
        """Reads labels from a csv file stored in self._csv_path

        Returns:
            np.array: Labels in an array.
        """
        df = pd.read_csv(self._csv_path)
        labels = df.iloc[:, -1].tolist()
        return labels

    def train_test_split(
        self,
        train_split: float = 0.8,
        shuffle: bool = False,
    ) -> Tuple[np.array, np.array] | np.array:
        """Split the dataset into training and test sets. Data can be\
            shuffled in the process.

        Args:
            train_split (float, optional): Percentage of data for training.\
            Defaults to 0.8.
            shuffle (bool, optional): Whether to shuffle the data.\
            Defaults to False.

        Raises:
            TypeError: If you have 1 datapoint it is impossible to split it.
            TypeError: Train_split attribute has to be between 0 and 1.

        Returns:
            Tuple[np.array, np.array] | np.array: Training and test sets.
        """
        if len(self.data) <= 1:
            raise TypeError("Number of datapoints has to be bigger than 1")

        if train_split < 0 or train_split > 1:
            raise TypeError("Train_split attribute has to be between 0 and 1.")

        if shuffle is True and self._labels_bool is True:
            data, labels = self._shuffle_data(self.data, self.labels)
        elif shuffle is True:
            data = self._shuffle_data(self.data, self.labels)
        else:
            data, labels = self.data, self.labels

        split = int(len(data) * train_split)

        train_data = data[:split]

        test_data = data[split:]

        if self._labels_bool is True:
            train_labels = labels[:split]
            test_labels = data[split:]

            return (
                self._if_lazy(train_data),
                train_labels,
                self._if_lazy(test_data),
                test_labels,
            )
        else:
            return self._if_lazy(train_data), self._if_lazy(test_data)

    def _shuffle_data(self, data: list, labels: list) -> list:
        """Helper function to help with shuffleing the data as\
        a part of train test split functionality. It takes the\
        indeces of the dataset and labels and shuffles them.

        Args:
            data (list): Data that you need shuffled.
            labels (list): Labels that need to be shuffled.

        Raises:
            TypeError: Data and labels have to be the same size.

        Returns:
            list: Returns a list of the shuffled data (and lables)

        """
        shuffled_indices = random.sample(range(len(data)), len(data))
        shuffled_data = [data[i] for i in shuffled_indices]

        if labels is not None:
            if len(data) != len(labels):
                raise TypeError("data and labels have to be the same size")
            shuffled_labels = [labels[i] for i in shuffled_indices]
            return shuffled_data, shuffled_labels
        else:
            return shuffled_data

    @property
    def root(self) -> str:
        """Getter for the root path of the dataset.

        Returns:
            str: Root path.
        """
        return self._root

    @root.setter
    def root(self, data_path: str) -> None:
        """Setter for the root path of the dataset.

        Args:
            data_path (str): New path where root should be located.
        """
        self._root = data_path

    @property
    def data_type(self) -> str:
        """Getter for the data type of the dataset.

        Returns:
            str: Returns the data type as a string.
        """
        return self._data_type
