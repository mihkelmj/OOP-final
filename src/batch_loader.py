from typing import List

import numpy as np


class BatchLoader:
    """
    A class for loading batches of data from a given dataset.

    Attributes:
        data (np.ndarray): The input data to be batch-loaded.
        batch_size (int): Size of each batch. Defaults to 10.
        shuffle (bool): Whether to shuffle the data before creating batches.
            Defaults to False.
        include_smaller (bool): Whether to include a smaller batch for the
            remaining data when len(data) is not divisible by batch_size.
            Defaults to True.

    """

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int = 10,
        shuffle: bool = False,
        include_smaller: bool = True,
    ) -> None:
        """
        Initialize the BatchLoader instance.

        Args:
            data (np.ndarray): The input data to be batch-loaded.
            batch_size (int, optional): Size of each batch. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the data before
                creating batches. Defaults to False.
            include_smaller (bool, optional): Whether to include a smaller
                batch for the remaining data when len(data) is not divisible
                by batch_size. Defaults to True.
        """
        self.data = data
        self._batch_size = batch_size
        self._include_smaller = include_smaller
        self._shuffle = shuffle
        self._indices = np.arange(len(data))
        self._current_batch = 0

    def __len__(self) -> int:
        """
        Return the number of batches.

        Returns:
            int: Number of batches.
        """
        if self._include_smaller:
            return len(self._indices) // self._batch_size + (
                1 if len(self._indices) % self._batch_size != 0 else 0
            )
        else:
            return len(self._indices) // self._batch_size

    def __next__(self) -> List[np.ndarray]:
        """
        Return the next batch of data.

        Returns:
            List[np.ndarray]: The next batch of data.
        """
        if self._current_batch >= len(self):
            raise StopIteration

        start = self._current_batch * self._batch_size
        end = start + self._batch_size
        if end > len(self.data):
            end = len(self.data)
        batch_indices = self._indices[start:end]
        batch_data = [self.data[i] for i in batch_indices]
        self._current_batch += 1
        return batch_data

    def __iter__(self) -> "BatchLoader":
        """
        Initialize the iterator.

        Returns:
            BatchLoader: Iterator object.
        """
        self._current_batch = 0
        return self

    def create_batches(self) -> List[List[np.ndarray]]:
        """
        Create all batches and return them as a list.

        Returns:
            List[List[np.ndarray]]: List of batches.
        """
        if self._shuffle:
            np.random.shuffle(self._indices)

        batches = [batch for batch in self]
        return batches
