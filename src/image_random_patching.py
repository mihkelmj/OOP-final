import os
import sys

import numpy as np

from abc_preprocessing import PreProcessingTechnique
from classification_dataset import ClassificationDataset
from custom_errors import UndefinedColorError
from regression_dataset import RegressionDataset

sys.path.append(os.getcwd() + "/src/")


class RandomPatching(PreProcessingTechnique):
    def __init__(self, height: int, width: int, color: str) -> None:
        """Random patching: given an input image of any size, fills a window\
           of the image with a pre-specified color. The top-left coordinate \
           of this window is sampled\
           randomly within the image. Let the user decide color,\
           height and width of this window at initialization.

        Args:
            height (int): Desired height of the center crop. Must be smaller/\
            equal of the picture height.
            width (int): Desired width of the center crop. Must be smaller/\
            equal of the picture width.
            color (str): 'red', 'green', 'black'
        """
        self.height = height
        self.width = width
        self._check_color(color)
        self.color = color
        self._processing_type = "image"

    def _transformation(
        self, old_data: ClassificationDataset | RegressionDataset
    ) -> list:
        """_transformation() is an a helper function that deals with\
            the overall creation of the new preprocessed dataset. It\
            applies the transformation iteratively to each entry in the \
            dataset given as an argument.

        Args:

            old_data (Dataset): a dataset of the type Dataset, that needs to \
            be preprocessed


        Returns:
            list: a list that consists of the preprocessed data entries
        """
        new_data = []
        for image_array in old_data.data:
            new_image_array = self._apply_random_patching(image_array)
            new_data.append(new_image_array)

        if old_data._labels_bool is True:
            return (new_data, old_data.labels)
        else:
            return new_data

    def _apply_random_patching(self, image_array: np.array) -> np.array:
        """_apply_random_patching() is a private helper function which applies\
            a np.array modification to a single datapoint. This creates\
            a randomly positioned patch of a prespecified color.

        Args:
            image_array (np.array): a np.array which represents a single \
            original datapoint

        Returns:
            np.array: a np.array which represents a single preprocessed \
            datapoint
        """
        # Get original image size (height, width, rgb colors)
        original_height, original_width, _ = image_array.shape

        # Sample top-left coordinates randomly
        top_left_x = np.random.randint(0, original_width - self.width + 1)
        top_left_y = np.random.randint(0, original_height - self.height + 1)

        # Create a copy of the image array
        patched_image = np.copy(image_array)

        # Set color based on user input

        if self.color == "red":
            color = [255, 0, 0]  # Red
        elif self.color == "green":
            color = [0, 255, 0]  # Green
        elif self.color == "black":
            color = [0, 0, 0]  # Black

        else:
            raise ValueError("Invalid color. Use 'red', 'green', or 'black'.")

        # Fill the patch with the specified color

        patched_image[
            top_left_y: top_left_y + self.height,
            top_left_x: top_left_x + self.width,
            :,
        ] = color

        return patched_image

    def _check_color(self, color: str) -> None:
        """_checkcolor() is a private helper function which checks\
            if the color is initialised as a prespecified color.

        Args:
            color (str): 'red', 'green', 'black'
        Raises:
            UndefinedColorError: if the color is not initialised as 'red',\
            'green', or 'black', UndefinedColorError is raised.
        """
        if color not in ("red", "green", "black"):
            raise UndefinedColorError(color)
