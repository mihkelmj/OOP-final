import os
import sys

import numpy as np

# importing from src
from abc_preprocessing import PreProcessingTechnique
from classification_dataset import ClassificationDataset
from regression_dataset import RegressionDataset

sys.path.append(os.getcwd() + "/src/")


class CenterCrop(PreProcessingTechnique):
    def __init__(self, height: int, width: int) -> None:
        """Center crop: given an input image of any size H x W,\
            returns a cropped image of size height x width.\
            If the specified height and width are greater\
            than the original image, the crop is not performed. In case, e.g.,\
            H>height, but W<width, then the crop is performed only on the\
             height dimension.

        Args:
            height (int): Desired height of the center crop.
            width (int): Desired width of the center crop.
        """
        self.height = height
        self.width = width
        self._processing_type = "image"

    def _transformation(
        self, old_data: ClassificationDataset | RegressionDataset
    ) -> list:
        """_transformation() is an a helper function that deals with\
            the overall creation of the new preprocessed dataset. It\
            applies the transformation iteratively to each entry in the\
            dataset given as an argument.

        Args:

            old_data (Dataset): a dataset of the type Dataset, that needs \
            to be preprocessed

        Returns:
            list: a list that consists of the preprocessed data entries
        """
        new_data = []
        for image_array in old_data.data:
            new_image_array = self._center_crop(image_array)
            new_data.append(new_image_array)

        return new_data

    def _center_crop(self, image_array: np.array) -> np.array:
        """_center_crop() is a private helper function which applies a\
            np.array modification to a single datapoint. This crops the\
            image numpy array according to the height and width specifications\
            specified above.

        Args:
            image_array (np.array): a np.array which represents a single\
            original datapoint

        Returns:
            np.array: a np.array which represents a single preprocessed\
            datapoint
        """
        # Get original image size (height, width, rgb colors)
        original_height, original_width, _ = image_array.shape

        # Calculate center coordinates
        center_x = original_width // 2
        center_y = original_height // 2

        # Calculate crop coordinates
        crop_x1 = max(center_x - self.width // 2, 0)
        crop_y1 = max(center_y - self.height // 2, 0)
        crop_x2 = min(center_x + self.width // 2, original_width)
        crop_y2 = min(center_y + self.height // 2, original_height)

        # Perform crop
        cropped_image = image_array
        if self.height < original_height:
            cropped_image = cropped_image[crop_y1:crop_y2, :, :]
        if self.width < original_width:
            cropped_image = cropped_image[:, crop_x1:crop_x2, :]
        return cropped_image
