import os
import sys
from copy import deepcopy

from abc_dataset import Dataset
from abc_preprocessing import PreProcessingTechnique
from classification_dataset import ClassificationDataset
from custom_errors import AudioPreprocessingError, ImagePreprocessingError
from regression_dataset import RegressionDataset

from .audio_random_cropping import RandomCropper
from .audio_resampling import Resampler
from .image_center_crop import CenterCrop
from .image_random_patching import RandomPatching

sys.path.append(os.getcwd() + "/src/")


class SequentialPreprocessing(PreProcessingTechnique):
    def __init__(self, *preprocessing_steps: PreProcessingTechnique) -> None:
        """A preprocessing class which implements a sequential pipeline of \
           preprocessing steps. This class takes as input (in the constructor)\
           a variable number of preprocessing steps and applies them \
           sequentially in the order they were passed.

        Args:
            *preprocessing_steps (PreProcessingTechnique): Preprocessing steps\
             to apply sequentially.
        """
        self.preprocessing_steps = preprocessing_steps

    def __call__(
        self, dataset: ClassificationDataset | RegressionDataset
    ) -> Dataset:
        """Applies the sequential preprocessing steps to the dataset given\
           as an input.

        Args:
            dataset (Dataset): a dataset of the type ClassificationDataset or
            RegressionDataset,\
                that is to be preprocessed.

        Returns:
            Dataset: Preprocessed dataset.
        """
        preprocessed_dataset = deepcopy(dataset)
        if preprocessed_dataset.lazy is True:
            preprocessed_dataset.data = preprocessed_dataset._if_lazy(
                preprocessed_dataset.data
            )
            preprocessed_dataset.lazy = False

        for processing_step in self.preprocessing_steps:

            self._data_type_checker(
                dataset=dataset, processing_step=processing_step
            )
            preprocessed_dataset = self._transformation(
                data=preprocessed_dataset, processingtechnique=processing_step
            )
        return preprocessed_dataset

    def _transformation(
        self,
        data: ClassificationDataset | RegressionDataset,
        processingtechnique: PreProcessingTechnique,
    ) -> list:
        """_transformation() is an a helper function that deals with\
            the overall creation of the new preprocessed dataset. It\
            applies eachk transformation passed as a variadic arguments
            iteratively to each entry in the dataset\
            given as an argument.

        Args:
            old_data (Dataset): a dataset of the type ClassificationDataset or
            RegressionDataset,\
                that needs to be preprocessed

        Returns:
            list: a list that consists of the preprocessed data entries
        """
        return processingtechnique(data)

    def _data_type_checker(
        self,
        dataset: ClassificationDataset | RegressionDataset,
        processing_step: PreProcessingTechnique,
    ) -> None:
        """_data_type_checker is a helper function which ensures the correct
        usage of\
            preprocessing tools. Resampler and RandomCropper can only be used
            on audio\
            data and RandomPatching and CenterCrop can only be used on image
            data.

        Args:
            dataset (ClassificationDataset | RegressionDataset): a dataset of
            the type\
                ClassificationDataset or RegressionDataset,\
                on which the preprocessing tool is about to be used.
            processing_step (PreProcessingTechnique): A processing tool of the\
                type Resampler, RandomCropper, RandomPatching or CenterCrop.

        Raises:
            AudioPreprocessingError: is raised if image preprocessing tools
            are used on audio data
            ImagePreprocessingError: is raised if audio preprocessing tools
            are used on image data
        """
        if dataset.data_type == "audio" and type(processing_step) not in (
            Resampler,
            RandomCropper,
        ):
            raise AudioPreprocessingError()
        elif dataset.data_type == "image" and type(processing_step) not in (
            RandomPatching,
            CenterCrop,
        ):
            raise ImagePreprocessingError()
        else:
            return
