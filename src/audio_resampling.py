import os
import sys

import librosa

# importing from src
from abc_preprocessing import PreProcessingTechnique
from classification_dataset import ClassificationDataset
from regression_dataset import RegressionDataset

sys.path.append(os.getcwd() + "/src/")


class Resampler(PreProcessingTechnique):
    def __init__(self, resampling_rate: int):
        """Resampling: Given an audio track and a sampling rate, it\
            returns the resampled audio track with a different \
                sampling rate.

        Args:
            _resampling_rate (int): rate at which the audio time series is\
                resampled
        """
        self._resampling_rate = resampling_rate
        self._processing_type = "audio"

    def _transformation(
        self, old_data: ClassificationDataset | RegressionDataset
    ) -> list:
        """_transformation() is an a helper function that deals with\
            the overall creation of the new preprocessed dataset. It\
            applies the transformation iteratively to each entry in the \
                dataset given as an argument.

        Args:
            old_data (Dataset): a dataset of the type ClassificationDataset or\
                RegressionDataset, that needs to be preprocessed

        Returns:
            list: a list that consists of the preprocessed data entries
        """
        new_data = []
        for audio_sample in old_data.data:
            new_audio_sample = self._resampling(audio_sample)
            new_data.append(new_audio_sample)
        return new_data

    def _resampling(self, audio_sample: tuple) -> tuple:
        """ audio_sample is a tuple of the nparray representation of the audio\
            file and the sampling rate

        Args:
            audio_sample (tuple): a tuple of the nparray representation\
                of the audio file and the sampling rate

        Returns:
            tuple: tuple consisting of resampled time series and the new \
                sampling rate
        """
        audio_nparray = audio_sample[0]
        sampling_rate = audio_sample[1]
        resampled_np_array = librosa.resample(
            y=audio_nparray,
            orig_sr=sampling_rate,
            target_sr=self._resampling_rate,
        )

        return (resampled_np_array, self._resampling_rate)
