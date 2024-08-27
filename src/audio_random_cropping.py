import os
import sys

import librosa
import numpy as np

# import from src
from abc_preprocessing import PreProcessingTechnique
from classification_dataset import ClassificationDataset
from regression_dataset import RegressionDataset

sys.path.append(os.getcwd() + "/src/")


class RandomCropper(PreProcessingTechnique):
    def __init__(self, crop_duration: int):
        """Random cropping: given an input audio track and a sampling rate,\
           crop the audio sample on a random starting point for a given
           duration\
           of n seconds. The output should be an audio track of duration n
           seconds\
           and the same sampling rate. If the track is shorter than n seconds,
           than\
           the original track is returned.

        Args:
            crop_duration (int): duration (n seconds) of the cropped audio
            files
        """
        self.crop_duration = crop_duration
        self._processing_type = "audio"

    def _transformation(
        self, old_data: ClassificationDataset | RegressionDataset
    ) -> list:
        """_transformation() is an a helper function that deals with\
            the overall creation of the new preprocessed dataset. It\
            applies the transformation iteratively to each entry in the
            dataset\
            given as an argument.

        Args:
            old_data (Dataset): a dataset of the type ClassificationDataset or
            RegressionDataset,\
                that needs to be preprocessed

        Returns:
            list: a list that consists of the preprocessed data entries
        """
        new_data = []
        for audio_sample in old_data.data:
            new_audio_sample = self._random_crop(audio_sample)
            new_data.append(new_audio_sample)
        return new_data

    def _random_crop(self, audio_sample: tuple) -> tuple:
        """audio_sample is a tuple of the nparray representation of\
            the audio file and the sampling rate

        Args:
            audio_sample (tuple): a data entry consisting of np.array
            representing\
                the time series and a sampling rate

        Returns:
            np.array: time series representation of a audio sample
        """
        audio_nparray = audio_sample[0]
        sampling_rate = audio_sample[1]
        audio_duration = librosa.get_duration(
            y=audio_nparray, sr=sampling_rate
        )

        # if the desired crop size is smaller than the current audio duration,
        # return the original audio sample
        if self.crop_duration >= audio_duration:
            return audio_sample

        # determining the last point when the cut can be made
        max_crop_time = audio_duration - self.crop_duration

        # determining the starting point
        start_time = np.random.uniform(0, max_crop_time)

        # Calculate sample indices for the random crop
        start_sample = int(start_time * sampling_rate)
        end_sample = start_sample + int(self.crop_duration * sampling_rate)

        # Crop the audio array
        cropped_audio_nparray = audio_nparray[start_sample:end_sample]

        return (cropped_audio_nparray, sampling_rate)
