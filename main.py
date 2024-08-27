import os
import sys

import soundfile as sf
from PIL import Image

from src.audio_random_cropping import RandomCropper
from src.audio_resampling import Resampler
from src.batch_loader import BatchLoader
from src.classification_dataset import ClassificationDataset
from src.image_center_crop import CenterCrop
from src.image_random_patching import RandomPatching
from src.regression_dataset import RegressionDataset
from src.sequential_processing import SequentialPreprocessing

sys.path.append(os.getcwd() + "/src/")


def main():

    # Showcasing a dataset with and without labels
    image_dataset_without_labels = RegressionDataset(
        data_path="regression_data/poster_data",
        data_type="image",
        labels=False,
        csv_path="regression_data/poster.csv",
        lazy=False,
    )
    print(f"Dataset without labels:\n{image_dataset_without_labels}")
    print(image_dataset_without_labels[0])

    image_dataset_with_labels = RegressionDataset(
        data_path="regression_data/poster_data",
        data_type="image",
        labels=True,
        csv_path="regression_data/poster.csv",
        lazy=False,
    )
    print(f"Dataset with labels:\n{image_dataset_with_labels}")
    print(image_dataset_with_labels[0])

    # Showcasing a dataset for classification and regression
    classification_dataset = ClassificationDataset(
        data_path="chess_data",
        data_type="image",
        labels=True,
        dataset_format="hierarchical",
        lazy=False,
    )
    print(f"Classification dataset:\n{classification_dataset}")
    print(classification_dataset[0])

    regression_dataset = RegressionDataset(
        data_path="song_data/songs",
        data_type="audio",
        labels=True,
        csv_path="song_data/songs.csv",
        lazy=False,
    )
    print(f"Regression dataset:\n{regression_dataset}")
    print(regression_dataset[0])

    # Showcasing a lazy and an eager dataloader
    lazy_dataset = ClassificationDataset(
        data_path="animal_data",
        data_type="audio",
        labels=True,
        dataset_format="hierarchical",
        lazy=True,
    )
    print(f"Lazy dataset:\n{lazy_dataset}")
    print(lazy_dataset[0])

    eager_dataset = ClassificationDataset(
        data_path="animal_data",
        data_type="audio",
        labels=True,
        dataset_format="hierarchical",
        lazy=False,
    )
    print(f"Eager dataset:\n{eager_dataset}")
    print(eager_dataset[0])

    # Showcasing a BatchLoader on top of two of these datasets
    batch_loader_1 = BatchLoader(
        data=sorted(regression_dataset.labels),
        shuffle=True,
        batch_size=8,
        include_smaller=False,
    )

    print("Regression dataset batches:\n")
    batches_1 = batch_loader_1.create_batches()
    print(len(batches_1))
    for i, batch in enumerate(batches_1):
        print(i, batch)

    batch_loader_2 = BatchLoader(
        data=sorted(classification_dataset.labels),
        shuffle=False,
        batch_size=8,
        include_smaller=True,
    )

    print("Classification dataset batches:\n")
    batches_2 = batch_loader_2.create_batches()
    print(len(batches_2))
    for i, batch in enumerate(batches_2):
        print(i, batch)

    # showcasing the image processing pipeline
    image_dataset = ClassificationDataset(
        data_path="chess_data",
        data_type="image",
        labels=False,
        dataset_format="hierarchical",
        lazy=True,
    )

    center_cropping = CenterCrop(height=250, width=250)
    random_patching = RandomPatching(height=25, width=25, color="red")
    sequential_image_processing = SequentialPreprocessing(
        center_cropping, random_patching
    )
    processed_dataset = sequential_image_processing(image_dataset)
    image = Image.fromarray(processed_dataset[0])
    image.save("processed_image.jpeg")

    # showcasing the audio processing pipeline
    audio_dataset = ClassificationDataset(
        data_path="song_data/songs",
        data_type="audio",
        labels=True,
        csv_path="song_data/songs.csv",
        dataset_format="csv",
        lazy=True,
    )

    random_cropping = RandomCropper(crop_duration=1)
    audio_resampling = Resampler(resampling_rate=30000)
    sequential_audio_processing = SequentialPreprocessing(
        random_cropping, audio_resampling
    )
    processed_audio_dataset = sequential_audio_processing(audio_dataset)
    audio_nparray = processed_audio_dataset.data[0][0]
    audio_sampling = processed_audio_dataset.data[0][1]
    sf.write("processed_audio.wav", audio_nparray, audio_sampling, "PCM_24")

    """
    # Showcasing the preprocessing pipeline on top of a BatchLoader output
    for batch in batches_1:
        batch = sequential_audio_processing(batch)"""

    """
    # Showcasing the preprocessing pipeline on top of a Dataset samples
    for i in range(5):
        new_data = sequential_audio_processing(regression_dataset[i])"""


if __name__ == "__main__":
    main()
