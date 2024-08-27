import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd() + "/src/")

from classification_dataset import ClassificationDataset
from regression_dataset import RegressionDataset


class TestClassificationImage(unittest.TestCase):
    def test_clas_lazy_labels(self):
        self.dataset = ClassificationDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=True,
            labels_path="regression_data/poster.csv",
            lazy=True,
            dataset_format="csv",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_clas_eager_labels(self):
        self.dataset = ClassificationDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=True,
            labels_path="regression_data/poster.csv",
            dataset_format="csv",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_clas_eager_no_labels(self):
        self.dataset = ClassificationDataset(
            data_path="chess_data",
            data_type="image",
            labels=False,
            dataset_format="hierarchical",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, np.ndarray)

    def test_clas_lazy_no_labels(self):
        self.dataset = ClassificationDataset(
            data_path="chess_data",
            data_type="image",
            labels=False,
            lazy=True,
            dataset_format="hierarchical",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, np.ndarray)

    def test_clas_wrong_format(self):
        with self.assertRaises(NameError) as context:
            self.dataset = ClassificationDataset(
                data_path="chess_data",
                data_type="image",
                labels=False,
                lazy=True,
                dataset_format="wadeva",
            )
        expected_error_msg = (
            "data format has to be one of 'csv' or 'hierarchical'"
        )
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, actual_error_msg)

    def test_clas_no_labels_labelpath(self):
        with self.assertRaises(NameError) as context:
            ClassificationDataset(
                data_path="regression_data/poster_data",
                data_type="image",
                labels=False,
                labels_path="regression_data/poster.csv",
                lazy=True,
                dataset_format="csv",
            )
        expected_error_msg = "labels is set to False but label_path is given.\
                Set labels to true or delete labels_path."
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, actual_error_msg)


class TestRegressionImage(unittest.TestCase):
    def test_reg_lazy_labels(self):
        self.dataset = RegressionDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=True,
            labels_path="regression_data/poster.csv",
            lazy=True,
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_reg_eager_labels(self):
        self.dataset = RegressionDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=True,
            labels_path="regression_data/poster.csv",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_reg_lazy_no_labels(self):
        self.dataset = RegressionDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=False,
            lazy=True,
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, np.ndarray)

    def test_reg_eager_no_labels(self):
        self.dataset = RegressionDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=False,
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, np.ndarray)

    def test_reg_labels_no_labelpath(self):
        with self.assertRaises(NameError) as context:
            RegressionDataset(
                data_path="regression_data/poster_data",
                data_type="image",
                labels=True,
                lazy=True,
            )
        expected_msg = "labels is set to True but no label_path is given.\
            Give label_path."
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_error_msg)

    def test_reg_no_labels_labelpath(self):
        with self.assertRaises(NameError) as context:
            RegressionDataset(
                data_path="regression_data/poster_data",
                data_type="image",
                labels=False,
                labels_path="regression_data/poster.csv",
                lazy=True,
            )
        expected_error_msg = "labels is set to False but label_path is given.\
                Set labels to true or delete labels_path."
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, actual_error_msg)


class TestRegressionAudio(unittest.TestCase):
    def test_reg_lazy_labels(self):
        self.dataset = RegressionDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=True,
            labels_path="song_data/songs.csv",
            lazy=True,
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_reg_eager_labels(self):
        self.dataset = RegressionDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=True,
            labels_path="song_data/songs.csv",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_reg_lazy_no_labels(self):
        self.dataset = RegressionDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=False,
            lazy=True,
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_reg_eager_no_labels(self):
        self.dataset = RegressionDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=False,
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_reg_labels_no_labelpath(self):
        with self.assertRaises(NameError) as context:
            RegressionDataset(
                data_path="song_data/songs",
                data_type="audio",
                labels=True,
                lazy=True,
            )
        expected_msg = "labels is set to True but no label_path is given.\
            Give label_path."
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_error_msg)

    def test_reg_no_labels_labelpath(self):
        with self.assertRaises(NameError) as context:
            RegressionDataset(
                data_path="song_data/songs",
                data_type="audio",
                labels=False,
                labels_path="song_data/songs.csv",
                lazy=True,
            )
        expected_error_msg = "labels is set to False but label_path is given.\
                Set labels to true or delete labels_path."
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, actual_error_msg)


class TestClassificationAudio(unittest.TestCase):
    def test_clas_lazy_labels(self):
        self.dataset = ClassificationDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=True,
            labels_path="song_data/songs.csv",
            lazy=True,
            dataset_format="csv",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_clas_eager_labels(self):
        self.dataset = ClassificationDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=True,
            labels_path="song_data/songs.csv",
            dataset_format="csv",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_clas_eager_no_labels(self):
        self.dataset = ClassificationDataset(
            data_path="animal_data",
            data_type="audio",
            labels=False,
            dataset_format="hierarchical",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_clas_lazy_no_labels(self):
        self.dataset = ClassificationDataset(
            data_path="animal_data",
            data_type="audio",
            labels=False,
            lazy=True,
            dataset_format="hierarchical",
        )
        actual_datapoint = self.dataset[0]
        self.assertIsInstance(actual_datapoint, tuple)

    def test_clas_wrong_format(self):
        with self.assertRaises(NameError) as context:
            self.dataset = ClassificationDataset(
                data_path="animal_data",
                data_type="audio",
                labels=False,
                lazy=True,
                dataset_format="wadeva",
            )
        expected_error_msg = (
            "data format has to be one of 'csv' or 'hierarchical'"
        )
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, actual_error_msg)

    def test_clas_no_labels_labelpath(self):
        with self.assertRaises(NameError) as context:
            ClassificationDataset(
                data_path="song_data/songs",
                data_type="audio",
                labels=False,
                labels_path="song_data/songs.csv",
                lazy=True,
                dataset_format="csv",
            )
        expected_error_msg = "labels is set to False but label_path is given.\
                Set labels to true or delete labels_path."
        actual_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, actual_error_msg)


class TestTrainTest(unittest.TestCase):
    def test_generic_reg_split_1(self):
        self.dataset = RegressionDataset(
            data_path="regression_data/poster_data",
            data_type="image",
            labels=True,
            labels_path="regression_data/poster.csv",
            lazy=True,
        )
        X_train, y_train, X_test, y_test = self.dataset.train_test_split(
            train_split=0.69, shuffle=True
        )
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertLessEqual(len(X_train), len(self.dataset))
        self.assertLessEqual(len(X_test), len(self.dataset))
        self.assertFalse(np.array_equal(X_train[0], X_test[0]))
        self.assertFalse(np.array_equal(y_train[0], y_test[0]))

    def test_generic_clas_split(self):
        self.dataset = ClassificationDataset(
            data_path="animal_data",
            data_type="audio",
            labels=False,
            dataset_format="hierarchical",
        )
        X_train, X_test = self.dataset.train_test_split(
            train_split=0.69, shuffle=True
        )
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertLessEqual(len(X_train), len(self.dataset))
        self.assertLessEqual(len(X_test), len(self.dataset))
        self.assertFalse(np.array_equal(X_train[0], X_test[0]))

    def test_generic_reg_split(self):
        self.dataset = ClassificationDataset(
            data_path="song_data/songs",
            data_type="audio",
            labels=True,
            labels_path="song_data/songs.csv",
            lazy=True,
            dataset_format="csv",
        )
        X_train, y_train, X_test, y_test = self.dataset.train_test_split(
            train_split=0.69, shuffle=True
        )
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertLessEqual(len(X_train), len(self.dataset))
        self.assertLessEqual(len(X_test), len(self.dataset))
        self.assertFalse(np.array_equal(X_train[0], X_test[0]))
        self.assertFalse(np.array_equal(y_train[0], y_test[0]))

    # lazy, shuffle, labels
    def split_lazy_shuffle_labels(self):
        pass

    # lazy, shuffle, no labels
    # lazy, no shuffle, no labels
    # eager, shuffle, no labels
    # eager, no shuffle, labels


if __name__ == "__main__":
    unittest.main()
