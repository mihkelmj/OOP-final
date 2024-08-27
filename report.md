# Report

## main.py

In the main program we showcase how to use our library.


In the main program we showcase how to use our library.


## abc_dataset.py

We have here 2 classes: 
- AbstractDataset which defines all necessary functionaly of different kinds of Dataset classes.
- Dataset which is a class to handle all the functionaly that a dataset would need.
  
Dataset class has attributes _root, data, labels, _data_type, _labels_bool, lazy and _csv_path. Data, labels and lazy are public attributes because depending on the task the dataset is used for these attributes may need to be changed within the class. Others are private since they should not be altered with. The user can define root directory but when he doesnt we assume that data is in the same folder. And they can always edit the path with root getter setter.
We have here 2 classes: 
- AbstractDataset which defines all necessary functionaly of different kinds of Dataset classes.
- Dataset which is a class to handle all the functionaly that a dataset would need.
  
Dataset class has attributes _root, data, labels, _data_type, _labels_bool, lazy and _csv_path. Data, labels and lazy are public attributes because depending on the task the dataset is used for these attributes may need to be changed within the class. Others are private since they should not be altered with. The user can define root directory but when he doesnt we assume that data is in the same folder. And they can always edit the path with root getter setter.

We have 2 ways to read in data to the class: 'csv' and 'hierachical'. I if you want to read in data from CSV we make the assumption that you have the labels in the second column of the file. When reading from folder we have certain supported types of files. For more info consult docstrings. 

The private method _if_lazy is used to conditionally load data either immediately or on-demand based on the lazy attribute. It works with either a single datapoint that needs to be loaded into memory or a list of datapoints. Then it calls helper methods to actually load the data in depending on the type of data.

The public method train_test_split splits the data into train and testing datasets based on their indices. We chose this approach since there is no need to 'move' all of the datapoints, just their indices will do. The shuffling is done by the method _shuffle_data.


We have 2 ways to read in data to the class: 'csv' and 'hierachical'. I if you want to read in data from CSV we make the assumption that you have the labels in the second column of the file. When reading from folder we have certain supported types of files. For more info consult docstrings. 

The private method _if_lazy is used to conditionally load data either immediately or on-demand based on the lazy attribute. It works with either a single datapoint that needs to be loaded into memory or a list of datapoints. Then it calls helper methods to actually load the data in depending on the type of data.

The public method train_test_split splits the data into train and testing datasets based on their indices. We chose this approach since there is no need to 'move' all of the datapoints, just their indices will do. The shuffling is done by the method _shuffle_data.



## batch_loader.py

The BatchLoader class is designed for efficiently loading batches of data from a given dataset. The key functionality of batch loader is to use it as an iterator and iterate over batches. Attributes of this class are detailed below.

Attributes:
- data (np.ndarray): The input data to be batch-loaded. A public attribute since the user might want to altered it.
- batch_size (int): Size of each batch. Defaults to 10.
- shuffle (bool): Whether to shuffle the data before creating batches. Defaults to False.
- include_smaller (bool): Whether to include a smaller batch for the remaining data when len(data) is not divisible by batch_size. Defaults to True.


The way to use BatchLoader is through the public method create_batches. This method creates and returns batches of data which then can be used further. 


The BatchLoader class is designed for efficiently loading batches of data from a given dataset. The key functionality of batch loader is to use it as an iterator and iterate over batches. Attributes of this class are detailed below.

Attributes:
- data (np.ndarray): The input data to be batch-loaded. A public attribute since the user might want to altered it.
- batch_size (int): Size of each batch. Defaults to 10.
- shuffle (bool): Whether to shuffle the data before creating batches. Defaults to False.
- include_smaller (bool): Whether to include a smaller batch for the remaining data when len(data) is not divisible by batch_size. Defaults to True.


The way to use BatchLoader is through the public method create_batches. This method creates and returns batches of data which then can be used further. 


## regression_dataset.py
The RegressionDataset class is pretty straight forward. It implements functionaly to use Dataset class with Regression type data. The only functionaly is to read in data and put safeguards for safe use.
The RegressionDataset class is pretty straight forward. It implements functionaly to use Dataset class with Regression type data. The only functionaly is to read in data and put safeguards for safe use.

## classification_dataset.py
The ClassificationDataset class is designed to handle classification datasets with two primary formats: 'csv' and 'hierarchical'. In the case of hierarchical data, the class assumes a folder structure where each subfolder contains data for a single class. For 'csv' format, the class relies on the abstract Dataset class's methods for reading data and labels from a CSV file.

The _create_data_object method is responsible for initializing the 'data' and 'labels' lists based on the specified dataset format, calling the appropriate helper method.

The _create_data_from_hierarchical method handles file retrieval from a hierarchical folder structure, iterating through class directories, creating numpy arrays from class folders, and handling labels based on user preferences.


The ClassificationDataset class is designed to handle classification datasets with two primary formats: 'csv' and 'hierarchical'. In the case of hierarchical data, the class assumes a folder structure where each subfolder contains data for a single class. For 'csv' format, the class relies on the abstract Dataset class's methods for reading data and labels from a CSV file.

The _create_data_object method is responsible for initializing the 'data' and 'labels' lists based on the specified dataset format, calling the appropriate helper method.

The _create_data_from_hierarchical method handles file retrieval from a hierarchical folder structure, iterating through class directories, creating numpy arrays from class folders, and handling labels based on user preferences.



## abc_preprocessing.py
This code provides a template for creating preprocessing techniques by defining a common interface through the abstract base class PreProcessingTechnique. Subclasses have to implement the actual preprocessing logic in the _transformation method. The provided __call__ method defines the overall process of applying a preprocessing technique to a dataset.
This code provides a template for creating preprocessing techniques by defining a common interface through the abstract base class PreProcessingTechnique. Subclasses have to implement the actual preprocessing logic in the _transformation method. The provided __call__ method defines the overall process of applying a preprocessing technique to a dataset.

## audio_resampling.py
Given an audio track and a sampling rate, the preprocessing tool returns the resampled audio track with a different sampling rate.
Given an audio track and a sampling rate, the preprocessing tool returns the resampled audio track with a different sampling rate.

## audio_random_crop.py
This is an implementation of random croping for audio data. Given an input audio track and a sampling rate it crops the audio sample on a random starting point for a given duration of n seconds. The output is an audio track of duration n seconds and the same sampling rate. If the track is shorter than n seconds, than the original track is returned. 
This is an implementation of random croping for audio data. Given an input audio track and a sampling rate it crops the audio sample on a random starting point for a given duration of n seconds. The output is an audio track of duration n seconds and the same sampling rate. If the track is shorter than n seconds, than the original track is returned. 
## image_center_crop.py
This is an implementation of image croping for audio data. Given an input image of any size H × W, it returns a cropped image of size height × width. If the specified height and width are greater than the original image, the crop is not performed. 
This is an implementation of image croping for audio data. Given an input image of any size H × W, it returns a cropped image of size height × width. If the specified height and width are greater than the original image, the crop is not performed. 
## image_random_patch.py
Given an input image of any size, this class will fill a window of the image with a pre-specified color that the user can choose. The top-left coordinate of this window is sampled randomly within the image. Let the user decide color, height and width of this window at initialization.
Given an input image of any size, this class will fill a window of the image with a pre-specified color that the user can choose. The top-left coordinate of this window is sampled randomly within the image. Let the user decide color, height and width of this window at initialization.
## sequential_processing.py

This class implements a sequential pipeline of preprocessing steps. This class takes as input (in the constructor) a variable number of preprocessing steps and applies them sequentially in the order they were passed.