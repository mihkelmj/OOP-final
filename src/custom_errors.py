class AudioPreprocessingError(Exception):

    def __init__(
        self,
        value,
        message="Audio data can only be preprocessed with RandomCutter and\
         Resampler tools!",
    ) -> None:
        self.value = value

        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class ImagePreprocessingError(Exception):

    def __init__(
        self,
        message="Image data can only be preprocessed with CenterCrop and \
        RandomPatcing tools!",
    ) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UndefinedColorError(Exception):
    def __init__(
        self, value, message="Undefined color is passed as an argument!"
    ) -> None:
        self.message = message
        self.value = value
        super().__init__(self.message, self.value)

    def __str__(self):
        return f"{self.message} ->> {self.value}"


class NoCSVPathError(Exception):
    def __init__(
        self,
        value,
        message="with dataset_format == 'csv' csv_path is required!",
    ) -> None:
        self.message = message
        self.value = value
        super().__init__(self.message, self.value)

    def __str__(self):
        return f"{self.message} ->> {self.value}"
