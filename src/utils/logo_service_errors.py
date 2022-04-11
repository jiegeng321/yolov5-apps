class Error(Exception):
    pass


class ImageDecodeError(Error):
    def __init__(self, message):
        self.message = message


class ImageFormatUnspportedError(Error):
    def __init__(self, format):
        self.message = 'image format {} is not supported'.format(format)
