from time import time


class Timer(object):
    def __init__(self):
        super().__init__()

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time()
        self.elapse = 1000*(self.end - self.start)
