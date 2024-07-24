from typing import Union
import numpy as np
from os.path import join, isdir
from os import makedirs


def ensure_directory_exists(func):
    def wrapper(self, name, *args, **kwargs):
        directory_path = join(self.storage_dir, name)
        if not isdir(directory_path):
            makedirs(directory_path)
        return func(self, name, *args, **kwargs)

    return wrapper
