from typing import Union, Optional, List
from os.path import isdir, isfile, join
from os import makedirs
import numpy as np

from camera import NotCalibratedError
from calibration_manager_utils import ensure_directory_exists
from frame_provider import IFrameProviderCalibrationDataManager


class FrameProviderCalibrationDataManager(IFrameProviderCalibrationDataManager):
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.is_mirrored = None
        self.anchor_point = None

        if not isdir(storage_dir):
            makedirs(storage_dir)

    @ensure_directory_exists
    def get_is_mirrored(self, name: str) -> bool:
        if self.is_mirrored is not None:
            return self.is_mirrored

        if not isfile(join(self.storage_dir, name, "_is_mirrored.txt")):
            raise FileNotFoundError(
                f"is_mirrored not found for {name}", FileNotFoundError
            )
        else:
            is_mirrored = np.loadtxt(join(self.storage_dir, name, "_is_mirrored.txt"))
            return bool(is_mirrored[0])

    @ensure_directory_exists
    def get_anchor_point(self, name: str) -> np.ndarray:
        if self.anchor_point is not None:
            return self.anchor_point

        if not isfile(join(self.storage_dir, name, "_anchor_point.txt")):
            raise FileNotFoundError(
                f"No saved anchor_point found for {name}", FileNotFoundError
            )
        else:
            anchor_point = np.loadtxt(join(self.storage_dir, name, "_anchor_point.txt"))
            return anchor_point

    @ensure_directory_exists
    def save_is_mirrored(self, name: str, is_mirrored: bool):
        self.is_mirrored = is_mirrored
        np.savetxt(
            join(self.storage_dir, name, "_camera_matrix.txt"),
            np.array([is_mirrored], dtype=bool),
        )

    @ensure_directory_exists
    def save_anchor_point(self, name: str, anchor_point: np.ndarray):
        self.anchor_point = anchor_point
        np.savetxt(
            join(self.storage_dir, name, "_anchor_point.txt"),
            anchor_point,
        )
