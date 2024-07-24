from typing import Union
from os.path import isdir, isfile, join
from os import makedirs
import numpy as np

from camera import IStereoCalibrationDataManager, NotCalibratedError
from calibration_manager_utils import ensure_directory_exists



class StereoCalibrationDataManager(IStereoCalibrationDataManager):
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir

        self.translation_matrix = None
        self.rotation_matrix = None

        if not isdir(storage_dir):
            makedirs(storage_dir)

    @ensure_directory_exists
    def get_rotation_matrix(self, name: str) -> np.ndarray:
        if self.rotation_matrix is not None:
            return self.rotation_matrix

        if not isfile(join(self.storage_dir, name, "_rotation_matrix.txt")):
            raise NotCalibratedError("No saved rotation-matrix found", NotCalibratedError)
        else:
            rotation_matrix = np.loadtxt(
                join(self.storage_dir, name, "_rotation_matrix.txt")
            )
            return rotation_matrix

    @ensure_directory_exists
    def get_translation_matrix(self, name: str) -> np.ndarray:
        if self.translation_matrix is not None:
            return self.translation_matrix

        if not isfile(join(self.storage_dir, name, "_translation_matrix.txt")):
            raise NotCalibratedError(
                "No saved translation_matrix found", NotCalibratedError
            )
        else:
            translation_matrix = np.loadtxt(
                join(self.storage_dir, name, "_translation_matrix.txt")
            )
            return translation_matrix

    @ensure_directory_exists
    def save_rotation_matrix(self, name: str, rotation_matrix: np.ndarray):
        self.rotation_matrix = rotation_matrix
        np.savetxt(join(self.storage_dir, name, "_rotation_matrix.txt"), rotation_matrix)

    @ensure_directory_exists
    def save_translation_matrix(
        self, name: str, translation_matrix: np.ndarray
    ):
        self.translation_matrix = translation_matrix
        np.savetxt(
            join(self.storage_dir, name, "_translation_matrix.txt"),
            translation_matrix,
        )

    @ensure_directory_exists
    def save_stereo_calibration_error(self, name: str, stereo_calibration_error: np.ndarray):
        self.stereo_calibration_error = stereo_calibration_error
        np.savetxt(
            join(self.storage_dir, name, "_stereo_calibration_error.txt"), stereo_calibration_error
        )
