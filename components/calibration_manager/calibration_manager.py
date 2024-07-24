from typing import Union, Optional, List
from os.path import isdir, isfile, join
from os import makedirs
import numpy as np

from camera.camera import ICalibrationDataManager
from camera.camera_utils import NotCalibratedError
from calibration_manager_utils import ensure_directory_exists


class CalibrationDataManager(ICalibrationDataManager):
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.camera_matrix = None
        self.optimized_camera_matrix = None
        self.distortion_matrix = None
        self.camera_resolution = None

        if not isdir(storage_dir):
            makedirs(storage_dir)

    @ensure_directory_exists
    def get_camera_matrix(self, name: str) -> np.ndarray:
        if self.camera_matrix is not None:
            return self.camera_matrix

        if not isfile(join(self.storage_dir, name, "_camera_matrix.txt")):
            raise FileNotFoundError("No saved camera-matrix found", FileNotFoundError)
        else:
            camera_matrix = np.loadtxt(
                join(self.storage_dir, name, "_camera_matrix.txt")
            )
            return camera_matrix

    @ensure_directory_exists
    def get_optimized_camera_matrix(self, name: str) -> np.ndarray:
        if self.optimized_camera_matrix is not None:
            return self.optimized_camera_matrix

        if not isfile(join(self.storage_dir, name, "_optimized_camera_matrix.txt")):
            raise FileNotFoundError(
                "No saved optimized_camera-matrix found", FileNotFoundError
            )
        else:
            optimized_camera_matrix = np.loadtxt(
                join(self.storage_dir, name, "_optimized_camera_matrix.txt")
            )
            return optimized_camera_matrix

    @ensure_directory_exists
    def get_distortion_matrix(self, name: str) -> np.ndarray:
        if self.distortion_matrix is not None:
            return self.distortion_matrix

        if not isfile(join(self.storage_dir, name, "_distortion_matrix.txt")):
            raise FileNotFoundError(
                "No saved distortion-matrix found", FileNotFoundError
            )
        else:
            distortion_matrix = np.loadtxt(
                join(self.storage_dir, name, "_distortion_matrix.txt")
            )
            return distortion_matrix

    @ensure_directory_exists
    def get_camera_resolution(self, name: str) -> np.ndarray:
        if self.camera_resolution is not None:
            return self.camera_resolution

        if not isfile(join(self.storage_dir, name, "_camera_resolution.txt")):
            raise FileNotFoundError("No saved camera_resolution found", FileNotFoundError)
        else:
            camera_resolution = np.loadtxt(
                join(self.storage_dir, name, "_camera_resolution.txt")
            )
            return camera_resolution

    @ensure_directory_exists
    def save_camera_matrix(self, name: str, camera_matrix: np.ndarray):
        self.camera_matrix = camera_matrix
        np.savetxt(join(self.storage_dir, name, "_camera_matrix.txt"), camera_matrix)

    @ensure_directory_exists
    def save_optimized_camera_matrix(
        self, name: str, optimized_camera_matrix: np.ndarray
    ):
        self.optimized_camera_matrix = optimized_camera_matrix
        np.savetxt(
            join(self.storage_dir, name, "_optimized_camera_matrix.txt"),
            optimized_camera_matrix,
        )

    @ensure_directory_exists
    def save_distortion_matrix(self, name: str, distortion_matrix: np.ndarray):
        self.distortion_matrix = distortion_matrix
        np.savetxt(
            join(self.storage_dir, name, "_distortion_matrix.txt"), distortion_matrix
        )

    @ensure_directory_exists
    def save_camera_resolution(self, name: str, camera_resolution: np.ndarray):
        self.camera_resolution = camera_resolution
        np.savetxt(join(self.storage_dir, name, "_camera_resolution.txt"), camera_resolution)

    def save_calibration_error(self, name: str, calibration_error: np.ndarray):
        np.savetxt(join(self.storage_dir, name, "_calibration_error.txt"), calibration_error)
