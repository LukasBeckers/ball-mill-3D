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

        if not isdir(storage_dir):
           makedirs(storage_dir)

    @ensure_directory_exists
    def get_camera_matrix(self, name: str) -> np.ndarray:
        if not isfile(join(self.storage_dir, name, "_camera_matrix.txt")):
            raise NotCalibratedError("No saved camera-matrix found", NotCalibratedError)
        else:
            camera_matrix = np.loadtxt(join(self.storage_dir, name, "_camera_matrix.txt"))
            return camera_matrix

    @ensure_directory_exists
    def get_optimized_camera_matrix(self, name: str) -> np.ndarray:
        if not isfile(join(self.storage_dir, name, "_optimized_camera_matrix.txt")):
            raise NotCalibratedError("No saved optimized_camera-matrix found", NotCalibratedError)
        else:
            optimized_camera_matrix = np.loadtxt(join(self.storage_dir, name, "_optimized_camera_matrix.txt"))
            return optimized_camera_matrix

    @ensure_directory_exists
    def get_distortion_matrix(self, name: str) -> np.ndarray:
        if not isfile(join(self.storage_dir, name, "_distortion_matrix.txt")):
            raise NotCalibratedError("No saved distortion-matrix found", NotCalibratedError)
        else:
            distortion_matrix = np.loadtxt(join(self.storage_dir, name, "_distortion_matrix.txt"))
            return distortion_matrix

    @ensure_directory_exists
    def save_camera_matrix(self, name:str, camera_matrix: np.ndarray):
        np.savetxt(join(self.storage_dir, name, "_camera_matrix.txt"), camera_matrix)

    @ensure_directory_exists
    def save_optimized_camera_matrix(self, name:str, optimized_camera_matrix: np.ndarray):
        np.savetxt(join(self.storage_dir, name, "_optimized_camera_matrix.txt"), optimized_camera_matrix)

    @ensure_directory_exists
    def save_distortion_matrix(self, name:str, distortion_matrix: np.ndarray):
        np.savetxt(join(self.storage_dir, name, "_distortion_matrix.txt"), distortion_matrix)
