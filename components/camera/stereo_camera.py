from abc import ABC, abstractmethod
from typing import Union, List, Tuple
import cv2
import numpy as np
from ultralytics.data.loaders import Image

from camera.camera import Camera
from camera.camera_utils import CornerDetectionError, generate_objectpoints, CornersOrdererError, ImagePointsExtractionError, StereoCalibrationError


class IStereoCornerOrderManager(ABC):
    @abstractmethod
    def oder_corners(self, corners0: np.ndarray, corners1: np.ndarray, frame0: np.ndarray, frame1:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class IStereoCalibrationDataManager(ABC):
    @abstractmethod
    def get_


class StereoCamera():
    def __init__(self, camera0: Camera, camera1: Camera, corner_order_manager:IStereoCornerOrderManager):
        self.camera0 = camera0
        self.camera1 = camera1
        self.corner_order_manger = corner_order_manager

        self.stereo_calibration_error = None
        self.translation_matrix = None
        self.rotation_matrix = None

    def _get_stereo_frames(self, video_name: str, index: int) -> Tuple[np.ndarray, np.ndarray]:
        frame1 = self.camera1.get_frame(video_path=video_name, index=index)
        frame0 = self.camera0.get_frame(video_path=video_name, index=index)
        return frame0, frame1

    def _detect_stereo_corners(self, frame0: np.ndarray, frame1: np.ndarray, rows_inner: int, columns_inner: int) -> Tuple[np.ndarray, np.ndarray]:
        try:
            corners0 = self.camera0.corner_detector.detect_corners(image=frame0, rows_inner=rows_inner, columns_inner=columns_inner)
        except CornerDetectionError as e:
            raise CornerDetectionError("Corner detection failed", e)

        try:
            corners1 = self.camera1.corner_detector.detect_corners(image=frame1, rows_inner=rows_inner, columns_inner=columns_inner)
        except CornerDetectionError as e:
            raise CornerDetectionError("Corner detection failed", e)
        return corners0, corners1

    def _get_imagepoints(self, video_name: str, rows_inner: int, columns_inner: int):

        frame0, frame1 = self._get_stereo_frames(video_name, index=30)

        try:
            corners0, corners1 = self._detect_stereo_corners(frame0, frame1, rows_inner, columns_inner)
        except CornerDetectionError as e:
            raise ImagePointsExtractionError("Corner detection failed", e)

        try:
            corners0, corners1 = self.corner_order_manger.oder_corners(corners0, corners1, frame0, frame1)
        except CornersOrdererError as e:
            raise ImagePointsExtractionError("Corners ordering failed", e)

        imagepoint0 = corners0
        imagepoint1 = corners1

        return imagepoint0, imagepoint1

    def stereo_calibrate(self,
                         video_files: List[str],
                         rows_inner: int=7,
                         columns_inner: int=9,
                         edge_length: float=0.005,
                         undistort: bool=False,
                         stereocalibration_flags = cv2.CALIB_FIX_PRINCIPAL_POINT):
        assert isinstance(video_files, list), "video_files should be of type list"
        assert isinstance(rows_inner, int), "rows should be of type int"
        assert isinstance(columns_inner, int), "columns should be of type int"
        assert isinstance(edge_length, float), "edge_length should be of type float"
        assert isinstance(undistort, bool), "undistort should be of type bool"

        imgpoints0 = [] # Pixel Coordinates
        imgpoints1 = []
        objpoints = [] # World Coordinates
        objpoint_one_image = generate_objectpoints(rows_inner=rows_inner, columns_inner=columns_inner, edge_length=edge_length)

        for video_name in video_files:
            try:
                imagepoint0, imagepoint1 = self._get_imagepoints(video_name, rows_inner, columns_inner)
            except ImagePointsExtractionError as e:
                print("Image point extraction failed")
                continue
            imgpoints0.append(imagepoint0)
            imgpoints1.append(imagepoint1)
            objpoints.append(objpoint_one_image)

        if objpoints == [] or imgpoints0 == [] or imgpoints1:
            raise StereoCalibrationError("Stereo calibration failed, no imagepoints were extracted, or no object_points were generated", StereoCalibrationError)

        example_frame = self.camera0.get_frame(video_files[0], index=30)
        height, width = example_frame.shape[:2]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
        stereo_calibration_error, camera_matrix0, distortion_matrix0, camera_matrix1, distortion_matrix1, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                      imgpoints0,
                                                                      imgpoints1,
                                                                      self.camera0.get_optimized_camera_matrix(),
                                                                      self.camera0.get_distortion_matrix(),
                                                                      self.camera1.get_optimized_camera_matrix(),
                                                                      self.camera1.get_distortion_matrix()
                                                                      (width, height),
                                                                      criteria=criteria,
                                                                      flags=stereocalibration_flags)

        optimized_camera_matrix0, roi = cv2.getOptimalNewCameraMatrix(camera_matrix0, distortion_matrix0, (width, height), 1,
                                                                   (width, height))
        optimized_camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(camera_matrix1, distortion_matrix1, (width, height), 1,
                                                                   (width, height))
        print(f'Stereo-calibration error: {stereo_calibration_error}')
        print(f'Translation Matrix: {T}')
        print(f'Rotation Matrix: {R}')

        self.

        return
