from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, List
import numpy as np
import cv2

from components.calibration_manager import calibration_manager

from .camera_utils import NotCalibratedError, generate_objectpoints, CornerDetectionError, CalibrationError


class ICalibrationDataManager(ABC):
    @abstractmethod
    def get_camera_matrix(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_distortion_matrix(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_optimized_camera_matrix(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def save_camera_matrix(self, name: str, camera_matrix: np.ndarray):
        pass

    @abstractmethod
    def save_calibration_error(self, name: str, calibration_error: np.ndarray):
        pass

    @abstractmethod
    def save_distortion_matrix(self, name: str, distortion_matrix: np.ndarray):
        pass

    @abstractmethod
    def save_optimized_camera_matrix(self, name: str, optimized_camera_matrix: np.ndarray):
        pass


class IFrameProvider(ABC):
    @abstractmethod
    def get_frame(self, video_path: str, i: int) -> np.ndarray:
        pass


class ICornerDetector(ABC):
    @abstractmethod
    def detect_corners(self, image: np.ndarray, rows_inner: int, columns_inner: int) -> np.ndarray:
        pass


class Camera():
    def __init__(self, name: str, calibration_manager: ICalibrationDataManager, frame_provider: IFrameProvider, corner_detector: ICornerDetector):
        assert isinstance(name, str), f"name must be str, is {type(name)}"
        assert isinstance(calibration_manager, ICalibrationDataManager), f"calibration_manager must be {ICalibrationDataManager}, is {type(calibration_manager)}"
        assert isinstance(frame_provider, IFrameProvider), f"frame_provide must be IFrameProvider, is {type(frame_provider)}"
        self.name = name
        self.calibration_manager = calibration_manager
        self.frame_provider = frame_provider
        self.corner_detector = corner_detector

    def get_camera_matrix(self):
        try:
            camera_matrix = self.calibration_manager.get_camera_matrix(self.name)
        except NotCalibratedError as e:
           raise NotCalibratedError(f"Camera {self.name} not jet calibrated", e)
        return camera_matrix

    def get_optimized_camera_matrix(self):
        try:
            optimized_camera_matrix = self.calibration_manager.get_optimized_camera_matrix(self.name)
        except NotCalibratedError as e:
            raise NotCalibratedError(f"Camera {self.name} not jet calibrated", e)
        return optimized_camera_matrix

    def get_distortion_matrix(self):
        try:
            distortion_matrix = self.calibration_manager.get_distortion_matrix(self.name)
        except NotCalibratedError as e:
            raise NotCalibratedError(f"Camera {self.name} not jet calibrated", e)
        return distortion_matrix

    def get_frame(self, video_path: str, index: int) -> np.ndarray:
        try:
            frame = self.frame_provider.get_frame(video_path=video_path, i=index)
        except IndexError as error:
            raise IndexError ("Error while loading video_frames", error)

        return frame

    def calibrate(self,
                    videos: List[str],
                    rows_inner: int=7,
                    columns_inner: int=9,
                    edge_length: float=0.005,
                    image_scaling: Union[float, int]=2,
                    ):
        assert isinstance(videos, list), "videos should be of type list"
        assert isinstance(rows_inner, int), "rows should be of type int"
        assert isinstance(columns_inner, int), "columns should be of type int"
        assert isinstance(edge_length, float), "edge_length should be of type float"
        assert isinstance(image_scaling, (float, int)), "image_scaling should be of type float"

        imgpoints = [] # Pixel coorinates in the image
        objpoints = [] # Defined coordinates in real space

        objpoint_one_image = generate_objectpoints(rows_inner=rows_inner, columns_inner=columns_inner, edge_length=edge_length)

        for video in videos:
            try:
                index = 30
                video_frame = self.frame_provider.get_frame(video_path=video, i=index)
            except IndexError as error:
                raise IndexError ("Error while loading video_frames", error)

            try:
                corners = self.corner_detector.detect_corners(image=video_frame, rows_inner=rows_inner, columns_inner=columns_inner)
            except CornerDetectionError as e:
                continue


            imgpoints.append(corners)
            objpoints.append(objpoint_one_image) # because corners were created by a set of similar images

        if imgpoints == [] or objpoints == []:
            raise CalibrationError("Calibration failed because no corners were detected", CalibrationError)


        video_frame = self.frame_provider.get_frame(video_path=videos[0], i=0)
        width = video_frame.shape[1]
        height = video_frame.shape[0]

        calibration_error, camera_matrix, distortion_matrix, rvecs, tvecs = cv2.calibrateCamera(np.array(objpoints), np.array(imgpoints), (width, height), None, None)
        optimized_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_matrix, (width, height), 1, (width, height))

        print('rmse:', calibration_error)
        print('camera matrix:\n', camera_matrix)
        print('optimized camera matrix:\n', optimized_camera_matrix)
        print('distortion coeffs:\n', distortion_matrix)

        self.calibration_manager.save_calibration_error(self.name, calibration_error)
        self.calibration_manager.save_camera_matrix(self.name, camera_matrix)
        self.calibration_manager.save_optimized_camera_matrix(self.name, optimized_camera_matrix)
        self.calibration_manager.save_distortion_matrix(self.name, distortion_matrix)

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        assert isinstance(img, np.ndarray), f"img must be of type np.ndarray, is {type(img)}."

        img = cv2.undistort(img,
                            self.get_camera_matrix(),
                            self.get_distortion_matrix(),
                            None,
                            self.get_optimized_camera_matrix())
        return np.array(img)
