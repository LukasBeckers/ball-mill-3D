from camera_utils import corner_detection, stereoCamera
from triangulation_utils import tiangulate


class Validator(stereoCamera):
    """
    Validates the stereo-calibration based on stereo-calibration checkerboard patterns. 
    """
    def __init__(self):
        super(stereoCamera).__init__()


    def add_calibration_frame(frame):
        "For reliable validation results only add frames from calibration videos NOT used in the stereo-calibration!"

        frame1, frame2 = self(frame)
    
    

