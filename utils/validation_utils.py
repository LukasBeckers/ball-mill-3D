from utils.camera_utils import corner_detection, stereoCamera
from utils.triangulation_utils import triangulate


class Validator(stereoCamera):

    def __init__(self):
        super(stereoCamera).__init__()
        self.frames1 = []
        self.frames2 = []
        self.corners1 = []
        self.corners2 = []
        self.points = []

    def add_calibration_frame(self, frame):
        "For reliable validation results only add frames from calibration videos NOT used in the stereo-calibration!"

        frame1, frame2 = self(frame)

        self.frames1.append(frame1)
        self.frames2.append(frame2)
        
    def _detect_corners(self, rows, columns, image_scaling=2):
        for frame1, frame2 in zip(self.frames1, self.frames2):

            corners1 = corner_detection(image_set=[frame1], image_scaling=image_scaling, cam=0, rows_inner=rows - 1, columns_inner = columns - 1, fallback_manual=True)
            corners2 = corner_detection(image_set=[frame2], image_scaling=image_scaling, cam=1, rows_inner=rows - 1, columns_inner = columns - 1, fallback_manual=True)
            
            self.corners1.extend(corners1 if corners1 is not None else False)
            self.corners2.extend(corners2 if corners2 is not None else False)

    def _triangulate(self):
        for corners1, corners2 in zip(self.corners1, self.corners2):
            if not corners1 or not corners2:
                self.points.append(False)
                continue

            frame_points = []
            for p1, p2 in zip(corners1, corners2):
                point = triangulate(self, p1, p2)
                frame_points.append(point)

            self.points.append(frame_points)

    def _calculate_distances(self):
        pass

            
            





    
    

