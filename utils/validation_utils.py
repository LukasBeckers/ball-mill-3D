from utils.camera_utils import corner_detection, stereoCamera, show_and_switch
from utils.triangulation_utils import triangulate


class Validator(stereoCamera):

    def __init__(self):
        super(stereoCamera).__init__()
        self.frames0 = []
        self.frames1 = []
        self.corners0 = []
        self.corners1 = []
        self.points = []

    def add_calibration_frame(self, frame):
        "For reliable validation results only add frames from calibration videos NOT used in the stereo-calibration!"

        frame0, frame1 = self(frame)

        self.frames0.append(frame0)
        self.frames1.append(frame1)
        
    def _detect_corners(self, rows, columns, image_scaling=2):
        for frame0, frame1 in zip(self.frames0, self.frames1):

            corners0 = corner_detection(image_set=[frame0], image_scaling=image_scaling, cam=0, rows_inner=rows - 1, columns_inner = columns - 1, fallback_manual=True)
            corners1 = corner_detection(image_set=[frame1], image_scaling=image_scaling, cam=1, rows_inner=rows - 1, columns_inner = columns - 1, fallback_manual=True)
            if corners0 is None or corners1 is None:
                self.corners0.append(None)
                self.corners1.append(None)
                continue
            else:
                corners0, corners1 = show_and_switch(img0=frame0,
                                                       img1=frame1, 
                                                       corners0=corners0[0],
                                                       corners1=corners1[0], 
                                                       rows_inner=rows - 1, 
                                                       columns_inner=columns - 1, 
                                                       image_scaling=image_scaling)
                self.corners0.append(corners0)
                self.corners1.append(corners1)


    def _triangulate(self):
        for corners0, corners1 in zip(self.corners0, self.corners1):
            if corners0 is None or corners1 is None:
                self.points.append(None)
                continue

            frame_points = []
            for p0, p1 in zip(corners0, corners1):
                point = triangulate(self, p0, p1)
                frame_points.append(point)

            self.points.append(frame_points)

    def _calculate_distances(self):
        pass

        """# Calculate distances between consecutive 3D points for the smaller subset
distances_smaller_subset = [np.linalg.norm(points_smaller_subset[i+1] - points_smaller_subset[i]) for i in range(len(points_smaller_subset) - 1)]
np.array([x for x in distances_smaller_subset if x < 0.01]).mean(), "Distance start - end (diagonaly) =", np.linalg.norm(coordinates[5]- coordinates[-6]), coordinates[0], coordinates[-1]"""


            
            





    
    

