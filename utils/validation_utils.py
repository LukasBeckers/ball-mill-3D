from numpy._typing import _16Bit
from utils.camera_utils import corner_detection, stereoCamera, show_and_switch
from utils.triangulation_utils import triangulate
from utils.detection_utils import Detector, draw_detections
from utils.video_utils import videoLoader
import numpy as np
import plotly.graph_objects as go
import cv2
from itertools import combinations


class chessboardValidator(stereoCamera):
    def __init__(self):
        super(stereoCamera).__init__()
        self.frames0 = []
        self.frames1 = []
        self.corners0 = []
        self.corners1 = []
        self.points = []
        self.rows_cols = []

    def _reset(self, full=False):
        if full:
            self.frames0 = []
            self.frames1 = []
            self.rows_cols = []
        self.corners0 = []
        self.corners1 = []
        self.points = []

    def add_calibration_frame(self, frame, rows, columns):
        "For reliable validation results only add frames from calibration videos NOT used in the stereo-calibration!"

        frame0, frame1 = self(frame)

        self.frames0.append(frame0)
        self.frames1.append(frame1)
        self.rows_cols.append([rows, columns])

    def _detect_corners(self, frame0, frame1, rows, columns, image_scaling=2):
        corners0 = corner_detection(image_set=[frame0], image_scaling=image_scaling, cam=0, rows_inner=rows - 1, columns_inner = columns - 1, fallback_manual=True)
        if corners0 is None:
            self.corners0.append(None)
            self.corners1.append(None)
            return
        corners1 = corner_detection(image_set=[frame1], image_scaling=image_scaling, cam=1, rows_inner=rows - 1, columns_inner = columns - 1, fallback_manual=True)
        if corners1 is None:
            self.corners0.append(None)
            self.corners1.append(None)
            return
        else:
            ret, corners0, corners1 = show_and_switch(img0=frame0,
                                      img1=frame1,
                                      corners0=corners0[0],
                                      corners1=corners1[0],
                                      rows_inner=rows - 1,
                                      columns_inner=columns - 1,
                                      image_scaling=image_scaling)

            if ret:
                self.corners0.append(corners0)
                self.corners1.append(corners1)
                return

            else:
                self.corners0.append(None)
                self.corners1.append(None)
                return

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

    def _calculate_distances(self, rows, columns):
        rows_inner = rows - 1
        columns_inner = columns - 1

        self.column_distances = []
        self.row_distances = []
        self.width_distances = []
        self.height_distances = []
        self.diagonal_distances = []

        # Calculate the "column" distances
        for frame_points in self.points:
            if frame_points is None:
                self.column_distances.append([])
                self.row_distances.append([])
                self.width_distances.append([])
                self.height_distances.append([])
                self.diagonal_distances.append([])
                continue

            col_dist = []
            for lower_index in range(0, len(frame_points), rows_inner):
                upper_index = lower_index + rows_inner
                col_points = frame_points[lower_index: upper_index]
                for point0, point1 in zip(col_points[:-1], col_points[1:]):
                    col_dist.append(np.linalg.norm(point1 - point0))
            self.column_distances.append(col_dist)

            # Calculate the "row" distances
            row_dist = []
            for start_row in range(rows_inner):
                row_points = frame_points[start_row::rows_inner]
                for point0, point1 in zip(row_points[:-1], row_points[1:]):
                    row_dist.append(np.linalg.norm(point1 - point0))
            self.row_distances.append(row_dist)

            # Calculate "width" distances
            width_dist = []
            for start_row in range(rows_inner):
                width_dist.append(np.linalg.norm(frame_points[start_row] - frame_points[start_row + (rows_inner * (columns_inner-1))]))
            self.width_distances.append(width_dist)

            # Calculate "height" distances
            height_dist = []
            for lower_index in range(0, len(frame_points), rows_inner):
                upper_index = lower_index + rows_inner
                col_points = frame_points[lower_index: upper_index]
                height_dist.append(np.linalg.norm(col_points[-1] - col_points[0]))
            self.height_distances.append(height_dist)

            # Calculate diagonal distances
            diag_dist = []
            diag_dist.append(np.linalg.norm(frame_points[-1] - frame_points[0]))
            diag_dist.append(np.linalg.norm(frame_points[rows_inner - 1] - frame_points[-rows_inner]))
            self.diagonal_distances.append(diag_dist)

    def _visualize(self):
        fig = go.Figure()

        for frame_points in self.points:
            if frame_points is None:
                continue

            coords = np.array(frame_points)

            connections = [[a, b] for a, b in zip(range(len(coords)-1), range(1, len(coords)))]

            x_range = [coords[:, 0].min(), coords[:, 0].max()]
            y_range = [coords[:, 1].min(), coords[:, 1].max()]
            z_range = [coords[:, 2].min(), coords[:, 2].max()]

            max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])

            x_mid = sum(x_range) / 2
            y_mid = sum(y_range) / 2
            z_mid = sum(z_range) / 2

            x_range = [x_mid - max_range / 2, x_mid + max_range / 2]
            y_range = [y_mid - max_range / 2, y_mid + max_range / 2]
            z_range = [z_mid - max_range / 2, z_mid + max_range / 2]

            for conn in connections:
                fig.add_trace(go.Scatter3d(
                    x=[coords[conn[0], 0], coords[conn[1], 0]],
                    y=[coords[conn[0], 1], coords[conn[1], 1]],
                    z=[coords[conn[0], 2], coords[conn[1], 2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Line'
                ))

            fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=1, color='red'),
            name='Points'
            ))

            fig.update_layout(
            title="3D Line Plot",
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate',
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            )
            )

        fig.show()

    def validate(self, image_scaling=2, new_frames=None):
        """
        Validate the calibration by detecting corners, triangulating points, and calculating distances.

        Args:
            new_frames (list, optional): List of new frames to be added for validation.
                                         Each frame should be a tuple containing (frame, rows, columns).
        """

        # Optionally add new frames
        if new_frames:
            for frame, rows, columns in new_frames:
                self.add_calibration_frame(frame, rows, columns)

        n_points = len(self.points)
        if n_points < len(self.rows_cols): # Checking if some frames are not labeled yet.
            for frame0, frame1, (rows, columns) in zip(self.frames0[n_points:], self.frames1[n_points:], self.rows_cols[n_points:]):
                self._detect_corners(frame0, frame1, rows, columns, image_scaling=image_scaling)

        self._triangulate()

        for rows, columns in self.rows_cols:
            self._calculate_distances(rows, columns)

        # Compute average lengths and standard deviation for each frame
        for i, (rows, columns) in enumerate(self.rows_cols):
            frame_points = self.points[i]
            if frame_points is None:
                print(f"Frame {i}: No valid points detected.")
                continue

            column_distances = self.column_distances[i]
            row_distances = self.row_distances[i]
            width_distances = self.width_distances[i]
            height_distances = self.height_distances[i]
            diagonal_distances = self.diagonal_distances[i]

            avg_col = np.mean(column_distances)
            std_col = np.std(column_distances)

            avg_row = np.mean(row_distances)
            std_row = np.std(row_distances)

            avg_width = np.mean(width_distances)
            std_width = np.std(width_distances)

            avg_height = np.mean(height_distances)
            std_height = np.std(height_distances)

            avg_diag = np.mean(diagonal_distances)
            std_diag = np.std(diagonal_distances)

            print(f"Frame {i}:")
            print(f"  Average Column Distance: {avg_col*100:.4f} cm (std: {std_col*100:.4f})")
            print(f"  Average Row Distance: {avg_row*100:.4f} cm (std: {std_row*100:.4f})")
            print(f"  Average Width Distance: {avg_width*100:.4f} cm (std: {std_width*100:.4f})")
            print(f"  Average Height Distance: {avg_height*100:.4f} cm (std: {std_height*100:.4f})")
            print(f"  Average Diagonal Distance: {avg_diag*100:.4f} cm (std: {std_diag*100:.4f})\n")

        # Visualize results
        self._visualize()


class stickerValidator(videoLoader, stereoCamera):
    """
    Calculates the distances between the stickers in a stereo-video.
    Measure these distances in the real object and compare to validate the stereo-vision results.
    """

    def __init__(self,
                stereo_config_name:str,
                model_name:str,
                n_stickers:int,
                max_dist_sticker:int=10,
                max_dist_ball:int=40,
                warmup_steps_ball:int=10):

        videoLoader.__init__(self)
        stereoCamera.__init__(self, name="stickerValidatorStereoCamera")

        self.detector0 = Detector(model_name=model_name, n_stickers=n_stickers, max_dist_sticker=max_dist_sticker,
            max_dist_ball=max_dist_ball, warmup_steps_ball=warmup_steps_ball)
        self.detector1 = Detector(model_name=model_name, n_stickers=n_stickers, max_dist_sticker=max_dist_sticker,
            max_dist_ball=max_dist_ball, warmup_steps_ball=warmup_steps_ball)

        self.load_from_yaml(stereo_config_name)
        self.n_stickers = n_stickers

    def validate_on(self, video_path:str):
        self.load_video(video_path=video_path)
        self._load_frames()
        self.acc_sticker_coords0 = {i: [] for i in range(self.n_stickers)}
        self.acc_sticker_coords1 = {i: [] for i in range(self.n_stickers)}
        for frame in self.frames[30:-30]:
            frame0, frame1 = self(frame)
            detection_results0 = self.detector0(frame0, mirror=True, return_yolo_output=True)
            detection_results1 = self.detector1(frame1, mirror=False, return_yolo_output=True)

            if detection_results0[1] is not None and detection_results1[1] is not None:
                frame0 = draw_detections(img=frame0, detection_results=detection_results0[2], stickercoords=detection_results0[1])
                frame1 = draw_detections(img=frame1, detection_results=detection_results1[2], stickercoords=detection_results1[1])

                cv2.imshow("Detection Results 0", frame0)
                cv2.imshow("Detection Results 1", frame1)
                cv2.waitKey(1)

                for key, coords in detection_results0[1].items():
                                self.acc_sticker_coords0[key].append(coords)

                for key, coords in detection_results1[1].items():
                                self.acc_sticker_coords1[key].append(coords)

        cv2.destroyAllWindows()
        self._triangulate()
        self._calculate_average_distances()
        self._print_distance_matrix()
        self.visualize_example_detection()
        self._print_distance_matrix()

    def _triangulate(self):
        self.sticker3dpoints = {}

        for i in range(self.n_stickers):
            coords0 = self.acc_sticker_coords0[i]
            coords1 = self.acc_sticker_coords1[i]

            points3d = [triangulate(self, coord0, coord1) for coord0, coord1 in zip(coords0, coords1)]
            self.sticker3dpoints[i] = points3d

    def _calculate_average_distances(self):
        self.avg_distances = np.zeros((self.n_stickers, self.n_stickers))

        for (i, j) in combinations(range(self.n_stickers), 2):
            distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(self.sticker3dpoints[i], self.sticker3dpoints[j])]
            average_distance = np.mean(distances)
            self.avg_distances[i, j] = self.avg_distances[j, i] = average_distance

    def _print_distance_matrix(self):
        print("Average Distance Matrix (in cm):")
        print("    " + "  ".join([f"Sticker {i}" for i in range(self.n_stickers)]))
        for i in range(self.n_stickers):
            row = [f"{self.avg_distances[i, j] * 100:.2f}" for j in range(self.n_stickers)]
            print(f"Sticker {i}  " + "  ".join(row))

    def visualize_example_detection(self):
        i = 30
        while self.frames is not None:
            frame0, frame1 = self(self.frames[i])  # Example frame
            detection_results0 = self.detector0(frame0, mirror=True, return_yolo_output=True)
            detection_results1 = self.detector1(frame1, mirror=False, return_yolo_output=True)

            if detection_results0[1] is not None and detection_results1[1] is not None:
                frame0 = draw_detections(img=frame0, detection_results=detection_results0[2], stickercoords=detection_results0[1])
                frame1 = draw_detections(img=frame1, detection_results=detection_results1[2], stickercoords=detection_results1[1])

                cv2.imshow("Example Detection 0", frame0)
                cv2.imshow("Example Detection 1", frame1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

            else:
                i += 1


class manualValidator(videoLoader, stereoCamera):
    """
    Allows users to mark two points on frame0 and frame1, triangulates these points, and displays the distance between the triangulated points.
    """

    def __init__(self, stereo_config_name: str):
        videoLoader.__init__(self)
        stereoCamera.__init__(self, name="ValidatorStereoCamera")

        self.load_from_yaml(stereo_config_name)

    def validate_on(self, video_path: str, frame_idx: int = 100):

        self.point0_frame0 = None
        self.point0_frame1 = None
        self.point1_frame0 = None
        self.point1_frame1 = None

        self.load_video(video_path=video_path)
        self._load_frames()

        cv2.namedWindow("Frame 0")
        cv2.setMouseCallback("Frame 0", self._on_mouse_click_frame0)

        cv2.namedWindow("Frame 1")
        cv2.setMouseCallback("Frame 1", self._on_mouse_click_frame1)

        self._display_frames(frame_idx)

        if (self.point0_frame0 is not None and self.point0_frame1 is not None and
            self.point1_frame0 is not None and self.point1_frame1 is not None):
            self._triangulate_and_display_distance()

        cv2.destroyAllWindows()

    def _display_frames(self, frame_idx):
        self.frame0, self.frame1 = self(self.frames[frame_idx])
        while True:
            display_frame0 = self.frame0.copy()
            display_frame1 = self.frame1.copy()

            if self.point0_frame0 is not None:
                cv2.circle(display_frame0, self.point0_frame0, 1, (0, 0, 255), -1)
            if self.point1_frame0 is not None:
                cv2.circle(display_frame0, self.point1_frame0, 1, (0, 255, 0), -1)

            if self.point0_frame1 is not None:
                cv2.circle(display_frame1, self.point0_frame1, 1, (0, 0, 255), -1)
            if self.point1_frame1 is not None:
                cv2.circle(display_frame1, self.point1_frame1, 1, (0, 255, 0), -1)

            cv2.imshow("Frame 0", display_frame0)
            cv2.imshow("Frame 1", display_frame1)

            key = cv2.waitKey(1)
            if key == 27:  # Escape key
                break

    def _on_mouse_click_frame0(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.point0_frame0 is None:
                self.point0_frame0 = (x, y)
                print(f"Point 0 on Frame 0: {self.point0_frame0}")
            elif self.point1_frame0 is None:
                self.point1_frame0 = (x, y)
                print(f"Point 1 on Frame 0: {self.point1_frame0}")

    def _on_mouse_click_frame1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.point0_frame1 is None:
                self.point0_frame1 = (x, y)
                print(f"Point 0 on Frame 1: {self.point0_frame1}")
            elif self.point1_frame1 is None:
                self.point1_frame1 = (x, y)
                print(f"Point 1 on Frame 1: {self.point1_frame1}")

    def _triangulate_and_display_distance(self):
        point0_3d = triangulate(self, self.point0_frame0, self.point0_frame1)
        point1_3d = triangulate(self, self.point1_frame0, self.point1_frame1)

        print(f"Triangulated 3D point 0: {point0_3d}")
        print(f"Triangulated 3D point 1: {point1_3d}")

        distance = np.linalg.norm(point0_3d - point1_3d)
        print(f"Distance between points: {distance * 100:.2f} cm")
