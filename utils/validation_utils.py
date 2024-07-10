from utils.camera_utils import corner_detection, stereoCamera, show_and_switch
from utils.triangulation_utils import triangulate
import numpy as np 
import plotly.graph_objects as go 


class Validator(stereoCamera):

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
            corners0, corners1 = show_and_switch(img0=frame0,
                                                   img1=frame1, 
                                                   corners0=corners0[0],
                                                   corners1=corners1[0], 
                                                   rows_inner=rows - 1, 
                                                   columns_inner=columns - 1, 
                                                   image_scaling=image_scaling)
            self.corners0.append(corners0)
            self.corners1.append(corners1)
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

    
    





