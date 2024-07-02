import cv2
import os
from ultralytics import YOLO

from utils.camera_utils import *


class stickerDetector():
    """
    Refines the sticker-detection results of the trained YOLOV8
    detector
    """

    def __init__(self, n_stickers, max_dist=10):
        self.ROIS = {}
        self.max_dist = max_dist
        self.first_frame = True
        self.n_stickers = n_stickers

    def __call__(self, detection_results, mirror=False):
        """
        detection_results = YOLOV8 results from detector trained

        to detect the red stickers.
        In contrast to the ballDetector, the stickerDetector uses no warmup-steps, because sticker detection tends to be very solid.
        Make sure that all stickers and only the stickers are detected in the first frame.
        """

        detection_results = detection_results[0]  # We only use single frame detection.
        boxes = detection_results.boxes.xyxy.tolist()
        classes = detection_results.boxes.cls

        coords = []
        for i, (cl, box) in enumerate(zip(classes, boxes)):
            coord = None
            if cl == 1:  # 0 = Ball, 1 = Red Sticker
                coord = np.array([box[0] + box[2], box[1] + box[3]]) / 2  # Box = xyxy creating average coord

            if coord is not None:
                coords.append(coord)

        if self.first_frame:
            if len(coords) == self.n_stickers:
                idxs = np.argsort([c[0] for c in coords])
                if mirror:
                    idxs = idxs[::-1]
                self.ROIS = {j: coords[i] for j, i in enumerate(idxs)}
                self.first_frame = False
                return None
            else:
                print(
                    f"First frame detection had {len(coords)} stickers detected, the real number of stickers should be {self.n_stickers}, skipping!")
                return None
        else:
            results = {}
            # choosing the coord for each roi, that is closest and checking if dist is over max dist
            for i, roi in self.ROIS.items():
                min_dist = 10E20
                candiate = None
                taken_coords = []
                for j, coord in enumerate(coords):
                    dist = np.sqrt((coord[0] - roi[0]) ** 2 + (coord[1] - roi[1]) ** 2)
                    if dist < min_dist:
                        if not j in taken_coords:  # Prevent double assignments
                            min_dist = dist
                            candidate = coord

                if min_dist > self.max_dist:
                    print("Over max dist in sticker detection!")
                    return None
                else:
                    results[i] = candidate
                    taken_coords.append(j)

        if len(results) == self.n_stickers:
            self.ROIS.update(results)
            return results

        print("Not enough stickers detected!")
        return None


class ballDetector():
    """
    Refines the ball-detection results of the trained YOLOV8
    detector
    """
    def __init__(self, max_dist=40, warmup_steps=10):
        self.ROI = None
        self.max_dist = max_dist
        self.warmup_regions = []
        self.warmup_steps = warmup_steps
        self.current_warmup_step = 0

    def _warmup(self, detection_results):
        """
        This function can be used to automatically determine the starting
        regions of intrest for the ball/balls.
        It works by removing outliers from the first self.warmup_steps
        detections and using the last valid detection as starting ROI.
        """
        detection_results = detection_results[0]  # We only use single frame detection.
        boxes = detection_results.boxes.xyxy.tolist()
        classes = detection_results.boxes.cls

        for cl, box in zip(classes, boxes):
            if  cl == 0:  # 0 = Ball, 1 = Red Sticker
                coord = np.array([box[0] + box[2], box[1] + box[3] ])/2 #Box = xyxy creating average coord
                self.warmup_regions.append(coord)
                self.current_warmup_step += 1
        # Maybe needs improvement if detection is bad!!
        # location gathering compleated, calculating starting position
        if self.current_warmup_step == self.warmup_steps:
            
            diffs = [[x2 - x1, y2 - y1] for [[x1, y1], [x2, y2]] in zip(self.warmup_regions[:-1], self.warmup_regions[1:])]
            dists = [np.sqrt(x**2 + y **2) for x, y in diffs]
            # removing all measurements that have an exessive distance to the previous measurement
            # because they are probably errors
            for i, dist in reversed(list(enumerate(dists))):
                if dist > self.max_dist:
                    self.warmup_regions.pop(i)

            self.ROI = self.warmup_regions[-1]
        return None


    def __call__(self, detection_results):
        """
        detection_results = YOLOV8 results from detector trained
        
        to detect the ball.
        """
        
        if self. current_warmup_step <= self.warmup_steps:
            print("Warming up", self.current_warmup_step)
            return self._warmup(detection_results)
        else:
            detection_results = detection_results[0]  # We only use single frame detection.
            boxes = detection_results.boxes.xyxy.tolist()
            classes = detection_results.boxes.cls
    
            for cl, box in zip(classes, boxes):
                coord = None
                if  cl == 0:  # 0 = Ball, 1 = Red Sticker
                    coord = np.array([box[0] + box[2], box[1] + box[3] ])/2 #Box = xyxy creating average coord
                if coord is not None:
                    # checking if the traveled distance is plausible
                    dist = np.sqrt((coord[0]-self.ROI[0])**2 + (coord[1] - self.ROI[1])**2)
                    if dist > self.max_dist:
                        self.max_dist += 10
                        print("Over max dist!!!")
                        return None
                    else:
                        self.max_dist = 20
                        self.ROI = coord
                        return coord
            
            print("Coord was None!!!", coord, classes, boxes)
            return None


if __name__=="__main__":
    pass




