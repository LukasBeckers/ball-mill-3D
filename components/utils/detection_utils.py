import cv2
import numpy as np
import os
from ultralytics import YOLO


def draw_detections(img: np.ndarray, detection_results, stickercoords=None):
    detection_results = detection_results[0]
    boxes = detection_results.boxes.xyxy.tolist()
    classes = detection_results.boxes.cls
    for cl, box in zip(classes, boxes):
        if  cl == 0:  # 0 = Ball, 1 = Red Sticker
            img = cv2.rectangle(img, np.array(box[:2], dtype=int), np.array(box[2:], dtype=int), [200, 200, 200], 4)
        if cl == 1:
            img = cv2.rectangle(img, np.array(box[:2], dtype=int), np.array(box[2:], dtype=int), [0, 0, 200], 4)

    if stickercoords is not None:
        for n, coord in stickercoords.items():
            cv2.putText(img, str(n), np.array(coord, dtype=int), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    return img


class Detector():
    """
    Incorporates the YOLO-model as well as the stickerDetector and the ballDetector
    """

    def __init__(self, model_name, n_stickers, max_dist_sticker=10, max_dist_ball=40, warmup_steps_ball=10):
        self.yolo_model = YOLO(f"../weights/{model_name}/best.pt")
        self.ballDetector = ballDetector(max_dist=max_dist_ball, warmup_steps=warmup_steps_ball)
        self.stickerDetector = stickerDetector(n_stickers=n_stickers, max_dist=max_dist_sticker)

    def __call__(self, image, mirror, return_yolo_output=False):
        detection_results = self.yolo_model(image, verbose=False)
        sticker_results = self.stickerDetector(detection_results, mirror=mirror)
        ball_results = self.ballDetector(detection_results)

        return (ball_results, sticker_results, detection_results) if return_yolo_output else (ball_results, sticker_results)


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

    def __call__(self, detection_results, mirror=False, verbose=False):
        """
        detection_results = YOLOV8 results

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
                idxs = np.argsort([c[0] for c in coords]) # Numerating Coords from
                if mirror:
                    idxs = idxs[::-1]
                self.ROIS = {j: coords[i] for j, i in enumerate(idxs)}
                self.first_frame = False
                return self.ROIS
            else:
                if verbose:
                    print(
                        f"First frame detection had {len(coords)} stickers detected, the real number of stickers should be {self.n_stickers}, skipping!")
                return None
        else:
            results = {}
            taken_coord_id = []
            # choosing the coord for each roi, that is closest and checking if dist is over max dist
            for i, roi in self.ROIS.items():
                min_dist = 10E20
                candidate = None
                candidate_id = None
                for j, coord in enumerate(coords):
                    dist = np.linalg.norm(roi - coord)
                    if dist < min_dist:
                        if not j in taken_coord_id:  # Prevent double assignments
                            min_dist = dist
                            candidate = coord
                            candidate_id = j

                if min_dist > self.max_dist:
                    if verbose:
                        print(f"Over max dist in sticker detection! ROI: {i}, {roi} min-dist: {min_dist}, threshold: {self.max_dist}")
                    return None
                else:
                    results[i] = candidate
                    taken_coord_id.append(candidate_id)


        if len(results) == self.n_stickers:
            self.ROIS.update(results)
            return results

        else:
            if verbose:
                print("Not enough stickers detected!")
            return None


class ballDetector():
    """
    Refines the ball-detection results of the trained YOLOV8
    detector
    """
    def __init__(self, max_dist=40, warmup_steps=10):
        self.ROI = []
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

    def __call__(self, detection_results, verbose=False):
        """
        detection_results = YOLOV8 results from detector trained

        to detect the ball.
        """

        if self. current_warmup_step <= self.warmup_steps:
            if verbose:
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
                        if verbose:
                            print("Over max dist in ball-detection!")
                        return None
                    else:
                        self.max_dist = 20
                        self.ROI = coord
                        return coord

            return None


if __name__=="__main__":
    pass
