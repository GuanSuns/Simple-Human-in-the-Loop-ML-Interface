import cv2
import numpy as np


class ObjectTracker:
    def __init__(self):
        self.trackers = None
        self.boxes = None

    def reset_trackers(self):
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        trackers = cv2.MultiTracker_create()
        self.trackers = trackers
        return trackers

    def add_new_trackers(self, current_obs, boxes, track_algo='csrt'):
        for box in boxes:
            self.add_new_tracker(current_obs, box, track_algo)

    def add_new_tracker(self, current_obs, box, track_algo='csrt'):
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        new_tracker = OPENCV_OBJECT_TRACKERS[track_algo]()
        self.trackers.add(new_tracker, current_obs, box)

    def _get_bounding_box(self, obs, color):
        """ Calculates the bounding box of a ndarray"""
        mask = np.all(obs == color, axis=-1)
        rows = np.any(mask, axis=0)
        cols = np.any(mask, axis=1)

        if len(np.where(rows)[0]) == 0 or len(np.where(cols)[0]) == 0:
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return [rmin, cmin, rmax-rmin, cmax-cmin]

    def _get_key_pickup_bounding_box(self, obs):
        key_pickup_agent_color = (127, 127, 127)
        key_pickup_key_color = (0, 0, 255)
        key_pickup_door_color = (0, 0, 0)

        boxes = []
        for color in [key_pickup_agent_color, key_pickup_key_color, key_pickup_door_color]:
            box = self._get_bounding_box(obs, color)
            if box is not None:
                boxes.append(box)
        return boxes

    def update(self, obs):
        (success, boxes) = self.trackers.update(obs)
        self.boxes = boxes
        return success, boxes

    def add_bounding_boxes(self, obs, boxes):
        # no need to call update fn again. just use the boxes.
        for box in boxes:
            (x, y, h, w) = [int(v) for v in box]
            cv2.rectangle(obs, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # as numpy
        return obs
