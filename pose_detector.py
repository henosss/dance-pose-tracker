import numpy as np
from ultralytics import YOLO

# YOLOv8-pose keypoint indices
KP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


class PoseDetector:
    def __init__(self, model_path="yolov8n-pose.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Return list of keypoint arrays (shape [17, 3]) for each person detected."""
        results = self.model(frame, conf=self.conf, verbose=False)
        people = []
        for r in results:
            if r.keypoints is None:
                continue
            kps = r.keypoints.data.cpu().numpy()  # [N, 17, 3]
            for person_kps in kps:
                people.append(person_kps)
        return people

    @staticmethod
    def angle(a, b, c):
        """Angle at point b formed by a-b-c (degrees)."""
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    @staticmethod
    def visible(kp, idx, thresh=0.3):
        return kp[idx, 2] > thresh

    @staticmethod
    def pt(kp, idx):
        return kp[idx, :2]
