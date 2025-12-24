import cv2
import numpy as np

class BallTracker:
    def __init__(self, max_missed=5):
        """
        max_missed: number of consecutive frames to predict
                    when detection is missing
        """
        self.kf = cv2.KalmanFilter(4, 2)

        # State: [x, y, vx, vy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Measurement: [x, y]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.initialized = False
        self.missed = 0
        self.max_missed = max_missed

    def init(self, x, y):
        self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        self.initialized = True
        self.missed = 0

    def update(self, x, y):
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
        self.missed = 0

    def predict(self):
        pred = self.kf.predict()
        self.missed += 1
        return int(pred[0]), int(pred[1])

    def lost(self):
        return self.missed >= self.max_missed
