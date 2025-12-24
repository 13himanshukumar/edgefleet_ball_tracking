import os
import cv2

def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("annotations", exist_ok=True)

def draw_trajectory(frame, trajectory):
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
