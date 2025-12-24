import argparse
import csv
import os
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import BallTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Cricket Ball Inference")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out_video", required=True, help="Output annotated video path")
    parser.add_argument("--out_csv", required=True, help="Output CSV annotation path")
    parser.add_argument("--model", default="models/yolov8_ball.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=1280)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YOLO model
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    writer = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    csv_file = open(args.out_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "x", "y", "visible"])

    # Tracker + trajectory
    tracker = BallTracker(max_missed=5)
    trajectory = deque(maxlen=50)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            device=0,
        )

        visible = 0
        cx, cy = -1, -1

        # ---------------- DETECTION ----------------
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            box = boxes[best_idx].xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if not tracker.initialized:
                tracker.init(cx, cy)
            else:
                tracker.update(cx, cy)

            visible = 1
            trajectory.append((cx, cy))

            # draw centroid
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # ---------------- TRACKING (PREDICTION) ----------------
        else:
            if tracker.initialized and not tracker.lost():
                px, py = tracker.predict()
                cx, cy = px, py
                visible = 1
                trajectory.append((px, py))

                # draw predicted centroid
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

            else:
                visible = 0
                cx, cy = -1, -1
                trajectory.clear()
                trajectory.append(None)

        # ---------------- DRAW TRAJECTORY ----------------
        for i in range(1, len(trajectory)):
            if trajectory[i - 1] is None or trajectory[i] is None:
                continue
            cv2.line(
                frame,
                trajectory[i - 1],
                trajectory[i],
                (255, 0, 0),
                2,
            )

        # ---------------- SAVE OUTPUTS ----------------
        csv_writer.writerow([frame_idx, cx, cy, visible])
        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()

    print("\n✅ Inference completed successfully")
    print(f"Video saved to: {args.out_video}")
    print(f"CSV saved to: {args.out_csv}")
    print(f"Frames processed: {frame_idx}/{total_frames}")


if __name__ == "__main__":
    main()
