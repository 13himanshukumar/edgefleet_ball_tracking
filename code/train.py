from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=1,
        device=0,          # GPU
        project="ball_training",
        name="yolov8_ball_gpu"
    )

if __name__ == "__main__":
    train()
