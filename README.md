🏏 Cricket Ball Tracking using YOLOv8

This project implements automatic cricket ball detection and tracking from a video using a fine-tuned YOLOv8 model.
The system generates:

🎥 An annotated output video with ball centroid and trajectory overlaid
📄 A CSV file containing per-frame ball coordinates and visibility

📁 Project Structure

edgefleet_ball_tracking/
│
├── code/                                # Core source code
│   ├── inference.py                    # Runs detection + tracking on input video
│   ├── tracker.py                      # Ball tracking logic (Kalman / smoothing)
│   ├── utils.py                        # Helper utilities
│   └── train.py                        # YOLO training script 
│
├── data/                                # Raw videos used in project
│   ├── train/                          # Training videos
│   └── test/                           # Test videos for inference
│
├── dataset/                             # YOLO-format dataset (from Roboflow)
│
├── ball_training/                       # YOLO training experiment outputs                  
│
├── models/                              
│   └── yolov8_ball.pt                  # Final trained ball detection model
│
├── examples/                           # Sample annotated frames showing ball centroid 
│   └──*.png                             # and trajectory overlay    
│
├── annotations/                         # Output CSV files (ball position per frame)
│   └── *.csv                            
│
├── results/                             
│   └── *.mp4                           # Processed videos with centroid & trajectory
│
├── README.md                            
├── report.pdf                          
└── requirements.txt                    


⚙️ Environment Setup
1️⃣ Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

⚠️ Note
If CUDA is not available, install CPU-only PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

🧠 Model Details
Architecture: YOLOv8
Framework: Ultralytics
Classes: ball
Input Size: 1280 × 1280
Output: Bounding box → centroid → trajectory

The model is fine-tuned specifically for small object (cricket ball) detection.

▶️ Inference
Run inference using:
python code/inference.py --video data/test/9.mov --out_video results/9.mp4 --out_csv annotations/9.csv

📤 Outputs Explained

🎥 Output Video
Original video with:
🔴 Red dot → Ball centroid
🔵 Blue polyline → Ball trajectory (last N frames)

📄 Output CSV Format
Column	Description
frame	Frame index (0-based)
x	X-coordinate of ball centroid
y	Y-coordinate of ball centroid
visible	1 if ball detected, else 0

Example:
frame,x,y,visible
0,642,381,1
1,650,389,1
2,-1,-1,0

visible = 0 → Ball not detected in that frame
Coordinates set to -1 when invisible 

🔄 Tracking Logic
The tracking pipeline works as follows:
YOLO detects the ball in each frame
Highest-confidence detection is selected
Centroid is computed from bounding box
A trajectory buffer (deque) stores recent centroids
Tracker smooths motion and handles brief missed detections
Trajectory is drawn frame-by-frame
If the ball is temporarily missed, the tracker maintains continuity using recent motion history.
When multiple detections are present in a frame, the detection with the highest confidence score is selected as the true ball candidate.

This ensures:
Reduced flickering
Smooth trajectory
Robust tracking across occlusions

📦 Model File
The trained YOLOv8 ball detection model is provided at:
models/yolov8_ball.pt
This model can be directly used for inference without retraining.

📄 Detailed Report
A detailed report (`report.pdf`) is included, covering:
- Model training decisions
- Hyperparameter choices
- Tracking logic and fallback handling
- Performance improvements and limitations
- Example qualitative results

🧪 Tested On
GPU: NVIDIA GeForce MX450 (2GB)
OS: Windows 11
Python: 3.12
CUDA: 12.1

👤 Author
Himanshu Kumar