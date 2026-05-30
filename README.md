# YOLOv8 Object Detection — Mobile Robot Vision System

Real-time object detection system built for a mobile robot project.
Uses YOLOv8s deep learning model trained on a custom dataset of 4 object classes.
Detection results are published as ROS2 topics for robot navigation integration.

---

## Detected Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | plastic_bottle | Plastic water/drink bottles |
| 1 | backpack | Backpacks and bags |
| 2 | perfume | Perfume bottles |
| 3 | clock | Wall clocks and alarm clocks |

---

## Model Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| overall | 0.953 | 0.913 | 0.954 | 0.771 |
| plastic_bottle | 0.980 | 0.873 | 0.950 | 0.765 |
| backpack | 0.937 | 0.993 | 0.993 | 0.946 |
| perfume | 0.970 | 0.955 | 0.980 | 0.633 |
| clock | 0.923 | 0.833 | 0.893 | 0.739 |

Inference speed: 6.5ms per image — Real-time FPS: 30-35

---

## Repository Structure

robot-vision-detection/
- best.pt — Trained YOLOv8s model (22.5MB)
- run_detection.py — Real-time detection script
- train_model.py — Training script with full comments
- augment_all_manual.py — Data augmentation script
- extract_frames.py — Video frame extraction script
- manual_dataset/ — Self-collected labeled images
  - plastic_bottle/ — 16 images + YOLO labels
  - backpack/ — 49 images + YOLO labels
  - perfume/ — 19 images + YOLO labels
  - clock/ — 76 images + YOLO labels
- README.md

---

## How to Run

Step 1 — Clone the repository
git clone https://github.com/behiry-design/robot-vision-detection.git
cd robot-vision-detection

Step 2 — Install requirements
pip install ultralytics opencv-python albumentations

Step 3 — Run real-time detection
python3 run_detection.py

Point your camera at any of the 4 objects. Press Q to quit.

---

## Dataset

The dataset was built from two sources:

1. Roboflow Public Datasets

| Class | Dataset | Used |
|-------|---------|------|
| plastic_bottle | Plastic Bottle 2.0 v15 | 300 train |
| backpack | Luggage v1 | 300 train |
| perfume | Perfume v4 (9 brands remapped to 1) | 300 train |
| clock | CLOCK v3 | 299 train |

2. Manual Dataset (included in this repo)

Self-photographed images in real deployment environment:

| Class | Images | Source |
|-------|--------|--------|
| plastic_bottle | 16 | Phone photos |
| backpack | 49 | Phone photos |
| perfume | 19 | Phone photos |
| clock | 76 | Video frames + phone photos |

Located in manual_dataset/ folder.
Each folder contains images and YOLO label files (.txt).

Label format: class_id x_center y_center width height
All values normalized 0 to 1.

3. Negative Samples

636 images of people and backgrounds with empty label files.
Teaches model that faces and rooms are NOT objects.
Eliminates false positives. Downloaded from COCO val2017.

Final Dataset Size:
- Train: 6,755 images
- Valid: 366 images
- Test: 343 images
- Total: 7,464 images

---

## Data Augmentation

Applied to all manual images using albumentations library.
Each original image generates 8 augmented copies.

| Transformation | Effect |
|----------------|--------|
| HorizontalFlip | Direction invariance |
| RandomBrightnessContrast | Different lighting conditions |
| Affine rotate/scale/shift | Different distances and angles |
| Blur | Camera focus variations |
| HueSaturationValue | Color variations |
| GaussNoise | Camera sensor noise |

Result: 573 manual images to 5,157 augmented images
See augment_all_manual.py for full code.

---

## Training

Model: YOLOv8s (11M parameters)
Chosen over YOLOv8n nano for better feature extraction
while still fitting in 4GB VRAM.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Epochs | 300 stopped at 182 | Early stopping at epoch 132 |
| Batch | 16 | Max safe for 4GB VRAM |
| Image size | 640x640 | Standard YOLOv8 benchmark |
| Learning rate | 0.001 | Stable convergence |
| Optimizer | SGD | Best for this dataset |
| Patience | 50 | Prevents premature stopping |
| Hardware | RTX 3050 4GB | Training time 7 hours |

See train_model.py for full training script with comments.

---

## ROS2 Integration

The model runs as a ROS2 node publishing detection results:

| Topic | Type | Content |
|-------|------|---------|
| /detected_object | String | Best detection class name |
| /detection_confidence | Float32 | Confidence score 0 to 1 |
| /object_center | Point | Pixel coordinates x y |
| /detections | String | Full JSON with all detections |

Run ROS2 node:
source /opt/ros/jazzy/setup.bash
python3 vision_node.py

Monitor topics:
ros2 topic echo /detected_object
ros2 topic echo /detection_confidence

---

## Notes

- Works on Windows, Linux, Mac
- No GPU required — CPU works but slower 5-10 FPS
- GPU recommended for 30+ FPS real-time detection
- Camera must be connected
- Confidence threshold 0.45 adjustable in script
- Future deployment on Raspberry Pi Camera v2

---

## Dependencies

ultralytics
opencv-python
albumentations
