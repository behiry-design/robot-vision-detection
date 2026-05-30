# YOLOv8 Object Detection Model
## Mobile Robot Vision System

Real-time object detection using YOLOv8s trained on custom dataset.

## Detected Classes
| ID | Class |
|----|-------|
| 0 | plastic_bottle |
| 1 | backpack |
| 2 | perfume |
| 3 | clock |

## Model Performance
| Class | mAP50 |
|-------|-------|
| overall | 0.954 |
| plastic_bottle | 0.950 |
| backpack | 0.993 |
| perfume | 0.980 |
| clock | 0.893 |

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/behiry-design/robot-vision-detection.git
cd robot-vision-detection
```

2. Install requirements:
```bash
pip install ultralytics opencv-python
```

3. Run detection:
```bash
python3 run_detection.py
```

4. Press Q to quit

## Notes
- Works on Windows, Linux, Mac
- No GPU required (CPU works, slower FPS)
- Camera must be connected
- Both files must be in same folder

## Model Details
- Architecture: YOLOv8s (11M parameters)
- Training: 182 epochs on RTX 3050 4GB VRAM
- Input size: 640x640
- FPS: 30-35 real-time
