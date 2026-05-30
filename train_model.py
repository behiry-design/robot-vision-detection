#!/usr/bin/env python3
"""
YOLOv8 Training Script
Mobile Robot Vision System
Classes: plastic_bottle, backpack, perfume, clock
"""

from ultralytics import YOLO

# ── Step 1: Load pretrained YOLOv8s model ─────────────────────────────────
# yolov8s = small model, 11M parameters
# Pretrained on COCO dataset (80 classes)
# We fine-tune it on our custom 4-class dataset
model = YOLO("yolov8s.pt")

# ── Step 2: Train on our custom dataset ───────────────────────────────────
model.train(
    # Dataset configuration file
    data="/home/behiry/robot_dataset/dataset2/data.yaml",

    # Number of training iterations
    # 300 set but early stopping triggers around epoch 132-182
    epochs=300,

    # Number of images processed together per iteration
    # batch=16 chosen based on 4GB VRAM limit
    # larger batch = more stable gradients
    batch=16,

    # Input image size - standard YOLOv8 size
    # larger = more detail but more VRAM
    imgsz=640,

    # Initial learning rate
    # 0.001 chosen after 0.01 caused divergence
    # smaller lr = slower but more stable learning
    lr0=0.001,

    # Optimizer - Stochastic Gradient Descent
    # SGD chosen from reference script recommendation
    optimizer="SGD",

    # Early stopping - stop if no improvement for 50 epochs
    # prevents wasted training time
    patience=50,

    # Don't cache images - dataset too large for RAM
    cache=False,

    # CPU threads for data loading
    workers=4,

    # GPU index - 0 = first GPU (RTX 3050)
    device=0,

    # Show detailed training progress
    verbose=True,

    # Experiment name - results saved to runs/detect/final_train1
    name="final_train1",
)

print("Training complete!")
print("Best model saved to: runs/detect/final_train1/weights/best.pt")
