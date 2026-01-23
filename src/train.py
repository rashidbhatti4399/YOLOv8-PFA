import torch
from ultralytics import YOLO
from custom_PFA import ParallelFusionAttention  # Your custom module

# Inject custom module into Ultralytics
import ultralytics.nn.tasks as tasks
tasks.ParallelFusionAttention = ParallelFusionAttention

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    try:
        model = YOLO("yolov8-PFA-DwConv.yaml").load("yolov8n.pt")
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Trying fallback initialization...")
        model = YOLO("yolov8-PFA-DWConv.yaml").load("yolov8n.pt")


    print("\nStarting single-phase training...")

    train_args = {
        "data": "RUOD.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 32,
        "device": device.type,
        "optimizer": "AdamW",
        "momentum": 0.9,
        "lr0": 2e-4,
        "weight_decay": 0.05,
        "cos_lr": True,
        "close_mosaic": 0,
        "augment": True,
    }

    results = model.train(**train_args)
    print("Training complete.")

