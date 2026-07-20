"""
Train YOLOv8 on the Vehicle_Detection dataset using transfer learning.
Base model: yolov8n.pt / yolov8s.pt

Usage:
    python train.py --data data/data.yaml --epochs 50 --imgsz 640 --batch 16 --model yolov8n.pt
"""

import argparse
from ultralytics import YOLO


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, base_model: str, device: str):
    
    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="models",
        name="itms_yolov8",
        patience=15,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        augment=True,
        val=True,
        exist_ok=True,
    )

    # Validate best weights
    metrics = model.val()
    print(metrics)

    # Export best.pt path
    print("Best weights saved at: models/itms_yolov8/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Pretrained COCO checkpoint: yolov8n/s/m/l/x.pt")
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.model, args.device)
