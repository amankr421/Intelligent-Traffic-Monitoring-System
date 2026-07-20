"""
Download the "Vehicle_Detection Computer Vision Model" dataset (v11, YOLOv8 format)
from Roboflow, by workspace "ProjectTraffic".

Get your free API key from: https://app.roboflow.com/settings/api
Then run:
    python download_dataset.py --api_key YOUR_API_KEY
"""

import argparse
import os
from roboflow import Roboflow


def download(api_key: str, out_dir: str = "data"):
    rf = Roboflow(api_key=api_key)

    # NOTE: update workspace/project slug if Roboflow renames them.
    project = rf.workspace("projecttraffic").project("vehicle_detection")
    version = project.version(11)

    dataset = version.download("yolov8", location=os.path.join(out_dir, "Vehicle_Detection"))
    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Your Roboflow API key")
    parser.add_argument("--out_dir", default="data", help="Output directory")
    args = parser.parse_args()

    download(args.api_key, args.out_dir)
