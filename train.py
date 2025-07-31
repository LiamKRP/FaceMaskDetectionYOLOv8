from ultralytics import YOLO
from utils import convert_voc_to_yolo, prepare_dataset, create_dataset_yaml
import argparse
import os

def main():
    model = YOLO("yolov8n.pt") 

    if not os.path.exists('runs'):
        os.mkdir('runs')

    project_dir = project_dir = os.path.join(os.getcwd(), "runs")
    # Train
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=32,
        project=project_dir,
        name='mask_detection_model_1',
        workers=4,
        device='mps'
    )

if __name__ == "__main__":
    main()

