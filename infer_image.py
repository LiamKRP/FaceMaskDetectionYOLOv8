from ultralytics import YOLO
import os
import argparse
import random
import matplotlib.pyplot as plt
import math
import cv2
from utils import getFreeImageId

def main():
    parser = argparse.ArgumentParser(description="Face Mask Detection Video Inference")
    parser.add_argument('--save_image', action="store_true")
    parser.add_argument('--number_images', type=int, default=5)
    args = parser.parse_args()

    print(args.number_images)

    save_output = args.save_image
    num_images = args.number_images

    model_path = os.path.abspath("runs/mask_detection_model_14/weights/best.pt")
    model = YOLO(model_path)

    test_directory_path = os.path.join('yolo_dataset', 'images', 'test')
    test_images = [os.path.join('yolo_dataset', 'images', 'test', f) for f in os.listdir(test_directory_path) if f.endswith((".jpg", ".png"))]

    annotated_images = []
    for i in range(num_images):
         idx = random.randint(0, len(test_images)-1)
         results = model(os.path.join(os.getcwd(), test_images[idx]))
         annotated_images.append(results[0].plot())
        
    
    # Show all images in one matplotlib figure
    n = len(annotated_images)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(5 * cols, 5 * rows))
    for i, img in enumerate(annotated_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()

    if save_output:
        if not os.path.exists('inference_images'):
            os.mkdir('inference_images')
        id = getFreeImageId()
        plt.savefig(f"inference_images/image{id}.jpg")
    plt.show()


if __name__ == "__main__":
    main()