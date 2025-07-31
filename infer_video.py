import cv2
from ultralytics import YOLO
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Face Mask Detection Video Inference")
    parser.add_argument('--save_video', action="store_true")
    args = parser.parse_args()

    model_path = os.path.abspath("runs/mask_detection_model_14/weights/best.pt")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)


    save_output = args.save_video
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(source=frame, persist=True, show=False, verbose=False, stream=False, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()

        cv2.imshow("Webcam YOLO Tracker", annotated_frame)

        if save_output:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()