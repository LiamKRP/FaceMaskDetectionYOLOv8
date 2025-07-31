from utils import prepare_dataset, convert_voc_to_yolo, create_dataset_yaml

def main():
    prepare_dataset()
    convert_voc_to_yolo('yolo_dataset/labels/train', 'yolo_dataset/labels/train', 'yolo_dataset/images/train')
    convert_voc_to_yolo('yolo_dataset/labels/val', 'yolo_dataset/labels/val', 'yolo_dataset/images/val')
    convert_voc_to_yolo('yolo_dataset/labels/test', 'yolo_dataset/labels/test', 'yolo_dataset/images/test')
    create_dataset_yaml()

if __name__ == "__main__":
    main()