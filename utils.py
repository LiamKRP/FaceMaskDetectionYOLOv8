import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil
import yaml

classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']

def prepare_dataset(train_split=0.7, val_split=0.2):
    # Get all the image files in the dataset
    file_names = [f for f in os.listdir('data/images') if f.endswith(('.png', 'jpg', 'jpeg'))]

    # split dataset between train, val and test
    random.shuffle(file_names)
    train_count = int(len(file_names) * train_split)
    train_images = file_names[:train_count]

    val_count = int(len(file_names) * val_split)
    val_images = file_names[train_count:val_count+train_count]

    print(val_images)

    test_count = int(len(file_names) * (1 - train_split + val_split))
    test_images = file_names[train_count+val_count:]

    print(train_count)
    print(val_count)
    print(test_count)

    # create directories
    if not os.path.exists('yolo_dataset'):
        os.mkdir("yolo_dataset")

    if not os.path.exists('yolo_dataset/images'):
        os.mkdir('yolo_dataset/images')
    
    if not os.path.exists('yolo_dataset/labels'):
        os.mkdir('yolo_dataset/labels')

    if not os.path.exists('yolo_dataset/images/train'):
        os.mkdir('yolo_dataset/images/train')
    
    if not os.path.exists('yolo_dataset/images/val'):
        os.mkdir('yolo_dataset/images/val')
    
    if not os.path.exists('yolo_dataset/images/test'):
        os.mkdir('yolo_dataset/images/test')

    if not os.path.exists('yolo_dataset/labels/train'):
        os.mkdir('yolo_dataset/labels/train')
    
    if not os.path.exists('yolo_dataset/labels/val'):
        os.mkdir('yolo_dataset/labels/val')
    
    if not os.path.exists('yolo_dataset/labels/test'):
        os.mkdir('yolo_dataset/labels/test')

    # move (not copy) images and labels to new directory
    train_images_dir = 'yolo_dataset/images/train'
    val_images_dir = 'yolo_dataset/images/val'
    test_images_dir = 'yolo_dataset/images/test'

    train_labels_dir = 'yolo_dataset/labels/train'
    val_labels_dir = 'yolo_dataset/labels/val'
    test_labels_dir = 'yolo_dataset/labels/test'

    for img in train_images:

        src_img = os.path.join('data/images/', img)
        src_label = os.path.join('data/annotations/', img.replace('.png', '.xml'))

        dst_img = os.path.join(train_images_dir, img)
        dst_label = os.path.join(train_labels_dir, img.replace('.png', '.xml'))

        if os.path.exists(src_label) & os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            print(f"Moved: {img} & {img}.")
        else:
            print(f"Skipping {img} because mask {img} was not found.")

    for img in val_images:

        src_img = os.path.join('data/images/', img)
        src_label = os.path.join('data/annotations/', img.replace('.png', '.xml'))

        dst_img = os.path.join(val_images_dir, img)
        dst_label = os.path.join(val_labels_dir, img.replace('.png', '.xml'))

        if os.path.exists(src_label):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            print(f"Moved: {img} & {img}.")
        else:
            print(f"Skipping {img} because mask {img} was not found.")

    for img in test_images:

        src_img = os.path.join('data/images/', img)
        src_label = os.path.join('data/annotations/', img.replace('.png', '.xml'))

        dst_img = os.path.join(test_images_dir, img)
        dst_label = os.path.join(test_labels_dir, img.replace('.png', '.xml'))

        if os.path.exists(src_label):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            print(f"Moved: {img} & {img}.")
        else:
            print(f"Skipping {img} because mask {img} was not found.")
    


def convert_voc_to_yolo(xml_folder, output_folder, images_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        image_file = root.find('filename').text
        image_path = os.path.join(images_folder, image_file)
        image = Image.open(image_path)
        w, h = image.size

        txt_file = os.path.join(output_folder, xml_file.replace('.xml', '.txt'))
        with open(txt_file, 'w') as f:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)

                xml_box = obj.find('bndbox')
                xmin = int(xml_box.find('xmin').text)
                xmax = int(xml_box.find('xmax').text)
                ymin = int(xml_box.find('ymin').text)
                ymax = int(xml_box.find('ymax').text)

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        os.remove(os.path.join(xml_folder, xml_file))

def create_dataset_yaml():
    dataset_folder = 'yolo_dataset'
    dataset_root = os.path.abspath(dataset_folder)

    dataset_yaml = {
        'path': dataset_root,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': classes
    }

    # Save YAML
    yaml_path = os.path.join(os.getcwd(), 'dataset.yaml')  # saves in current working directory
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"dataset.yaml generated at: {yaml_path}")
    print(f"With dataset root: {dataset_root}")

def getFreeImageId():
    path = 'inference_images'

    i = 1
    while True:
        if os.path.exists(os.path.join(path, f'image{i}.jpg')):
            i += 1
        else:
            return i