from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys


def yolo_predict(image, model):
    """
    Draws points at the center of bounding boxes detected by YOLO on the template image.

    Parameters:
    - image: The cropped image to run YOLO inference on.
    - template_image: The template image where points will be drawn.
    - model: The YOLO model used for inference.

    Returns:
    - template_image: The template image with points drawn at the centers of bounding boxes.
    - centers: A list of tuples representing the coordinates of the centers of the bounding boxes.
    """
    results = model(image, verbose=False)
    centers = []
    n_classes = 0
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        n_classes = len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            label = int(labels[i])
            confidence = result.boxes.conf[i]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # cv2.circle(template_image, (center_x, center_y), 5, (0, 255, 0), -1)  # Green point
            centers.append((center_x, center_y))

    return centers, n_classes

def yolo_crop(image, model, padding=25, padding_box=0, blackout_boxes=[]):
    """
    Crops around metal boxes and blackouts class 0 with some padding.
    
    Parameters:
    - image: Input image.
    - model: YOLO model used for inference.
    - padding: Padding to apply around the detected bounding boxes.
    
    Returns:
    - mask: Image with cropped areas for metal and blackout applied for class 0.
    """
    results = model(image, verbose=False)
    n_classes = []
    metal_boxes = []
    class_0_count = 0
    class_1_count = 0
    
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        
        class_0_count += (labels == 0).sum().item()
        class_1_count += (labels == 1).sum().item()
        
        metal_boxes.extend(boxes[labels == 1])

    n_classes = [class_0_count, class_1_count]

    mask = np.zeros_like(image)

    for box in metal_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return mask, n_classes

def yolo_transform_directory(image_path, output_path, model):
    image = cv2.imread(image_path)
    results = model(image_path)

    metal_boxes = []
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        
        metal_boxes.extend(boxes[labels == 1])

    padding = 40
    mask = np.zeros_like(image)

    for box in metal_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        # Keep the "metal" part visible, black out the rest
        mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    cv2.imwrite(output_path, mask)

def process_directory(input_dir, output_dir, model):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                output_path = os.path.join(output_subdir, file)
                try:
                    yolo_transform_directory(img_path, output_path, model)
                except Exception as e:
                    print(f'Error processing {img_path}: {e}')
                    continue

                # TODO:
                # Setup logging for each subdirectory
                # setup_logging(output_subdir)

if __name__ == '__main__':
    model = YOLO('../dataset/runs/detect/train6/weights/best.pt')
    input_dir = '../test_all'
    output_dir = '../test_data/'
    process_directory(input_dir, output_dir, model)