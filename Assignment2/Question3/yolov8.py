from ultralytics import YOLO
import cv2
import numpy as np

# Step 1: Define the COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Step 2: Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for the nano model, suitable for faster inference

# Step 3: Specify the path to the input image
image_path = 'indonesia3.jpeg'  # Replace with the actual image path

# Step 4: Run YOLOv8 inference
results = model(image_path, conf=0.3, iou=0.3)  # Set confidence and IoU thresholds

# Helper function to crop the bounding box with padding
def crop_bounding_box(image, bbox):
    # Flatten the bounding box if it's a 2D array
    if isinstance(bbox, np.ndarray) and bbox.ndim > 1:
        bbox = bbox.flatten()

    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox)  # Ensure integer values for pixel indices

    # Add padding to ensure full flag is captured
    h, w = image.shape[:2]
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2]

# Helper function to apply a mask to remove background noise
def apply_mask(cropped):
    mask = cv2.inRange(cropped, (0, 0, 0), (255, 255, 255))  # Adjust range as needed
    return cv2.bitwise_and(cropped, cropped, mask=mask)

# Step 5: Define a robust function to analyze flags and classify them as Indonesia or Poland
def analyze_flag_robust(image, bbox):
    # Crop bounding box with padding
    cropped = crop_bounding_box(image, bbox)

    # Apply mask to remove background noise
    cropped = apply_mask(cropped)

    # Split the cropped region into multiple horizontal slices to handle waving flags
    h, w = cropped.shape[:2]
    sections = np.array_split(cropped, 4, axis=0)  # Divide into 4 horizontal slices

    red_pixels = []
    white_pixels = []

    for section in sections:
        cropped_hsv = cv2.cvtColor(section, cv2.COLOR_BGR2HSV)
        red_lower = np.array([0, 70, 50])  # Adjusted red range
        red_upper = np.array([10, 255, 255])
        white_lower = np.array([0, 0, 220])  # Adjusted white range
        white_upper = np.array([180, 30, 255])

        red_count = cv2.inRange(cropped_hsv, red_lower, red_upper).sum()
        white_count = cv2.inRange(cropped_hsv, white_lower, white_upper).sum()
        red_pixels.append(red_count)
        white_pixels.append(white_count)

    # Final classification based on red/white pixel distribution
    if sum(red_pixels[:2]) > sum(white_pixels[:2]) and sum(white_pixels[2:]) > sum(red_pixels[2:]):
        return "Indonesia"
    elif sum(white_pixels[:2]) > sum(red_pixels[:2]) and sum(red_pixels[2:]) > sum(white_pixels[2:]):
        return "Poland"
    else:
        return "Unknown flag"

# Step 6: Load the image using OpenCV for further processing
image = cv2.imread(image_path)

# Step 7: Process each detected object
print("Detections:")
for result in results:
    for box in result.boxes:
        cls = int(box.cls.cpu().numpy().item())  # Get the class ID
        conf = float(box.conf.cpu().numpy().item())  # Get the confidence score
        coords = box.xyxy.cpu().numpy()  # Get the bounding box coordinates

        # Flatten the bounding box coordinates if necessary
        coords = coords.flatten()

        # Map class ID to class name
        class_name = COCO_CLASSES[cls]
        print(f"Class: {class_name}, Confidence: {conf:.2f}, Bounding Box: {coords}")

        # If the detected class is "kite" or "umbrella," analyze it as a potential flag
        if class_name in ["kite", "umbrella"]:
            flag_type = analyze_flag_robust(image, coords)
            print(f"Detected flag type: {flag_type}")

# Step 8: Display the annotated image
annotated_image = results[0].plot()  # Generate the annotated image with detections
cv2.imshow('YOLOv8 Detections', annotated_image)

# Step 9: Wait for a key press and close the OpenCV window
cv2.waitKey(0)
cv2.destroyAllWindows()








