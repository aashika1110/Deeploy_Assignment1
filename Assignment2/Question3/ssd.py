import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np

# Define COCO class labels
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Load the SSD model pretrained on the COCO dataset
model = ssd300_vgg16(pretrained=True)
model.eval()  # Set the model to evaluation mode since we're not training

# Function to preprocess the input image
def preprocess_image(image_path):
    """
    Loads and preprocesses an image for SSD model input.
    """
    # Open the image and convert to RGB (required format for the model)
    image = Image.open(image_path).convert("RGB")
    # Convert the image to a PyTorch tensor and add a batch dimension
    input_image = F.to_tensor(image).unsqueeze(0)
    return image, input_image

# Function to run object detection on the image
def detect_objects(model, input_image, confidence_threshold=0.1):
    """
    Runs inference on the input image and filters results by confidence threshold.
    """
    # Perform object detection using the model
    outputs = model(input_image)

    # Extract bounding boxes, labels, and confidence scores from the model output
    boxes = outputs[0]["boxes"].detach().numpy()
    labels = outputs[0]["labels"].detach().numpy()
    scores = outputs[0]["scores"].detach().numpy()

    # Filter results based on the confidence threshold
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:  # Only keep detections above the threshold
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)

    return filtered_boxes, filtered_labels, filtered_scores

# Function to visualize the detections on the image
def visualize_detections(image, boxes, labels, scores, target_classes):
    """
    Draws bounding boxes and class labels on the image for the detected objects.
    """
    # Convert the PIL image to a NumPy array for OpenCV
    image_cv = np.array(image)

    # Loop through the detected objects
    for box, label, score in zip(boxes, labels, scores):
        # Get the class name for the detected object
        label_name = COCO_CLASSES[label]
        # Only draw boxes for relevant classes (e.g., flag-like objects)
        if label_name in target_classes:
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box)
            # Draw the bounding box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add a label with the class name and confidence score
            cv2.putText(
                image_cv, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    # Display the image with detections
    cv2.imshow("Detected Flags", cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the image window

# Main function to run the script
def main(image_path):
    """
    Main function to detect and visualize flag-like objects in an image.
    """
    # Step 1: Preprocess the image
    image, input_image = preprocess_image(image_path)

    # Step 2: Run object detection
    boxes, labels, scores = detect_objects(model, input_image)

    # Step 3: Visualize the results
    # Specify target classes that might resemble flags (e.g., cloth, umbrella, kite)
    target_classes = ["umbrella", "kite", "cloth"]
    visualize_detections(image, boxes, labels, scores, target_classes)

# Run the script
if __name__ == "__main__":
    # Replace this with the path to your test image
    test_image_path = "polandday.jpg"
    main(test_image_path)
