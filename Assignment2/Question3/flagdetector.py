import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
import cv2
import numpy as np

def run_ssd_model(image_path, confidence_threshold=0.1):
    """
    Run SSD model to detect objects in an image.
    :param image_path: Path to the input image
    :param confidence_threshold: Minimum confidence score for displaying detections
    :return: None
    """
    # === Step 1: Load the SSD model ===
    # Pre-trained on COCO dataset
    model = ssd300_vgg16(pretrained=True).eval()

    # === Step 2: Load and preprocess the image ===
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the file path.")
        return

    # Convert the image to RGB (required by torchvision)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a tensor
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    # === Step 3: Run the SSD model ===
    with torch.no_grad():
        detections = model(image_tensor)[0]  # Get detections for the first (and only) image

    # === Step 4: Process detections ===
    for i, (box, score, label) in enumerate(zip(detections['boxes'], detections['scores'], detections['labels'])):
        if score >= confidence_threshold:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            # Draw the bounding box on the original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add a label and confidence score
            cv2.putText(image, f"Object {i+1}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # === Step 5: Display the results ===
    cv2.imshow("SSD Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Main Program ===
if __name__ == "__main__":
    # Replace 'example_flag.jpg' with your image file
    run_ssd_model("indonesiaf.png")

