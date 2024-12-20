import cv2
import torch
import torchvision.transforms as T

# Load the pre-trained SSD model
# ssdlite320_mobilenet_v3_large is a lightweight version optimized for speed
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()  # Set the model to evaluation mode (no training)

# Define the path to the input video
video_path = 'test_video.webm'
cap = cv2.VideoCapture(video_path)  # Open the video file

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties: frame rate, width, and height
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer to save the processed video
out = cv2.VideoWriter('output_ssd.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Define a transform to preprocess frames (convert to tensor)
transform = T.Compose([
    T.ToTensor()  # Converts a NumPy array (frame) to a PyTorch tensor
])

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), exit the loop
        break

    # Preprocess the frame (convert it to a tensor and add batch dimension)
    img_tensor = transform(frame).unsqueeze(0)  # Shape: [1, C, H, W]

    # Run SSD inference on the frame
    with torch.no_grad():  # No gradients needed during inference
        predictions = model(img_tensor)

    # Extract the predictions
    boxes = predictions[0]['boxes']  # Bounding box coordinates
    labels = predictions[0]['labels']  # Class labels
    scores = predictions[0]['scores']  # Confidence scores

    # Loop through all detections and draw them on the frame
    for i in range(len(scores)):
        if scores[i] > 0.25:  # Default confidence threshold (50%)
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[i])

            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box

            # Display the label and confidence score
            label = f"Label {labels[i]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the processed frame (optional)
    cv2.imshow('SSD Output', frame)

    # Write the processed frame to the output video file
    out.write(frame)

    # Allow quitting the video display by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources after processing
cap.release()  # Close the video file
out.release()  # Finalize the output video file
cv2.destroyAllWindows()  # Close all OpenCV windows
