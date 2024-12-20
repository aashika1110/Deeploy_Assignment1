import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# Load the pre-trained SSD model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the COCO class labels
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Path to the input video
video_path = 'test_video.webm'
cap = cv2.VideoCapture(video_path)  # Open the video file

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Define the output video writer
out = cv2.VideoWriter('output_ssd_coco.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Define a transform to preprocess the frames
transform = T.Compose([
    T.ToTensor()  # Converts the frame (NumPy array) to a PyTorch tensor
])

# Set the batch size
batch_size = 4  # Process 4 frames at a time
frames_batch = []  # Temporary storage for frames in the current batch

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # End of video
        break

    # Add the current frame to the batch
    frames_batch.append(frame)

    # If the batch is full, process it
    if len(frames_batch) == batch_size:
        # Preprocess each frame in the batch
        batch_tensors = torch.stack([transform(frame) for frame in frames_batch])  # Create a batch of tensors

        # Run SSD inference on the batch
        with torch.no_grad():  # No gradient computation needed
            predictions = model(batch_tensors)

        # Process predictions for each frame in the batch
        for i, frame in enumerate(frames_batch):
            boxes = predictions[i]['boxes']  # Bounding box coordinates
            labels = predictions[i]['labels']  # Class labels
            scores = predictions[i]['scores']  # Confidence scores

            # Draw detections on the frame
            for j in range(len(scores)):
                if scores[j] > 0.5:  # Confidence threshold
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, boxes[j])

                    # Map the label index to the COCO class name
                    label = COCO_CLASSES[labels[j]]

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
                    cv2.putText(frame, f"{label}: {scores[j]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the processed frame in a window
            cv2.imshow('SSD Output', frame)

            # Write the processed frame to the output video
            out.write(frame)

            # Allow quitting with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                exit()

        # Clear the batch after processing
        frames_batch = []

# Process remaining frames in the batch if the video ends
if len(frames_batch) > 0:
    batch_tensors = torch.stack([transform(frame) for frame in frames_batch])
    with torch.no_grad():
        predictions = model(batch_tensors)

    for i, frame in enumerate(frames_batch):
        boxes = predictions[i]['boxes']
        labels = predictions[i]['labels']
        scores = predictions[i]['scores']

        for j in range(len(scores)):
            if scores[j] > 0.25:
                x1, y1, x2, y2 = map(int, boxes[j])
                label = COCO_CLASSES[labels[j]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label}: {scores[j]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the final batch frames
        cv2.imshow('SSD Output', frame)
        out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


