import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the confidence threshold
model.conf = 0.5  # Only detect objects with confidence > 50%

# Set the IoU threshold for Non-Maximum Suppression (NMS)
model.iou = 0.4  # Overlapping boxes with IoU > 40% are suppressed

# Path to the input video
video_path = 'test_video.webm'
cap = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

# Define the output video writer for the results
out = cv2.VideoWriter('output_yolo_tweaked.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  # End of video
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Render results on the frame
    result_frame = results.render()[0]

    # Write the processed frame to the output video
    out.write(result_frame)

    # Optional: Display the frame
    cv2.imshow('YOLO Processed', result_frame)

    # Allow exiting by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()



