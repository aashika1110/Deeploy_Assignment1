import cv2  # OpenCV for video processing
import torch  # PyTorch to load and use the YOLOv5 model

# Load the pre-trained YOLOv5 model using Torch Hub
# 'yolov5s' refers to the small, fast version of the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Path to the input video file
video_path = 'test_video.webm'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)  # Open the video file

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()  # Exit the program if the video cannot be opened

# Get properties of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of video frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of video frames

# Define the output video writer to save the processed video
out = cv2.VideoWriter(
    'output_yolo.avi',  # Output file name
    cv2.VideoWriter_fourcc(*'XVID'),  # Codec for output video
    fps,  # Frames per second
    (width, height)  # Frame size
)

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), break the loop
        break

    # Pass the current frame to the YOLO model for detection
    results = model(frame)

    # YOLO returns results as images with bounding boxes and labels drawn on them
    result_frame = results.render()[0]  # Annotated frame

    # Display the processed frame in a window (optional)
    cv2.imshow('YOLOv5', result_frame)

    # Write the processed frame to the output video file
    out.write(result_frame)

    # Allow the user to quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
cap.release()  # Close the video file
out.release()  # Release the output file
cv2.destroyAllWindows()  # Close all OpenCV windows
