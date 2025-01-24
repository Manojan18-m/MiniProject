import os
from ultralytics import YOLO
import cv2
import numpy as np

# Define paths
VIDEO_PATH = os.path.join('.', 'muddy_Road pothole.mp4')  # Path to input video
OUTPUT_VIDEO_PATH = os.path.join('.', 'outputs')  # Path to save output video

# Load the YOLO model
model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

# Define confidence threshold and class names
threshold = 0.5  # Confidence threshold
class_name_dict = {0: 'pothole'}  # Add more classes if needed

# Open the input video
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video: {VIDEO_PATH}")
    exit()

# Get video properties (frame width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter to save the output video
out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
    fps,
    (frame_width, frame_height)
)

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        print("End of video or error reading frame.")
        break

    # Perform inference on the current frame
    results = model(frame)[0]

    # Loop over detected objects
    for result in results:
        if result.boxes:
            for box in result.boxes:
                conf = box.conf.item()
                cls_id = box.cls.item()  # Class ID

                if conf >= threshold:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Draw a polygon (rectangle) around the detected object
                    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Put class name and confidence on the frame
                    label = f"{class_name_dict.get(cls_id, 'Unknown')} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"Class: {cls_id}, Confidence: {conf}, Box: ({x1}, {y1}, {x2}, {y2})")

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {OUTPUT_VIDEO_PATH}")