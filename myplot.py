import cv2
import os

# Define the paths to the directories containing the label files
real_data_dir = "dataset/10_txt"
detection_data_dir = "runs/detect/exp4/labels"
tracking_data_dir = "runs/track/exp16/labels"

# Define the paths to the video file
video_file = "runs/track/exp16/output_video10_3.avi"

# Load the video
cap = cv2.VideoCapture(video_file)

# Define colors for bounding boxes (BGR format)
real_color = (0, 255, 0)    # Green color for real data bounding box
detection_color = (255, 0, 0)  # Red color for detection data bounding box
tracking_color = (0, 0, 255)  # Blue color for tracking data bounding box
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for ID labels
font_scale = 0.4

if_real = True
if_detect = True
if_track = True

# Adjust playback speed
fps = cap.get(cv2.CAP_PROP_FPS)  # Get current frames per second (FPS)
# new_fps = fps / 5  # New desired frames per second (1/5 of the original)
# cap.set(cv2.CAP_PROP_FPS, new_fps)  # Set new frames per second

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
output_video = cv2.VideoWriter('video/less.avi', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))                                                          # Iterate over each frame in the video
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame index
    frame_idx += 1

    # Define the origin point for drawing text labels (bottom-left corner)
    text_origin = (10, frame.shape[0] - 10)  # (x, y) format

    if if_real:
        # Load real data label file if it exists
        real_label_file = os.path.join(real_data_dir, f"{frame_idx:06d}.txt")
        if os.path.exists(real_label_file):
            with open(real_label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:  # Check if data is valid
                        class_id = int(data[0])
                        x, y, w, h = map(float, data[1:])
                        # Convert to pixel coordinates if needed and draw bounding box
                        x1, y1 = int((x - w / 2) * frame.shape[1]), int((y - h / 2) * frame.shape[0])
                        x2, y2 = int(x1 + w * frame.shape[1]), int(y1 + h * frame.shape[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), real_color, 2)
                        # Draw text label
                        cv2.putText(frame, f"Real: {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}", text_origin, font, font_scale*2,
                                    real_color, 1, cv2.LINE_AA)
                        # Update text_origin for the next label
                        text_origin = (text_origin[0], text_origin[1] - 20)
    if if_detect:
        # Load detection data label file if it exists
        detection_label_file = os.path.join(detection_data_dir, f"output_video10_{frame_idx}.txt")
        if os.path.exists(detection_label_file):
            with open(detection_label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:  # Check if data is valid
                        class_id = int(data[0])
                        x, y, w, h = map(float, data[1:])
                        # Convert to pixel coordinates if needed and draw bounding box
                        x1, y1 = int((x - w / 2) * frame.shape[1]), int((y - h / 2) * frame.shape[0])
                        x2, y2 = int(x1 + w * frame.shape[1]), int(y1 + h * frame.shape[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), detection_color, 2)
                        # Draw text label
                        cv2.putText(frame, f"Detection: {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}", text_origin, font, font_scale*2,
                                    detection_color, 1, cv2.LINE_AA)
                        # Update text_origin for the next label
                        text_origin = (text_origin[0], text_origin[1] - 20)

    if if_track:
        # Load tracking data label file if it exists
        tracking_label_file = os.path.join(tracking_data_dir, f"output_video10_3_{frame_idx}.txt")
        if os.path.exists(tracking_label_file):
            with open(tracking_label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5 or len(data) == 6:  # Check if data is valid
                        class_id = int(data[0])
                        x, y, w, h = map(float, data[1:5])
                        # Convert to pixel coordinates if needed and draw bounding box
                        x1, y1 = int((x - w / 2) * frame.shape[1]), int((y - h / 2) * frame.shape[0])
                        x2, y2 = int(x1 + w * frame.shape[1]), int(y1 + h * frame.shape[0])
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), tracking_color, 2)
                        # Draw text label
                        cv2.putText(frame, f"Tracking: {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}", text_origin, font, font_scale*2,
                                    tracking_color, 1, cv2.LINE_AA)
                        # Update text_origin for the next label
                        text_origin = (text_origin[0], text_origin[1] - 20)

    # Show the frame with bounding boxes
    output_video.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
