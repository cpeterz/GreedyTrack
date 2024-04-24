import cv2
import os

def video_to_images(video_file, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Failed to open video file")
        return

    # Read and save each frame as an image
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        output_file = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_file, frame)

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Successfully saved {frame_count} frames as images in {output_folder}")

# Specify the path to the input video file
video_file = "video/less.avi"

# Specify the path to the output folder
output_folder = "myimages/4"

# Convert the video to images
video_to_images(video_file, output_folder)
