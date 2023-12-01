#!/usr/bin/env python3
import cv2
import numpy as np

# Load the video
video_path = "/Users/yashrajshanker/Desktop/Fall 2023/Research/OpenCV/11-21-1.mov" # Path File
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"The video at {video_path} could not be opened.")
    exit(1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'XVID' for AVI
output_path = "/Users/yashrajshanker/Desktop/Fall 2023/Research/OpenCV/processed_video.mp4"  # Change to your desired output path
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, img = cap.read()
    if not ret:
        break  # Break the loop if there are no more frames

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    ret, binary = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contour(s) and draw the guide wire
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Example: only draw contours larger than 100 in area
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    out.write(img)
    # Display the results
    cv2.imshow('Guide Wire Detected', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()