import cv2
import numpy as np
import time

# Parameters
min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 250  # Adjust line height according to the frame size
matches = []
vehicles = 0


# Function to get the centroid of a bounding box
def get_centroid(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy


# Capture from the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or adjust if using a different camera
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # Set height

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize variables for FPS calculation
time_deltas = []
fps = 0

# Read two initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

prev_time = time.time()  # Initialize the previous time

while ret:
    # Flip the frame horizontally
    frame1 = cv2.flip(frame1, 1)

    current_time = time.time()
    time_elapsed = current_time - prev_time
    prev_time = current_time

    # Calculate FPS based on time elapsed
    if time_elapsed > 0:
        time_deltas.append(time_elapsed)
        # Keep only the last 10 time intervals
        if len(time_deltas) > 10:
            time_deltas.pop(0)
        # Calculate the average time delta and FPS
        avg_time_delta = sum(time_deltas) / len(time_deltas)
        fps = 1 / avg_time_delta

    # Calculate the absolute difference between the two frames
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours in the frame
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        # Draw bounding boxes and count the objects if the contour is valid
        if contour_valid:
            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
            cv2.line(frame1, (0, line_height), (frame1.shape[1], line_height), (0, 255, 0), 2)

            centroid = get_centroid(x, y, w, h)
            matches.append(centroid)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

            # Check if the detected object crosses the counting line
            for (cx, cy) in matches:
                if (line_height - offset) < cy < (line_height + offset):
                    vehicles += 1
                    matches.remove((cx, cy))

    # Display the count of detected objects and FPS
    cv2.putText(frame1, f"Total Objects Detected: {vehicles}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
    cv2.putText(frame1, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 70, 0), 2)

    # Show the frame with detections
    cv2.imshow("Object Detection", frame1)

    # Break loop on 'ESC' key press
    if cv2.waitKey(1) == 27:
        break

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

# Release resources
cv2.destroyAllWindows()
cap.release()
