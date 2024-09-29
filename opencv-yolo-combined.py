import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLO model from the specified path
model = YOLO('/home/pardusumsu/code/Counting-Sheep/drone-detect.pt')

# Parameters
min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 250
vehicles = 0
frame_count = 0
detection_interval = 5  # Run reliability check every 10 frames
min_trackers = 3
reliability_check_interval = 50  # YOLO-based reliability check interval

# Flags to control the counting mode
use_opencv_tracking = True  # Start with OpenCV-only tracking

# OpenCV Tracker (using CSRT tracker for YOLO-supported tracking)
trackers = cv2.legacy.MultiTracker_create()

# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize variables for FPS calculation
fps = 0
prev_time = time.time()

# Read initial frames for OpenCV-only counting
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Get the current time to calculate FPS
    current_time = time.time()
    time_elapsed = current_time - prev_time
    prev_time = current_time

    # Calculate FPS based on time elapsed
    if time_elapsed > 0:
        fps = 1 / time_elapsed

    # Check which mode is active
    if use_opencv_tracking:
        # OpenCV-only tracking
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

            if not contour_valid:
                continue

            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
            centroid = (x + w // 2, y + h // 2)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

            # Check if the centroid crosses the line
            if (line_height - offset) < centroid[1] < (line_height + offset):
                vehicles += 1

        # Draw the counting line
        cv2.line(frame1, (0, line_height), (frame1.shape[1], line_height), (0, 255, 0), 2)

        # Prepare for next frame in OpenCV-only mode
        display_frame = frame1.copy()  # Use frame1 as the display frame
        frame1 = frame2
        ret, frame2 = cap.read()
        frame2 = cv2.flip(frame2, 1)  # Flip the next frame as well

    else:
        # YOLO-Supported Tracking Mode
        # Update trackers
        success, boxes = trackers.update(frame)
        display_frame = frame.copy()  # Use the current frame as the display frame

        # Periodically run YOLO for reliability check
        if frame_count % reliability_check_interval == 0 or not success:
            results = model(frame)
            yolo_detections = results[0].boxes.data.cpu().numpy() if len(results) > 0 else []

            # Check consistency between YOLO detections and tracker predictions
            yolo_boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, _, _ in yolo_detections]
            tracker_boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in boxes]

            # Check the reliability by comparing YOLO detections with tracker predictions
            reliability_ok = len(yolo_boxes) == len(tracker_boxes) and all(
                [cv2.norm(np.array(y), np.array(t)) < 50 for y, t in zip(yolo_boxes, tracker_boxes)]
            )

            # If reliability is not ok, reset the trackers with YOLO detections
            if not reliability_ok:
                print("Tracker reliability failed. Resetting with YOLO detections.")
                trackers = cv2.legacy.MultiTracker_create()
                for x1, y1, x2, y2 in yolo_boxes:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    trackers.add(tracker, frame, (x1, y1, x2 - x1, y2 - y1))

        # Draw tracked boxes
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            centroid = (x + w // 2, y + h // 2)
            cv2.circle(display_frame, centroid, 5, (0, 255, 0), -1)

            # Check if the centroid crosses the line
            if (line_height - offset) < centroid[1] < (line_height + offset):
                vehicles += 1

        # Draw the counting line
        cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Display common annotations
    cv2.putText(display_frame, f"Total Objects Detected: {vehicles}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the unified display frame
    cv2.imshow("Object Counting", display_frame)

    # Check for key presses to switch modes
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('1'):  # Press '1' for OpenCV-only tracking
        use_opencv_tracking = True
        print("Switched to OpenCV-only tracking.")
    elif key == ord('2'):  # Press '2' for YOLO-supported tracking
        use_opencv_tracking = False
        print("Switched to YOLO-supported tracking.")

    # Increment frame count
    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

