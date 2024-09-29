import cv2
import numpy as np
import time
import imutils

min_contour_width = 40
min_contour_height = 40
offset = 3
line_height = 200
matches = []
vehicles = 0


def get_centrolid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture("data/output_conveyor_belt-2.mp4")
cap.set(3, 1920)
cap.set(4, 1080)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Get the frame size dynamically
frame_height, frame_width = frame1.shape[:2]

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
out = cv2.VideoWriter('data/output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Initialize variables for FPS calculation
time_deltas = []
fps = 0

ret, frame2 = cap.read()
prev_time = time.time()  # Initialize the previous time

while ret:
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

    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    ret, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((20, 20)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and draw bounding rectangles
    # Also draw a line in the middle of the frame
    # and count the number of centroids that cross the line
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (
                h >= min_contour_height)

        if not contour_valid:
            continue
        cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        cv2.line(frame1, (frame_width // 2 - 200, line_height), (frame_width // 2 + 200, line_height), (0, 255, 0), 2)
        centrolid = get_centrolid(x, y, w, h)
        matches.append(centrolid)
        cv2.circle(frame1, centrolid, 5, (0, 255, 0), -1)
        cx, cy = get_centrolid(x, y, w, h)
        for (x, y) in matches:
            if y < (line_height + offset) and y > (line_height - offset) and x < frame_width // 2 + 200 and x > frame_width // 2 - 200:
                vehicles = vehicles + 1
                matches.remove((x, y))

    cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)
    cv2.putText(frame1, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 70, 0), 2)

    # cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("Vehicle Detection", 1920 - frame1.shape[1], 1080 - frame1.shape[0])
    cv2.imshow("Vehicle Detection", frame1)
    out.write(frame1)  # Write the frame to the output video

    # Resize for display
    th = imutils.resize(th, width=300, height=200)
    dilated = imutils.resize(dilated, width=300, height=200)
    closing = imutils.resize(closing, width=300, height=200)

    # Display output
    cv2.imshow("threshold", th)
    cv2.moveWindow("threshold", 0, 0)

    cv2.imshow("Dilated", dilated)
    cv2.moveWindow("Dilated", 1920 - 300, 0)

    cv2.imshow("Closing", closing)
    cv2.moveWindow("Closing", 0, 1080 - 200)
    cv2.waitKey(30)
    if cv2.waitKey(1) == 27:
        break
    frame1 = frame2
    ret, frame2 = cap.read()

cv2.destroyAllWindows()
cap.release()
out.release()  # Release the VideoWriter object
