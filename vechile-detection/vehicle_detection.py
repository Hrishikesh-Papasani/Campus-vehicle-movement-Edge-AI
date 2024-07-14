import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8n.pt')

# Define vehicle classes
vehicle_classes = ['car', 'threewheeler', 'truck', 'bus', 'cycle']

# Initialize counters
count_in = {cls: 0 for cls in vehicle_classes}
count_out = {cls: 0 for cls in vehicle_classes}

# Initialize previous positions
previous_positions = {}

# Define the axis (horizontal line at the middle of the frame)
axis_y = None  # This will be set once the frame dimensions are known

# Function to detect and count vehicles
def process_frame(frame, direction):
    global count_in, count_out, previous_positions, axis_y
    height, width, _ = frame.shape
    if axis_y is None:
        axis_y = height // 2  # Set the axis at the middle of the frame

    results = model(frame)
    df = results.pandas().xyxy[0]

    # Filter by vehicle classes
    vehicles = df[df['name'].isin(vehicle_classes)]
    
    current_positions = {}
    for idx, vehicle in vehicles.iterrows():
        cls = vehicle['name']
        x_center = (vehicle['xmin'] + vehicle['xmax']) / 2
        y_center = (vehicle['ymin'] + vehicle['ymax']) / 2
        current_positions[idx] = (cls, x_center, y_center)
        
        if idx in previous_positions:
            prev_cls, prev_x, prev_y = previous_positions[idx]
            if prev_y < axis_y <= y_center:
                count_in[cls] += 1
            elif prev_y > axis_y >= y_center:
                count_out[cls] += 1

    previous_positions = current_positions

    # Draw the axis
    cv2.line(frame, (0, axis_y), (width, axis_y), (0, 255, 0), 2)
    return frame

# Initialize video streams
cap_in = cv2.VideoCapture(0)  # Webcam or local camera
cap_out = cv2.VideoCapture('https://192.168.1.40:8080/video')  # Update the URL on your machine

# Check if video streams are opened successfully
if not cap_in.isOpened():
    print("Error: Could not open incoming camera stream.")
    exit()
if not cap_out.isOpened():
    print("Error: Could not open outgoing camera stream.")
    exit()

try:
    while cap_in.isOpened() and cap_out.isOpened():
        ret_in, frame_in = cap_in.read()
        ret_out, frame_out = cap_out.read()

        if not ret_in:
            print("Error: Failed to read frame from incoming camera stream.")
            break
        if not ret_out:
            print("Error: Failed to read frame from outgoing camera stream.")
            break

        # Process frames
        frame_in = process_frame(frame_in, 'in')
        frame_out = process_frame(frame_out, 'out')

        # Display frames (optional)
        cv2.imshow('Incoming', frame_in)
        cv2.imshow('Outgoing', frame_out)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted by user.")

finally:
    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()

    # Print final counts
    print("Final Incoming counts:", count_in)
    print("Final Outgoing counts:", count_out)
