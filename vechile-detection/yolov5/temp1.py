import cv2
import torch

# Load the YOLOv5 model
model_path = 'D:/Yamini_pro/yolov5/runs/train/exp12/weights/best.pt'  # Update this to your model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Define vehicle classes
vehicle_classes = ['car', 'motorbike', 'threewheeler', 'truck', 'bus', 'cycle']

# Initialize counters
count_in = {cls: 0 for cls in vehicle_classes}
count_out = {cls: 0 for cls in vehicle_classes}

# Function to detect and count vehicles
def process_frame(frame, direction):
    global count_in, count_out
    results = model(frame)
    df = results.pandas().xyxy[0]
    
    # Filter by vehicle classes
    vehicles = df[df['name'].isin(vehicle_classes)]
    
    # Update counts
    for idx, vehicle in vehicles.iterrows():
        cls = vehicle['name']
        if direction == 'in':
            count_in[cls] += 1
        else:
            count_out[cls] += 1
    
    return frame

# Initialize video streams
cap_in = cv2.VideoCapture('incoming_camera_stream_url')
cap_out = cv2.VideoCapture('outgoing_camera_stream_url')

while cap_in.isOpened() and cap_out.isOpened():
    ret_in, frame_in = cap_in.read()
    ret_out, frame_out = cap_out.read()
    
    if not ret_in or not ret_out:
        break
    
    # Process frames
    frame_in = process_frame(frame_in, 'in')
    frame_out = process_frame(frame_out, 'out')
    
    # Display frames (optional)
    cv2.imshow('Incoming', frame_in)
    cv2.imshow('Outgoing', frame_out)
    
    # Print counts
    print("Incoming counts:", count_in)
    print("Outgoing counts:", count_out)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_in.release()
cap_out.release()
cv2.destroyAllWindows()
