import cv2
import subprocess
import re
import signal
import psutil  # Import psutil module to manage subprocesses
from collections import defaultdict
import threading

def kill_process_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

def run_detection(source, source_name, vehicle_counts, overall_count, consecutive_no_detections, stop_event):
    # Define the command to run
    command = f'python detect.py --weights "D:/Yamini_pro/yolov5/runs/train/exp12/weights/best.pt" --img 640 --conf 0.25 --source {source}'

    # Start the subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    vehicle_classes = ["car", "threewheel", "bus", "truck", "motorbike", "van"]

    try:
        for line in iter(process.stdout.readline, b""):
            if stop_event.is_set():
                break
            line = line.decode("utf-8").strip()
            print(f"{source_name}: {line}")

            detected = False
            for vehicle_class in vehicle_classes:
                match = re.search(rf"\b{vehicle_class}\b", line, re.IGNORECASE)
                if match:
                    vehicle_counts[vehicle_class] += 1
                    overall_count[0] += 1
                    consecutive_no_detections[0] = 0
                    print(f"Detected {source_name} {vehicle_class}. Count so far: {vehicle_counts[vehicle_class]}")
                    detected = True
                    break

            if not detected:
                if re.search(r"\bno detections\b", line, re.IGNORECASE):
                    consecutive_no_detections[0] += 1
                    if consecutive_no_detections[0] >= 100:
                        print(f"Stopping {source_name} process due to no detections...")
                        kill_process_tree(process.pid)
                        break
    except KeyboardInterrupt:
        print(f"Keyboard interrupt detected in {source_name}. Exiting...")
    finally:
        process.kill()  # Kill the process if it's still running

def capture_and_display(source, window_name, stop_event):
    # Open the video source
    cap = cv2.VideoCapture(source)
    
    # Check if the source opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Set the desired frame width and height (e.g., 640x480)
    frame_width = 640
    frame_height = 480

    # Main loop to capture and display frames
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from {source}")
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Display the frame in the specified window
        cv2.imshow(window_name, frame)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    # Release the video source and close the window
    cap.release()
    cv2.destroyWindow(window_name)

def ambuscan():
    # Define the sources
    webcam_source = 0  # Local webcam
    ip_camera_source = "http://192.0.0.4:8080/video"  # IP camera feed URL

    # Initialize variables
    vehicle_counts_webcam = defaultdict(int)
    vehicle_counts_ip_camera = defaultdict(int)
    overall_count_webcam = [0]
    overall_count_ip_camera = [0]
    consecutive_no_detections_webcam = [0]
    consecutive_no_detections_ip_camera = [0]

    # Stop event to control the threads
    stop_event = threading.Event()

    # Create threads for capturing and displaying frames
    thread_webcam_capture = threading.Thread(target=capture_and_display, args=(webcam_source, "Webcam", stop_event))
    thread_ip_camera_capture = threading.Thread(target=capture_and_display, args=(ip_camera_source, "IP Camera", stop_event))

    # Create threads for vehicle detection
    thread_webcam_detection = threading.Thread(target=run_detection, args=(webcam_source, "Webcam", vehicle_counts_webcam, overall_count_webcam, consecutive_no_detections_webcam, stop_event))
    thread_ip_camera_detection = threading.Thread(target=run_detection, args=(ip_camera_source, "IP Camera", vehicle_counts_ip_camera, overall_count_ip_camera, consecutive_no_detections_ip_camera, stop_event))

    # Start the threads
    thread_webcam_capture.start()
    thread_ip_camera_capture.start()
    thread_webcam_detection.start()
    thread_ip_camera_detection.start()

    # Wait for the capture threads to finish
    thread_webcam_capture.join()
    thread_ip_camera_capture.join()
    
    # Set the stop event to end detection threads if capture threads have finished
    stop_event.set()

    # Wait for the detection threads to finish
    thread_webcam_detection.join()
    thread_ip_camera_detection.join()

    # Print final counts
    print("\nFinal webcam vehicle counts:")
    for vehicle_class, count in vehicle_counts_webcam.items():
        print(f"{vehicle_class}: {count}")
    print(f"Overall webcam vehicle count: {overall_count_webcam[0]}")

    print("\nFinal IP camera vehicle counts:")
    for vehicle_class, count in vehicle_counts_ip_camera.items():
        print(f"{vehicle_class}: {count}")
    print(f"Overall IP camera vehicle count: {overall_count_ip_camera[0]}")

ambuscan()
