import subprocess
import re
import signal
import psutil  # Import psutil module to manage subprocesses
from collections import defaultdict

def kill_process_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()

def ambuscan():
    # Define the command to run
    command = 'python detect.py --weights "D:/Yamini_pro/yolov5/runs/train/exp12/weights/best.pt" --img 640 --conf 0.25 --source=0'

    # Start the subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    
    # Initialize variables
    consecutive_no_detections = 0

    # Dictionary to store the count of each vehicle class
    vehicle_counts = defaultdict(int)
    vehicle_classes = ["car", "threewheel", "bus", "truck", "motorbike", "van"]

    # Overall count of vehicles
    overall_count = 0

    try:
        for line in iter(process.stdout.readline, b""):
            line = line.decode("utf-8").strip()
            print(line)
            
            detected = False
            # Check for detections in the line
            for vehicle_class in vehicle_classes:
                match = re.search(rf"\b{vehicle_class}\b", line, re.IGNORECASE)
                if match:
                    vehicle_counts[vehicle_class] += 1
                    overall_count += 1
                    consecutive_no_detections = 0
                    print(f"Detected {vehicle_class}. Count so far: {vehicle_counts[vehicle_class]}")
                    detected = True
                    break
            
            if not detected:
                if re.search(r"\bno detections\b", line, re.IGNORECASE):
                    consecutive_no_detections += 1
                    if consecutive_no_detections >= 100:
                        print("Stopping program due to no detections...")
                        # Terminate the subprocess and its children
                        kill_process_tree(process.pid)
                        break
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    finally:
        # Clean up resources
        process.kill()  # Kill the process if it's still running

        # Print final counts
        print("\nFinal vehicle counts:")
        for vehicle_class, count in vehicle_counts.items():
            print(f"{vehicle_class}: {count}")
        print(f"Overall vehicle count: {overall_count}")

ambuscan()
