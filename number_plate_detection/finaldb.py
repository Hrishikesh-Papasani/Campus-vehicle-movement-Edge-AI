import cv2
import os
from ultralytics import YOLO
import easyocr
import sqlite3
from datetime import datetime

# Load the trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Adjust the path as necessary

def connect_to_database():
    conn = sqlite3.connect('vehicle_records.db')  # Change the path as necessary
    return conn

def create_table():
    conn = connect_to_database()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_records (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 vehicle_type TEXT,
                 vehicle_number TEXT UNIQUE,
                 timestamp_in TEXT,
                 timestamp_out TEXT)''')
    conn.commit()
    conn.close()

def insert_vehicle_data(vehicle_type, vehicle_number):
    conn = connect_to_database()
    c = conn.cursor()
    c.execute("SELECT * FROM vehicle_records WHERE vehicle_number=?", (vehicle_number,))
    existing_data = c.fetchone()
    if existing_data:
        print("Vehicle number already exists. Skipping insertion.")
        conn.close()
        return

    timestamp_in = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    timestamp_out = ""  # Will be updated when the vehicle exits

    c.execute("INSERT INTO vehicle_records (vehicle_type, vehicle_number, timestamp_in, timestamp_out) VALUES (?, ?, ?, ?)",
              (vehicle_type, vehicle_number, timestamp_in, timestamp_out))
    conn.commit()
    conn.close()
    print("Data inserted successfully.")

# Create the table if it doesn't exist
create_table()

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

min_area = 500
count = 0

# Specify the directory to save images
output_dir = "C:\\Users\\Sai Kishore\\Desktop\\Numplate\\plates"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

reader = easyocr.Reader(['en'])

detection_counts = {}

try:
    while True:
        success, img = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        img_roi = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                area = (x2 - x1) * (y2 - y1)

                if area > min_area:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Number Plate", (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                    img_roi = img[y1:y2, x1:x2]

                    gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cv2.imshow("ROI", thresh_roi)
                    result = reader.readtext(thresh_roi)

                    if result:
                        text = result[0][-2]
                        text=str.upper(text)
                        print(f"Detected text: {text}")

                        if text in detection_counts:
                            detection_counts[text] += 1
                        else:
                            detection_counts[text] = 1

                        if detection_counts[text] > 5:
                            insert_vehicle_data("Unknown", text)
                            detection_counts[text] = 0  # Reset the count after insertion

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('s') and img_roi is not None:
            save_path = os.path.join(output_dir, f"scanned_img_{count}.jpg")
            cv2.imwrite(save_path, img_roi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            count += 1
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
