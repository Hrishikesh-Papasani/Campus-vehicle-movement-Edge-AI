import cv2
import easyocr
from flask import Flask, render_template, Response

# Load the trained YOLOv5 model
model = YOLO('yolov8n.pt')  # Adjust the path as necessary

app = Flask(__name__)

reader = easyocr.Reader(['en'])

def detect_number_plate():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    min_area = 500

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)

                    if area > min_area:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        img_roi = img[y1:y2, x1:x2]

                        gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                        _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        result = reader.readtext(thresh_roi)

                        if result:
                            text = str.upper(result[0][-2])
                            print(f"Detected text: {text}")

                            # Display the detected text on the image
                            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            _, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_number_plate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
