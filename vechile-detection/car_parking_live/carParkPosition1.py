import cv2
import pickle

width, height = 107, 48

# Load the existing parking positions or initialize an empty list if file not found
try:
    with open('CarParkPosition1.pkl', 'rb') as f:
        posList = pickle.load(f)
except (FileNotFoundError, EOFError):
    posList = []

# Define the mouse callback function
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                break  # Ensure only one position is removed

    # Save the updated list of positions
    with open('CarParkPosition1.pkl', 'wb') as f:
        pickle.dump(posList, f)

# Main loop
while True:
    img = cv2.imread("park_img.jpg")
    if img is None:
        print("Error: Image file not found. Please check the path.")
        break

    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    
    key = cv2.waitKey(1)
    if key == 27:  # Exit on 'Esc' key
        break

cv2.destroyAllWindows()
