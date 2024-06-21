# live_writing_with_recognition.py
import cv2
import numpy as np
import torch
from torch import nn
from model import Network

# Load the pre-trained CNN model
model = Network()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Load the saved HSV values
hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

kernel = np.ones((5, 5), np.uint8)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

x1, y1 = 0, 0
noise_thresh = 500

def preprocess_roi(roi):
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=0)
    return torch.tensor(roi, dtype=torch.float32)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_range = hsv_value[0]
    upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        # Increase the ROI by 20%
        increase_percentage = 0.2
        w = int(w * (1 + increase_percentage))
        h = int(h * (1 + increase_percentage))
        x2 = max(x2 - int(w * increase_percentage / 2), 0)
        y2 = max(y2 - int(h * increase_percentage / 2), 0)
        w = min(w, frame.shape[1] - x2)
        h = min(h, frame.shape[0] - y2)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            canvas = cv2.line(canvas, (x1, y1), (x2, y2), [0, 255, 0], 4)
        
        x1, y1 = x2, y2
    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)

    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == ord('z'):
        if x2 and y2:
            roi = canvas[y2:y2+h, x2:x2+w]
            if roi.size > 0:
                roi_preprocessed = preprocess_roi(roi)
                with torch.no_grad():
                    output = model(roi_preprocessed)
                    digit = torch.argmax(output, dim=1).item()
                print(f"Predicted Digit: {digit}")

    if key & 0xFF == ord('c'):
        canvas = None

cv2.destroyAllWindows()
cap.release()
