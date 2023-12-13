import threading
import cv2
from deepface import DeepFace
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

faceMatch = False

referenceImage = cv2.imread("G:\Projects\Python\Apps\LiveCameraRecognition\example.jpeg")

def checkFace(frame):
    global faceMatch

    try:
        if DeepFace.verify(frame, referenceImage.copy(), enforce_detection=False)['verified']:
            faceMatch = True
        else:
            faceMatch = False

    except ValueError:
        faceMatch = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=checkFace, args=(frame.copy(),)).start()
            except ValueError:
                pass
            counter += 1

            if faceMatch:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

            cv2.imshow("Live", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    cv2.destroyAllWindows()

