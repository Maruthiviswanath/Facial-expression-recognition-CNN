import cv2
import numpy as np
import os
from model import ERModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
)

model = ERModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.current_emotion = "Detecting"
        self.current_confidence = 0.0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, fr = self.video.read()
        if not success:
            return None, self.current_emotion, self.current_confidence

        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis]

            # ✅ ONLY SAFE WAY
            emotion, confidence = model.predict_emotion(roi)

            self.current_emotion = emotion
            self.current_confidence = confidence

            cv2.putText(
                fr,
                f"{emotion} ({int(confidence)}%)",
                (x, y - 10),
                font,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)

            break  # detect ONE face only

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes(), self.current_emotion, self.current_confidence
