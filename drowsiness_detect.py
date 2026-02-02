import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

from config import *
from eye_utils import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
from mouth_utils import preprocess_mouth, get_mouth_roi
from alert import AlertSystem

#Loading models
mouth_model = tf.keras.models.load_model(MOUTH_MODEL_PATH)
alert = AlertSystem(ALARM_SOUND_PATH)

#MediaPipe Face Landmarker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

#Temporal buffers
ear_history = deque(maxlen=EAR_HISTORY)
mouth_history = deque(maxlen=MOUTH_HISTORY)

eye_closed_start = None
prev_time = time.time()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = face_landmarker.detect_for_video(
        mp_image, int(time.time() * 1000)
    )

    eye_state = "AWAKE"
    eye_color = (0, 255, 0)
    mouth_state = "NOT YAWNING"
    mouth_color = (0, 255, 0)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        #Eyes (EAR)
        left_eye  = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]

        left_ear  = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        ear_history.append(ear)
        smooth_ear = np.mean(ear_history)

        eye_closed = smooth_ear < EAR_THRESHOLD

        if eye_closed:
            eye_state = "SLEEPY"
            eye_color = (0, 0, 255)
        else:
            eye_state = "AWAKE"
            eye_color = (0, 255, 0)

        #Draw eye landmarks
        for idx_list in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
            for i in idx_list:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                cv2.circle(frame, (x, y), 2, eye_color, -1)

        #Mouth
        mouth_roi, mouth_box = get_mouth_roi(frame, landmarks, w, h)
        mouth_input = preprocess_mouth(mouth_roi)

        if mouth_input is not None:
            pred = mouth_model.predict(mouth_input, verbose=0)[0][0]
            mouth_history.append(pred)

        smooth_mouth = np.mean(mouth_history) if mouth_history else 0

        if smooth_mouth > YAWN_THRESHOLD:
            mouth_state = "YAWNING"
            mouth_color = (0, 0, 255)

        #Drowsiness logic
        now = time.time()
        if eye_closed:
            if eye_closed_start is None:
                eye_closed_start = now
            elif now - eye_closed_start > CLOSED_EYE_SECONDS:
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "DRIVER DROWSINESS DETECTED",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    3
                )
                alert.play()
        else:
            eye_closed_start = None

        #Drawing mouth box
        if mouth_box:
            x1, y1, x2, y2 = mouth_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    #UI
    cv2.putText(frame, eye_state, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, eye_color, 2)

    cv2.putText(frame, mouth_state, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mouth_color, 2)

    fps = int(1 / (time.time() - prev_time))
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
