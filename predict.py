import argparse
import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 

IMG_SIZE           = 80
EAR_THRESHOLD      = 0.23
YAWN_THRESHOLD     = 0.5
MOUTH_MODEL_PATH   = "mouth_cnn.h5"
FACE_LANDMARKER_PATH = "face_landmarker.task"
 
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78]

 
 
def load_models():
    """Load the saved CNN mouth model and MediaPipe face landmarker."""
    print("[INFO] Loading mouth CNN model from:", MOUTH_MODEL_PATH)
    mouth_model = tf.keras.models.load_model(MOUTH_MODEL_PATH)
 
    print("[INFO] Loading MediaPipe Face Landmarker from:", FACE_LANDMARKER_PATH)
    base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
 
    return mouth_model, face_landmarker
 
 
def eye_aspect_ratio(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR) from 6 facial landmark points.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A low EAR (< threshold) indicates the eye is closed.
    """
    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
 
    A = dist(eye_points[1], eye_points[5])
    B = dist(eye_points[2], eye_points[4])
    C = dist(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)
 
 
def get_mouth_roi(frame, landmarks, w, h, pad=35):
    """
    Extract the mouth region of interest (ROI) from the frame
    using MediaPipe landmark indices.
    """
    xs = [int(landmarks[i].x * w) for i in MOUTH_IDX]
    ys = [int(landmarks[i].y * h) for i in MOUTH_IDX]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return frame[y1:y2, x1:x2]
 
 
def preprocess_mouth(roi):
    """
    Apply the EXACT same preprocessing used during training:
      - Resize to IMG_SIZE x IMG_SIZE (80x80)
      - Normalize pixel values to [0, 1]
      - Reshape to (1, 80, 80, 3) for model input
    """
    if roi is None or roi.size == 0:
        return None
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized / 255.0
    return roi_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)
 
 
def run_inference(image_path: str, mouth_model, face_landmarker):
    """
    Run full drowsiness inference on a single static image.
    Returns a dict with eye state, mouth state, confidence, and final verdict.
    """
 
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)
 
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 

    result = face_landmarker.detect(mp_image)
 
    if not result.face_landmarks:
        print("[WARN] No face detected in the image.")
        return {
            "eye_state": "NO FACE DETECTED",
            "ear": None,
            "mouth_state": "N/A",
            "mouth_confidence": None,
            "verdict": "UNABLE TO DETERMINE — no face found"
        }
 
    landmarks = result.face_landmarks[0]
 
    
    left_eye  = [landmarks[i] for i in LEFT_EYE_IDX]
    right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
    left_ear  = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear       = (left_ear + right_ear) / 2.0
 
    eye_closed = ear < EAR_THRESHOLD
    eye_state  = "CLOSED (SLEEPY)" if eye_closed else "OPEN (AWAKE)"
 

    mouth_roi   = get_mouth_roi(frame, landmarks, w, h)
    mouth_input = preprocess_mouth(mouth_roi)
 
    mouth_state      = "N/A"
    mouth_confidence = None
 
    if mouth_input is not None:
    
        raw_pred = mouth_model.predict(mouth_input, verbose=0)[0][0]
        mouth_confidence = float(raw_pred)
        is_yawning = mouth_confidence > YAWN_THRESHOLD
        mouth_state = "YAWNING" if is_yawning else "NOT YAWNING"
 
    
    if eye_closed and mouth_state == "YAWNING":
        verdict = "DROWSY — Eyes closed AND yawning detected"
    elif eye_closed:
        verdict = "DROWSY — Eyes closed (prolonged closure)"
    elif mouth_state == "YAWNING":
        verdict = "DROWSY — Yawning detected"
    else:
        verdict = "ALERT — No drowsiness indicators"
 
    return {
        "eye_state": eye_state,
        "ear": ear,
        "mouth_state": mouth_state,
        "mouth_confidence": mouth_confidence,
        "verdict": verdict
    }
 
 
def print_results(results: dict):
    """Print human-readable, formatted prediction results."""
    print("\n" + "=" * 50)
    print("   DROWSINESS DETECTION — INFERENCE RESULT")
    print("=" * 50)
 
    ear_str = f"{results['ear']:.3f}" if results['ear'] is not None else "N/A"
    print(f"  Eye State    : {results['eye_state']:<22} (EAR: {ear_str})")
 
    if results["mouth_confidence"] is not None:
        conf_pct = results["mouth_confidence"] * 100
        print(f"  Mouth State  : {results['mouth_state']:<22} | Confidence: {conf_pct:.1f}%")
    else:
        print(f"  Mouth State  : {results['mouth_state']}")
 
    print("-" * 50)
    print(f"  Final Verdict: {results['verdict']}")
    print("=" * 50 + "\n")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection — Static Image Inference"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to an input face image (e.g., test_face.jpg)"
    )
    args = parser.parse_args()
 

    mouth_model, face_landmarker = load_models()
 
   
    print(f"\n[INFO] Running inference on: {args.image}")
    results = run_inference(args.image, mouth_model, face_landmarker)
 
    
    print_results(results)
 
 
if __name__ == "__main__":
    main()