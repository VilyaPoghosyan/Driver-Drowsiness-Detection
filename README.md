Driver Drowsiness Detection System, Real-Time Fatigue Monitoring Using AI
A real-time Driver Drowsiness Detection System that identifies early signs of fatigue using computer vision and deep learning, aimed at improving road safety by preventing fatigue-related accidents.

Features
* Eye Monitoring: Detects prolonged eye closure using Eye Aspect Ratio (EAR) with temporal smoothing.
* Yawning Detection: Monitors yawning behavior using a CNN-based mouth classifier.
* Real-Time Inference: Processes live webcam input with minimal latency.
* Smart Alerts: Triggers visual and audio alerts only after sustained drowsiness, reducing false alarms.
* Driver-Focused: Tracks only the driver’s face, ignoring passengers and background activity.

How It Works (Pipeline)
Webcam → Face Landmarker → Eye & Mouth Analysis → Temporal Logic → Drowsiness Alert
* Eyes: Computes EAR from facial landmarks and tracks it over time to detect sustained eye closure.
* Mouth: A trained CNN classifies mouth regions as yawning or not yawning.
* Decision Logic: Alerts are triggered only when drowsiness indicators persist for a defined duration.

Datasets Used
* Eye Behavior: MRL Eye Dataset — includes open and closed eyes, with and without spectacles.
* Yawning Detection: Kaggle Yawn Dataset — labeled yawning / non-yawning images.

Tech Stack
* Languages & Libraries: Python, TensorFlow/Keras, OpenCV, NumPy
* Face Analysis: MediaPipe Face Landmarker
* Key Concepts: CNNs, real-time inference, temporal smoothing, geometric eye metrics (EAR), threshold-based alerting

