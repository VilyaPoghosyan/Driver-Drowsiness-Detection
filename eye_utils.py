import numpy as np

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points):
    def dist(a, b):
        return np.linalg.norm(
            np.array([a.x, a.y]) - np.array([b.x, b.y])
        )

    A = dist(eye_points[1], eye_points[5])
    B = dist(eye_points[2], eye_points[4])
    C = dist(eye_points[0], eye_points[3])

    return (A + B) / (2.0 * C)
