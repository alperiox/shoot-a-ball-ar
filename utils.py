from mediapipe import solutions

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2

import numpy as np

def normalized_to_pixel_coordinates(
    normalized_x, normalized_y, image_width, image_height
) -> tuple[int, int]:
    px = solutions.drawing_utils._normalized_to_pixel_coordinates(
        normalized_x, normalized_y, image_width, image_height
    )
    return px


def process_landmarks(results, frame) -> list[list[tuple]]:
    # returns the coordinates of the landmarks

    hand_landmarks_list = results.hand_landmarks
    processed_landmarks_list = []

    h, w, _ = frame.shape

    for landmarks in hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in landmarks
            ]
        )
        landmarks_w_pixel_coordinates = []
        for landmark in landmarks:
            scaled_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, w, h)
            landmarks_w_pixel_coordinates.append(scaled_px)

        processed_landmarks_list.append(landmarks_w_pixel_coordinates)

    return processed_landmarks_list


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image


def prepare_detector():
    # define the detector

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    return detector

