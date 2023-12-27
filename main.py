import cv2
from mediapipe import solutions

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2
import numpy as np


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


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


# get the current camera

cap = cv2.VideoCapture(1)
# get the webcam resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

position = (600, 400)
radius = 15

circle_check = lambda x, y: (x - position[0]) ** 2 + (y - position[1]) ** 2 <= radius**2

interaction_radius = 45
interaction_check = (
    lambda x, y: (x - position[0]) ** 2 + (y - position[1]) ** 2
    < interaction_radius**2 - 5
)
acceleration = 0
velocity = 0
friction = 0.5
g_vector = (0, -20)

# check flags
draw_line_check = False
ball_shot_check = False

# variables to store the previous locations of thumb and index finger
prev_thumb_coords = None
prev_index_coords = None

# ellipse constants (aka the hoop)
hoop_position = (1000, 250)
hoop_radius = (60, 25)
hoop_angle = -30
hoop_color = (0, 0, 255)
hoop_thickness = 2

# check for the hoop
hoop_equation = (lambda x, y: 
    (x - hoop_position[0]) ** 2 / hoop_radius[0] ** 2
    + (y - hoop_position[1]) ** 2 / hoop_radius[1] ** 2)
hoop_check = lambda x, y: ( hoop_equation(x,y) <= 1 )
hoop_flag = False
score = 0

debug = False

# prepare the detector
detector = prepare_detector()

while cap.isOpened():
    success, image = cap.read()

    # get the active frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # mirror the image
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # draw a circle at the initial position wiht initial radius
    image = cv2.circle(image, position, radius, (0, 0, 255), -1)

    if debug:
        # draw a circle at the initial position with interaction radius
        image = cv2.circle(image, position, interaction_radius, (0, 255, 0), 1)

    # draw an ellipse to use as a basketball hoop

    image = cv2.ellipse(
        image,
        hoop_position,
        hoop_radius,
        hoop_angle,
        0,
        360,
        hoop_color,
        hoop_thickness,
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    detection_result = detector.detect(image)
    np_image = image.numpy_view()
    processed_landmarks = process_landmarks(detection_result, np_image)
    annotated_image = np_image
    if debug:
        annotated_image = draw_landmarks_on_image(np_image, detection_result)

    if debug:
        # write the score to the right edge
        annotated_image = cv2.putText(
            annotated_image,
            f"score: {score}",
            (width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            (0, 0, 0),
            FONT_THICKNESS,
        )

    # check if any hand is detected
    if len(processed_landmarks) > 0:
        # check if the thumbs and index fingers are in the interaction circle, the indices for the landmarks are 4 and 8
        # I'll use the first hand for now
        thumb_coords = processed_landmarks[0][4]
        index_coords = processed_landmarks[0][8]

        if thumb_coords is None or index_coords is None:
            continue

        if prev_thumb_coords is None:
            prev_thumb_coords = thumb_coords
        if prev_index_coords is None:
            prev_index_coords = index_coords

        # write the previous and current coordinates of the thumb and index finger to the screen
        if debug:
            annotated_image = cv2.putText(
                annotated_image,
                f"prev thumb: {prev_thumb_coords}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"prev index: {prev_index_coords}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"thumb: {thumb_coords}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"index: {index_coords}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )

        # distance between the index finger and the thumb
        distance = np.sqrt(
            (thumb_coords[0] - index_coords[0]) ** 2
            + (thumb_coords[1] - index_coords[1]) ** 2
        )
        if debug:
            # write the distance to the screen
            annotated_image = cv2.putText(
                annotated_image,
                f"distance: {distance:.4f}",
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )

        thumb_in_circle = interaction_check(thumb_coords[0], thumb_coords[1])
        index_in_circle = interaction_check(index_coords[0], index_coords[1])

        if debug:
            # also write the interaction check results to the screen
            annotated_image = cv2.putText(
                annotated_image,
                f"thumb in circle: {thumb_in_circle}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"index in circle: {index_in_circle}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )

        if (
            thumb_in_circle and index_in_circle
        ):  # meaning that both the thumb and index finger are in the interaction circle and now can be used to control the ball
            # use the previous position of the thumb and index finger, then calculate the new center for the circle
            # if previous coordinates are None, then set it to the current positions

            # calculate the new center using the average change in the thumb and index finger positions

            # reset the previous force vector
            draw_line_check = False
            ball_shot_check = False

            if distance < 30:
                draw_line_check = True
                velocity = 0 # reset the velocity since we're holding the ball now
                ball_shot_check = False

            else:
                ### Moving the circle using the average change in the thumb and index finger positions
                # calculate the average change in the thumb and index finger positions
                thumb_change = (
                    thumb_coords[0] - prev_thumb_coords[0],
                    thumb_coords[1] - prev_thumb_coords[1],
                )
                index_change = (
                    index_coords[0] - prev_index_coords[0],
                    index_coords[1] - prev_index_coords[1],
                )

                # calculate the average change in the thumb and index finger positions
                avg_change = (
                    (thumb_change[0] + index_change[0]) / 2,
                    (thumb_change[1] + index_change[1]) / 2,
                )

                # calculate the new center
                new_center = (
                    int(position[0] + avg_change[0]),
                    int(position[1] + avg_change[1]),
                )

                if debug:
                    # write the change and average change along with the new center
                    annotated_image = cv2.putText(
                        annotated_image,
                        f"thumb change: {thumb_change}",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE,
                        (0, 0, 0),
                        FONT_THICKNESS,
                    )
                    annotated_image = cv2.putText(
                        annotated_image,
                        f"index change: {index_change}",
                        (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE,
                        (0, 0, 0),
                        FONT_THICKNESS,
                    )
                    annotated_image = cv2.putText(
                        annotated_image,
                        f"avg change: ({avg_change[0]:.4f}, {avg_change[1]:.4f})",
                        (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE,
                        (0, 0, 0),
                        FONT_THICKNESS,
                    )
                    annotated_image = cv2.putText(
                        annotated_image,
                        f"new center: {new_center}",
                        (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE,
                        (0, 0, 0),
                        FONT_THICKNESS,
                    )
                # set the new center
                position = new_center

        # update the previous coordinates
        prev_thumb_coords = thumb_coords
        prev_index_coords = index_coords

        if draw_line_check:
            # start a line from the center of the circle to the middle of the thumb and index finger
            velocity = 0
            ball_shot_check = 0

            annotated_image = cv2.line(
                annotated_image,
                position,
                (
                    (thumb_coords[0] + index_coords[0]) // 2,
                    (thumb_coords[1] + index_coords[1]) // 2,
                ),
                (0, 0, 255),
                2,
            )

            # calculate the force vector
            force_vector = (
                -((thumb_coords[0] + index_coords[0]) // 2 - position[0]),
                -((thumb_coords[1] + index_coords[1]) // 2 - position[1]),
            )

            if debug:
                
                # write the force vector to the screen
                annotated_image = cv2.putText(
                    annotated_image,
                    f"force vector: ({force_vector[0]:.4f}, {force_vector[1]:.4f})",
                    (10, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE,
                    (0, 0, 0),
                    FONT_THICKNESS,
                )

            # put the vector to the center of the circle
            force_vector_coords = (
                force_vector[0] + position[0],
                force_vector[1] + position[1],
            )
            # put the force vector to the point
            annotated_image = cv2.line(
                annotated_image, position, force_vector_coords, (255, 0, 0), 2
            )

            if distance >= 30:
                ball_shot_check = True
                draw_line_check = False

    if ball_shot_check:
        # since we let go of the thumb and index finger, we can now move the circle using the force vector
        friction_vector = (
            -force_vector[0] * friction / fps,
            -force_vector[1] * friction / fps,
        )
        force_vector = (
            force_vector[0] + friction_vector[0],
            force_vector[1] + friction_vector[1],
        )
        # add the gravity vector to the force vector
        force_vector = (
            force_vector[0] - ( g_vector[0] / fps),
            force_vector[1] - ( g_vector[1] / fps),
        )

        # I want to move it smoothly so we will use an acceleration variable to control the speed of the circle in the direction of the force vector

        # use the force vector to calculate the acceleration
        acceleration = np.sqrt(force_vector[0] ** 2 + force_vector[1] ** 2) / 100
        # calculate the velocity
        velocity = acceleration

        if debug:
            # write the velocity and acceleration to the screen
            annotated_image = cv2.putText(
                annotated_image,
                f"velocity: {velocity:.4f}",
                (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"acceleration: {acceleration:.4f}",
                (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"friction vector: ({friction_vector[0]:.4f},{friction_vector[1]:.4f})",
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )
            annotated_image = cv2.putText(
                annotated_image,
                f"force vector: ({force_vector[0]:.4f}, {force_vector[1]:.4f})",
                (10, 480),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )

        # calculate the new position
        new_position = (
            (position[0] + int(velocity * force_vector[0])),
            (position[1] + int(velocity * force_vector[1])),
        )
        position = new_position

        if hoop_check(position[0], position[1]):
            score += 1

        if debug:
            # write the new position to the screen
            annotated_image = cv2.putText(
                annotated_image,
                f"new position: {new_position}",
                (10, 510),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                (0, 0, 0),
                FONT_THICKNESS,
            )

        # if the circle is outside of the screen, then calculate the angle of the force vector and reflect it
        # just get the normal vector, then use the reflection formula: r = d - 2(d.n)n where d is the force vector and n is the normal vector

        # get the normal vector of the surface that it touched
        if position[0]-radius < 0 and ((0 < position[1]-radius) and (position[1]+radius<height)):  # left side
            normal_vector = (1, 0)
        elif position[0]+radius > width and ((0 < position[1]-radius) and (position[1]+radius<height)):  # right side
            normal_vector = (-1, 0)
        elif position[1]-radius < 0 and ((0 <= position[0]-radius) and (position[1]+radius < width)):  # top side
            normal_vector = (0, -1)
        elif position[1]+radius > height and ((0 <= position[0]-radius) and (position[1]+radius < width)):  # bottom side
            normal_vector = (0, 1)
        else:
            normal_vector = None

        if normal_vector is not None:
            # calculate the reflection vector
            reflection_vector = (
                force_vector[0]
                - 2
                * (
                    force_vector[0] * normal_vector[0]
                    + force_vector[1] * normal_vector[1]
                )
                * normal_vector[0],
                force_vector[1]
                - 2
                * (
                    force_vector[0] * normal_vector[0]
                    + force_vector[1] * normal_vector[1]
                )
                * normal_vector[1],
            )

            if debug:
                # write the reflection vector to the screen
                annotated_image = cv2.putText(
                    annotated_image,
                    f"reflection vector: ({reflection_vector[0]:.4f}, {reflection_vector[1]:.4f})",
                    (10, 540),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE,
                    (0, 0, 0),
                    FONT_THICKNESS,
                )
            # set the force vector to the reflection vector
            force_vector = reflection_vector


        # if velocity is 0, then stop the ball
        if velocity <= 0:
            ball_shot_check = False

    if debug:
        # show the frame rate
        annotated_image = cv2.putText(
            annotated_image,
            f"fps: {fps}",
            (10, 570),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            (0, 0, 0),
            FONT_THICKNESS,
        )

    # show the annotated frame

    cv2.imshow("Demo", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
    if cv2.waitKey(5) & 0xFF == ord("r"):
        draw_line_check = False
        ball_shot_check = False
        position = (600, 400)
    if cv2.waitKey(5) & 0xFF == ord("d"):
        debug = not debug

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
