import cv2
import os
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Calculate point a and point b distance
def calculate_distance(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    distance = np.linalg.norm(a - b)

    return distance


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angles = np.abs(radians * 180.0 / np.pi)

    if angles > 180.0:
        angles = 360 - angles

    return angles


url = 'rtsp://root:2858@*%*@120.110.115.130/axis-media/media.amp'
urls = 'rtsp://admin:lab515password@120.110.115.137:554/media/video1'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(urls, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)


# Curl counter variables
counter = 0
trigger = False
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Get Size of frame
        width = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x * width,
                     landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y * height]
            index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * width,
                     landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * height]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            distances = calculate_distance(index, mouth)

            # Visualize angle
            cv2.putText(image, str(distances),
                        tuple(np.multiply(index, [width, height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic
            # if angle > 160:
            #     stage = "down"
            # if angle < 30 and stage == 'down':
            #     stage = "up"
            #     counter += 1
            #     print(counter)
            if distances <= 20:
                if not trigger:
                    trigger = True
                    counter += 1
            elif distances >= 60:
                if trigger:
                    trigger = False

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()