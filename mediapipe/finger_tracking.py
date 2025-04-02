import cv2
import numpy as np
from hand_writing_detector import HandWritingDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

detector = HandWritingDetector()
trajectory = []
MAX_TRAJECTORY_POINTS = 50
prev_index_pos = None
WRITING_MOTION_THRESHOLD = 15
JITTER_THRESHOLD = 3

def smart_smooth(current_pos, prev_pos):
    """智能平滑函数，区分有意运动和抖动"""
    if prev_pos is None:
        return current_pos

    distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))

    if distance > WRITING_MOTION_THRESHOLD:
        return current_pos
    elif distance < JITTER_THRESHOLD:
        return prev_pos
    else:
        alpha = 0.3
        return (int(alpha * current_pos[0] + (1 - alpha) * prev_pos[0]),
                int(alpha * current_pos[1] + (1 - alpha) * prev_pos[1]))

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # frame = cv2.flip(frame, 1)
        is_writing = detector.process(frame)
        current_index_pos = detector.index_tip_position

        if is_writing and current_index_pos != (0, 0):
            smoothed_pos = smart_smooth(current_index_pos, prev_index_pos)
            prev_index_pos = smoothed_pos

            trajectory.append(smoothed_pos)
            if len(trajectory) > MAX_TRAJECTORY_POINTS:
                trajectory.pop(0)

            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

            cv2.circle(frame, smoothed_pos, 10, (0, 0, 255), -1)
        else:
            trajectory = []
            prev_index_pos = None

        status_text = "Writing: YES" if is_writing else "Writing: NO"
        status_color = (0, 255, 0) if is_writing else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("Smart Finger Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
