import cv2
import mediapipe as mp
import numpy as np

class HandWritingDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        初始化手势书写检测器
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.is_writing = False
        self.index_tip_position = (0, 0)
        self.landmarks = None

    @staticmethod
    def calculate_distance(point1, point2):
        """计算两点之间的欧式距离"""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    def is_finger_straight(self, finger_tip, finger_root, finger_joints, threshold=0.85):
        """
        判断手指是否伸直
        """
        if self.landmarks is None:
            return False

        tip_to_root = self.calculate_distance(self.landmarks[finger_tip], self.landmarks[finger_root])

        finger_length = sum(self.calculate_distance(self.landmarks[finger_joints[i]], self.landmarks[finger_joints[i + 1]])
                            for i in range(len(finger_joints) - 1))

        return tip_to_root > finger_length * threshold if finger_length >= 0.05 else False

    def update_writing_status(self):
        """更新书写状态"""
        if self.landmarks is None:
            self.is_writing = False
            return

        # thumb_straight = self.is_finger_straight(4, 1, [1, 2, 3, 4], 0.8)
        # index_straight = self.is_finger_straight(8, 5, [5, 6, 7, 8])
        middle_bent = not self.is_finger_straight(12, 9, [9, 10, 11, 12])
        ring_bent = not self.is_finger_straight(16, 13, [13, 14, 15, 16])
        pinky_bent = not self.is_finger_straight(20, 17, [17, 18, 19, 20])

        self.is_writing =  middle_bent and ring_bent and pinky_bent

    def process(self, frame):
        """处理输入帧并检测手势"""
        self.frame_shape = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        self.landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.landmarks = hand_landmarks.landmark
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                self.update_writing_status()
                self.index_tip_position = (int(self.landmarks[8].x * self.frame_shape[1]),
                                           int(self.landmarks[8].y * self.frame_shape[0]))

        return self.is_writing
