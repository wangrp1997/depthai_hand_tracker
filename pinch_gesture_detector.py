import numpy as np
from collections import deque

class PinchGestureDetector:
    def __init__(self, distance_threshold=0.1, confidence_window=30):
        self.distance_threshold = distance_threshold
        self.confidence_window = confidence_window
        # 为每个手指存储历史距离数据
        self.distance_history = {
            'index': deque(maxlen=confidence_window),
            'middle': deque(maxlen=confidence_window),
            'ring': deque(maxlen=confidence_window),
            'pinky': deque(maxlen=confidence_window)
        }
        
    def calculate_finger_distance(self, hand, finger_tip_idx):
        """计算拇指和其他手指指尖之间的3D空间距离"""
        if hand is None or hand.norm_landmarks is None:
            return None
            
        # 使用3D坐标 [x,y,z]
        thumb_tip = hand.norm_landmarks[4]  # 拇指指尖索引
        finger_tip = hand.norm_landmarks[finger_tip_idx]
        
        # 计算3D空间中的欧氏距离
        # norm_landmarks包含了x,y,z三个坐标值
        distance_3d = np.sqrt(np.sum((thumb_tip - finger_tip) ** 2))
        return distance_3d
        
    def detect_pinch(self, hand):
        """检测捏合手势并返回详细信息"""
        if hand is None:
            return None
            
        # 手指指尖索引
        finger_indices = {
            'index': 8,    # 食指指尖
            'middle': 12,  # 中指指尖
            'ring': 16,    # 无名指指尖
            'pinky': 20    # 小指指尖
        }
        
        results = {}
        for finger_name, tip_idx in finger_indices.items():
            distance = self.calculate_finger_distance(hand, tip_idx)
            if distance is not None:
                self.distance_history[finger_name].append(distance)
                
                # 计算置信度
                if len(self.distance_history[finger_name]) >= self.confidence_window:
                    mean_distance = np.mean(self.distance_history[finger_name])
                    std_distance = np.std(self.distance_history[finger_name])
                    confidence = 1.0 / (1.0 + std_distance)  # 标准差越小，置信度越高
                    
                    results[finger_name] = {
                        'distance': distance,
                        'is_pinching': distance < self.distance_threshold,
                        'confidence': confidence,
                        'mean_distance': mean_distance,
                        'std_distance': std_distance
                    }
                
        return results
        
    def get_calibration_data(self, pinch_results):
        """根据捏合检测结果生成校准数据"""
        if not pinch_results:
            return None
            
        calibration_data = {}
        for finger_name, data in pinch_results.items():
            if data['confidence'] > 0.8 and data['is_pinching']:  # 只在高置信度时提供校准数据
                calibration_data[finger_name] = {
                    'correction_factor': data['mean_distance'] / self.distance_threshold,
                    'confidence': data['confidence']
                }
                
        return calibration_data if calibration_data else None 