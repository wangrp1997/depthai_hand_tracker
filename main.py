#!/usr/bin/env python3


from HandTrackerRenderer import HandTrackerRenderer
from pinch_gesture_detector import PinchGestureDetector
import argparse
import numpy as np
from collections import Counter
import cv2
import socket
import json
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                    help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_tracker.add_argument('-p', '--pinch', action="store_true",
                    help="Enable pinch gesture detection and calibration")
parser_tracker.add_argument('--pinch_threshold', type=float, default=0.1,
                    help="Distance threshold for pinch detection (default=0.1)")
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
parser.add_argument('--server_ip', type=str, default='172.30.83.97',
                    help="WSL2 服务器IP地址")
parser.add_argument('--server_port', type=int, default=12345,
                    help="WSL2 服务器端口")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerEdge import HandTracker
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTracker import HandTracker


tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        use_handedness_average=not args.use_last_handedness,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

# 初始化计数器
finger_counter = Counter()
total_frames = 0
FRAMES_TO_COUNT = 30  # 统计30帧内的结果

# 如果启用了捏合检测，初始化检测器
pinch_detector = PinchGestureDetector(
    distance_threshold=args.pinch_threshold
) if args.pinch else None

# 创建Socket客户端
client_socket = None
reconnect_attempts = 0
MAX_RECONNECT_ATTEMPTS = 3

def connect_to_server(server_ip, server_port):
    global client_socket, reconnect_attempts
    try:
        if client_socket:
            client_socket.close()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)  # 设置超时时间
        client_socket.connect((server_ip, server_port))
        print(f"已连接到服务器 {server_ip}:{server_port}")
        reconnect_attempts = 0  # 重置重连次数
    except Exception as e:
        print(f"连接服务器失败: {e}")
        client_socket = None
        if reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            reconnect_attempts += 1
            print(f"尝试重新连接 ({reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})...")
            time.sleep(2)  # 等待2秒后重试
            connect_to_server(server_ip, server_port)

def send_pinch_data(data):
    global client_socket
    if client_socket:
        try:
            json_data = json.dumps(data)
            client_socket.send(json_data.encode('utf-8'))
        except Exception as e:
            print(f"发送数据失败: {e}")
            client_socket = None
            # 尝试重新连接
            connect_to_server(args.server_ip, args.server_port)

# 在新线程中连接服务器
threading.Thread(target=connect_to_server, args=(args.server_ip, args.server_port)).start()

while True:
    # Run hand tracker on next frame
    # 'bag' contains some information related to the frame 
    # and not related to a particular hand like body keypoints in Body Pre Focusing mode
    # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
    frame, hands, bag = tracker.next_frame()
    if frame is None: break

    # 处理捏合手势检测
    if args.pinch and hands:
        for i, hand in enumerate(hands):
            pinch_results = pinch_detector.detect_pinch(hand)
            if pinch_results:
                # 为左右手设置不同的显示位置
                height = frame.shape[0]
                width = frame.shape[1]
                x_pos = 10 if hand.handedness < 0.5 else width - 250
                base_y = height - 30  # 从底部向上显示
                
                # 显示手的类型（左/右）
                hand_type = "left" if hand.handedness < 0.5 else "right"
                cv2.putText(frame, f"{hand_type} Hand", 
                          (x_pos, base_y - 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示每个手指的捏合状态并发送数据
                for j, (finger_name, data) in enumerate(pinch_results.items()):
                    color = (0, 255, 0) if data['is_pinching'] else (0, 0, 255)
                    text = f"{finger_name}: {data['distance']:.2f}"
                    if 'confidence' in data:
                        text += f" ({data['confidence']:.2f})"
                    
                    cv2.putText(frame, text,
                              (x_pos, base_y - 75 + j * 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 发送捏合数据到ROS2节点
                    if data['is_pinching'] and 'confidence' in data and data['confidence'] > 0.8:
                        send_data = {
                            'hand': hand_type,
                            'finger': finger_name,
                            'is_pinching': True,
                            'distance': float(data['distance']),
                            'confidence': float(data['confidence'])
                        }
                        send_pinch_data(send_data)

    # Draw hands
    frame = renderer.draw(frame, hands, bag)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

if client_socket:
    client_socket.close()
renderer.exit()
tracker.exit()
