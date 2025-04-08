import depthai as dai
import cv2
import numpy as np

# 创建pipeline
pipeline = dai.Pipeline()

# 创建左单色相机节点
monoLeft = pipeline.createMonoCamera()
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# 创建右单色相机节点
monoRight = pipeline.createMonoCamera()
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# 创建立体深度节点
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# 设置置信度阈值
stereo.initialConfig.setConfidenceThreshold(245)
# 设置左右匹配检查
stereo.setLeftRightCheck(True)
# 设置亚像素级别
stereo.setSubpixel(True)

# 连接相机到立体深度节点
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# 创建输出节点
xoutLeft = pipeline.createXLinkOut()
xoutLeft.setStreamName("left")
monoLeft.out.link(xoutLeft.input)

xoutRight = pipeline.createXLinkOut()
xoutRight.setStreamName("right")
monoRight.out.link(xoutRight.input)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# 连接设备并启动pipeline
with dai.Device(pipeline) as device:
    # 获取输出队列
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # 获取左右相机和深度图像
        inLeft = qLeft.get()
        inRight = qRight.get()
        inDepth = qDepth.get()
        
        # 转换深度图像为可视化格式
        depth_frame = inDepth.getFrame()
        depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_frame = np.asarray(depth_frame, dtype=np.uint8)
        depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

        # 显示图像
        cv2.imshow("left", inLeft.getCvFrame())
        cv2.imshow("right", inRight.getCvFrame())
        cv2.imshow("depth", depth_frame)

        if cv2.waitKey(1) == ord("q"):
            break