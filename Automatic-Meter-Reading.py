import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageChops
from sympy.geometry import Point, Line, Circle
import torch

# 加载 YOLOv5 模型（需要提前安装 yolov5 库并下载预训练模型）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 定义电表刻度信息（根据实际情况调整）
min_value = 0
max_value = 100
total_degrees = 360

def process_frame(frame):
    # 保存原始帧的副本
    original_frame = frame.copy()

    # 颜色空间转换
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 显示原始图像
    cv2.imshow("org", frame)

    # 使用 PIL 处理指针图像
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_none = Image.open('none.jpg')
    pointer = ImageChops.difference(img_none, pil_frame)
    pointer.save('pointer.jpg')

    # 读取指针图像
    pointer_frame = cv2.imread('pointer.jpg')
    pointer_frame = cv2.cvtColor(pointer_frame, cv2.COLOR_RGB2BGR)

    # 显示指针图像
    cv2.imshow("pointer", pointer_frame)

    # 高斯去噪
    kernel = np.ones((6, 6), np.float32) / 36
    gray_cut_filter2D = cv2.filter2D(frame[0:frame.shape[0], 0:frame.shape[1]], -1, kernel)

    # 指针图像灰度化和二值化
    gray_pointer = cv2.cvtColor(pointer_frame, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(gray_pointer, 80, 100, cv2.THRESH_BINARY)
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_RGB2BGR)

    # 显示高斯去噪后二值化的指针
    cv2.imshow("gaussian", thresh1)

    # Canny 边缘检测
    edges = cv2.Canny(thresh1, 50, 200, apertureSize=3)
    cv2.imshow("canny", edges)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=60)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.drawMarker(frame, (x2, y2), (255, 0, 0), thickness=2, markerType=cv2.MARKER_STAR, line_type=cv2.LINE_8,
                           markerSize=20)

        # 求直线交点
        x1, y1, x2, y2 = lines[0][0]
        l = Line(Point(x1, y1), Point(x2, y2))
        x1, y1, x2, y2 = lines[1][0]
        l1 = Line(Point(x1, y1), Point(x2, y2))
        x_zhizhen = np.float64(l1.intersection(l))
        print("指针端点", x_zhizhen)

        # 圆检测
        org_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        org_gray = cv2.medianBlur(org_gray, 5)
        circles = cv2.HoughCircles(org_gray, cv2.HOUGH_GRADIENT, 1, 10, param1=150, param2=90, minRadius=465,
                                   maxRadius=485)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x = 0
            y = 0
            j = 0
            for i in circles[0, :]:
                # 绘制外圆
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 10)
                x = (x * j + i[0]) / (j + 1)
                y = (y * j + i[1]) / (j + 1)
                j = j + 1
                # 绘制圆心
                y = y + 8
                x = x + 1
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 5)

            cv2.imshow("yuanxin", frame)
            print("中心为：", i[0] - 8, i[1] - 4)

            # 图像零点坐标
            org_x0, org_y0 = 340, 719
            pointer_1 = original_frame.copy()
            cv2.line(pointer_1, (int(x_zhizhen[0][0]), int(x_zhizhen[0][1])), (i[0] - 8, i[1] - 4), (255, 255, 255), 3)
            cv2.line(pointer_1, (org_x0, org_y0), (i[0] - 8, i[1] - 4), (255, 255, 255), 3)
            cv2.imshow("jiaodu", pointer_1)

            # 向量计算
            vector_AB = np.array([int(x_zhizhen[0][0]) - (i[0] - 8), int(x_zhizhen[0][1]) - (i[1] - 4)])
            vector_BC = np.array([org_x0 - (i[0] - 8), org_y0 - (i[1] - 4)])

            # 向量的点积
            dot_product = np.dot(vector_AB, vector_BC)

            # 计算向量 AB 和向量 BC 的范数（模）
            norm_AB = np.linalg.norm(vector_AB)
            norm_BC = np.linalg.norm(vector_BC)

            # 计算夹角的余弦值
            cos_theta = dot_product / (norm_AB * norm_BC)

            # 使用反余弦函数计算夹角的弧度值
            theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))

            # 将弧度值转换为角度
            theta_degrees = np.degrees(theta_radians)
            print("度数为", theta_degrees)

            # 根据角度计算电表读数
            meter_reading = (theta_degrees / total_degrees) * (max_value - min_value) + min_value
            print("电表读数为：", meter_reading)

    # 使用 YOLOv5 进行数字识别
    results = model(frame)
    results.print()  # 打印检测结果
    results.show()  # 显示检测结果

    return frame

# 动态读取视频
cap = cv2.VideoCapture(0)  # 如果是摄像头输入，参数为 0；如果是视频文件，参数为视频文件路径

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
