import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageChops
from pylab import *
from sympy.geometry import Point, Line, Circle

# 读取图像
def read_image(filepath):
    try:
        cv_image = cv2.imread(filepath)
        if cv_image is None:
            raise FileNotFoundError(f"无法读取图像文件: {filepath}")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        pil_image = Image.open(filepath)
        return cv_image, pil_image
    except Exception as e:
        print(f"读取图像时出错: {e}")
        return None, None

# 保存指针图像
def save_pointer_image(original_pil_image, none_image_path):
    try:
        none_image = Image.open(none_image_path)
        pointer_image = ImageChops.difference(none_image, original_pil_image)
        pointer_image.save('pointer.jpg')
        return pointer_image
    except Exception as e:
        print(f"保存指针图像时出错: {e}")
        return None

# 图像处理
def process_image(original_image, pointer_image):
    # 高斯去噪
    kernel = np.ones((6, 6), np.float32) / 36
    gray_cut_filter2D = cv2.filter2D(original_image[0:original_image.shape[0], 0:original_image.shape[1]], -1, kernel)

    # 指针图像灰度化和二值化
    gray_pointer = cv2.cvtColor(pointer_image, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(gray_pointer, 80, 100, cv2.THRESH_BINARY)
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_RGB2BGR)

    return gray_cut_filter2D, thresh1

# 边缘检测和直线检测
def detect_edges_and_lines(thresh1):
    # canny边缘检测
    edges = cv2.Canny(thresh1, 50, 200, apertureSize=3)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=60)
    return edges, lines

# 求直线交点
def find_intersection_point(lines):
    x1, y1, x2, y2 = lines[0][0]
    l = Line(Point(x1, y1), Point(x2, y2))
    x1, y1, x2, y2 = lines[1][0]
    l1 = Line(Point(x1, y1), Point(x2, y2))
    intersection_point = np.float64(l1.intersection(l))
    return intersection_point

# 圆检测
def detect_circles(original_image):
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.medianBlur(original_gray, 5)
    circles = cv2.HoughCircles(original_gray, cv2.HOUGH_GRADIENT, 1, 10, param1=150, param2=90, minRadius=465, maxRadius=485)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x = 0
        y = 0
        j = 0
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(original_image, (i[0], i[1]), i[2], (0, 0, 255), 10)
            x = (x * j + i[0]) / (j + 1)
            y = (y * j + i[1]) / (j + 1)
            j = j + 1
            # draw the center of the circle
            y = y + 8
            x = x + 1
            cv2.circle(original_image, (int(x), int(y)), 2, (0, 0, 255), 5)
        return original_image, x, y
    else:
        return original_image, None, None

# 计算角度
def calculate_angle(intersection_point, center_x, center_y, zero_x, zero_y):
    # 向量
    vector_AB = np.array([int(intersection_point[0][0]) - center_x, int(intersection_point[0][1]) - center_y])
    vector_BC = np.array([zero_x - center_x, zero_y - center_y])
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
    return theta_degrees

def main():
    filepath = "0.jpg"
    none_image_path = "none.jpg"

    # 读取图像
    original_image, original_pil_image = read_image(filepath)
    if original_image is None or original_pil_image is None:
        return

    # 保存指针图像
    pointer_image = save_pointer_image(original_pil_image, none_image_path)
    if pointer_image is None:
        return

    # 图像处理
    gray_cut_filter2D, thresh1 = process_image(original_image, pointer_image)

    # 边缘检测和直线检测
    edges, lines = detect_edges_and_lines(thresh1)

    # 绘制检测出的直线和端点
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.drawMarker(original_image, (x2, y2), (255, 0, 0), thickness=2, markerType=cv2.MARKER_STAR, line_type=cv2.LINE_8,
                       markerSize=20)

    # 求直线交点
    intersection_point = find_intersection_point(lines)
    print("指针端点", intersection_point)

    # 圆检测
    original_image, center_x, center_y = detect_circles(original_image)
    if center_x is None or center_y is None:
        return
    print("中心为：", center_x - 8, center_y - 4)

    # 计算角度
    zero_x, zero_y = 340, 719  # 图像零点坐标 第一个刻度不是很均匀 实际选为零刻度下标点位置
    pointer_1 = cv2.imread(filepath)
    cv2.line(pointer_1, (int(intersection_point[0][0]), int(intersection_point[0][1])), (center_x - 8, center_y - 4), 255, 3)
    cv2.line(pointer_1, (zero_x, zero_y), (center_x - 8, center_y - 4), 255, 3)
    theta_degrees = calculate_angle(intersection_point, center_x - 8, center_y - 4, zero_x, zero_y)
    print("度数为", theta_degrees)

    # 显示图像
    cv2.imshow("org", original_image)
    cv2.imshow("pointer", pointer_image)
    cv2.imshow("gaussian", thresh1)
    cv2.imshow("canny", edges)
    cv2.imshow("houghlines", original_image)
    cv2.imshow("yuanxin", original_image)
    cv2.imshow("jiaodu", pointer_1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
