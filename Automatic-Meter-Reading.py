import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont,ImageChops
from pylab import *
from sympy.geometry import Point, line
from sympy import Point, Line, Circle

filepath="0.jpg"
org=cv2.imread(filepath)
org=cv2.cvtColor(org,cv2.COLOR_RGB2BGR) #原图

cv2.imshow("org",org)
#subplot(331)
#imshow(org)
#title("原图")

org_1=Image.open('0.jpg')
img_none=Image.open('none.jpg')
pointer=ImageChops.difference(img_none,org_1)
pointer.save('pointer.jpg') #保存指针图像

frame=cv2.imread('pointer.jpg')
frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) #指针

cv2.imshow("pointer",frame)

kernel = np.ones((6, 6), np.float32) / 36
gray_cut_filter2D = cv2.filter2D(org[0:org.shape[0], 0:org.shape[1]], -1, kernel)   #高斯去噪
gray_pointer=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #灰度化
ret, thresh1 = cv2.threshold(gray_pointer, 80, 100, cv2.THRESH_BINARY)  #二值化
thresh1=cv2.cvtColor(thresh1,cv2.COLOR_RGB2BGR)

cv2.imshow("gaussian",thresh1)  #高斯去噪后二值化的指针

edges=cv2.Canny(thresh1,50,200,apertureSize=3)  #canny边缘检测

cv2.imshow("canny",edges)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=60)  #霍夫直线检测 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(org, (x1, y1), (x2, y2), (0, 0, 255),3)
    cv2.drawMarker(org,(x2,y2),(255, 0, 0),thickness=2,markerType=cv2.MARKER_STAR,line_type=cv2.LINE_8,markerSize=20)   #打印检测出的直线与线上端点
cv2.imshow('houghlines',org)

x1, y1, x2, y2 = lines[0][0]
l = Line(Point(x1, y1),Point( x2, y2))
x1, y1, x2, y2 = lines[1][0]
l1 = Line(Point(x1, y1),Point( x2, y2))
x_zhizhen = np.float64(l1.intersection(l)) #sympy库求直线交点
print("指针端点",x_zhizhen) #输出指针尖的坐标值

org_gray=cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
org_gray=cv2.medianBlur(org_gray,5)
circles = cv2.HoughCircles(org_gray,cv2.HOUGH_GRADIENT,1,10,param1=150,param2=90,minRadius=465,maxRadius=485)
circles = np.uint16(np.around(circles))
x = 0
y = 0
j=0
for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(org,(i[0],i[1]),i[2],(0,0,255),10)
        x = (x*j+i[0])/(j+1)
        y = (y*j+i[1])/(j+1)
        j=j+1
    # draw the center of the circle
        y=y+8
        x=x+1
        cv2.circle(org, (int(x), int(y)), 2, (0, 0, 255), 5)
cv2.imshow("yuanxin",org)
print("中心为：",i[0]-8,i[1]-4)

org_x0, org_y0 = 340,719  #图像零点坐标 第一个刻度不是很均匀 实际选为零刻度下标点位置
pointer_1=cv2.imread("0.jpg")
cv2.line(pointer_1, (int(x_zhizhen[0][0]),int(x_zhizhen[0][1])), (i[0]-8,i[1]-4), 255, 3)
cv2.line(pointer_1, (org_x0,org_y0), (i[0]-8,i[1]-4), 255, 3)
cv2.imshow("jiaodu",pointer_1)

#向量
vector_AB=np.array([int(x_zhizhen[0][0])-(i[0]-8),int(x_zhizhen[0][1])-(i[1]-4)])
vector_BC = np.array([org_x0-(i[0]-8),org_y0-(i[1]-4)])
#向量的点积
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
print("度数为",theta_degrees)
