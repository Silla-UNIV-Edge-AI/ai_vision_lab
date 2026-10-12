"""
[실습 1-CV] 키포인트 결과를 창으로 보기
▶ 실행 : python lab1_first_pose_cv.py   (아무 키나 누르면 종료)
"""
import cv2
from ultralytics import YOLO

pose = YOLO("yolo11n-pose.pt")("bus.jpg", verbose=False)[0]

cv2.imshow("pose", pose.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
