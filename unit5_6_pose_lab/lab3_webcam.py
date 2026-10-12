"""
[실습 3] 웹캠으로 실시간 포즈 추정
교안: 5단원 5.3

▶ 준비 : Docker 컨테이너 안에서 실행 (./run_docker.sh 로 들어감) – 교안 5.1.2
▶ 실행 : python3 lab3_webcam.py
▶ 결과 : 웹캠 화면에 점과 뼈대 (q 키로 종료)
"""
# 실습 3. 웹캠으로 실시간 포즈 추정
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)                    # 0번 카메라(웹캠) 열기

while True:
    ok, frame = cap.read()                   # 1) 카메라에서 사진 한 장 받기
    if not ok:
        break
    result = model(frame, verbose=False)[0]  # 2) AI로 관절 찾기
    image = result.plot()                    # 3) 점과 뼈대 그리기
    cv2.imshow("Pose", image)                # 4) 화면에 보여주기
    if cv2.waitKey(1) == ord("q"):           # q 키를 누르면 끝
        break

cap.release()
cv2.destroyAllWindows()
