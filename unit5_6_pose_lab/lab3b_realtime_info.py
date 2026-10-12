"""
[실습 3-B] 실시간 정보 표시 – 사람 수 · 추론 시간 · FPS
교안: 5단원 5.4

▶ 준비 : Docker 컨테이너 안에서 실행, USB 웹캠 (/dev/video0)
▶ 실행 : python3 lab3b_realtime_info.py
▶ 결과 : 웹캠 화면 위에 사람 수 · 추론 시간(ms) · FPS · 코 좌표 표시 (q 키로 종료)
"""
import time

import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")     # 6단원 실습 7 뒤에는 "yolo11n-pose.engine"으로 바꿔 비교
cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ok, frame = cap.read()                       # 1) 카메라에서 사진 한 장 받기
    if not ok:
        break
    result = model(frame, verbose=False)[0]      # 2) AI로 관절 찾기
    image = result.plot()                        # 3) 점과 뼈대 그리기

    # 4) 숫자 정보 계산
    now = time.time()
    fps = 1 / (now - prev)                       # 1초에 몇 장을 처리하나
    prev = now
    people = len(result.boxes)                   # 찾은 사람 수
    ms = result.speed["inference"]               # AI 계산(추론)에 걸린 시간

    info = f"People {people} | Infer {ms:.0f} ms | FPS {fps:.1f}"
    cv2.putText(image, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)        # 검은 테두리
    cv2.putText(image, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)    # 주홍 글자 (BGR)

    # 5) 첫 번째 사람의 코(0번)가 잘 보이면 좌표 표시 – 4단원 실습 2의 숫자를 실시간으로
    if people > 0 and result.keypoints.conf[0][0] > 0.5:
        x, y = result.keypoints.xy[0][0].tolist()
        label, pos = f"nose ({x:.0f}, {y:.0f})", (int(x) + 10, int(y) - 25)
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)      # 검은 테두리
        cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)    # 초록 글자

    cv2.imshow("Pose Info", image)               # 6) 화면에 보여 주기
    if cv2.waitKey(1) == ord("q"):               # q 키를 누르면 끝
        break

cap.release()
cv2.destroyAllWindows()
