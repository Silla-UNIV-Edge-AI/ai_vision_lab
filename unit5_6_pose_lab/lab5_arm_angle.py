"""
[실습 5] 팔꿈치 각도 실시간 측정
교안: 6단원 6.2

▶ 준비 : Docker 컨테이너 안에서 실행 (./run_docker.sh 로 들어감) – 교안 5.1.2
▶ 실행 : python3 lab5_arm_angle.py
▶ 결과 : 오른쪽 팔꿈치 옆에 각도 표시 (q 키로 종료)
"""
# 실습 5. 팔꿈치 각도 실시간 측정
import cv2
from ultralytics import YOLO
from angle import get_angle

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, verbose=False)[0]
    image = result.plot()

    if len(result.boxes) > 0:                     # 사람이 1명 이상 있으면
        p = result.keypoints.xy[0].tolist()       # 첫 번째 사람의 점 17개
        s = result.keypoints.conf[0].tolist()     # 각 점의 신뢰도

        if s[6] > 0.5 and s[8] > 0.5 and s[10] > 0.5:   # 세 점이 잘 보일 때만
            shoulder, elbow, wrist = p[6], p[8], p[10]  # 오른쪽 어깨-팔꿈치-손목
            angle = get_angle(shoulder, elbow, wrist)

            x, y = int(elbow[0]), int(elbow[1])
            cv2.putText(image, f"{angle:.0f} deg", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)        # 검은 테두리
            cv2.putText(image, f"{angle:.0f} deg", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)    # 주홍 글자 (BGR)

    cv2.imshow("Arm Angle", image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
