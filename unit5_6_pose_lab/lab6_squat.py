"""
[실습 6] 스쿼트 횟수 세기
교안: 6단원 6.3

▶ 준비 : Docker 컨테이너 안에서 실행 (./run_docker.sh 로 들어감) – 교안 5.1.2
▶ 실행 : python3 lab6_squat.py
▶ 결과 : 화면 왼쪽 위에 Squat 횟수 (q 키로 종료)
"""
# 실습 6. 스쿼트 횟수 세기
import cv2
from ultralytics import YOLO
from angle import get_angle

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)

count = 0          # 스쿼트 횟수
state = "up"       # 현재 자세: "up"(서 있음) 또는 "down"(앉음)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, verbose=False)[0]
    image = result.plot()

    if len(result.boxes) > 0:
        p = result.keypoints.xy[0].tolist()
        s = result.keypoints.conf[0].tolist()

        if s[12] > 0.5 and s[14] > 0.5 and s[16] > 0.5:
            hip, knee, ankle = p[12], p[14], p[16]    # 오른쪽 엉덩이-무릎-발목
            angle = get_angle(hip, knee, ankle)

            if angle < 100:                           # 충분히 앉으면
                state = "down"
            if angle > 160 and state == "down":       # 앉았다가 다시 일어서면
                state = "up"
                count = count + 1                     # 1회 추가!

            cv2.putText(image, f"Knee {angle:.0f} deg  [{state}]", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(image, f"Squat: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("Squat Counter", image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
