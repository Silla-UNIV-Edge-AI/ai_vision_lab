"""
[실습 2] 키포인트 숫자 읽기와 번호 확인
교안: 4단원 4.8 (PC 기초 실습)

▶ 준비 : pip install ultralytics, bus.jpg
▶ 실행 : python lab2_keypoints.py
▶ 결과 : 첫 번째 사람의 17개 관절 좌표 · 신뢰도 출력 + numbers.jpg(관절 위에 번호)
"""
import cv2
from ultralytics import YOLO

NAMES = ["코", "왼눈", "오른눈", "왼귀", "오른귀",
         "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치", "왼손목", "오른손목",
         "왼엉덩이", "오른엉덩이", "왼무릎", "오른무릎", "왼발목", "오른발목"]

model = YOLO("yolo11n-pose.pt")
result = model("bus.jpg", verbose=False)[0]
print("찾은 사람 수:", len(result.boxes))

points = result.keypoints.xy[0].tolist()       # 첫 번째 사람의 점 17개 (x, y)
scores = result.keypoints.conf[0].tolist()     # 각 점의 신뢰도 (0~1)

# 1) 숫자로 읽기
for i in range(17):
    x, y = points[i]
    print(f"{i:2d} {NAMES[i]:6s} x={x:6.1f}  y={y:6.1f}  신뢰도={scores[i]:.2f}")

# 2) 그림으로 확인하기 – 관절 위에 번호 찍기
image = result.orig_img.copy()                 # 아무것도 그리지 않은 원본 사진
for i, (x, y) in enumerate(points):
    if scores[i] < 0.5:                        # 잘 안 보이는 점은 건너뛰기
        continue
    if i == 0:
        color = (0, 200, 0)                    # 0번 코 = 초록
    elif i % 2 == 1:
        color = (0, 0, 255)                    # 홀수 = 그 사람의 왼쪽 = 빨강
    else:
        color = (255, 0, 0)                    # 짝수 = 그 사람의 오른쪽 = 파랑
    cv2.circle(image, (int(x), int(y)), 5, color, -1)
    cv2.putText(image, str(i), (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite("numbers.jpg", image)
print("numbers.jpg 저장 – 빨강(홀수) = 왼쪽, 파랑(짝수) = 오른쪽")
