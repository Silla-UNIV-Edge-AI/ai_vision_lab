"""
[실습 2-CV] 키포인트 숫자 읽기 + 결과를 창으로 보기
▶ 실행 : python lab2_keypoints_cv.py   (아무 키나 누르면 종료)
"""
import cv2
from ultralytics import YOLO

NAMES = ["코", "왼눈", "오른눈", "왼귀", "오른귀",
         "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치", "왼손목", "오른손목",
         "왼엉덩이", "오른엉덩이", "왼무릎", "오른무릎", "왼발목", "오른발목"]

SKELETON = [(0,1),(0,2),(1,3),(2,4),                     # 얼굴
            (5,6),(5,7),(7,9),(6,8),(8,10),              # 팔
            (5,11),(6,12),(11,12),                       # 몸통
            (11,13),(13,15),(12,14),(14,16)]             # 다리

result = YOLO("yolo11n-pose.pt")("bus.jpg", verbose=False)[0]
all_points = result.keypoints.xy.tolist()
all_scores = result.keypoints.conf.tolist()

for p, (points, scores) in enumerate(zip(all_points, all_scores)):
    print(f"\n=== 사람 {p} ===")
    for i in range(17):
        x, y = points[i]
        print(f"{i:2d} {NAMES[i]:6s} x={x:6.1f}  y={y:6.1f}  신뢰도={scores[i]:.2f}")

image = result.orig_img.copy()
for points, scores in zip(all_points, all_scores):
    for a, b in SKELETON:
        if scores[a] < 0.5 or scores[b] < 0.5:
            continue
        pa = (int(points[a][0]), int(points[a][1]))
        pb = (int(points[b][0]), int(points[b][1]))
        cv2.line(image, pa, pb, (0, 255, 255), 2)
    for i, (x, y) in enumerate(points):
        if scores[i] < 0.5:
            continue
        color = (0, 200, 0) if i == 0 else ((0, 0, 255) if i % 2 else (255, 0, 0))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)
        cv2.putText(image, str(i), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow("keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
