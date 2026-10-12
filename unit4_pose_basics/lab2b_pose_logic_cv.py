"""
[실습 2-B CV] 한 사람(첫 번째)만 자세 판단 + 창으로 보기
▶ 실행 : python lab2b_pose_logic_cv.py   (아무 키나 누르면 종료)
"""
import cv2
from ultralytics import YOLO

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

result = YOLO("yolo11n-pose.pt")("bus.jpg", verbose=False)[0]
points = result.keypoints.xy[0].tolist()         # 첫 번째 사람만
scores = result.keypoints.conf[0].tolist()
box    = result.boxes.xyxy[0].tolist()           # 첫 번째 사람의 박스

# 1) y는 아래로 갈수록 커진다
print("코 y =", round(points[0][1]), "/ 오른발목 y =", round(points[16][1]))

# 2) 신뢰도 0.5 미만은 "못 찾은 점"
for i in range(17):
    if scores[i] < 0.5:
        print("믿기 어려운 점:", i, "번, 신뢰도", round(scores[i], 2))

# 3) 오른손을 들었나? (손목이 어깨보다 위면 든 것)
wrist_y, shoulder_y = points[10][1], points[6][1]
print("오른손목 y =", round(wrist_y), "/ 오른어깨 y =", round(shoulder_y))
status = "RIGHT HAND UP" if wrist_y < shoulder_y else "RIGHT HAND DOWN"
print(status)

# 4) 첫 번째 사람만 그리기
image = result.orig_img.copy()
x1, y1, x2, y2 = map(int, box)
cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
for a, b in SKELETON:
    if scores[a] < 0.5 or scores[b] < 0.5:
        continue
    cv2.line(image, (int(points[a][0]), int(points[a][1])),
                    (int(points[b][0]), int(points[b][1])), (0, 255, 255), 2)
for i, (x, y) in enumerate(points):
    if scores[i] < 0.5:
        continue
    cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

cv2.putText(image, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
cv2.imshow("pose logic (person 0)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
