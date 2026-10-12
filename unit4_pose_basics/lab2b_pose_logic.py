"""
[실습 2-B] 숫자로 자세 판단하기 – y 비교 · 신뢰도 · 손 들기
▶ 실행 : python lab2b_pose_logic.py
"""
from ultralytics import YOLO

# 모델 불러오기 (사람마다 관절 17개를 찾아줌)
model = YOLO("yolo11n-pose.pt")
result = model("bus.jpg", verbose=False)[0]

# 첫 번째 사람의 관절만 꺼내기
# xy    : (x, y) 좌표    conf : 신뢰도(0~1)
points = result.keypoints.xy[0].tolist()
scores = result.keypoints.conf[0].tolist()

# 1) 주의! 화면에서는 y가 아래로 갈수록 커진다.
#    → 머리(코)는 y가 작고, 발은 y가 크다.
print("코 y =", round(points[0][1]), "/ 오른발목 y =", round(points[16][1]))

# 2) 신뢰도 0.5 미만은 "못 찾은 점"으로 취급 (좌표가 엉터리일 수 있음)
for i in range(17):
    if scores[i] < 0.5:
        print("믿기 어려운 점:", i, "번, 신뢰도", round(scores[i], 2))

# 3) 오른손을 들었나?
#    손목이 어깨보다 "위" = y가 더 작다  → 들었다
wrist_y    = points[10][1]      # 오른손목
shoulder_y = points[6][1]       # 오른어깨
print("오른손목 y =", round(wrist_y), "/ 오른어깨 y =", round(shoulder_y))
if wrist_y < shoulder_y:
    print("오른손을 들었습니다")
else:
    print("오른손을 내렸습니다")
