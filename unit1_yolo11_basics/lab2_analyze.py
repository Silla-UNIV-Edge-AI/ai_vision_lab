"""
[STEP 4] 탐지 결과를 코드로 읽기 – 이름별 개수 · 박스 좌표 · 신뢰도
교안: 1단원 1.3 STEP 4

▶ 준비 : STEP 2(lab1_detect.py)를 먼저 실행해 test_image_dir 폴더 만들기
▶ 실행 : python lab2_analyze.py
▶ 결과 : 사진별 {이름: 개수}와 test1.jpg의 박스 정보 출력
"""
import os
from collections import Counter
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model(source='test_image_dir/*.jpg', verbose=False)   # 저장 · 로그 없이 예측

# ① 사진마다 찾은 물체 세기
for r in results:
    nums = r.boxes.cls.cpu().numpy().astype(int)   # 박스마다 클래스 번호
    names = [model.names[n] for n in nums]         # 번호 → 이름
    print(os.path.basename(r.path), dict(Counter(names)))

# ② 박스 하나하나 꺼내기 (첫 번째 사진 test1.jpg)
print('\n[test1.jpg 박스]')
for box in results[0].boxes:
    name = model.names[int(box.cls)]
    conf = float(box.conf)                         # 신뢰도 0~1
    x1, y1, x2, y2 = box.xyxy[0].tolist()          # 왼쪽 위 · 오른쪽 아래 좌표(픽셀)
    print(f'{name:10s} {conf:.2f}  ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})')
