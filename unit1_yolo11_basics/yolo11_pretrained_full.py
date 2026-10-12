"""
[부록 A] 1단원 전체 실습 코드 – 1.3 STEP 2~4를 한 파일로
교안: 부록 A (단계별 설명은 1.3절)

▶ 준비 : pip install ultralytics, test_image_dir.zip을 같은 폴더에 두기
▶ 실행 : python yolo11_pretrained_full.py
▶ 결과 : runs/detect/predict 폴더의 결과 사진 4장, 사진별 물체 개수와 박스 정보 출력
"""
import os
import zipfile
from collections import Counter
from ultralytics import YOLO

# ===== STEP 2. 준비 · 모델 불러오기 · 예측 =====
if not os.path.isdir('test_image_dir'):
    zipfile.ZipFile('test_image_dir.zip').extractall('test_image_dir')
model = YOLO('yolo11n.pt')
results = model(source='test_image_dir/*.jpg', save=True)
print('결과 폴더 :', results[0].save_dir)          # STEP 3 : 이 폴더의 사진을 열어 확인

# ===== STEP 4. 결과를 코드로 읽기 =====
for r in results:
    names = [model.names[int(c)] for c in r.boxes.cls]
    print(os.path.basename(r.path), dict(Counter(names)))
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f'   {model.names[int(box.cls)]:12s} {float(box.conf):.2f}  ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})')
