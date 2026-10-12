"""
[STEP 2] 사진 4장 한 번에 탐지하기 – 준비 · 모델 불러오기 · 예측
교안: 1단원 1.3 STEP 2

▶ 준비 : pip install ultralytics, test_image_dir.zip을 같은 폴더에 두기
▶ 실행 : python lab1_detect.py
▶ 결과 : runs/detect/predict 폴더에 박스가 그려진 사진 4장
"""
import os
import zipfile
from ultralytics import YOLO

# ① 실습 사진 준비 : test_image_dir.zip → test_image_dir 폴더
if not os.path.isdir('test_image_dir'):
    zipfile.ZipFile('test_image_dir.zip').extractall('test_image_dir')
print('사진 :', sorted(os.listdir('test_image_dir')))

# ② 사전 학습 모델 불러오기 (처음 실행 때 yolo11n.pt 자동 다운로드)
model = YOLO('yolo11n.pt')
print('클래스 수 :', len(model.names))                    # 80 (COCO)
print('0번, 63번 :', model.names[0], model.names[63])     # 번호 → 이름

# ③ 예측 – 핵심은 이 한 줄!
results = model(source='test_image_dir/*.jpg', save=True)
print('결과 폴더 :', results[0].save_dir)
