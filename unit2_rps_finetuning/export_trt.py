"""
[2단원 보충] best.pt → TensorRT 엔진(best.engine) 만들기 – 교안 3.4 ③과 같은 변환
▶ 준비 : best.pt(2단원 결과), Jetson의 Docker 컨테이너
▶ 실행 : ./run_docker.sh export   (약 8~10분, 처음 한 번)
▶ 결과 : best.onnx · best.engine (FP16, imgsz 416)
"""
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# 학습한 가중치 (스크립트 위치 기준 경로)
PT_PATH = Path(__file__).parent / 'best.pt'
IMGSZ = 416  # 교안의 학습 크기(2.2절)와 같은 값. 다른 크기로 학습한 best.pt 라면 그 값으로 바꿉니다

model = YOLO(PT_PATH)

# TensorRT 엔진 변환 (FP16, 이 Jetson 전용 - 다른 장비/TRT 버전에서는 다시 만들어야 함)
# 중간 단계로 같은 이름의 ONNX 파일(best.onnx)도 함께 만들어집니다.
# Jetson에서 수 분 ~ 10분 정도 걸립니다.
engine_path = model.export(format='engine', imgsz=IMGSZ, half=True, device=0)   # → best.engine (app_trt.py 가 읽는 이름)
print(f"TensorRT 엔진 저장: {engine_path}")

# 엔진이 제대로 동작하는지 간단히 확인
trt_model = YOLO(engine_path, task='detect')
dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
trt_model(dummy, verbose=False)
print(f"엔진 테스트 OK - 클래스: {trt_model.names}")
