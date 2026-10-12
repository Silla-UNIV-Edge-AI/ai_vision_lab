"""
[실습 7] TensorRT로 속도 높이기 (Jetson 전용)
교안: 6단원 6.4

▶ 준비 : Docker 컨테이너 안에서 실행 (./run_docker.sh 로 들어감) – 교안 5.1.2
▶ 실행 : python3 lab7_tensorrt.py
▶ 결과 : yolo11n-pose.engine 생성 후 웹캠 실행
"""
# 실습 7. TensorRT로 속도 올리기 (Jetson에서 실행)
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
model.export(format="engine", half=True)      # .pt -> .engine (몇 분 걸림)

fast_model = YOLO("yolo11n-pose.engine")      # 변환된 엔진 불러오기
fast_model.predict(source=0, show=True)       # 웹캠으로 실행 (속도 비교해 보기)
