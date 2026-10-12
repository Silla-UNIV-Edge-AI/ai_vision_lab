"""
[2단원 보충] 가위 · 바위 · 보 실시간 탐지 앱 – ONNX 모델 (교안 3.4 ④ 실측 사례처럼 속도 비교)
▶ 준비 : best.onnx (./run_docker.sh export 로 만듦), USB 웹캠, Jetson의 Docker 컨테이너
▶ 실행 : ./run_docker.sh app
▶ 결과 : 박스 · 이름 · FPS 가 표시된 웹캠 창 (q 키로 종료)
"""
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import time

# Scissors 박스 색상: 주황색 (#FC6D2F, OpenCV는 BGR 순서) – 기본색이 거의 흰색이라 흰 바탕에서 안 보임
SCISSORS_COLOR = (47, 109, 252)

# model = YOLO('best.onnx')
model = YOLO('best.onnx', task='detect')

cap = cv2.VideoCapture(0)
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    result = model(frame, conf=0.5, verbose=False)[0]

    # Scissors만 주황색, 나머지 클래스는 기본 색상
    annotator = Annotator(frame.copy())
    for box in result.boxes:
        cls = int(box.cls)
        name = result.names[cls]
        color = SCISSORS_COLOR if name == 'Scissors' else colors(cls, True)
        annotator.box_label(box.xyxy[0], f"{name} {float(box.conf):.2f}", color=color)
    annotated_frame = annotator.result()

    # FPS 계산 (숫자가 깜빡이지 않도록 이전 값과 섞어서 부드럽게)
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
    prev_time = now

    cv2.putText(annotated_frame, f"FPS {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLO Inference', annotated_frame)

    # q 키로 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
