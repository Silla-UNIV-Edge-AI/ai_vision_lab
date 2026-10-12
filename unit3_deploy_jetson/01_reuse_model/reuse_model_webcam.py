"""
[3.1] 저장한 모델로 웹캠 실시간 예측
▶ 준비 : best.pt(2단원 결과), USB 웹캠
▶ 실행 : python reuse_model_webcam.py
▶ 결과 : 예측 창 (창을 클릭하고 q 를 누르면 종료)
"""
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

colors.palette[2] = (252, 109, 47)   # Scissors(2번) 기본색이 거의 흰색이라 흰 바탕에서 안 보임 → 주황색 (RGB)

model = YOLO('best.pt')

for result in model(0, imgsz=416, conf=0.5, show=True, stream=True):   # 0 = 첫 번째 웹캠
    pass
