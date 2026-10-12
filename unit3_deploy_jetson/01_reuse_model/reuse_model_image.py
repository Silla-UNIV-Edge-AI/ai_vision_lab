"""
[3.1] 저장한 모델로 사진 예측
▶ 준비 : best.pt(2단원 결과), python prepare_my_hand.py 로 만든 사진
▶ 실행 : python reuse_model_image.py
▶ 결과 : 예측 결과 창 (창을 클릭하고 아무 키나 누르면 종료)
"""
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

colors.palette[2] = (252, 109, 47)   # Scissors(2번) 기본색이 거의 흰색이라 흰 바탕에서 안 보임 → 주황색 (RGB)

model = YOLO('best.pt')

result = model('my_hand_scissors.jpg', imgsz=416, conf=0.5)[0]   # 학습 크기 416, 신뢰도 0.5 이상만

cv2.imshow('result', result.plot())   # 박스를 그린 그림을 창으로 보기
cv2.waitKey(0)                        # 키를 누를 때까지 기다림
