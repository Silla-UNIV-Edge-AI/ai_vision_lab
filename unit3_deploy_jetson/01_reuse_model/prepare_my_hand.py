"""
[3.1] 잘라 둔 손 사진 3장을 예측해서 한 창에 나란히 보기
▶ 준비 : best.pt(2단원 결과), my_hand_scissors.jpg · my_hand_rock.jpg · my_hand_paper.jpg
▶ 실행 : python prepare_my_hand.py
▶ 결과 : 예측 결과 창 (창을 클릭하고 아무 키나 누르면 종료)
"""
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

colors.palette[2] = (252, 109, 47)   # Scissors(2번) 기본색이 거의 흰색이라 흰 바탕에서 안 보임 → 주황색 (RGB)

model = YOLO('best.pt')

images = ['my_hand_scissors.jpg', 'my_hand_rock.jpg', 'my_hand_paper.jpg']
results = model(images, imgsz=416, conf=0.5)   # 학습 크기 416, 신뢰도 0.5 이상만

cv2.imshow('my_hand', cv2.hconcat([r.plot() for r in results]))   # 결과 3장을 나란히 보기
cv2.waitKey(0)
