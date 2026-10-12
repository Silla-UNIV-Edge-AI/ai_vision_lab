"""
[1.4.1] 예측 옵션 바꿔 보기 – classes · conf · imgsz
교안: 1단원 1.4.1

▶ 준비 : STEP 2(lab1_detect.py)를 먼저 실행
▶ 실행 : python lab3_options.py
▶ 결과 : 옵션에 따라 달라지는 박스 수 출력 (사람만 표시한 test4는 runs/detect에 저장)
"""
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# ① 사람(0번)만, 신뢰도 0.5 이상 → 멀리 있는 작은 사람(0.38)은?
for conf in (0.25, 0.5):
    r = model('test_image_dir/test4.jpg', classes=[0], conf=conf, verbose=False)[0]
    print(f'test4  사람만 conf={conf} : {len(r.boxes)}명')
model('test_image_dir/test4.jpg', classes=[0], conf=0.5, save=True, verbose=False)

# ② 입력 크기 키우기 → 뒤편의 작은 얼룩말이 더 잡힐까?
for size in (640, 1280):
    r = model('test_image_dir/test2.jpg', imgsz=size, verbose=False)[0]
    print(f'test2  imgsz={size} : 박스 {len(r.boxes)}개')
