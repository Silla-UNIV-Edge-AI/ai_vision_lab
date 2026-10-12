"""
[1.4.2] 입력 소스 바꾸기 – 인터넷 사진(강의 예제 zidane.jpg)과 웹캠
교안: 1단원 1.4.2

▶ 준비 : 인터넷 연결, (웹캠) 카메라와 모니터
▶ 실행 : python lab4_sources.py          → 인터넷 사진
         python lab4_sources.py webcam   → 웹캠 실시간 탐지 (결과 창에서 q 키로 종료)
▶ 결과 : 사람 · 넥타이가 탐지된 결과 창, 웹캠 화면 위 실시간 박스
"""
import sys
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

if 'webcam' not in sys.argv:
    # ① 인터넷 사진 : source에 주소(URL)를 넣으면 내려받아 바로 탐지
    results = model('https://ultralytics.com/images/zidane.jpg', save=True)
    results[0].show()                    # 박스가 그려진 결과 창
else:
    # ② 웹캠 : source=0(첫 번째 카메라), stream=True로 프레임을 하나씩 처리
    for r in model(0, show=True, stream=True, verbose=False):
        print(f'{len(r.boxes)}개 탐지', end='\r')
