"""
[5.2] Jetson 첫 실행 – GPU 확인과 추론 시간 재기 (결과를 화면 창으로 보기)
교안: 5단원 5.2

▶ 준비 : Docker 컨테이너 안에서 실행 (./run_docker.sh 로 들어감) – 교안 5.1.2, bus.jpg
▶ 실행 : python3 jetson_check_cv.py
▶ 결과 : GPU 사용 여부 · 회차별 추론 시간 출력, 결과 창 (창을 클릭하고 아무 키나 누르면 종료)
"""
import cv2
import torch
from ultralytics import YOLO

# 1) GPU를 쓸 수 있는지 확인 (Jetson 컨테이너에서는 True)
print("GPU 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))

# 2) 같은 사진을 5번 추론하며 시간 재기
model = YOLO("yolo11n-pose.pt")
for i in range(5):
    result = model("bus.jpg", verbose=False)[0]
    print(f"{i + 1}회차 추론 시간: {result.speed['inference']:.1f} ms")

# 3) 결과를 화면에 보여 주기 – 4단원 PC 결과(사람 4명)와 같은지 확인
print("찾은 사람 수:", len(result.boxes), "명")
cv2.imshow("Jetson Pose", result.plot())   # 점과 뼈대를 그린 그림을 창으로 보기
cv2.waitKey(0)                             # 키를 누를 때까지 기다림
