"""
[실습 1] 사진 한 장으로 첫 포즈 추정 – 객체 탐지 vs 키포인트 검출
교안: 4단원 4.8 (PC 기초 실습)  →  5단원 5.2에서 Jetson으로 다시 실행

▶ 준비 : PC에서 pip install ultralytics (1단원과 같은 환경), bus.jpg
▶ 실행 : python lab1_first_pose.py
▶ 결과 : detect.jpg(박스만) · pose.jpg(박스 + 관절 점) 저장, 결과 숫자 출력
"""
from ultralytics import YOLO

# 1) 두 모델 불러오기 (처음 실행할 때 자동 다운로드)
det_model = YOLO("yolo11n.pt")          # 1단원의 객체 탐지 모델
pose_model = YOLO("yolo11n-pose.pt")    # 키포인트 검출 모델

# 2) 같은 사진을 두 모델에 넣기
det = det_model("bus.jpg", verbose=False)[0]
pose = pose_model("bus.jpg", verbose=False)[0]

# 3) 결과 그림 저장
det.save("detect.jpg")     # 박스만
pose.save("pose.jpg")      # 박스 + 관절 점 + 뼈대

# 4) 결과 숫자 비교
print("[객체 탐지] 찾은 물체:", [det.names[int(c)] for c in det.boxes.cls])
print("[키포인트]  찾은 사람:", len(pose.boxes), "명")
print("[키포인트]  관절 좌표 모양:", tuple(pose.keypoints.xy.shape))   # (사람 수, 17, 2)
