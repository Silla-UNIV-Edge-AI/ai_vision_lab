"""
[3.6.2 ⑥] Thumbs-up / Thumbs-down 예측 – 사진 또는 USB 웹캠
교안: 3단원 3.6.2 (핵심 명령 – 예측)

▶ 준비 : train.ipynb로 만든 모델 (best.engine → best.onnx → best.pt 순으로 있는 것을 자동 선택)
▶ 실행 : (Docker 컨테이너 안에서는 python3 predict.py … – 교안 3.6.2)
    python predict.py 사진.jpg                    # 결과 → 사진과 같은 폴더에 사진_result.jpg
    python predict.py 사진.jpg --show             # 결과 창으로 보기
    python predict.py --webcam                    # USB 웹캠 실시간 (q 종료, FPS 표시)
    python predict.py --webcam --cam 1            # /dev/video1 사용 (Docker: run_docker.sh 의 CAM 도 바꾸고 컨테이너를 다시 만듦)
    python predict.py 사진.jpg --conf 0.3         # 신뢰도 기준 조정 (기본 0.5)
    python predict.py 사진.jpg --model best.onnx --device cpu   # CPU 추론 (.engine은 GPU 전용)
▶ 결과 : 모델: best.engine
         Thumbs-up    conf=0.93 box=(120, 85, 410, 460)
         저장: 사진_result.jpg
"""
import argparse
import os

import cv2
from ultralytics import YOLO

# 1) 명령 옵션 읽기
parser = argparse.ArgumentParser(description="Thumbs-up / Thumbs-down 예측")
parser.add_argument("image", nargs="?", help="예측할 사진 파일")
parser.add_argument("--webcam", action="store_true", help="USB 웹캠 실시간 예측")
parser.add_argument("--cam", type=int, default=0, help="카메라 번호 (기본 0 = /dev/video0)")
parser.add_argument("--model", help="모델 파일 (기본: best.engine → best.onnx → best.pt)")
parser.add_argument("--conf", type=float, default=0.5, help="신뢰도 기준 (기본 0.5)")
parser.add_argument("--device", help="0 = GPU, cpu = CPU")
parser.add_argument("--show", action="store_true", help="결과를 창으로 보기")
args = parser.parse_args()

# 2) 모델 고르기 – 이 폴더 → 학습 폴더 순으로, 빠른 형식부터
weights = "runs/detect/thumb/weights"
candidates = [args.model] if args.model else [
    "best.engine", f"{weights}/best.engine", "best.onnx", f"{weights}/best.onnx", f"{weights}/best.pt"]
model_path = next((m for m in candidates if os.path.exists(m)), None)
if model_path is None:
    raise SystemExit("모델 파일이 없습니다. train.ipynb로 먼저 학습 · 변환하세요.")
print("모델:", model_path)
model = YOLO(model_path, task="detect")

# 3-A) 웹캠 실시간 예측 – stream=True 로 프레임을 하나씩 받아서 창에 표시
if args.webcam:
    if not cv2.VideoCapture(args.cam).isOpened():
        raise SystemExit(f"카메라 {args.cam}번을 열 수 없습니다. ls /dev/video* 로 번호를 확인하세요.")
    for result in model(args.cam, conf=args.conf, device=args.device, stream=True, verbose=False):
        image = result.plot()
        fps = 1000 / sum(result.speed.values())          # 전처리+추론+후처리 시간(ms)으로 FPS 계산
        cv2.putText(image, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Thumbs", image)
        if cv2.waitKey(1) == ord("q"):                    # q 키로 종료
            break

# 3-B) 사진 한 장 예측 – 결과 출력 · 저장
elif args.image:
    result = model(args.image, conf=args.conf, device=args.device, verbose=False)[0]
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        print(f"{result.names[int(box.cls)]:12s} conf={float(box.conf):.2f} box=({x1}, {y1}, {x2}, {y2})")

    out_path = os.path.splitext(args.image)[0] + "_result.jpg"
    image = result.plot()
    cv2.imwrite(out_path, image)
    print("저장:", out_path)
    if args.show:
        cv2.imshow("Thumbs", image)
        cv2.waitKey(0)
else:
    parser.print_help()
