# 1단원 터미널 명령 모음 (복사해서 한 줄씩 실행)

# ── 1.3 STEP 1 설치 ──
conda create -n yolo11_env python=3.10 -y  # 가상환경 yolo11_env 만들기 (처음 한 번만)
conda activate yolo11_env                    # 켜기 → 프롬프트 앞에 (yolo11_env) 표시
pip install ultralytics                      # (yolo11_env) 안에 Ultralytics 설치

cd ai_vision_lab/unit1_yolo11_basics   # 실습 폴더로 이동 (터미널을 열면 홈 폴더에서 시작)

# ── 1.3 실습 파일 실행 ──
python lab1_detect.py                # STEP 2  준비 · 모델 · 예측
python lab2_analyze.py               # STEP 4  결과를 코드로 읽기

# ── 1.4 실습 확장 ──
python lab3_options.py               # 1.4.1   conf · classes · imgsz
python lab4_sources.py               # 1.4.2   인터넷 사진(zidane.jpg)
python lab4_sources.py webcam        # 1.4.2   웹캠 (결과 창에서 q 키로 종료)

# ── 1.4.3 명령어(CLI) 한 줄로 실행 ──
yolo predict model=yolo11n.pt source=test_image_dir save=True
yolo predict model=yolo11n.pt source=test_image_dir/test4.jpg classes=0 conf=0.5
yolo predict model=yolo11n.pt source=0 show=True
yolo predict model=yolo11n.pt source=video.mp4

# ── 1.4.4 다른 작업(Task) 모델 ──
yolo predict model=yolo11n-seg.pt source=test_image_dir/test4.jpg save=True
