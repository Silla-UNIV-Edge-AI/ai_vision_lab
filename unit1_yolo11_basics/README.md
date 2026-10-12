# 📘 1단원 · 객체 탐지와 YOLO 첫걸음

> 교안 1.3 · 1.4 · 부록 A | 실행 환경: PC · conda `yolo11_env`

---

## 1. 준비 (1.3 STEP 1)
```bash
conda create -n yolo11_env python=3.10 -y  # 가상환경 yolo11_env 만들기 (처음 한 번만)
conda activate yolo11_env                    # 켜기 → 프롬프트 앞에 (yolo11_env) 표시
pip install ultralytics                      # (yolo11_env) 안에 Ultralytics 설치

cd ai_vision_lab/unit1_yolo11_basics   # 실습 폴더로 이동 (터미널을 열면 홈 폴더에서 시작)
```
터미널을 새로 열 때마다 `conda activate yolo11_env`를 먼저 실행합니다. 이 폴더에서 아래 순서대로 `python 파일이름.py`로 실행합니다. 실습 사진은 `test_image_dir.zip`(사진 4장)에 들어 있고, `lab1_detect.py`가 처음 실행될 때 자동으로 풉니다.

---

## 2. 실습 파일

| 순서 | 파일 | 하는 일 | 교안 |
|---|---|---|---|
| ① | `lab1_detect.py` | 사진 준비 → `yolo11n.pt` 불러오기 → 4장 한 번에 예측 (runs/detect/predict) | 1.3 STEP 2 |
| – | (파일 없음) | 결과 사진 읽기 – 박스 · 클래스 · 신뢰도 | 1.3 STEP 3 |
| ② | `lab2_analyze.py` | 사진별 {이름: 개수}, 박스의 신뢰도 · 좌표 꺼내기 | 1.3 STEP 4 |
| ③ | `lab3_options.py` | `classes` · `conf` · `imgsz` 옵션 비교 | 1.4.1 |
| ④ | `lab4_sources.py` | 인터넷 사진(강의 예제 zidane.jpg) / `webcam` 인자를 주면 웹캠 실시간 탐지 | 1.4.2 |
| CLI | `terminal_commands.sh` | 설치 · 실행 순서 · CLI · 분할 모델 명령 모음 | 1.3 STEP 1, 1.4.3, 1.4.4 |
| 한 번에 | `yolo11_pretrained_full.py` | ① + ②를 한 파일로 | 부록 A |

```bash
python lab4_sources.py           # 인터넷 사진 (인터넷 연결 필요)
python lab4_sources.py webcam    # 웹캠 (카메라 · 모니터 필요, 결과 창에서 q 키로 종료)
```

---

## 3. 기대 결과 (lab2_analyze.py)
```
test1.jpg {'laptop': 1, 'mouse': 1, 'keyboard': 1, 'cell phone': 1}
test2.jpg {'giraffe': 1, 'zebra': 4}
test3.jpg {'sandwich': 2, 'dining table': 1, 'cup': 2, 'bowl': 1}
test4.jpg {'person': 2, 'horse': 1, 'truck': 1}

[test1.jpg 박스]
laptop     0.96  (148, 19) - (548, 336)
...
```
**lab3_options.py** : test4 사람만 conf=0.25 → 2명, conf=0.5 → 1명 / test2 imgsz=640 → 박스 5개, imgsz=1280 → 9개

신뢰도 숫자는 버전 · 장치에 따라 소수점 둘째 자리에서 조금 다를 수 있습니다 (Ultralytics 8.4, CPU 기준).

---

**다음 단계** → [`../unit2_rps_finetuning/`](../unit2_rps_finetuning/) · 내 데이터로 YOLO 학습하기 (Colab)
