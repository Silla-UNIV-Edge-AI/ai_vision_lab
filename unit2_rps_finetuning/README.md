# 📘 2단원 · 내 데이터로 YOLO 학습하기 — 가위 · 바위 · 보

> 교안 2.2 · 부록 B | 실행 환경: Google Colab (T4 GPU)

---

## 1. 실행 방법 (Google Colab)
1. colab.research.google.com → **파일 → 노트북 업로드** → `RPS_finetuning_step_by_step.ipynb`
2. **런타임 → 런타임 유형 변경 → T4 GPU** (STEP 1)
3. 왼쪽 🔑 **보안 비밀** 에 `ROBOFLOW_API_KEY` 추가 + 노트북 액세스 켜기 (Roboflow 가입: 교안 2.1.3, 키 발급: 2.2 STEP 2-1)
4. 셀 ①부터 차례로 실행 (학습 셀 ⑦은 T4 기준 약 45분)
5. 셀 ⑭에서 `best.pt`를 내려받아 3단원에서 사용

| 파일 | 내용 |
|---|---|
| `RPS_finetuning_step_by_step.ipynb` | STEP 1~8, 셀 ①~⑭ (교안과 같은 번호, 설명 포함) |
| `RPS_finetuning_full.ipynb` | 부록 B 전체 코드 – 한 번에 다시 실행할 때 |

- 데이터셋: Roboflow Universe `roboflow-58fyf / rock-paper-scissors-sxsw` version 14 (노트북이 자동으로 내려받음)
- 셀 ①이 안 되면 셀 ①-대안(다운로드 링크 + curl)을 사용합니다.
- 학습 뒤 런타임이 다시 시작되면 STEP 7의 「모델 다시 불러오기」 셀을 먼저 실행하세요.

---

## 2. (보충) Jetson에서 실시간 앱으로 실행하기 (Docker)
교안 3.4절 ④ 「실측 사례」처럼 ONNX와 TensorRT의 실시간 속도(FPS)를 비교해 보는 두 앱입니다. 변환 규칙은 교안 3.4절 ③과 같습니다(`imgsz=416`, FP16, 파일 이름 `best.onnx` · `best.engine`).
Colab에서 내려받은 `best.pt`를 이 폴더에 넣고, USB 카메라를 연결한 뒤 이 폴더에서 실행합니다.

```bash
cd ~/ai_vision_lab/unit2_rps_finetuning
./run_docker.sh export    # ① TensorRT 엔진 만들기 (약 8~10분, 처음 한 번) → best.onnx · best.engine
./run_docker.sh app       # ONNX 모델로 실시간 추론
./run_docker.sh app_trt   # TensorRT 엔진으로 실시간 추론 (가장 빠름)
```

| 명령 | 내용 |
|---|---|
| `./run_docker.sh export` | `export_trt.py` – `best.pt` → `best.onnx` · `best.engine` (FP16, imgsz 416) |
| `./run_docker.sh app` | `app.py` – `best.onnx` 로 추론 |
| `./run_docker.sh app_trt` | `app_trt.py` – `best.engine` 으로 추론 |
| `./run_docker.sh shell` | 컨테이너 bash 진입 |
| `./run_docker.sh rm` | 컨테이너 삭제 (다음 실행 때 새로 만듦) |

- 컨테이너 이름은 `rps`, 이미지는 `ultralytics-jetson:gui` 입니다. `ai_vision_lab` 폴더가 컨테이너의 `/workspace` 로 연결됩니다.
- 추론 창은 **q** 키로 닫습니다 (교안의 다른 웹캠 실습과 같음).
- `ai_vision_lab` 폴더의 위치를 옮겼거나 예전 스크립트로 만든 컨테이너가 남아 있으면 `./run_docker.sh rm` 으로 지운 뒤 다시 실행하세요 (폴더 연결은 컨테이너를 만들 때 정해집니다).
- 예전 규칙(imgsz 480)으로 만든 `best.onnx` · `best_.engine` 이 남아 있다면 지우고 `./run_docker.sh export` 를 다시 실행하세요.
- 처음 실행할 때 onnxruntime-gpu를 Jetson용 1.24.0으로 바꿔 설치합니다 (이미지에 든 버전은 numpy 2와 맞지 않음).
- `.engine` 파일은 만든 환경의 TensorRT 버전에서만 동작합니다 (컨테이너 10.7, 호스트 conda 환경 10.3). 그래서 `app_trt` 는 컨테이너에서 `export` 를 한 뒤에만 실행되고, 그 전에는 export 하라는 안내를 띄웁니다.
- 창이 뜨지 않으면 호스트에서 `xhost +SI:localuser:root` 를 실행하세요 (스크립트가 매번 자동 실행합니다).

---

**다음 단계** → [`../unit3_deploy_jetson/`](../unit3_deploy_jetson/) · 학습한 모델을 Jetson에 배포하기
