# 📘 3단원 · 학습한 모델 활용과 엣지 배포 — Jetson Orin Nano

> 교안 3.1 ~ 3.6 | 실행 환경: Colab + Jetson Orin Nano (Docker)

Jetson 공통 준비(JetPack 6 · DisplayPort 화면 · 웹캠 · 성능 모드)는 교안 **부록 F**를 먼저 확인하세요.
이 폴더의 위치는 `~/ai_vision_lab/unit3_deploy_jetson` 이고, 절마다 하위 폴더(01 → 3.1 · 3.2절, 02 → 3.4절, 03 → 3.6절)에서 실습합니다.

---

## 1. 파일 한눈에 보기

| 폴더 / 파일 | 하는 일 | 교안 |
|---|---|---|
| `01_reuse_model/reuse_model_image.py` | 2단원 best.pt 다시 불러와 사진 1장 예측 | 3.1.2 |
| `01_reuse_model/prepare_my_hand.py` | 손 사진 3장(my_hand_scissors · rock · paper)을 한 번에 예측해 한 창에 나란히 보기 | 3.1.3 |
| `01_reuse_model/reuse_model_webcam.py` | 2단원 best.pt 로 웹캠 실시간 예측 | 3.1.4 |
| `01_reuse_model/colab_cli_commands.txt` | 학습 · 검증 · 추론을 CLI 한 줄로 (Colab) | 3.2 |
| `02_jetson_export/jetson_export_and_run.sh` | 성능 모드 → TensorRT FP16 변환 → .pt · .onnx · .engine 웹캠 추론 비교 → jtop | 3.3 · 3.4 ③ |
| `02_jetson_export/timing_template.py` | 전처리 · 추론 · 후처리 구간별 시간 재기 (틀) | 3.4 ④ |
| `03_thumbs_project/` | 👍/👎 응용 프로젝트 (아래) | 3.6 |
| `run_docker.sh` | Docker 컨테이너(`ai_vision`)에 들어가기 – 3 · 5 · 6단원 공용 (아래) | 3.1.1 · 3.7 |

---

## 2. Docker 컨테이너에서 실행 (`run_docker.sh`) — 교안 3.1.1
각 폴더에 같은 `run_docker.sh` 가 들어 있습니다(5 · 6단원 `unit5_6_pose_lab` 의 것과도 같은 파일). 규칙은 하나입니다.
**`./run_docker.sh` 로 컨테이너에 들어간 뒤, 그 안에서 평소처럼 명령을 실행하고 `exit` 로 나옵니다.**
들어갈 때 있던 폴더가 그대로 컨테이너의 작업 폴더가 됩니다. USB 웹캠은 먼저 연결해 두세요.
```bash
cd ~/ai_vision_lab/unit3_deploy_jetson/01_reuse_model
cp ~/Downloads/best.pt .                 # Colab에서 받은 best.pt 복사 (2단원 STEP 8)
chmod +x ../*/run_docker.sh              # 실행 권한 주기 (처음 한 번)
./run_docker.sh                          # 들어가기 → 프롬프트가 root@… 로 바뀌면 성공 (처음에는 준비하느라 몇 분)
python3 reuse_model_image.py             #   (컨테이너 안) 사진 1장 – 아무 키로 종료
python3 prepare_my_hand.py               #   (컨테이너 안) 사진 3장을 한 창에 – 아무 키로 종료
python3 reuse_model_webcam.py            #   (컨테이너 안) 웹캠 – q 종료
exit                                     # 나오기

cd ../02_jetson_export                   # sudo nvpmodel · jetson_clocks · jtop 은 컨테이너 밖(Jetson 터미널)에서
cp ../01_reuse_model/best.pt .           # 3.1절의 모델 가져오기
./run_docker.sh                          # 들어가기
yolo export model=best.pt format=engine half=True imgsz=416
yolo predict model=best.engine source=0 imgsz=416 show=True
exit

cd ../03_thumbs_project
./run_docker.sh                          # 들어가기 (아래는 모두 컨테이너 안)
export ROBOFLOW_API_KEY=내키
python3 download_dataset.py
jupyter notebook                         # Jetson 브라우저에서 http://localhost:8888 → train.ipynb (종료 Ctrl+C)
python3 predict.py --webcam
exit
```
| 명령 | 입력하는 곳 | 뜻 |
|---|---|---|
| `./run_docker.sh` | Jetson 터미널 (실습 폴더) | 컨테이너에 들어가기 – 컨테이너가 없으면 만들고 들어감 |
| `exit` | 컨테이너 안 | 나오기 – 컨테이너는 남아 있어 다음에는 바로 들어감 |
| `docker rm -f ai_vision` | Jetson 터미널 | 컨테이너 지우기 – 카메라 · 화면 문제나 폴더 위치를 옮겼을 때 다시 만들기 |

`run_docker.sh` 는 30여 줄이고 위에서 아래로 읽으면 됩니다: 0) 실습 폴더 · 웹캠 확인 → 1) 화면 연결 → 2) 컨테이너 만들기(처음 한 번) → 3) 들어가기 → 4) 나온 뒤 결과 파일 소유권 되돌리기.

- 컨테이너 이름은 `ai_vision`, 이미지는 `ultralytics-jetson:gui` 입니다. `~/ai_vision_lab` 폴더가 컨테이너의 `/workspace` 로 연결됩니다.
- 컨테이너를 처음 만들 때 `~/ai_vision_lab/setup_container.sh` 가 한 번 실행되어 ONNX Runtime 교체, 결과 폴더 설정, roboflow · Jupyter Notebook 설치를 합니다(몇 분, 인터넷 필요). "설치 실패"가 나오면 컨테이너 안에서 `bash /workspace/setup_container.sh` 를 다시 실행하세요.
- 웹캠은 `/dev/video0` 을 씁니다. 번호가 다르면 `run_docker.sh` 의 `CAM` 줄을 고친 뒤 컨테이너를 지우고 다시 들어갑니다.
- 컨테이너(TensorRT 10.7)에서 만든 `.engine`은 호스트 conda 환경(TensorRT 10.3, 교안 부록 F ③)에서 쓸 수 없습니다. 실행할 쪽에서 변환하세요.
- 예전 스크립트로 만든 컨테이너(`unit3`)가 남아 있으면 `docker rm -f unit3` 로 지워도 됩니다.

---

## 3. 03_thumbs_project — 순서
```bash
pip install -r requirements.txt            # 호스트 conda 환경만. Docker 컨테이너는 설치하지 말 것 (처음 만들 때 설치됨)
export ROBOFLOW_API_KEY=내키
python download_dataset.py                 # ① dataset/ 폴더 받기 (CC BY 4.0)
jupyter notebook train.ipynb               # ②~⑤ 환경 확인 → 학습 → 평가 → ONNX · TensorRT 변환
python predict.py dataset/test/images/사진이름.jpg   # ⑥ 사진 예측 → 사진이름_result.jpg
python predict.py --webcam                 # ⑥ 웹캠 실시간 (q 종료, FPS 표시)
```
- 위 명령은 호스트 conda 환경 기준입니다. Docker 컨테이너에서는 `./run_docker.sh` 로 들어가 `python3 …` · `jupyter notebook` 으로 실행합니다 (위 "Docker 컨테이너에서 실행" 참고).
- `predict.py`는 모델을 지정하지 않으면 `best.engine → best.onnx → best.pt` 순서로, 이 폴더와 `runs/detect/thumb/weights/`에서 찾아 씁니다. 노트북 ⑤가 변환 파일을 이 폴더로 복사합니다.
- `.engine` 파일은 **변환한 그 Jetson 전용**입니다. 다른 장비에서는 다시 변환하세요.
- 수업에서는 안내 · 시연을 하고, 학습(20 epoch)은 수업 외 과제로 진행합니다(교안 3.6).

---

**다음 단계** → [`../unit4_pose_basics/`](../unit4_pose_basics/) · 자세 추정(Pose) 기초 실습 (PC)
