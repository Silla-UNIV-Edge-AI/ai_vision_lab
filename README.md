# ai_vision_lab · 딥러닝과 Edge AI 실습 자료

> 「AI 비전 실습 통합 교안」(3부 8단원)의 **모든 실습 코드와 실습 사진**을 단원 순서대로 모은 저장소입니다.

![python](https://img.shields.io/badge/Python-3.10-blue)
![ultralytics](https://img.shields.io/badge/Ultralytics-8.4-ff69b4)
![platform](https://img.shields.io/badge/Platform-PC%20%7C%20Jetson%20Orin%20Nano-success)
![license](https://img.shields.io/badge/Use-Educational-lightgrey)

폴더 이름의 **단원 번호가 곧 교안의 단원 번호**이고, 각 파일 맨 위에 **교안 위치 · 준비 · 실행 · 결과**가 적혀 있습니다.

> 📂 **폴더 위치** — 이 `ai_vision_lab` 폴더를 **홈 폴더**에 둡니다.
> - Jetson: `~/ai_vision_lab`
> - PC: 터미널(Anaconda Prompt)을 열었을 때의 폴더 아래 `ai_vision_lab`
>
> 교안의 모든 경로는 이 위치를 기준으로 합니다.

---

## 📑 목차

1. [폴더 한눈에 보기](#1-폴더-한눈에-보기)
2. [수업 순서대로 따라 하기](#2-수업-순서대로-따라-하기)
3. [실행 환경](#3-실행-환경-자세한-내용-교안-부록-f)
4. [확인한 내용](#4-확인한-내용)
5. [사진 출처](#5-사진-출처)
6. [다운로드 · 사용 방법](#6-다운로드--사용-방법)

---

## 1. 폴더 한눈에 보기

| 폴더 | 교안 | 실행 장소 | 주요 파일 |
|---|---|---|---|
| [`unit1_yolo11_basics/`](unit1_yolo11_basics/) | Ⅰ부 1단원 · 부록 A | PC 또는 Jetson | `lab1_detect.py` ~ `lab4_sources.py`, `yolo11_pretrained_full.py`, 실습 사진 `test_image_dir.zip` |
| [`unit2_rps_finetuning/`](unit2_rps_finetuning/) | Ⅰ부 2단원 · 부록 B | Google Colab (T4 GPU) | `RPS_finetuning_step_by_step.ipynb`(셀 ①~⑭), `RPS_finetuning_full.ipynb` · (보충) Jetson 실시간 앱 `app.py` · `app_trt.py` |
| [`unit3_deploy_jetson/`](unit3_deploy_jetson/) | Ⅰ부 3단원 | Colab + Jetson Orin Nano (Docker) | `01_reuse_model/`(3.1 · 3.2), `02_jetson_export/`(3.4), `03_thumbs_project/`(3.6 👍/👎 응용 프로젝트), `run_docker.sh` |
| [`unit4_pose_basics/`](unit4_pose_basics/) | Ⅱ부 4단원 | PC (1단원 환경) | 기초 실습 `lab1_first_pose.py` · `lab2_keypoints.py` · `lab2b_pose_logic.py`와 창으로 보는 `_cv.py` 3개, `bus.jpg`, 기대 결과 그림 |
| [`unit5_6_pose_lab/`](unit5_6_pose_lab/) | Ⅱ부 5 · 6단원 | Jetson Orin Nano (Docker) | `run_docker.sh`, `jetson_check.py` · `jetson_check_cv.py`, `lab3_webcam.py` · `lab3b_realtime_info.py`, `angle.py`, `lab5` ~ `lab7`, `bus.jpg` |
| [`unit7_8_face_recognition/`](unit7_8_face_recognition/) | Ⅲ부 7 · 8단원 | PC(Anaconda) → Jetson(Miniforge) | `lab0a` ~ `lab6`, 미니 프로젝트 2개, 실습 사진 `img/` |

> **실습 번호 메모**
> - 1단원 실습 파일은 `lab1`~`lab4` (1.3 STEP 2 · 4, 1.4.1 · 1.4.2)
> - 자세 추정은 **단원마다 번호를 따로** 셉니다. 4단원 PC 기초 실습은 실습 1 · 2 · 2-B (각 CV 실습 포함), 5단원 Jetson은 실습 3 · 3-B, 6단원은 실습 4~7.
> - 7 · 8단원 얼굴 인식은 실습 0 ~ 7.

---

## 2. 수업 순서대로 따라 하기

```text
1단원  unit1_yolo11_basics       PC에서 conda 환경 yolo11_env 만들기 → lab1 → lab2 (→ lab3 · lab4 확장)
2단원  unit2_rps_finetuning      Colab에 노트북 업로드 → 셀 ①부터 차례로 실행 → best.pt 내려받기
3단원  unit3_deploy_jetson       Jetson에서 ./run_docker.sh 로 컨테이너 진입 → best.pt 재사용(01) → TensorRT 변환(02) → 👍/👎 프로젝트(03)
4단원  unit4_pose_basics         PC(conda yolo11_env)에서 lab1 → lab2 → lab2b (박스 vs 관절 · 숫자 · 번호 · 자세 판단)
5단원  unit5_6_pose_lab          3단원과 같은 ./run_docker.sh 로 컨테이너 진입 → jetson_check → jetson_check_cv → lab3 → lab3b
6단원  unit5_6_pose_lab          angle.py → lab5 → lab6 → lab7 (speed_compare.sh 로 속도 비교)
7단원  unit7_8_face_recognition  environment.yml 로 환경 → check_env.py → lab0a … lab3
8단원  unit7_8_face_recognition  lab4 … lab6 → Jetson에서 lab6 → 미니 프로젝트
```

---

## 3. 실행 환경 (자세한 내용: 교안 부록 F)

| 단원 | 환경 | 설치 방법 |
|---|---|---|
| **1** | PC · conda 가상환경 `yolo11_env` (Python 3.10) | `conda create -n yolo11_env python=3.10 -y` → `conda activate yolo11_env` → `pip install ultralytics` (교안 1.3 STEP 1) |
| **2** | Google Colab | 노트북 셀 안에서 설치 (Roboflow API Key 필요 — 교안 2.1.3) |
| **3** | Jetson Orin Nano (JetPack 6) · Docker 컨테이너 | 각 실습 폴더의 `run_docker.sh` (교안 3.1.1, 이미지 `ultralytics-jetson:gui`) + 이 폴더의 `setup_container.sh` |
| **4** | 1단원과 같은 PC 환경 (conda `yolo11_env`) | 추가 설치 없음 |
| **5 · 6** | 3단원과 같은 Docker 컨테이너 | `unit5_6_pose_lab/run_docker.sh` (3단원과 같은 파일, 교안 5.1.2) |
| **7 · 8** | conda 가상환경 `face_env` (PC: Anaconda, Jetson: Miniforge) | `conda env create -f unit7_8_face_recognition/environment.yml` |

### 메모

- `setup_container.sh`(이 폴더 바로 아래)는 Jetson 컨테이너를 처음 만들 때 `run_docker.sh`가 **한 번** 실행하는 준비 스크립트입니다(ONNX Runtime 교체 · roboflow · Jupyter 설치). **옮기거나 지우지 마세요.**
- 모델 파일(`yolo11n.pt`, `yolo11n-pose.pt`, `yolo11n-seg.pt`)은 처음 실행할 때 **자동으로 내려받아지므로** 이 묶음에 넣지 않았습니다(인터넷 필요).
- 2 · 3단원 데이터셋(가위바위보, Thumbs-up/down)은 용량이 커서 넣지 않았습니다. 노트북과 `download_dataset.py`가 Roboflow에서 직접 받습니다.
- 웹캠 실습은 창이 뜨면 **q 키**로 끝냅니다. Jetson에서는 모니터(DisplayPort)가 연결되어 있어야 창이 보입니다.

---

## 4. 확인한 내용

- **1단원 스크립트 전체**, **4단원 기초 실습**(1 · 2 · 2-B), **5 · 6단원** `jetson_check` · 실습 3 · 3-B · 4~6(CPU로 대신 확인), **3단원 `predict.py`** 는 Ultralytics 8.4(CPU)에서 실제로 실행해 교안의 결과와 일치함을 확인했습니다.
  - 예: `test1.jpg → laptop · mouse · keyboard · cell phone`, 각도 `90.0 / 180.0 / 5.7`
- **7 · 8단원** 코드는 `face_recognition` 1.3.0으로 확인했습니다.
- **Colab 노트북(2단원)** 과 **Jetson 전용 코드**(TensorRT 변환, `lab7_tensorrt.py`)는 문법만 점검했으므로, 수업 전에 강사가 한 번 실행해 보기를 권장합니다.

> 참고: Ultralytics 8.4부터 결과 폴더 이름이 `predict-2`, `predict-3`처럼 하이픈(-)이 붙습니다(이전 버전은 `predict2`). 1단원 코드는 실행할 때마다 결과 폴더 경로를 출력합니다.

---

## 5. 사진 출처

[`CREDITS.md`](CREDITS.md)를 참고하세요. 얼굴 사진은 수업용이며, **다른 사람의 사진은 반드시 본인의 동의를 받은 뒤** 사용하세요.

---

## 6. 다운로드 · 사용 방법

```bash
# ① 저장소를 홈 폴더에 복제
cd ~
git clone https://github.com/Silla-UNIV-Edge-AI/ai_vision_lab.git

# ② 단원 폴더로 이동해서 실습
cd ai_vision_lab/unit1_yolo11_basics
# 각 단원 README.md 의 안내를 따르세요
```

Jetson에서는 `~/ai_vision_lab`에 그대로 두면 모든 `run_docker.sh`가 바로 동작합니다.
PC에서는 터미널(Anaconda Prompt)을 열었을 때의 폴더에 두세요.

---

## 🙋 문의

- 수업 자료 관련 문의: 신라대학교 사범대학 컴퓨터교육과 딥러닝과 Edge-AI 교과 담당 교수(jhkim3217@gmail.com)
- 코드 오류 · 개선 제안: [Issues](../../issues) 에 올려 주세요
