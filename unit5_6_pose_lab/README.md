# 📘 5 · 6단원 · YOLO11-pose Jetson 실습

> 교안 5.1 ~ 6.4 | 실행 환경: Jetson Orin Nano (Docker)

5단원부터는 **Jetson Orin Nano**에서 실행합니다. 4단원 PC 기초 실습([`../unit4_pose_basics/`](../unit4_pose_basics/))을 먼저 해 두세요.

---

## 1. 준비 — Jetson에서 (교안 5.1.2, 그림 5-1 Docker 구성도)
3단원과 **같은 컨테이너 · 같은 스크립트**(`run_docker.sh`)를 씁니다.
실습 파일은 모두 **이 폴더(`~/ai_vision_lab/unit5_6_pose_lab`)** 에 있고, 여기서 바로 실습합니다.

1. USB 웹캠을 **먼저** 연결하고 이 폴더로 이동합니다: `cd ~/ai_vision_lab/unit5_6_pose_lab`
2. 처음 한 번만 실행 권한을 줍니다: `chmod +x run_docker.sh`
3. 속도를 재기 전에 성능 모드를 켭니다: `sudo nvpmodel -m 2 && sudo jetson_clocks` (교안 3.4절 ③)
4. `./run_docker.sh`로 컨테이너에 들어갑니다. 프롬프트가 `root@…:/workspace/unit5_6_pose_lab#`로 바뀌면 성공입니다.
5. 컨테이너 안(`/workspace/unit5_6_pose_lab` = 이 폴더)에서 `python3 파일이름.py`로 실행하고, 끝나면 `exit`로 나옵니다.

---

## 2. 실습 파일

| 단계 | 파일 | 하는 일 | 교안 |
|---|---|---|---|
| 첫 실행 | `jetson_check.py`, `jetson_check_cv.py` | GPU 사용 확인 · 추론 시간 재기 · PC와 같은 결과인지 확인 (cv는 결과를 창으로) | 5.2 |
| 실습 3 | `lab3_webcam.py` | 웹캠 실시간 포즈 | 5.3 |
| 실습 3-B | `lab3b_realtime_info.py` | 화면에 사람 수 · 추론 시간 · FPS · 코 좌표 표시 | 5.4 |
| 실습 4 | `angle.py` | 세 점으로 각도 (실습 5 · 6에서 import) | 6.1 |
| 실습 5 | `lab5_arm_angle.py` | 오른쪽 팔꿈치 각도 | 6.2 |
| 실습 6 | `lab6_squat.py` | 스쿼트 횟수 세기 | 6.3 |
| 실습 7 | `lab7_tensorrt.py`, `speed_compare.sh` | TensorRT 변환 · 속도 비교 | 6.4 |

---

## 3. 기대 결과

- `jetson_check.py` → `GPU 사용 가능: True`, GPU 이름(Orin), 3회차 이후 추론 시간이 PC보다 짧음, 사람 4명
- `angle.py` → `90.0`, `180.0`, `5.710593137499633`
- 웹캠 실습은 **q 키**로 종료합니다. 관절 번호는 교안 그림 4-4 / 부록 C 참고.
- `GPU 사용 가능: False`가 나오면 `exit` 후 `docker rm -f ai_vision` 으로 컨테이너를 지우고 다시 들어가세요(`--runtime=nvidia` 필요, 교안 3.1.1 · 부록 F).
- 카메라가 열리지 않으면 `ls /dev/video*` 로 번호를 확인하고(0번이 아니면 `run_docker.sh` 의 `CAM` 줄 수정), `exit` 후 `docker rm -f ai_vision` 으로 컨테이너를 지우고 다시 들어갑니다(카메라는 컨테이너를 만들 때 연결됩니다).

---

**다음 단계** → [`../unit7_8_face_recognition/`](../unit7_8_face_recognition/) · 얼굴 인식 (PC → Jetson)
