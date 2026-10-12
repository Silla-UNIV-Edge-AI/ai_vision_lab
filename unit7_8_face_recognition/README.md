# 📘 7 · 8단원 · 내 얼굴을 알아보는 AI 만들기

> 교안 7 · 8단원 (얼굴 인식) | 실행 환경: PC(Anaconda) → Jetson(Miniforge) · conda `face_env`

「AI 비전 실습 통합 교안」 7 · 8단원(얼굴 인식)의 모든 실습 코드입니다.
파일 이름 순서(lab0 → lab6)가 곧 수업 순서입니다.

---

## 1. 처음 한 번만: 환경 만들기 (교안 7.2절)

```bash
cd ai_vision_lab/unit7_8_face_recognition   # 실습 폴더로 이동 (Jetson: cd ~/ai_vision_lab/unit7_8_face_recognition)
conda env create -f environment.yml     # 가상환경(face_env) + 라이브러리 한 번에 설치
conda activate face_env                 # 실습할 때마다 먼저 실행
python check_env.py                     # '모든 준비 완료!'가 나오면 성공
python webcam_test.py                   # 웹캠 화면이 뜨면 성공 (q로 종료)
```

> OpenCV는 **4.x**(`opencv-python<5`)를 사용합니다. OpenCV 5부터는 실습 0의 `cv2.CascadeClassifier`가 빠져 있습니다.

## 2. 실습 사진 (교안 7.2.9)

실습에 쓰는 사진은 모두 `img` 폴더에 들어 있습니다. 원본 강의(SKT FLY AI 2022)의 사진을 중심으로 구성했습니다.

| 폴더 | 사진 | 사용하는 실습 |
|---|---|---|
| `img/haar/` | `single.jpg`(오바마), `group.jpg`(오바마·바이든) | 실습 0-A, 0-B |
| `img/known/` | `bill.jpg`, `steve.jpg`, `elon.jpg`, `messi.jpg`, `obama.jpg`, `biden.jpg` | 실습 2, 4, 5, 6 |
| `img/unknown/` | `messi2.jpg` (같은 사람일까?) | 실습 4 |
| `img/groups/` | `team1.jpg`(4명), `team2.jpg`(11명), `bill-steve-elon.jpg`, `obama-biden.jpg` | 실습 0-A 비교, 1, 3, 5 |

사진 출처는 `img/CREDITS.md`에 있습니다. 내 사진(`img/known/me.jpg`)을 넣고 코드의 `known_people`에 한 줄을 추가하면 나도 인식됩니다. 다른 사람의 사진은 반드시 본인의 동의를 받은 뒤 사용하세요.

## 3. 실습 파일 한눈에 보기

모든 파일은 이 폴더(`unit7_8_face_recognition`)에서 `python 파일이름.py`로 실행합니다.

| 순서 | 파일 | 하는 일 | 교안(절) |
|---|---|---|---|
| 점검 | `check_env.py` | 설치 확인 | 7.2.7 |
| 점검 | `webcam_test.py` | 웹캠 확인 | 7.2.7 |
| 실습 0-A | `lab0a_haar_face.py` | Haar Cascade로 사진 속 얼굴 찾기 | 7.3.4 |
| 실습 0-B | `lab0b_haar_face_eye.py` | 얼굴 안(ROI)에서 눈까지 찾기 | 7.3.5 |
| 실습 0-C | `lab0c_haar_webcam.py` | 웹캠으로 실시간 얼굴 찾기 | 7.3.6 |
| 실습 1 | `lab1_find_faces.py` | face_recognition으로 사람 수 세기 | 7.4절 |
| 실습 2 | `lab2_landmarks.py` | 눈·코·입에 68개 점 찍기 | 7.5절 |
| 실습 3 | `lab3_crop_faces.py` | 얼굴만 오려서 저장하기 | 7.6절 |
| 실습 4 | `lab4_compare.py` | 두 사진이 같은 사람인지 판별 | 8.1절 |
| 실습 4 | `lab4_compare_cv2.py` | OpenCV로 읽어서 비교 (BGR → RGB) | 8.1.2 |
| 실습 5 | `lab5_name_tag_cv2.py` | 단체 사진에 영어 이름표 붙이기 | 8.2.2 |
| 실습 5 | `lab5_name_tag_pil.py` | 한글 이름표 붙이기 (Pillow) | 8.2.3 |
| 실습 6·7 | `lab6_webcam.py` | 웹캠 실시간 얼굴 인식 (Jetson Orin Nano도 동일) | 8.3절, 8.4절 |
| 미니 프로젝트 | `smile_selfie.py` | 웃으면 자동으로 찍히는 셀카 (실습 0 응용) | 8.5절 |
| 미니 프로젝트 | `attendance.py` | 얼굴 인식 출석부 (실습 6 응용) | 8.5절 |

## 4. 코드 읽는 법

모든 파일은 같은 모양으로 되어 있습니다.

```python
"""
[실습 번호] 제목
교안: 통합 교안 단원.절.항

▶ 준비 : 필요한 사진·장치
▶ 실행 : python 파일이름.py
▶ 결과 : 무엇이 보이면 성공인지
"""
import ...

# 1) 첫 번째 단계
# 2) 두 번째 단계
# ...
```

- 단계 번호 `# 1) # 2) …`는 교안의 「코드 한 줄씩 풀이」 순서와 같습니다.
- 창이 뜨는 실습은 사진이면 **아무 키**, 웹캠이면 **q 키**로 끝냅니다.
- 좌표 형식은 두 가지입니다. Haar Cascade(실습 0)는 `(x, y, w, h)`, face_recognition(실습 1~)은 `(top, right, bottom, left)`입니다.

## 5. 자주 만나는 오류

| 증상 | 해결 |
|---|---|
| `Please install face_recognition_models ...` | 최신 setuptools 문제 → `pip install "setuptools<81"` |
| `module 'cv2' has no attribute 'CascadeClassifier'` | OpenCV 5가 설치됨 → `pip install "opencv-python<5"` |
| `IndexError: list index out of range` | 사진에서 얼굴을 못 찾음 → 정면·밝은 사진으로 교체 |
| `ModuleNotFoundError` | `conda activate face_env`를 먼저 실행 |
| 웹캠 화면이 안 뜸 | `VideoCapture(0)`을 `(1)`로 바꿔 보기 |
| 한글 이름이 `???`로 깨짐 | `lab5_name_tag_pil.py` 사용 |

더 많은 해결 방법은 교안 8.6절을 참고하세요.

---

**교안의 마지막 단원입니다.** 수고하셨습니다 🎉
전체 흐름 복습은 [최상위 README](../README.md)의 「수업 순서대로 따라 하기」 섹션을 참고하세요.
