# 📘 4단원 · 인체 키포인트 검출의 이해 — PC 기초 실습

> 교안 4.8 | 실행 환경: PC · 1단원과 같은 conda `yolo11_env`

4단원은 **기초 이론(4.1~4.7) + PC 기초 실습(4.8)** 입니다. Jetson 실습 환경은 5단원에서 만들므로, 여기 코드는 **1단원에서 만든 PC 환경(conda `yolo11_env`) 그대로** 실행합니다.

---

## 1. 실행 방법

```bash
conda activate yolo11_env            # 1단원의 가상환경 켜기 → 프롬프트 앞에 (yolo11_env)
cd ai_vision_lab/unit4_pose_basics   # 실습 폴더로 이동
python lab1_first_pose.py            # 이어서 lab1_first_pose_cv.py (결과를 창으로)
python lab2_keypoints.py             # 이어서 lab2_keypoints_cv.py
python lab2b_pose_logic.py           # 이어서 lab2b_pose_logic_cv.py
```

---

## 2. 실습 파일

| 실습 | 파일 | 확인하는 이론 | 교안 | 결과 |
|---|---|---|---|---|
| 실습 1 | `lab1_first_pose.py` | 객체 탐지(박스) vs 키포인트 검출(박스 + 관절 점) | 4.8.2 (← 4.1.3) | `detect.jpg`, `pose.jpg`, 결과 모양 (4, 17, 2) |
| 실습 1-CV | `lab1_first_pose_cv.py` | 결과 그림을 파일 대신 창으로 보기 | 4.8.3 | 결과 창 (아무 키로 종료) |
| 실습 2 | `lab2_keypoints.py` | 17개 번호 · 좌우 규칙, 좌표 · 신뢰도 | 4.8.4 (← 4.3 · 4.4) | 17줄 좌표 출력 + `numbers.jpg` |
| 실습 2-CV | `lab2_keypoints_cv.py` | 모든 사람의 점 · 번호 · 뼈대를 직접 그려 창으로 보기 | 4.8.5 | 결과 창 |
| 실습 2-B | `lab2b_pose_logic.py` | y는 아래로 커짐, 신뢰도 거르기, 좌표로 손 들기 판단 | 4.8.6 (← 4.4 · 4.3.2) | 터미널 출력 |
| 실습 2-B CV | `lab2b_pose_logic_cv.py` | 판단 결과를 첫 번째 사람 위에 그려 창으로 보기 | 4.8.7 | 결과 창 |

`expected_results/` 폴더에 실행하면 나와야 하는 그림(detect.jpg · pose.jpg · numbers.jpg)이 있습니다. 내 결과와 비교해 보세요.

---

## 3. 기대 출력
```
# lab1
[객체 탐지] 찾은 물체: ['bus', 'person', 'person', 'person', 'person']
[키포인트]  찾은 사람: 4 명
[키포인트]  관절 좌표 모양: (4, 17, 2)

# lab2b
코 y = 442 / 오른발목 y = 858
믿기 어려운 점: 3 번, 신뢰도 0.43
오른손목 y = 534 / 오른어깨 y = 499
오른손을 내렸습니다
```
---

**다음 단계** → [`../unit5_6_pose_lab/`](../unit5_6_pose_lab/) · 같은 모델 · 같은 사진을 Jetson의 GPU로 실행하고(`jetson_check.py`), 웹캠 실시간으로 넓힙니다.
