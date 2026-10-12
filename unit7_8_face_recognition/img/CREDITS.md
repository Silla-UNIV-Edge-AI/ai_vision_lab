# 실습 사진 목록과 출처

| 폴더 / 파일 | 내용 | 쓰는 실습 | 출처 |
|---|---|---|---|
| `known/bill.jpg`, `known/steve.jpg`, `known/elon.jpg` | 빌 게이츠 · 스티브 잡스 · 일론 머스크 (한 사람씩) | 5, 6 | 원본 강의 자료 ① |
| `known/messi.jpg` | 메시 | 2, 4, 5, 6 | 원본 강의 자료 ① |
| `known/obama.jpg`, `known/biden.jpg` | 오바마 · 바이든 | 5 | 미국 정부 사진 ② |
| `unknown/messi2.jpg` | 메시 (다른 사진) | 4 | 원본 강의 자료 ① |
| `haar/single.jpg` | 오바마 (한 사람) | 0-A | 미국 정부 사진 ② |
| `haar/group.jpg` | 오바마 · 바이든 | 0-B | 미국 정부 사진 ② |
| `groups/team1.jpg` | 4명 단체 사진 | 3 | 원본 강의 자료 ① |
| `groups/team2.jpg` | 11명 단체 사진 | 0-A 비교, 1 | 원본 강의 자료 ① |
| `groups/bill-steve-elon.jpg` | 빌 게이츠 · 일론 머스크 · 스티브 잡스 | 5 | 원본 강의 자료 ① |
| `groups/obama-biden.jpg` | 오바마 · 바이든 | 5 (바꿔 보기) | 미국 정부 사진 ② |

① SKT FLY AI Challenger 2022 「AI 프로젝트 수업」(김종현, 2022. 8.) Face Recognition 파트에서 추출했습니다.
   (예제 저장소: https://github.com/SKT-FLY-AI-2022/face_recognition_examples)
   `bill-steve-elon.jpg`와 세 사람의 사진은 강의 결과 화면에서 이름표를 지우고 잘라 낸 것이라, 이마 부분이 조금 흐릿합니다.
   **이 사진들은 수업용입니다. 교안·자료를 외부에 배포할 때는 원저작권을 확인하세요.**

② ageitgey/face_recognition 예제 이미지 (미국 백악관·정부 공식 사진, 퍼블릭 도메인)
   https://github.com/ageitgey/face_recognition/tree/master/examples

## 내 사진으로 바꾸기

`known/me.jpg`처럼 내 사진을 넣고 코드의 `known_people`에 한 줄을 추가하면 됩니다.
다른 사람의 사진은 반드시 본인의 동의를 받은 뒤 사용하세요.
