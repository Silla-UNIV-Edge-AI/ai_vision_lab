#!/bin/bash
# [3.6.2 핵심 명령] predict.py 사용 예 (교안 3.6.2 의 상자와 같은 내용)
# ※ 한꺼번에 실행하는 스크립트가 아닙니다. 필요한 줄만 복사해서 실행하세요.
#   (호스트 conda 환경이라면 컨테이너에 들어가지 않고 python predict.py ... 로)
echo "한 줄씩 복사해서 실행하세요: cat $0"
exit 0

# Jetson 터미널에서 – 프롬프트가 root@… 이면(이미 컨테이너 안) 아래 두 줄은 건너뜀
cd ~/ai_vision_lab/unit3_deploy_jetson/03_thumbs_project
./run_docker.sh                              # 들어가기

# 컨테이너 안에서
ls dataset/test/images                       # 테스트 이미지 이름 확인
# 테스트 이미지 → 사진 옆에 이미지이름_result.jpg 로 저장 (긴 이름은 Tab 키로 자동 완성)
python3 predict.py dataset/test/images/이미지이름.jpg
python3 predict.py my_photo.jpg --show       # 내 사진 – 결과 창으로 보기 (아무 키로 닫기)
python3 predict.py --webcam                  # 웹캠 실시간 (q 종료, FPS 표시)
python3 predict.py my_photo.jpg --conf 0.3   # 신뢰도 기준 (기본 0.5)
# CPU 추론 (.engine 은 GPU 전용)
python3 predict.py my_photo.jpg --model best.onnx --device cpu
exit                                         # 나오기

# 출력 예
# 모델: best.engine
# Thumbs-up    conf=0.93 box=(120, 85, 410, 460)
# 저장: my_photo_result.jpg

# 참고: 내 사진은 my_photo.jpg 라는 이름으로 이 폴더에 넣어 둡니다.
#       다른 카메라 번호(--cam 1)를 쓰려면 run_docker.sh 의 CAM 줄을 고치고
#       docker rm -f ai_vision 으로 컨테이너를 지운 뒤 다시 들어갑니다.
