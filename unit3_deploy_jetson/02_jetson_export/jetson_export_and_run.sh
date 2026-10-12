#!/bin/bash
# [3.4 ③] Jetson Orin Nano에서 제 성능을 내는 실행 순서
# ※ 한꺼번에 실행하는 스크립트가 아닙니다. 이 폴더(02_jetson_export)의 터미널에 한 줄씩 복사해서 실행하세요.
#   ([호스트] = Jetson 터미널에서, [컨테이너] = ./run_docker.sh 로 들어간 컨테이너 안에서)
echo "한 줄씩 복사해서 실행하세요: cat $0"
exit 0

# 0) [호스트] 3.4절 실습 폴더로 이동하고 3.1절의 모델 가져오기
cd ~/ai_vision_lab/unit3_deploy_jetson/02_jetson_export
cp ../01_reuse_model/best.pt .

# 1) [호스트] 전원 모드 최대 + 클럭 고정 (JetPack 6.2는 Orin Nano에서 MAXN SUPER 모드 지원)
sudo nvpmodel -q           # 현재 모드 확인 (모드 번호는 기기마다 다름 → /etc/nvpmodel.conf 의 NAME=MAXN_SUPER 번호)
sudo nvpmodel -m 2         # MAXN SUPER 모드 번호로 변경 (Orin Nano Super 개발자 키트는 2)
sudo jetson_clocks         # 클럭 최대 고정

# 2) [호스트] 컨테이너에 들어가기 → 프롬프트가 root@… 로 바뀜
./run_docker.sh

# 3) [컨테이너] 반드시 Jetson에서 TensorRT FP16 엔진으로 변환 (Colab에서 만든 engine은 호환 안 됨)
#    약 8~10분 걸립니다. 중간에 한동안 출력이 없어도 멈춘 것이 아니니 기다리세요. → best.onnx 와 best.engine 이 함께 생김
yolo export model=best.pt format=engine half=True imgsz=416
ls -lh best.*              # 세 파일이 모두 있는지 확인

# 4) [컨테이너] 세 형식을 같은 웹캠으로 차례로 실행 → 속도 기록 (source=0 → 첫 번째 카메라, 창은 q 키로 닫는다)
#    순서: PyTorch 원본(.pt) → ONNX(.onnx) → TensorRT FP16(.engine)
#    결과 영상(runs/detect/predict/0.avi)을 저장하지 않으려면 명령 끝에 save=False 를 붙입니다.
yolo predict model=best.pt     source=0 imgsz=416 show=True
yolo predict model=best.onnx   source=0 imgsz=416 show=True
yolo predict model=best.engine source=0 imgsz=416 show=True
exit                       # 나오기 → Jetson 터미널로 돌아옴

# 5) [호스트] GPU·메모리 사용량 실시간 확인 (jtop 은 컨테이너가 아니라 호스트에서)
sudo pip3 install -U jetson-stats     # 처음 한 번만 설치
sudo reboot                           # 설치 후 재부팅해야 jtop 서비스가 켜짐
jtop                                  # 재부팅 후 실행 (q 로 종료)
