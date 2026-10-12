#!/bin/bash
# Jetson 실습(3 · 5 · 6단원) – Docker 컨테이너에 들어가기
#   사용법 : 실습 폴더에서 ./run_docker.sh  →  python3 파일.py · yolo ...  →  exit
#   지우기 : docker rm -f ai_vision   (컨테이너를 다시 만들고 싶을 때)

IMAGE="ultralytics-jetson:gui"      # 실습용 Docker 이미지
NAME="ai_vision"                    # 컨테이너 이름
CAM="/dev/video0"                   # USB 웹캠 (ls /dev/video* 로 확인)
LAB_DIR="$HOME/ai_vision_lab"       # 실습 코드 폴더 → 컨테이너 안에서는 /workspace
HERE="/workspace${PWD#$LAB_DIR}"    # 지금 폴더 → 컨테이너 안에서 같은 폴더

# 0) 실행 전 확인 – 실습 폴더에서 실행했는지, 웹캠이 연결되어 있는지
[[ "$PWD" == "$LAB_DIR"* ]] || { echo "~/ai_vision_lab 안에서 실행하세요."; exit 1; }
[ -e "$CAM" ]               || { echo "USB 웹캠($CAM)을 먼저 연결하세요."; exit 1; }

# 1) 화면 – 컨테이너에서 띄운 창이 Jetson 모니터에 보이게
export DISPLAY=${DISPLAY:-:1}
xhost +SI:localuser:root > /dev/null

# 2) 컨테이너 만들기 (처음 한 번) – GPU · 화면 · 실습 폴더 · 웹캠을 차례로 이어 줌
if ! docker ps -a | grep -qw "$NAME"; then
    docker run -d --name $NAME --ipc=host --network=host \
        --runtime=nvidia \
        -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$LAB_DIR":/workspace \
        --device $CAM \
        $IMAGE sleep infinity > /dev/null || exit 1
    docker exec $NAME bash /workspace/setup_container.sh    # 처음 한 번 준비 (몇 분)
fi

# 3) 들어가기 – 컨테이너를 켜고, 지금 폴더에서 시작 (나올 때는 exit)
docker start $NAME > /dev/null
docker exec -it -e DISPLAY -w "$HERE" $NAME bash

# 4) 나온 뒤 – 컨테이너(root)가 만든 파일을 내 계정 것으로 되돌리기
docker exec $NAME chown -R $(id -u):$(id -g) "$HERE"
