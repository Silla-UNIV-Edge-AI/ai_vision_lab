#!/bin/bash
# 2단원 RPS 스크립트를 Docker 컨테이너에서 실행 (Jetson Orin, USB 카메라 + GUI 창)
#
# 사용법:
#   ./run_docker.sh export    # export_trt.py  - TensorRT 엔진 만들기 (약 10분) → best.onnx · best.engine
#   ./run_docker.sh app       # app.py         - ONNX 모델로 실시간 추론 (q 종료)
#   ./run_docker.sh app_trt   # app_trt.py     - TensorRT 엔진으로 실시간 추론 (q 종료)
#   ./run_docker.sh shell     # 컨테이너 bash 진입
#   ./run_docker.sh rm        # 컨테이너 삭제
#
# 주의: 컨테이너의 TensorRT(10.7)는 호스트 conda 환경의 TensorRT(10.3)와 버전이 달라
#       엔진(.engine) 파일을 서로 바꿔 쓸 수 없습니다. 컨테이너에서 쓰려면 컨테이너에서
#       ./run_docker.sh export 로 다시 만드세요. (같은 파일 이름이라 덮어씁니다)

IMAGE="ultralytics-jetson:gui"
NAME="rps"
HOST_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"   # 이 스크립트가 있는 2단원 폴더 (~/ai_vision_lab/unit2_rps_finetuning)
LAB_DIR="$(dirname "$HOST_DIR")"               # ai_vision_lab → /workspace 로 마운트 (다른 단원도 접근 가능)
UNIT_DIR="/workspace/$(basename "$HOST_DIR")"  # 컨테이너 안 작업 폴더
ENGINE="best.engine"                           # export_trt.py 가 만들고 app_trt.py 가 읽는 엔진
ENGINE_MARK=".engine_built_in_docker"         # 컨테이너에서 엔진을 만들었다는 표시

# numpy 2 와 호환되는 Jetson(JetPack 6, CUDA 12.6)용 onnxruntime-gpu
# (이미지에 든 1.23.0 은 numpy 1.x 로 빌드되어 import 가 실패함)
ORT_WHEEL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d98/0b934b9a29c1a/onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"

# 1. X11 디스플레이 설정 (컨테이너의 root 가 화면을 쓰도록 허용)
export DISPLAY=${DISPLAY:-:1}
xhost +SI:localuser:root > /dev/null 2>&1

# 2. 컨테이너 준비: 없으면 생성, 멈춰 있으면 시작
ensure_container() {
    if [ -z "$(docker ps -aq -f name=^${NAME}$)" ]; then
        echo ">> 컨테이너 '${NAME}' 를 새로 생성합니다."
        # 연결된 USB 카메라(/dev/video*) 모두 넘기기
        CAMS=""
        for dev in /dev/video*; do
            [ -e "$dev" ] && CAMS="$CAMS --device $dev"
        done
        docker run -d --name "$NAME" --ipc=host --runtime=nvidia \
            -e DISPLAY="$DISPLAY" \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v "$LAB_DIR:/workspace" \
            -w "$UNIT_DIR" \
            $CAMS \
            "$IMAGE" sleep infinity > /dev/null || exit 1
    elif [ -z "$(docker ps -q -f name=^${NAME}$)" ]; then
        docker start "$NAME" > /dev/null || exit 1
    fi

    # onnxruntime 이 import 안 되면 한 번만 교체 설치
    if ! docker exec "$NAME" python3 -c "import onnxruntime" > /dev/null 2>&1; then
        echo ">> onnxruntime-gpu 를 Jetson 용 1.24.0 으로 교체합니다 (최초 1회)."
        docker exec "$NAME" pip install -q --root-user-action=ignore --no-deps --force-reinstall "$ORT_WHEEL" || exit 1
    fi
}

run_py() {
    docker exec -it -w "$UNIT_DIR" "$NAME" python3 "$@"
}

case "$1" in
    export)
        ensure_container
        run_py export_trt.py && touch "$HOST_DIR/$ENGINE_MARK"
        ;;
    app)
        ensure_container
        run_py app.py
        ;;
    app_trt)
        ensure_container
        if [ ! -f "$HOST_DIR/$ENGINE" ] || [ ! -f "$HOST_DIR/$ENGINE_MARK" ] || \
           [ "$HOST_DIR/$ENGINE" -nt "$HOST_DIR/$ENGINE_MARK" ]; then
            echo "!! $ENGINE 이(가) 컨테이너에서 만든 엔진이 아닙니다. 먼저 ./run_docker.sh export 를 실행하세요."
            exit 1
        fi
        run_py app_trt.py
        ;;
    shell)
        ensure_container
        docker exec -it -w "$UNIT_DIR" "$NAME" bash
        ;;
    rm)
        docker rm -f "$NAME"
        ;;
    *)
        sed -n '4,9p' "$0" | sed 's/^# //'
        exit 1
        ;;
esac
