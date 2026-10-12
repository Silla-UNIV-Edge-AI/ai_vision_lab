#!/bin/bash
# 컨테이너를 처음 만들 때 run_docker.sh 가 한 번 실행합니다 (컨테이너 안에서 실행 · 인터넷 필요)
# 실패했다면 컨테이너 안에서 다시 실행:  bash /workspace/setup_container.sh
PIP="pip install -q --root-user-action=ignore"

echo ">> 1/3 ONNX Runtime 확인"
# 이미지에 든 onnxruntime 은 numpy 2 와 맞지 않음 → import 가 안 되면 Jetson 용 1.24.0 으로 교체
ORT="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d98/0b934b9a29c1a/onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"
python3 -c "import onnxruntime" 2> /dev/null || $PIP --no-deps --force-reinstall "$ORT" || FAIL=1

echo ">> 2/3 결과 폴더 설정 – runs/ 를 실행한 폴더 아래에 저장"
yolo settings runs_dir=runs > /dev/null

echo ">> 3/3 roboflow · Jupyter Notebook 설치 (3.6절용 · 몇 분 걸립니다)"
# roboflow 를 그냥 설치하면 opencv-python-headless 가 딸려 와 cv2 창이 안 뜨므로 --no-deps
$PIP notebook pillow-avif-plugin python-dotenv requests-toolbelt filetype || FAIL=1
$PIP --no-deps roboflow || FAIL=1

# jupyter notebook 만 입력하면 되도록 설정 → 주소는 http://localhost:8888
mkdir -p /root/.jupyter
cat > /root/.jupyter/jupyter_server_config.py << 'CONFIG'
c.ServerApp.allow_root = True       # 컨테이너 안은 root 계정
c.ServerApp.open_browser = False    # 브라우저는 Jetson 쪽에서 직접 엶
c.IdentityProvider.token = ""       # 주소에 토큰을 붙이지 않음
CONFIG

if [ -z "$FAIL" ]; then
    echo ">> 준비 완료"
else
    echo ">> 설치 실패 – 인터넷 연결을 확인한 뒤, 컨테이너 안에서 다시 실행하세요:"
    echo "   bash /workspace/setup_container.sh"
fi
