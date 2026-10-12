"""
[3.6.2 ②] Thumbs-up / Thumbs-down 데이터셋 내려받기 (Roboflow Universe, CC BY 4.0)
교안: 3단원 3.6.2

▶ 준비 : pip install roboflow, 환경 변수 ROBOFLOW_API_KEY 설정 (교안 2.1.3 · 2.2 STEP 2-1)
         Linux/Jetson: export ROBOFLOW_API_KEY=내키   /  Windows: set ROBOFLOW_API_KEY=내키
▶ 실행 : python download_dataset.py
▶ 결과 : dataset/ 폴더 (data.yaml, train/valid/test)
"""
import os
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("ROBOFLOW_API_KEY 가 없습니다. 먼저 export ROBOFLOW_API_KEY=내키 (Windows: set ROBOFLOW_API_KEY=내키)")

rf = Roboflow(api_key=api_key)
project = rf.workspace("mendeteksi-thumbs-up-thumbs-down").project("thumbs-up-thumbs-down-zobmd")
dataset = project.version(3).download("yolov11", location="dataset")
print("저장 위치:", dataset.location)
print(open(os.path.join("dataset", "data.yaml"), encoding="utf-8").read())   # nc: 2, names 순서 확인
