#!/bin/bash
# [6.4] 속도 비교 기록표 – 터미널 로그의 "… ms inference" 값을 기록하세요
# ※ ./run_docker.sh 로 들어간 컨테이너 안에서 한 줄씩 실행합니다.
yolo predict model=yolo11n-pose.pt source=0 show=True
yolo export model=yolo11n-pose.pt format=engine half=True
yolo predict model=yolo11n-pose.engine source=0 show=True
# (선택) yolo11s 비교
# yolo export model=yolo11s-pose.pt format=engine half=True
# yolo predict model=yolo11s-pose.engine source=0 show=True
