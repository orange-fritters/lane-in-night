#!/bin/bash

# 1. 일단 100개 폴더에 저장하는 코드
## 바꿔줄 거는 뒤에 "/path/to/model/pth"
## 지금은 얘로 돼있음 '/home/mindong/lane-in-night/result/exp_3/006_0.550.pth'
## 결과는 대충 report/result_night_fine_tuned/ 에 저장됨

/home/mindong/lane-in-night/research/bin/python /home/mindong/lane-in-night/visualize_night.py \
  --model_dir "/path/to/model"

# 2. 100개 각자 3 by 4 image로 만들어주는 코드
## 바꿔줄 거 없음
## 결과는 report/figures_night_fine_tuned/ 에 저장됨

/home/mindong/lane-in-night/research/bin/python /home/mindong/lane-in-night/make_figure.py


# 3. 비디오 csv 만들어주는 거
## 바꿔줄 거 있음
## --root 뒤에는 폴더 경로 --first 뒤에는 첫 번째 이미지 번호 --last 뒤에는 마지막 이미지 번호 --version 뒤에는 버전 번호
## version 번호 기억해둬야 함 ㅇㅋ?
## 바로 밑에 쓸 거
/home/mindong/lane-in-night/research/bin/python /home/mindong/lane-in-night/data_processor_video.py \
  --root 'data/lane_detected/Training/Raw/c_1280_720_night_train_1' \
  --first 9936538 \
  --last 9936598 \
  --version 2

# 4. video 만들어주는 코드
## 바꿔줄 거 많음
## 대충 밑에 파일들 눈치껏 만져보셈

/home/mindong/lane-in-night/research/bin/python /home/mindong/lane-in-night/video_maker.py \
  --root 'data/lane_detected/Training/Raw/c_1280_720_night_train_1' \
  --csv 'image_paths_vid.csv' \
  --model_pth 'result/exp_3/006_0.550.pth' \
  --video_name 'output_video.mp4' \
  --version 2 \
  --img_height 1280 \
  --img_width 720


# 5. figures zip 해주는 코드

zip -r figures_night_fine_tuned.zip report/figures_night_fine_tuned/




