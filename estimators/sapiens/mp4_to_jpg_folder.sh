#!/usr/bin/env bash

ffmpeg -i /home/gsantm/scripts/pose_estimators/sapiens/test.mp4 -qscale:v 2 /home/gsantm/scripts/pose_estimators/sapiens/sapiens_frames/frame_%06d.jpg