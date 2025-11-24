module load a100
module load cuda/11.8.0
module load anaconda3/2024.02-1
module load mamba/24.9.0-0

conda activate alphapose

CONFIG="/home/gsantm/repositories/AlphaPose/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
CHECKPOINT="/home/gsantm/repositories/AlphaPose/pretrained_models/multi_domain_fast50_regression_256x192.pth"
VIDEO_NAME="/home/gsantm/scripts/pose_estimators/alphapose/test.mp4"

cd /home/gsantm/repositories/AlphaPose

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --video ${VIDEO_NAME} \
    --outdir examples/res --save_video

ffmpeg -i /home/gsantm/repositories/AlphaPose/examples/res/AlphaPose_test.mp4 \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  /home/gsantm/repositories/AlphaPose/examples/res/AlphaPose_test_h264.mp4
