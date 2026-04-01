# OpenPifPaf estimator

Runs pose estimation using [OpenPifPaf](https://openpifpaf.github.io/) with the
`shufflenetv2k30-wholebody` checkpoint, producing 133-keypoint COCO-WholeBody
output (same format as mmposewholebody).

## Keypoints

133 keypoints total:
- 17 body (COCO)
- 68 face
- 21 left hand
- 21 right hand
- 6 foot

Output shape: `(frames, people, 133, 2)` — x, y coordinates only (no z).
Each keypoint's detection confidence is stored separately in the pose body's
`confidence` field, shape `(frames, people, 133)`.

Note: frames where no person is detected are omitted from the output, so the
frame count may be less than the total number of video frames.

## Installation
Due to its dependency on `torchvision<0.15`, OpenPifPaf is only compatible with `Python<3.11`. 

## Arguments

| Argument | Description |
|---|---|
| `--device cpu\|gpu` | Select inference device (default: auto-detect; CPU is used if no CUDA GPU is available) |

## Notes

- The model checkpoint (`shufflenetv2k30-wholebody`, ~100 MB) is downloaded
  automatically on first run and cached in `~/.cache/torch/hub/checkpoints/`.
- For GPU inference on Linux, install a CUDA-enabled torch before running:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
