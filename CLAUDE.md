# CLAUDE.md — project conventions and notes

## Project structure

- `install.sh` — top-level installer, dispatches to `estimators/<name>/install_<name>.sh`
- `videos_to_poses.sh` — top-level runner, dispatches to `estimators/<name>/run_<name>.sh`
- `estimators/<name>/` — one directory per pose estimator, containing:
  - `install_<name>.sh` — estimator-specific install script
  - `run_<name>.sh` — estimator-specific run script
  - `README.md` — documents estimator-specific arguments
- `tools/<name>/` — all downloaded/installed artifacts for an estimator go here (gitignored), e.g. `tools/openpose/`. This includes cloned repos, venvs, container images, etc.

## Shell script conventions

All scripts use `#!/usr/bin/env bash` and `set -euo pipefail`.

**Self-locating pattern** — every script computes its own absolute path so it can be called from anywhere:
```bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
```
In estimator scripts the convention is:
```bash
OPENPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$OPENPOSE_DIR")"
TOOLS=$REPO_DIR/tools
```

**Argument parsing** — use a `while [[ $# -gt 0 ]]` loop with `case`/`shift 2` for key-value args and `shift` for flags. Reject unknown arguments with `exit 1`.

**Forwarding a known optional arg as an array** — for optional named args that must be forwarded explicitly (not via PASSTHROUGH), build an array and expand it:
```bash
DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG=(--device "$DEVICE")
fi
some_script "${DEVICE_ARG[@]}"
```
This avoids quoting issues and works correctly when the arg is absent (empty array expands to nothing).

**Passing through unknown args** — `videos_to_poses.sh` consumes only `--type`, `--input`, `--output`, `--device` and collects everything else into a `PASSTHROUGH` array, which is forwarded verbatim to the estimator script. For each unknown `--flag`, the next token is also captured if it does not start with `--` (i.e. it is a value):
```bash
PASSTHROUGH=()
--*)
    PASSTHROUGH+=("$1")
    if [[ $# -gt 1 && "$2" != --* ]]; then
        PASSTHROUGH+=("$2")
        shift 2
    else
        shift
    fi
    ;;
# call with:
some_script "${PASSTHROUGH[@]}"
```
Estimator-specific scripts are responsible for accepting or rejecting any argument they receive.

**Forwarding `--slurm`** — build a `SLURM_ARG` variable and pass it unquoted:
```bash
SLURM_ARG=""
if [ "$USE_SLURM" = true ]; then SLURM_ARG="--slurm"; fi
some_script $SLURM_ARG
```

## Device support

- `videos_to_poses.sh` accepts `--device cpu|gpu` (optional), validates the value, and passes it explicitly to the estimator script via a `DEVICE_ARG` array.
- Each estimator run script accepts `--device` and fails immediately if the requested device is unsupported:
  - GPU-only estimators (openpose, alphapose): fail if `--device cpu`
  - CPU-only estimators (mediapipe): fail if `--device gpu`
  - Estimators that support both: accept any value without failing
- The device check should appear right after argument parsing, before any pre-flight checks. This ensures it fails fast and cleanly even when the estimator is not installed.

## Testing: asserting a command fails

To assert that a script exits non-zero in a bash test (useful for validation tests):
```bash
assert_fails() {
    local description="$1"
    shift
    if bash "$@" 2>/dev/null; then
        echo "FAIL: expected failure but succeeded: $description" >&2
        exit 1
    fi
    echo "OK: $description"
}
```
Validation tests (argument rejection, device/estimator mismatches) require no install or data download, so they can run as a standalone CI job with no dependencies.

## SLURM conventions

- `--slurm` is an optional flag on both `install.sh` and `videos_to_poses.sh`, forwarded to estimator scripts.
- **Install scripts** must always accept `--slurm`, even if it makes no difference for that estimator (i.e. silently ignored). This keeps the top-level interface uniform.
- **Run scripts** must fail with a clear error if `--slurm` is passed but no SLURM submission code exists yet for that estimator. Do not silently ignore it.
- Some upstream scripts use `$SLURM_SUBMIT_DIR` to locate their own files, so they must be called via `sbatch` from their repo root. Use a subshell to avoid affecting the calling script's working directory:
  ```bash
  (cd $SOME_REPO && sbatch scripts/some_script.sh)
  ```
- Scripts that self-locate via `$0` (using `dirname "$0"`) do NOT need a `cd` before calling them.

## Python virtual environments

Prefer the standard library approach over conda or other heavier tools:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
The venv should live inside `tools/<estimator name>/` alongside any other installed artifacts.

## alphapose estimator

- Upstream repo: https://github.com/bricksdont/alphapose-singularity-uzh
- Cloned to: `$TOOLS/alphapose/alphapose-singularity-uzh`
- Install produces four artifacts (all inside the cloned repo):
  - The cloned repo directory itself
  - `alphapose.sif` — Singularity container image (~8 GB, pulled from ghcr)
  - `venv/` — Python virtual environment
  - `data/models/` — model weights
- Container is pulled with `singularity/apptainer pull` (not built); `install_alphapose.sh` detects whichever of `apptainer` or `singularity` is available.
- `slurm_build_container.sh` uses `$SLURM_SUBMIT_DIR` → must be submitted via `(cd repo && sbatch ...)`.
- `setup_venv.sh`, `download_models.sh`, `batch_to_pose.sh`, `slurm_submit.sh` all self-locate → no `cd` needed.
- alphapose-specific arguments: `--chunks N` and `--lowprio` (both only valid with `--slurm`); `--keypoints 136|133` (valid with or without `--slurm`).

## openpose estimator

- Upstream repo: https://github.com/bricksdont/openpose-singularity-uzh
- Cloned to: `$TOOLS/openpose/openpose-singularity-uzh`
- Install produces three artifacts (all inside the cloned repo):
  - The cloned repo directory itself
  - `openpose.sif` — Singularity container image
  - `venv/` — Python virtual environment
- `run_openpose.sh` checks for all three before running and fails with a clear message if any are missing (with a note about SLURM build jobs still in progress for the `.sif` check).
- `slurm_build_container.sh` uses `$SLURM_SUBMIT_DIR` → must be submitted via `(cd repo && sbatch ...)`.
- `build_container.sh`, `setup_venv.sh`, `batch_to_pose.sh`, `slurm_submit.sh` all self-locate → no `cd` needed.
- openpose-specific argument: `--chunks N` (only valid together with `--slurm`; enforced in `run_openpose.sh`).