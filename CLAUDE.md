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

**Forwarding extra args** — use a bash array + `read -ra` to safely split a string of extra arguments so they expand as separate tokens:
```bash
EXTRA=()
--extra) read -ra EXTRA <<< "$2"; shift 2 ;;
# call with:
some_script "${EXTRA[@]}"
```
Do NOT pass extra args as an unquoted string variable (`$ARGS`) — it breaks on spaces. Do NOT double-quote it — it passes as one token.

**Forwarding `--slurm`** — build a `SLURM_ARG` variable and pass it unquoted:
```bash
SLURM_ARG=""
if [ "$USE_SLURM" = true ]; then SLURM_ARG="--slurm"; fi
some_script $SLURM_ARG
```

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