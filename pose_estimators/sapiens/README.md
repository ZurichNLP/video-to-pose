# **Summary of Environment Setup Issues**

This report describes the sequence of failures encountered when attempting to build and run the SAPIENS framework on a cluster using conda, CUDA modules, PyTorch, and OpenMMLab dependencies. Multiple issues arose due to incompatibilities between PyTorch, CUDA, mmcv, mmpose, NumPy, and isolated builds triggered during `pip install -e`.

---

## **1. Environment Creation and Initial State**

A fresh conda environment (`python=3.10`) was created using the SAPIENS installation script.
The script:

* Removed any previous environment.

* Installed PyTorch, torchvision, and torchaudio via:

  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  ```

* Installed CUDA-dependent submodules of the SAPIENS repository in editable mode.

* Installed OpenMMLab packages (`mmcv`, `mmdet`, `mmpose`, `mmengine`).

The cluster uses software modules, including:

* `anaconda3/2024.02-1`
* `cuda/12.2.1`
* `a100` GPU constraint module

This introduced potential mixing between conda-installed tools and system-wide modules.

---

## **2. Error: Missing CUDA_HOME**

During the editable installation of `sapiens/cv`, a CUDA extension build failed:

```
OSError: CUDA_HOME environment variable is not set.
```

**Cause:**
The cluster’s CUDA module was not loaded, so environment variables such as `CUDA_HOME` were missing.

**Resolution:**
Loading the appropriate CUDA module:

```
module load cuda/12.2.1
```

---

## **3. PyTorch Installed with the Wrong CUDA Version (Critical Root Cause)**

Even after installing PyTorch using:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

the solver incorrectly resolved **CUDA 11.8 builds**:

```
pytorch-2.1.2-cuda118
```

**Cause:**

* The cluster’s conda resolver pulled PyTorch binaries from **conda-forge**, which only provides CUDA 11.8 builds for PyTorch 2.1.x.
* Even though `pytorch-cuda=12.1` was requested, conda silently replaced it with a CUDA 11.8 variant to satisfy solver constraints.

**Impact:**

* Installed PyTorch ABI = CUDA 11.8
  → But the environment + mmcv wheels expect **CUDA 12.1**.
* This mismatch caused:

  * mmcv `_ext.so` symbol resolution errors
  * failures to build mmcv from source
  * misaligned CUDA kernels in SAPIENS submodules
  * cascade of further incompatibilities

This was one of the **primary causes** of the environment failure.

---

## **4. Error: NumPy Binary Incompatibility**

Running the pose demo produced:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Cause:**
NumPy `2.x` breaks ABI compatibility with C/Cython extensions used by:

* `xtcocotools`
* OpenMMLab packages
* Older compiled ops inside the SAPIENS modules

**Resolution:**
Downgrade NumPy:

```
pip install numpy==1.26.4
```

which restored compatibility with binary extensions.

---

## **5. Error: mmcv C++ Extension Symbol Error**

Importing MMCV operations triggered:

```
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocation...
```

**Cause:**
`mmcv` had been installed using a wheel **compiled for a different PyTorch minor version or CUDA version**.
Different PyTorch minor versions expose different C++ symbols, so precompiled `.so` files fail to load.

This error is documented in MMPose issue #1939.

**Failed Fixes Tried:**

* Installing mmcv via:

  ```
  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/.../torch2.5.0/index.html
  ```

  → Still mismatched.

* Building `mmcv-lite` or `mmcv-full` from source without forcing `MMCV_WITH_OPS=1`.

* Installing mmcv-full 1.7.2
  → Incompatible with mmpose (assertion requires `mmcv>=2.0.0rc4,<2.2.0`).

---

## **6. Error: Conda Installed Torch with Wrong CUDA Version**

Attempting to install PyTorch 2.1.x with CUDA 12.1 via:

```
conda install pytorch==2.1.2 torchvision==0.16.1 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

resulted in:

```
pytorch-2.1.2-cuda118 ...
```

**Cause:**
Conda resolved dependencies using **higher-priority conda-forge packages**, which only provide PyTorch 2.1.2 compiled with **CUDA 11.8**, not CUDA 12.1.

This mismatch caused:

* mmcv wheels (built for cu121) to fail
* Further compilation failures for C++ extensions

---

## **7. Error: Building mmcv from Source Fails**

Trying to run:

```
pip install "mmcv>=2.0.0rc4,<2.2.0" --no-binary
```

produced pip syntax errors, and once fixed, mmcv failed to build due to:

* mismatched CUDA versions
* Python 3.10 ABI issues
* dependency conflicts
* lack of explicit `MMCV_WITH_OPS=1`

---

## **8. Final State**

After all fixes, the environment reached near-functional state:

* SAPIENS modules installed
* CUDA and torch operational
* NumPy reverted to ABI-compatible version (1.26.x), resolving xtcocotools and other C-extension incompatibilities caused by NumPy 2.x

The **remaining blocker** was mmcv/mmpose failing to load due to persistent ABI mismatches between:

* torch version
* mmcv wheel version
* CUDA version
* numpy and Cython extensions

---

## **9. Root Causes (Condensed)**

1. CUDA module not loaded → missing CUDA_HOME
2. pip isolated builds → torch unavailable during compilation
3. NumPy 2.x → ABI breakage for C extensions
4. mmcv installed with the wrong torch/CUDA ABI
5. Conda installing a fallback PyTorch build with unexpected CUDA version
6. Version constraints between mmpose, mmengine, mmdet, and mmcv not respected
7. Mixing pip, conda, anaconda module, and system CUDA libraries