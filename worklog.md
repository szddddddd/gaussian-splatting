## 2026-06-04 13:15 - Inspect Map-Free integration scope

### Goal
- Understand the existing scene loading flow and the requested Map-Free `seq0` training support before making code changes.

### Files inspected
- `README.md`: confirm repository data loading conventions and training entrypoints.
- `agent_plans/mapfree_plan1.md`: extract the exact implementation and test requirements.
- `scene/dataset_readers.py`: inspect existing Colmap/Blender readers, camera conventions, and point cloud initialization paths.
- `scene/__init__.py`: inspect scene type auto-detection and callback dispatch.
- `scene/colmap_loader.py`: confirm `qvec2rotmat` convention and existing point cloud readers.
- `/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/intrinsics.txt`: inspect real intrinsics file format.
- `/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/poses.txt`: inspect real pose file format.

### Changes made
- `worklog.md`: created the worklog file required by repository instructions.

### Commands run
```bash
git status --short --branch --untracked-files=no
rg --files -g 'README*' -g 'AGENTS.md' -g 'agent_plans/**' | sed -n '1,120p'
find . -maxdepth 2 -type d | sort | sed -n '1,120p'
sed -n '1,220p' README.md
sed -n '1,260p' agent_plans/mapfree_plan1.md
find /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000 -maxdepth 2 | sort | sed -n '1,120p'
sed -n '1,260p' scene/dataset_readers.py
sed -n '1,240p' scene/__init__.py
sed -n '260,520p' scene/dataset_readers.py
sed -n '1,40p' /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/intrinsics.txt
sed -n '1,40p' /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/poses.txt
date -u '+%Y-%m-%d %H:%M'
sed -n '1,260p' scene/colmap_loader.py
rg -n "train_test_exp|source_path|depths" arguments scene utils train.py | sed -n '1,200p'
find /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0 -maxdepth 1 -type f -name 'frame_*.jpg' | wc -l
wc -l /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/intrinsics.txt /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/poses.txt
find /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000 -path '*/sparse/*' | sort | sed -n '1,40p'
```

### Results

* Repository currently has a pre-existing modification in `.gitignore`.
* Existing scene detection only supports Colmap (`sparse`) and Blender (`transforms_train.json`).
* Local Map-Free scene `s00000` contains `intrinsics.txt`, `poses.txt`, `poses_device.txt`, `overlaps.npz`, and `seq0`.
* Local `seq0` contains 580 jpg frames. `intrinsics.txt` and `poses.txt` each contain 1159 lines, so alignment must be performed by `frame_path`.
* No local `sparse/0` point cloud was found under the target Map-Free scene, so a fallback random point cloud path will be needed.

### Current state

* The required implementation scope is clear and tightly bounded to scene loading.
* No code changes beyond creating `worklog.md` have been made yet.

### Next recommended step

* Implement a Map-Free reader in `scene/dataset_readers.py` and add auto-detection in `scene/__init__.py`, then validate with a reader snippet and a short training smoke test.

## 2026-06-04 13:20 - Implement Map-Free seq0 reader and scene auto-detection

### Goal
- Add minimal Map-Free train-scene loading support for `s00000/seq0` without changing training core logic.

### Files inspected
- `scene/dataset_readers.py`: identify the smallest insertion points for a new dataset reader and point cloud fallback.
- `scene/__init__.py`: identify where to extend dataset auto-detection.

### Changes made
- `scene/dataset_readers.py`: added Map-Free path detection, scene-root resolution, `intrinsics.txt`/`poses.txt` parsing, stable 80-frame sampling, camera construction, eval split support, and point cloud fallback generation using `getNerfppNorm(train_cam_infos)`.
- `scene/dataset_readers.py`: registered `sceneLoadTypeCallbacks["MapFree"] = readMapFreeSceneInfo`.
- `scene/__init__.py`: added Map-Free auto-detection ahead of Colmap/Blender dispatch so both `.../s00000` and `.../s00000/seq0` work.

### Commands run
```bash
git diff -- scene/dataset_readers.py scene/__init__.py
git status --short --branch --untracked-files=no
sed -n '1,260p' scene/dataset_readers.py
```

### Results

* Map-Free reader now resolves either a scene root containing `intrinsics.txt`, `poses.txt`, and `seq0/`, or a direct `seq0` path whose parent contains that metadata.
* Camera poses follow the same convention as the Colmap reader: `R = qvec2rotmat(qvec).T`, `T = tvec`.
* Point cloud initialization now reuses `sparse/0/points3D.{ply,bin,txt}` if present, otherwise generates `mapfree_points3D.ply` with 100000 random points centered and scaled by the training-camera normalization radius.

### Current state

* The code changes are in place and scoped to `scene/dataset_readers.py` and `scene/__init__.py`.
* Validation still needs to be run in the project conda environment.

### Next recommended step

* Run a reader-only Python snippet against `s00000/seq0`, then run a 30-iteration smoke training job to confirm end-to-end loading.

## 2026-06-04 13:20 - Validate Map-Free reader and smoke training

### Goal
- Verify that the new reader loads `s00000/seq0` correctly and that `train.py` can consume it for a short run.

### Files inspected
- `scene/dataset_readers.py`: verify line-level implementation after the patch.
- `scene/__init__.py`: verify dataset dispatch order after the patch.

### Changes made
- `worklog.md`: recorded validation commands, environment discrepancy, and observed results.

### Commands run
```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian-splatting && python - <<\"PY\"
from pathlib import Path
from scene.dataset_readers import readMapFreeSceneInfo, isMapFreeScenePath

seq_path = Path("/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0")
scene_root_path = seq_path.parent

assert isMapFreeScenePath(seq_path)
assert isMapFreeScenePath(scene_root_path)

scene_info = readMapFreeSceneInfo(str(seq_path), images=None, depths="", eval=False, train_test_exp=False)
train_cams = scene_info.train_cameras
assert len(train_cams) == 80, len(train_cams)
assert len(scene_info.test_cameras) == 0
assert scene_info.point_cloud is not None
assert Path(scene_info.ply_path).exists(), scene_info.ply_path
for cam in train_cams:
    assert Path(cam.image_path).exists(), cam.image_path
    assert cam.FovX > 0 and cam.FovY > 0
    assert cam.R.shape == (3, 3), cam.R.shape
    assert cam.T.shape == (3,), cam.T.shape
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda info --envs'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<\"PY\"
from pathlib import Path
from scene.dataset_readers import readMapFreeSceneInfo, isMapFreeScenePath

seq_path = Path("/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0")
scene_root_path = seq_path.parent

assert isMapFreeScenePath(seq_path)
assert isMapFreeScenePath(scene_root_path)

scene_info = readMapFreeSceneInfo(str(seq_path), images=None, depths="", eval=False, train_test_exp=False)
train_cams = scene_info.train_cameras
assert len(train_cams) == 80, len(train_cams)
assert len(scene_info.test_cameras) == 0
assert scene_info.point_cloud is not None
assert Path(scene_info.ply_path).exists(), scene_info.ply_path
for cam in train_cams:
    assert Path(cam.image_path).exists(), cam.image_path
    assert cam.FovX > 0 and cam.FovY > 0
    assert cam.R.shape == (3, 3), cam.R.shape
    assert cam.T.shape == (3,), cam.T.shape

print({
    "train_cameras": len(train_cams),
    "test_cameras": len(scene_info.test_cameras),
    "ply_path": scene_info.ply_path,
    "first_image": train_cams[0].image_name,
    "last_image": train_cams[-1].image_name,
    "radius": float(scene_info.nerf_normalization["radius"]),
})
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<\"PY\"
from scene.dataset_readers import readMapFreeSceneInfo

scene_info = readMapFreeSceneInfo(
    "/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000",
    images=None,
    depths="",
    eval=True,
    train_test_exp=False,
)
print({
    "train_cameras": len(scene_info.train_cameras),
    "test_cameras": len(scene_info.test_cameras),
    "first_test": scene_info.test_cameras[0].image_name if scene_info.test_cameras else None,
})
assert len(scene_info.train_cameras) == 70, len(scene_info.train_cameras)
assert len(scene_info.test_cameras) == 10, len(scene_info.test_cameras)
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python train.py -s /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0 --model_path output/mapfree_s00000_seq0_smoke --iterations 30'
git status --short --branch --untracked-files=no
nl -ba scene/dataset_readers.py | sed -n '140,260p'
nl -ba scene/dataset_readers.py | sed -n '450,520p'
nl -ba scene/__init__.py | sed -n '36,56p'
date -u '+%Y-%m-%d %H:%M'
```

### Results

* `conda activate gaussian-splatting` failed because that environment name does not exist locally.
* `conda info --envs` showed the available project environment is `gaussian_splatting`, which matches the repository README naming.
* Reader-only validation passed for the direct `seq0` path:
  * aligned frames: 580
  * sampled frames: 80
  * train cameras: 80
  * test cameras: 0
  * generated point cloud: `/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/mapfree_points3D.ply`
  * first sampled image: `seq0_frame_00000`
  * last sampled image: `seq0_frame_00579`
* Reader validation also passed for the scene-root path with `eval=True`:
  * train cameras: 70
  * test cameras: 10
  * split followed `idx % 8 == 0` on the sampled 80 frames
* Smoke training passed:
  * command reached `Training complete.`
  * the training entrypoint recognized Map-Free automatically
  * the run initialized from the generated random point cloud and saved Gaussians at iteration 30

### Current state

* Map-Free `s00000/seq0` can now be read directly by the Gaussian Splatting training entrypoint.
* Both `-s .../s00000` and `-s .../s00000/seq0` are supported.
* No existing sparse point cloud was present under the target scene, so the current validation used the generated random point cloud.

### Next recommended step

* If you want stronger coverage, run a longer training job on `s00000/seq0` and inspect the produced reconstruction quality; the data-loading path itself is already validated.

## 2026-06-04 13:22 - Record final repository state

### Goal
- Capture the final git working tree state after the Map-Free changes and validation run.

### Files inspected
- `.gitignore`: pre-existing modified file visible in git status.
- `AGENTS.md`: pre-existing untracked file visible in git status.
- `agent_plans/`: pre-existing untracked directory visible in git status.

### Changes made
- `worklog.md`: appended final repository state information.

### Commands run
```bash
git status --short --branch
date -u '+%Y-%m-%d %H:%M'
```

### Results

* Final git status shows:
  * modified: `.gitignore`, `scene/__init__.py`, `scene/dataset_readers.py`
  * untracked: `AGENTS.md`, `agent_plans/`, `worklog.md`
* `AGENTS.md` and `agent_plans/` were already present in the workspace context and were not modified by this task.

### Current state

* The requested Map-Free loader changes are complete and validated.
* The repository still contains unrelated pre-existing working tree state outside this task.

### Next recommended step

* Commit only the intended scene loader changes and `worklog.md` if you want to preserve this integration cleanly.

## 2026-06-05 05:19 - Inspect TartanAir integration scope

### Goal
- Verify the local TartanAir layout, inspect the existing scene readers, and confirm the camera transform convention before patching.

### Files inspected
- `scene/dataset_readers.py`: inspect existing helper patterns, fallback point cloud generation, and callback registration.
- `scene/__init__.py`: inspect automatic dataset detection and dispatch order.
- `utils/graphics_utils.py`: confirm `getWorld2View2` expects `R` stored transposed and `T` in world-to-camera form.
- `worklog.md`: append a new entry without overwriting previous records.
- `/etc/codex/skills/karpathy-guidelines/SKILL.md`: follow the repository's minimal-change coding guidance.

### Changes made
- `worklog.md`: appended the TartanAir inspection record.

### Commands run
```bash
pwd
git status --short
rg --files | sed -n '1,120p'
sed -n '1,220p' AGENTS.md
sed -n '1,220p' /etc/codex/skills/karpathy-guidelines/SKILL.md
sed -n '1,320p' scene/dataset_readers.py
sed -n '320,620p' scene/dataset_readers.py
sed -n '620,780p' scene/dataset_readers.py
sed -n '1,260p' scene/__init__.py
sed -n '1,240p' utils/graphics_utils.py
sed -n '1,200p' worklog.md
find /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000 -maxdepth 2 -type d | sort
find /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_left -maxdepth 1 -type f | sort | sed -n '1,5p'
find /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right -maxdepth 1 -type f | sort | sed -n '1,5p'
find /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_left -maxdepth 1 -type f | wc -l
find /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right -maxdepth 1 -type f | wc -l
wc -l /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/pose_left.txt /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/pose_right.txt
date -u '+%Y-%m-%d %H:%M'
```

### Results

* The target TartanAir trajectory exists locally with `image_left/`, `image_right/`, `pose_left.txt`, and `pose_right.txt`.
* `image_left` and `image_right` each contain 1197 frames, and each pose file contains 1197 lines.
* `getWorld2View2` stores `Rt[:3, :3] = R.transpose()`, so new reader code must pass `R = R_w2c.T` and `T = T_w2c`.

### Current state

* The required implementation scope is constrained to `scene/dataset_readers.py` and `scene/__init__.py`.
* The real dataset layout matches the requested auto-detection pattern.

### Next recommended step

* Add the TartanAir helper functions, reader, and scene dispatch, then validate with `py_compile`, a direct reader smoke test, and a short training run.

## 2026-06-05 05:25 - Implement and validate TartanAir reader

### Goal
- Add direct TartanAir v1 `image_left` / `image_right` scene loading and verify that `train.py -s <image_dir>` works end-to-end.

### Files inspected
- `scene/dataset_readers.py`: add TartanAir path detection, pose conversion, frame sampling, and random point cloud fallback.
- `scene/__init__.py`: add TartanAir auto-detection in the scene dispatch flow.
- `/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_left`: verify left camera image directory exists.
- `/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right`: verify right camera image directory exists.
- `/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/pose_left.txt`: verify left pose file exists and frame count matches.
- `/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/pose_right.txt`: verify right pose file exists and frame count matches.

### Changes made
- `scene/dataset_readers.py`: added `_is_tartanair_image_dir`, `isTartanAirScenePath`, `resolveTartanAirScenePath`, `_get_tartanair_image_paths`, `_sampleTartanAirFrames`, `_quat_xyzw_to_rotmat`, `_buildTartanAirCameraInfos`, `_loadTartanAirPointCloud`, and `readTartanAirSceneInfo`.
- `scene/dataset_readers.py`: registered `sceneLoadTypeCallbacks["TartanAir"] = readTartanAirSceneInfo`.
- `scene/__init__.py`: added automatic TartanAir path detection so `train.py -s .../image_left` and `train.py -s .../image_right` dispatch to the new reader.
- `worklog.md`: appended implementation and validation details.

### Commands run
```bash
sed -n '130,320p' scene/dataset_readers.py
sed -n '640,760p' scene/dataset_readers.py
sed -n '1,80p' scene/__init__.py
git status --short
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python -m py_compile scene/dataset_readers.py scene/__init__.py'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from pathlib import Path
from scene.dataset_readers import resolveTartanAirScenePath, _get_tartanair_image_paths, _sampleTartanAirFrames

for path_str in [
    "/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_left",
    "/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right",
]:
    scene_paths = resolveTartanAirScenePath(path_str)
    image_paths = _get_tartanair_image_paths(scene_paths["image_dir"])
    pose_lines = sum(1 for _ in open(scene_paths["pose_file"], "r"))
    sample_indices = _sampleTartanAirFrames(image_paths)
    print(scene_paths["scene_label"])
    print({
        "images": len(image_paths),
        "pose_lines": pose_lines,
        "sampled": len(sample_indices),
        "sample_first": sample_indices[:5].tolist(),
        "sample_last": sample_indices[-5:].tolist(),
    })
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from scene.dataset_readers import readTartanAirSceneInfo
import numpy as np
p = "/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right"
s = readTartanAirSceneInfo(p, None, "", False, False)
print("train", len(s.train_cameras), "test", len(s.test_cameras))
c = s.train_cameras[0]
print(c.image_path, c.width, c.height, c.FovX, c.FovY)
print(c.R)
print(c.T)
assert len(s.train_cameras) == 100
assert len(s.test_cameras) == 0
assert np.isfinite(c.R).all()
assert np.isfinite(c.T).all()
assert c.FovX > 0 and c.FovY > 0
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && CUDA_VISIBLE_DEVICES=1 python train.py -s /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right -m /data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/output/tartanair_abandonedfactory_Hard_P000_right_test --iterations 1000 -r 4 --port 6010'
git status --short
rg -n "_is_tartanair_image_dir|isTartanAirScenePath|resolveTartanAirScenePath|_get_tartanair_image_paths|_sampleTartanAirFrames|_quat_xyzw_to_rotmat|_buildTartanAirCameraInfos|_loadTartanAirPointCloud|readTartanAirSceneInfo|TartanAir" scene/dataset_readers.py scene/__init__.py
date -u '+%Y-%m-%d %H:%M'
```

### Results

* `py_compile` passed for `scene/dataset_readers.py` and `scene/__init__.py`.
* Local directory checks and helper-based sampling checks passed:
  * `image_left`: 1197 images, 1197 pose lines, sampled 100 frames, first indices `[0, 12, 24, 36, 48]`, last indices `[1147, 1159, 1171, 1183, 1196]`
  * `image_right`: 1197 images, 1197 pose lines, sampled 100 frames, first indices `[0, 12, 24, 36, 48]`, last indices `[1147, 1159, 1171, 1183, 1196]`
* Direct reader smoke test passed for `image_right`:
  * generated random point cloud: `/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/tartanair_points3D_right.ply`
  * loaded 100 train cameras, 0 test cameras
  * first camera image: `/data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right/000000_right.png`
  * first camera size: `640x480`
  * first camera FOV: `FovX=1.5707963267948966`, `FovY=1.2870022175865687`
  * first camera `R` and `T` were finite and contained no NaNs
* Short training run passed:
  * `train.py` automatically recognized the TartanAir image directory layout
  * loader reported `images=1197, sampled=100, train=100, test=0, point_cloud=random`
  * training initialized from 50000 random points
  * run completed successfully with `[ITER 1000] Saving Gaussians` and `Training complete.`

### Current state

* `train.py -s /data/new_disk7/szd/worldmodel_dataset/tartanair/abandonedfactory/Hard/P000/image_right` now works without adding a new CLI loader flag.
* The TartanAir reader only consumes the selected `image_left` or `image_right` folder and its matching pose file, without merging cameras or trajectories.
* The generated fallback point cloud now exists for the tested right-camera trajectory.

### Next recommended step

* If needed, run a longer training job on the specific TartanAir trajectory you care about and inspect the produced reconstruction quality; the data-loading path itself is validated.
## 2026-06-06 12:55 - CUDA COLMAP environment inspection

### Goal
- Inspect the current Docker/Linux/CUDA/NVIDIA/build environment before installing or compiling CUDA-enabled COLMAP.

### Files inspected
- `AGENTS.md`: confirm repository workflow, conda activation rule, git safety, and worklog format.
- `README.md`: confirm repository uses COLMAP as an external SfM/calibration dependency.
- `worklog.md`: confirm existing log content before appending new records.

### Changes made
- `worklog.md`: appended this environment-inspection entry.

### Commands run
```bash
pwd
git status --short
rg --files -g 'README*' -g 'AGENTS.md' -g 'worklog.md'
ls
sed -n '1,220p' AGENTS.md
tail -n 80 worklog.md
sed -n '1,120p' README.md
date -u '+%Y-%m-%d %H:%M'
```

### Results

* Repository root is `/data/new_disk7/szd/worldmodel_dataset/gaussian-splatting`.
* Git status before COLMAP work: `worklog.md` already modified and `run_muyu_3dgs_pipeline.sh` untracked.
* AGENTS instructions require preserving prior worklog entries and activating `gaussian_splatting` before Python/build/test commands.

### Current state

* No COLMAP installation or build action has been taken yet.
* Existing user/local worktree changes are being preserved.

### Next recommended step

* Run the requested Docker/Linux/CUDA/NVIDIA/compiler/COLMAP environment checks, then decide whether GPU access is available.

## 2026-06-06 12:56 - Docker CUDA and existing COLMAP check

### Goal
- Determine whether the container can access NVIDIA GPUs and whether the installed COLMAP has CUDA support.

### Files inspected
- `worklog.md`: append environment findings without deleting previous entries.

### Changes made
- `worklog.md`: appended the Docker/CUDA/COLMAP inspection result.

### Commands run
```bash
bash -lc 'set +e
printf "===== /etc/os-release =====\n"; cat /etc/os-release
printf "===== uname -a =====\n"; uname -a
printf "===== which colmap =====\n"; which colmap || true
printf "===== colmap -h =====\n"; colmap -h || true
printf "===== colmap -v =====\n"; colmap -v || true
printf "===== nvidia-smi =====\n"; nvidia-smi || true
printf "===== nvcc --version =====\n"; nvcc --version || true
printf "===== CUDA_HOME =====\n"; echo ${CUDA_HOME:-}
printf "===== ldconfig cuda libs =====\n"; ldconfig -p | grep cuda || true
printf "===== gcc --version =====\n"; gcc --version
printf "===== g++ --version =====\n"; g++ --version
printf "===== cmake --version =====\n"; cmake --version
printf "===== python --version =====\n"; python --version || true
printf "===== conda info =====\n"; conda info || true
printf "===== /usr/local/cuda* =====\n"; ls -ld /usr/local/cuda* 2>/dev/null || true
printf "===== colmap path candidates =====\n"; ls -l /usr/bin/colmap /usr/local/bin/colmap 2>/dev/null || true
printf "===== dpkg colmap ownership =====\n"; if command -v colmap >/dev/null 2>&1; then dpkg -S $(command -v colmap) || true; fi'
ls -l /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.4
ls -l /usr/local/cuda-12.4/bin/nvcc /usr/local/cuda-12/bin/nvcc /usr/local/cuda/bin/nvcc
apt-cache policy colmap cmake ninja-build libceres-dev cuda-toolkit-12-4 cuda-compiler-12-4 nvidia-cuda-toolkit
dpkg -l colmap git cmake ninja-build build-essential libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libsuitesparse-dev libatlas-base-dev libopenblas-dev 2>/dev/null
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python --version && which python && conda info --envs'
```

### Results

* OS: Ubuntu 22.04.4 LTS on Linux kernel `5.4.0-169-generic`.
* Existing `colmap` is `/usr/bin/colmap`, owned by apt package `colmap`, version `3.7-2`.
* Existing `colmap -h` reports `COLMAP 3.7 ... without CUDA`; `colmap -v` is not a valid command for this old package.
* GPU access works inside the container: `nvidia-smi` reports 4 x NVIDIA RTX A6000 GPUs, driver `525.125.06`, CUDA runtime `12.4`.
* `nvcc` is not on PATH because `/usr/local/cuda` points to `/tmp/tmpl7y_ktqb/`, but `/usr/local/cuda-12.4/bin/nvcc` exists.
* CUDA libraries are registered through `/usr/local/cuda-12/targets/x86_64-linux/lib` and `/usr/local/cuda-12.4`.
* GCC/G++ are `11.4.0`; CMake is `3.22.1`; Ninja is installed.
* `gaussian_splatting` conda env exists and uses Python `3.7.12`.
* Several COLMAP build dependencies are installed, but the dependency set still needs completion/checking.

### Current state

* Docker GPU access is available, so building CUDA-enabled COLMAP is feasible.
* Current apt COLMAP is CPU-only and should be superseded by a source install in `/usr/local/bin`.
* Need to use `/usr/local/cuda-12.4` explicitly for compilation.

### Next recommended step

* Install any missing build dependencies, clone official COLMAP source under `/opt/colmap`, inspect its CMake options, and configure CUDA with architecture `86` for RTX A6000.

## 2026-06-06 12:58 - COLMAP build dependency installation

### Goal
- Install the system packages needed to compile COLMAP with CUDA support on Ubuntu 22.04.

### Files inspected
- `worklog.md`: append dependency installation record.

### Changes made
- System apt packages: installed missing COLMAP build dependencies, including FLANN, FreeImage, OpenBLAS, Qt5/OpenGL, and related transitive libraries.
- `worklog.md`: appended dependency installation result.

### Commands run
```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake ninja-build build-essential libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libsuitesparse-dev libopenblas-dev
ls -ld /opt /opt/colmap /opt/colmap/build 2>/dev/null
git status --short
nproc
/usr/local/cuda-12.4/bin/nvcc --version
```

### Results

* `apt-get update` succeeded using Ubuntu TUNA mirrors and NVIDIA CUDA repo.
* Dependency install succeeded.
* Newly installed key packages include `libflann-dev`, `libfreeimage-dev`, `libopenblas-dev`, `qtbase5-dev`, and `libqt5opengl5-dev`.
* GCC 11 packages were upgraded to Ubuntu 22.04 patch version `11.4.0-1ubuntu1~22.04.3`.
* `/opt/colmap` did not exist before cloning.
* `nproc` reports `64`.
* `/usr/local/cuda-12.4/bin/nvcc --version` reports CUDA compilation tools `12.4`, `V12.4.131`.

### Current state

* Build dependencies are present.
* Existing `/usr/bin/colmap` remains the apt CPU-only package.
* CUDA toolkit is usable when addressed through `/usr/local/cuda-12.4`.

### Next recommended step

* Clone official COLMAP into `/opt/colmap`, inspect current CMake options, configure with CUDA enabled and `CMAKE_CUDA_ARCHITECTURES=86`, then build.

## 2026-06-06 13:00 - COLMAP source clone and initial CMake configure

### Goal
- Clone official COLMAP source and configure a CUDA-enabled build.

### Files inspected
- `/opt/colmap/CMakeLists.txt`: confirm current CMake option names, especially `CUDA_ENABLED`.
- `/opt/colmap/cmake/FindDependencies.cmake`: confirm CUDA detection and dependency behavior.
- `/opt/colmap/doc/install.rst`: confirm current Ubuntu dependency guidance and CUDA architecture guidance.
- `worklog.md`: append source/configure result.

### Changes made
- `/opt/colmap`: cloned official COLMAP repository.
- System apt packages: installed missing `libqt5svg5-dev`.
- `/opt/colmap/build`: initial CMake build directory was created by the failed configure attempt.
- `worklog.md`: appended clone/configure record.

### Commands run
```bash
git clone https://github.com/colmap/colmap.git /opt/colmap
git -C /opt/colmap rev-parse --short HEAD
git -C /opt/colmap describe --tags --always --dirty
rg -n "option\\(|CUDA|CMAKE_CUDA|COLMAP_CUDA|CUDA_ENABLED|CUDAToolkit" CMakeLists.txt cmake src -g 'CMakeLists.txt' -g '*.cmake'
sed -n '1,220p' CMakeLists.txt
rg -n "Install|Ubuntu|CUDA|ninja|cmake" README.md INSTALL.md doc -g '*.md' -g '*.rst'
sed -n '68,150p' doc/install.rst
sed -n '130,320p' cmake/FindDependencies.cmake
sed -n '320,520p' cmake/FindDependencies.cmake
rg -n "FAISS|ONNX|POSELIB|PoseLib|ONNX_ENABLED|FAISS" src cmake CMakeLists.txt
rg -n "find_package\\(|OpenImageIO|FreeImage|GMock|Curl|OpenSSL|Ceres|SQLite|FLANN|LZ4" cmake src CMakeLists.txt
dpkg -l libgmock-dev libqt5svg5-dev libcurl4-openssl-dev libssl-dev libopenimageio-dev openimageio-tools libceres-dev 2>/dev/null
apt-cache policy libgmock-dev libqt5svg5-dev libcurl4-openssl-dev libssl-dev libopenimageio-dev openimageio-tools
DEBIAN_FRONTEND=noninteractive apt-get install -y libqt5svg5-dev
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && export CUDAHOSTCXX=/usr/bin/g++ && cmake -S /opt/colmap -B /opt/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_ENABLED=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 -DONNX_ENABLED=OFF -DTESTS_ENABLED=OFF -DBLA_VENDOR=OpenBLAS'
```

### Results

* Official COLMAP cloned at commit `1ba44581`, described as `3.12.0-662-g1ba44581`; source version is `4.1.0.dev0`.
* Current source uses `CUDA_ENABLED` as the CUDA option; CUDA support is enabled when CUDAToolkit is found.
* RTX A6000 compute capability is Ampere `8.6`, so `CMAKE_CUDA_ARCHITECTURES=86` is appropriate.
* ONNX support is not required for SIFT GPU/SfM, so it is being disabled to avoid unnecessary runtime downloads/build surface.
* Initial CMake configure failed before CUDA detection because conda selected compiler wrappers:
  * `C compiler identification is GNU 10.4.0`
  * `CXX compiler identification is GNU 10.4.0`
  * compiler path was `/root/miniconda3/envs/gaussian_splatting/bin/x86_64-conda-linux-gnu-c++`
  * error: `Could NOT find Boost (missing: graph program_options)`

### Current state

* Source is cloned and dependencies are mostly present.
* Initial configure failure is attributed to conda compiler wrapper/system Boost detection mismatch, not lack of GPU/CUDA.

### Next recommended step

* Reconfigure in an activated conda shell while explicitly setting `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, and CUDA paths to the system CUDA 12.4 toolkit.

## 2026-06-06 13:03 - Successful CUDA COLMAP CMake configure

### Goal
- Resolve CMake configuration issues and generate build files for CUDA-enabled COLMAP.

### Files inspected
- `/opt/colmap/cmake/FindDependencies.cmake`: determine FAISS and dependency behavior.
- `/opt/colmap/src/thirdparty/CMakeLists.txt`: identify fetched FAISS version and CMake requirement.
- `worklog.md`: append configuration result.

### Changes made
- System apt packages: installed `libfaiss-dev` to avoid fetching FAISS v1.14.1, which requires CMake >= 3.24.
- `/opt/colmap/build`: regenerated CMake build files with CUDA enabled.
- `worklog.md`: appended successful configure record.

### Commands run
```bash
rm -rf /opt/colmap/build
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && export CUDAHOSTCXX=/usr/bin/g++ && cmake -S /opt/colmap -B /opt/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCUDA_ENABLED=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DONNX_ENABLED=OFF -DTESTS_ENABLED=OFF -DBLA_VENDOR=OpenBLAS'
sed -n '1,140p' cmake/FindDependencies.cmake
sed -n '50,110p' src/thirdparty/CMakeLists.txt
apt-cache policy cmake libfaiss-dev
rg -n "FAISS|faiss|COLMAP_FAISS|FeatureDescriptorIndex" src/colmap/feature src/colmap -g '*.cc' -g '*.h' -g 'CMakeLists.txt'
DEBIAN_FRONTEND=noninteractive apt-get install -y libfaiss-dev
rm -rf /opt/colmap/build
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && export CUDAHOSTCXX=/usr/bin/g++ && cmake -S /opt/colmap -B /opt/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr -DCUDA_ENABLED=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DONNX_ENABLED=OFF -DTESTS_ENABLED=OFF -DFETCH_FAISS=OFF -DBLA_VENDOR=OpenBLAS'
```

### Results

* Reconfigure with explicit system GCC/G++ got past Boost and detected CUDA, but failed when fetched FAISS v1.14.1 required CMake `3.24.0`; system CMake is `3.22.1`.
* Installed Ubuntu `libfaiss-dev` package version `1.7.2-5`.
* Final CMake configure succeeded with:
  * GCC/G++: `/usr/bin/gcc`, `/usr/bin/g++` version `11.4.0`
  * CUDA compiler: `/usr/local/cuda-12.4/bin/nvcc`
  * `Enabling CUDA support (version: 12.4.131, archs: 86)`
  * `Enabling GPU support (OpenGL: ON, CUDA: ON)`
  * ONNX disabled
  * tests disabled
  * system Qt5, SQLite, Zlib, GMP, and MPFR selected via `CMAKE_PREFIX_PATH=/usr`
* CMake warned `BLA_VENDOR` was unused by current COLMAP source.

### Current state

* `/opt/colmap/build` contains valid Ninja build files for CUDA-enabled COLMAP.
* No install has been performed yet.

### Next recommended step

* Build with Ninja using moderate parallelism, then install to `/usr/local` and verify that `/usr/local/bin/colmap` takes precedence over apt `/usr/bin/colmap`.

## 2026-06-06 13:12 - Switch to COLMAP 3.11.1 for Ubuntu 22.04 compatibility

### Goal
- Resolve build incompatibilities from current COLMAP master and choose a stable source version that can compile with the existing Ubuntu 22.04 toolchain.

### Files inspected
- `/opt/colmap/src/colmap/feature/index.cc`: identify FAISS API incompatibility.
- `/opt/colmap/src/thirdparty/CMakeLists.txt`: identify fetched FAISS and PoseLib behavior.
- `/opt/colmap/cmake/FindDependencies.cmake`: confirm CUDA and PoseLib find behavior.
- `worklog.md`: append build failure and version switch record.

### Changes made
- `/opt/colmap`: checked out official tag `3.11.1`.
- `/opt/colmap/build`: removed and regenerated during configuration attempts.
- System apt packages: installed `libfaiss-dev`, later found unsuitable for current master but harmless.
- `worklog.md`: appended compatibility decision record.

### Commands run
```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && cmake --build /opt/colmap/build --parallel 16'
git -C /opt/colmap tag --sort=-v:refname | head -n 30
git -C /opt/colmap branch --all --verbose --no-abbrev | head -n 40
git -C /opt/colmap show 3.12.0:CMakeLists.txt | sed -n '1,90p'
git -C /opt/colmap show 3.12.0:src/colmap/feature/index.cc | sed -n '120,150p'
git -C /opt/colmap show 3.11.1:CMakeLists.txt | sed -n '1,90p'
git -C /opt/colmap show 3.11.1:src/colmap/feature/index.cc | sed -n '120,150p'
git -C /opt/colmap show 3.12.6:src/thirdparty/CMakeLists.txt | sed -n '50,95p'
git -C /opt/colmap show 3.12.6:src/colmap/feature/index.cc | sed -n '130,145p'
git -C /opt/colmap show 3.11.1:src/thirdparty/CMakeLists.txt | sed -n '1,100p'
git -C /opt/colmap show 3.11.1:cmake/FindDependencies.cmake | sed -n '120,250p'
git -C /opt/colmap show 3.11.1:src/colmap/feature/CMakeLists.txt | sed -n '1,95p'
git -C /opt/colmap status --short
git -C /opt/colmap checkout 3.11.1
rm -rf /opt/colmap/build
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && export CUDAHOSTCXX=/usr/bin/g++ && cmake -S /opt/colmap -B /opt/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr -DCUDA_ENABLED=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DTESTS_ENABLED=OFF'
apt-cache policy poselib libposelib-dev
git -C /opt/colmap show 3.11.1:src/thirdparty/CMakeLists.txt | sed -n '55,75p'
ls -ld /opt/PoseLib /opt/poselib 2>/dev/null
git -C /opt/colmap status --short && git -C /opt/colmap describe --tags --always --dirty
```

### Results

* Building current COLMAP master failed at `src/colmap/feature/index.cc` because it expects `faiss::SearchParametersIVF`; Ubuntu 22.04 `libfaiss-dev` 1.7.2 exposes `faiss::IVFSearchParameters`.
* Current master's bundled FAISS requires CMake >= 3.24, while system CMake is 3.22.1.
* Official tag `3.11.1` avoids FAISS and links FLANN instead, while still supporting `CUDA_ENABLED`.
* `3.11.1` CMake detected:
  * `Enabling CUDA support (version: 12.4.131, archs: 86)`
  * `Enabling GPU support (OpenGL: ON, CUDA: ON)`
* `3.11.1` configure then failed only while FetchContent tried to download PoseLib from `codeload.github.com`; the proxy returned an SSL EOF during the redirected zip download.
* APT has no usable `poselib` package; no existing `/opt/PoseLib` source directory was present.

### Current state

* `/opt/colmap` is on clean detached HEAD at official tag `3.11.1`.
* CUDA detection is working.
* Need to supply PoseLib locally to avoid the failing CMake zip download path.

### Next recommended step

* Clone PoseLib with git, checkout COLMAP 3.11.1's expected commit `8028473d92c9347794a0e3d3541863b5cbb15743`, install it to `/usr/local`, then configure COLMAP with `FETCH_POSELIB=OFF`.

## 2026-06-06 13:15 - Local PoseLib install and final COLMAP configure

### Goal
- Provide PoseLib locally and configure COLMAP 3.11.1 without network FetchContent downloads.

### Files inspected
- `/opt/colmap/src/thirdparty/CMakeLists.txt`: identify PoseLib commit required by COLMAP 3.11.1.
- `worklog.md`: append PoseLib install and final configure record.

### Changes made
- `/opt/PoseLib`: cloned official PoseLib repository and checked out commit `8028473d92c9347794a0e3d3541863b5cbb15743`.
- `/usr/local/lib/libPoseLib.a`, `/usr/local/include/PoseLib`, `/usr/local/lib/cmake/PoseLib`: installed PoseLib.
- `/opt/colmap/build`: regenerated build files for COLMAP 3.11.1 using local PoseLib.
- `worklog.md`: appended PoseLib/configure record.

### Commands run
```bash
git clone https://github.com/PoseLib/PoseLib.git /opt/PoseLib
git -C /opt/PoseLib checkout 8028473d92c9347794a0e3d3541863b5cbb15743
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && cmake -S /opt/PoseLib -B /opt/PoseLib/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH=/usr -DTESTS_ENABLED=OFF -DMARCH_NATIVE=OFF'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && cmake --build /opt/PoseLib/build --parallel 16 && cmake --install /opt/PoseLib/build'
rm -rf /opt/colmap/build
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && export CUDAHOSTCXX=/usr/bin/g++ && cmake -S /opt/colmap -B /opt/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH="/usr/local;/usr" -DCUDA_ENABLED=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DTESTS_ENABLED=OFF -DFETCH_POSELIB=OFF'
```

### Results

* PoseLib build and install succeeded.
* COLMAP 3.11.1 final configure succeeded with:
  * `Found FreeImage`, `Found FLANN`, `Found LZ4`
  * `Found Ceres version: 2.0.0 installed in: /usr`
  * `Enabling CUDA support (version: 12.4.131, archs: 86)`
  * `Enabling GPU support (OpenGL: ON, CUDA: ON)`
  * `FETCH_POSELIB=OFF` using installed PoseLib.
* CMake emitted a developer warning that this older COLMAP/CMake policy ignores `CUDAToolkit_ROOT`; CUDA was still found through the explicit `CMAKE_CUDA_COMPILER` and system CUDA paths.

### Current state

* `/opt/colmap/build` is configured for CUDA-enabled COLMAP 3.11.1.
* Ready to build and install COLMAP.

### Next recommended step

* Run Ninja build, install to `/usr/local`, run `ldconfig`, and verify `/usr/local/bin/colmap` reports CUDA support.

## 2026-06-06 13:23 - Diagnose COLMAP link failure from conda CUDA libraries

### Goal
- Diagnose final link failure after COLMAP objects compiled successfully.

### Files inspected
- `/opt/colmap/build/CMakeCache.txt`: inspect CUDA library paths selected by CMake.
- `worklog.md`: append link failure diagnosis.

### Changes made
- `worklog.md`: appended link failure diagnosis.

### Commands run
```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && cmake --build /opt/colmap/build --parallel 16'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && env | sort | rg "^(CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|LD_LIBRARY_PATH|LIBRARY_PATH|CPATH|CMAKE|CUDA|CUDACXX|CUDAHOSTCXX|PATH|CONDA)="'
ls -l /root/miniconda3/envs/gaussian_splatting/lib/libcudart* /root/miniconda3/envs/gaussian_splatting/lib/libcurand* /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart* /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcurand* 2>/dev/null
rg -n "CUDAToolkit|CUDA_LIBRARIES|CUDA_curand|CUDA::curand|CUDA_TOOLKIT_ROOT|find_package\\(CUDA" /opt/colmap/cmake /opt/colmap/build/CMakeCache.txt
rg -n "CMAKE_CUDA|CUDAToolkit|CUDA_.*LIB|CUDA_TOOLKIT" /opt/colmap/build/CMakeCache.txt
```

### Results

* COLMAP 3.11.1 compiled through CUDA and C++ object generation, but final link failed:
  * undefined references to `cudaGetDeviceProperties_v2`
* Link command used conda CUDA libraries:
  * `/root/miniconda3/envs/gaussian_splatting/lib/libcudart.so`
  * `/root/miniconda3/envs/gaussian_splatting/lib/libcurand.so`
* The active conda environment exports `CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, and `LDFLAGS` that add conda includes/libs and rpaths.
* `gaussian_splatting` conda contains CUDA 11.6 runtime symlinks, while compilation uses system CUDA 12.4:
  * conda `libcudart.so.11.6.55`
  * system `/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so.12.4.127`
* CMake cache selected many `CUDA_*_LIBRARY` values from the conda environment despite `nvcc` coming from `/usr/local/cuda-12.4`.

### Current state

* The build failure is caused by mixed CUDA runtimes from conda and system CUDA, not missing GPU access or missing CUDA support.
* Need to rebuild with conda activated but with conda build/link variables unset and CUDA library cache variables explicitly pointed at `/usr/local/cuda-12.4`.

### Next recommended step

* Remove `/opt/colmap/build`, reconfigure while unsetting `CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`, `LIBRARY_PATH`, `CPATH`, and setting `CUDA_TOOLKIT_ROOT_DIR`, `CUDA_cudart_LIBRARY`, `CUDA_curand_LIBRARY`, etc. to system CUDA 12.4 paths.

## 2026-06-06 13:28 - Install and verify CUDA-enabled COLMAP

### Goal
- Build, install, and verify that `colmap` is globally available and CUDA/GPU-enabled.

### Files inspected
- `/opt/colmap/build/CMakeCache.txt`: verify CUDA runtime library paths and absence of conda compiler/linker flags.
- `worklog.md`: append final installation and verification record.

### Changes made
- `/usr/local/bin/colmap`: installed CUDA-enabled COLMAP 3.11.1.
- `/usr/local/lib/libcolmap_*.a`, `/usr/local/include/colmap`, `/usr/local/share/colmap`: installed COLMAP libraries, headers, and CMake metadata.
- Dynamic linker cache: refreshed with `ldconfig`.
- `worklog.md`: appended final installation and verification result.

### Commands run
```bash
rm -rf /opt/colmap/build
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH && export CC=/usr/bin/gcc && export CXX=/usr/bin/g++ && export CUDACXX=/usr/local/cuda-12.4/bin/nvcc && export CUDAHOSTCXX=/usr/bin/g++ && export CUDA_HOME=/usr/local/cuda-12.4 && export CUDA_PATH=/usr/local/cuda-12.4 && export PATH=/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && cmake -S /opt/colmap -B /opt/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_PREFIX_PATH="/usr/local;/usr" -DCUDA_ENABLED=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDA_cudart_LIBRARY=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so -DCUDA_curand_LIBRARY=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcurand.so -DCUDA_cudart_static_LIBRARY=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart_static.a -DCUDA_curand_static_LIBRARY=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcurand_static.a -DTESTS_ENABLED=OFF -DFETCH_POSELIB=OFF'
rg -n "CUDA_cudart_LIBRARY|CUDA_curand_LIBRARY|CUDA_cudart_static_LIBRARY|CUDA_curand_static_LIBRARY|CMAKE_EXE_LINKER_FLAGS|CMAKE_CXX_FLAGS:|CMAKE_C_FLAGS:" /opt/colmap/build/CMakeCache.txt
rg -n "root/miniconda3|usr/local/cuda-12.4" /opt/colmap/build/CMakeCache.txt
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH && export PATH=/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && cmake --build /opt/colmap/build --parallel 16'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH && export PATH=/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && cmake --install /opt/colmap/build'
ldconfig
bash -lc 'set +e
cd /tmp
printf "===== which colmap =====\n"; which colmap
printf "===== command -v all candidates =====\n"; type -a colmap
printf "===== colmap -h =====\n"; colmap -h
printf "===== colmap -v =====\n"; colmap -v
printf "===== ldd cuda =====\n"; ldd $(which colmap) | grep -i cuda || true
printf "===== ldd curand/cudart =====\n"; ldd $(which colmap) | grep -Ei "cudart|curand" || true
printf "===== feature gpu options =====\n"; colmap feature_extractor -h | grep -i gpu || true
printf "===== matcher gpu options =====\n"; colmap exhaustive_matcher -h | grep -i gpu || true
printf "===== feature use_gpu option exact =====\n"; colmap feature_extractor -h | grep -F -- "--SiftExtraction.use_gpu" || true
printf "===== matching use_gpu option exact =====\n"; colmap exhaustive_matcher -h | grep -F -- "--SiftMatching.use_gpu" || true
printf "===== nvidia-smi summary =====\n"; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true'
bash -lc 'set +e
cd /tmp
printf "===== colmap help version line =====\n"; colmap -h 2>&1 | head -n 3
printf "===== feature use_gpu option exact =====\n"; colmap feature_extractor -h 2>&1 | grep -F -- "--SiftExtraction.use_gpu" || true
printf "===== matching use_gpu option exact =====\n"; colmap exhaustive_matcher -h 2>&1 | grep -F -- "--SiftMatching.use_gpu" || true
printf "===== gpu option grep =====\n"; colmap feature_extractor -h 2>&1 | grep -i gpu || true; colmap exhaustive_matcher -h 2>&1 | grep -i gpu || true'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH && export PATH=/usr/local/bin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin && tmpdir=$(mktemp -d /tmp/colmap_gpu_smoke.XXXXXX) && echo "tmpdir=$tmpdir" && colmap feature_extractor --database_path "$tmpdir/database.db" --image_path /data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/assets --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 0 --SiftExtraction.max_num_features 256'
git status --short
which colmap && colmap -h | head -n 3
bash -lc 'set +e; colmap -v 2>&1'
```

### Results

* Final configure selected system CUDA 12.4 for the critical runtime libraries:
  * `CUDA_cudart_LIBRARY=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so`
  * `CUDA_curand_LIBRARY=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcurand.so`
* Final COLMAP build succeeded.
* Install succeeded and placed executable at `/usr/local/bin/colmap`.
* `which colmap` from `/tmp` returns `/usr/local/bin/colmap`; `type -a colmap` shows it precedes `/usr/bin/colmap`.
* `colmap -h` reports:
  * `COLMAP 3.11.1 -- Structure-from-Motion and Multi-View Stereo`
  * `(Commit 682ea9ac on 2024-12-06 with CUDA)`
* `colmap -v` is not a valid command in COLMAP 3.11.1 and returns `Command '-v' not recognized`; version/CUDA status must be read from `colmap -h`.
* `ldd $(which colmap) | grep -i cuda` shows `libcudart.so.12` resolved from `/usr/local/cuda-12/targets/x86_64-linux/lib/libcudart.so.12`.
* GPU options are present:
  * `--SiftExtraction.use_gpu arg (=1)`
  * `--SiftExtraction.gpu_index arg (=-1)`
  * `--SiftMatching.use_gpu arg (=1)`
  * `--SiftMatching.gpu_index arg (=-1)`
* `nvidia-smi` reports 4 x NVIDIA RTX A6000 GPUs with driver `525.125.06`.
* Runtime smoke test on temporary `/tmp/colmap_gpu_smoke.*` database printed `Creating SIFT GPU feature extractor`, confirming GPU SIFT path starts successfully. Some image errors followed because the repository `assets/` directory contains mixed image dimensions and an SVG while the smoke command used `--ImageReader.single_camera 1`; this does not invalidate CUDA/GPU verification.
* Final repo git status only shows `worklog.md` modified.

### Current state

* CUDA-enabled COLMAP is installed globally and takes precedence over the old apt CPU-only `/usr/bin/colmap`.
* GPU access is available in the container.
* Existing apt `colmap` package remains installed but is shadowed by `/usr/local/bin/colmap`.

### Next recommended step

* Use `/usr/local/bin/colmap` normally from any working directory. For actual datasets, use `--SiftExtraction.use_gpu 1` and `--SiftMatching.use_gpu 1`; optionally set `--SiftExtraction.gpu_index` / `--SiftMatching.gpu_index` to a specific RTX A6000 index.
## 2026-06-07 13:40 - Inspect UnrealStereo4K layout and existing readers

### Goal
- Understand the local UnrealStereo4K sample scene and the current gaussian-splatting scene reader registration before adding a new reader.

### Files inspected
- `scene/dataset_readers.py`: inspected existing `CameraInfo`, `SceneInfo`, TartanAir, MapFree, BlendedMVS, Colmap, and Blender reader patterns.
- `scene/__init__.py`: inspected scene type detection and callback dispatch order.
- `/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000`: inspected local UnrealStereo4K scene directory layout.
- `/etc/codex/skills/karpathy-guidelines/SKILL.md`: used local coding guidance for scoped implementation and verification.

### Changes made
- None yet.

### Commands run
```bash
pwd && ls -lah && git status --short && test -f worklog.md && tail -80 worklog.md || true
ls -lah /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000
find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000 -maxdepth 2 -type f | head -50
sed -n '1,260p' scene/dataset_readers.py
sed -n '1,260p' scene/__init__.py
sed -n '1,220p' /etc/codex/skills/karpathy-guidelines/SKILL.md
find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0 -maxdepth 1 -type f | sed -n '1,10p' && find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image1 -maxdepth 1 -type f | sed -n '1,10p'
find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics0 -maxdepth 1 -type f | sed -n '1,10p' && find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics1 -maxdepth 1 -type f | sed -n '1,10p'
find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000 -maxdepth 2 -iregex '.*\(intrinsics\|camera\|metadata\|readme\|calib\).*' -print
rg -n "def read|PointCloud|sceneLoadTypeCallbacks|train_test_exp|TartanAir|MapFree|BlendedMVS" scene/dataset_readers.py
printf 'Image0 count '; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0 -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.raw' \) | wc -l
printf 'Image1 count '; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image1 -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.raw' \) | wc -l
printf 'Image0 suffixes\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0 -maxdepth 1 -type f | sed 's/.*\.//' | sort | uniq -c
printf 'Image1 suffixes\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image1 -maxdepth 1 -type f | sed 's/.*\.//' | sort | uniq -c
printf 'Image0 natural first/last\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0 -maxdepth 1 -type f -printf '%f\n' | sort -V | sed -n '1,5p;$p'
printf 'Image1 natural first/last\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image1 -maxdepth 1 -type f -printf '%f\n' | sort -V | sed -n '1,5p;$p'
printf 'Extrinsics0 count '; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics0 -maxdepth 1 -type f | wc -l
printf 'Extrinsics1 count '; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics1 -maxdepth 1 -type f | wc -l
printf 'Extrinsics0 suffixes\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics0 -maxdepth 1 -type f | sed 's/.*\.//' | sort | uniq -c
printf 'Extrinsics1 suffixes\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics1 -maxdepth 1 -type f | sed 's/.*\.//' | sort | uniq -c
printf 'Extrinsics0 natural first/last\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics0 -maxdepth 1 -type f -printf '%f\n' | sort -V | sed -n '1,5p;$p'
printf 'Extrinsics1 natural first/last\n'; find /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics1 -maxdepth 1 -type f -printf '%f\n' | sort -V | sed -n '1,5p;$p'
printf 'Extrinsics0/00000.txt\n'; sed -n '1,20p' /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics0/00000.txt
printf 'Extrinsics0/00001.txt\n'; sed -n '1,20p' /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics0/00001.txt
printf 'Extrinsics1/00000.txt\n'; sed -n '1,20p' /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics1/00000.txt
printf 'Extrinsics1/00001.txt\n'; sed -n '1,20p' /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Extrinsics1/00001.txt
file /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0/00000.png /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image1/00000.png
find /data/new_disk7/szd/worldmodel_dataset/unreal4k -maxdepth 3 -iregex '.*\(intrinsics\|camera\|metadata\|readme\|calib\).*' -print | head -100
sed -n '220,980p' scene/dataset_readers.py
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from pathlib import Path
from PIL import Image
for side in ["Image0", "Image1"]:
    p = Path("/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000") / side / "00000.png"
    with Image.open(p) as im:
        print(side, im.size, im.mode)
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from pathlib import Path
root = Path("/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000")
for side in [0,1]:
    imgs = {p.stem for p in (root / f"Image{side}").glob("*.png")}
    exts = {p.stem for p in (root / f"Extrinsics{side}").glob("*.txt")}
    print(f"side {side}", len(imgs), len(exts), "missing_ext", sorted(list(imgs-exts))[:5], "missing_img", sorted(list(exts-imgs))[:5])
PY'
```

### Results

* Repository already had modified `worklog.md` and untracked `scannetpp.log`; no target source files were modified before this task.
* Local scene `/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000` contains `Image0`, `Image1`, `Extrinsics0`, `Extrinsics1`, `Disp0`, and `Disp1`.
* `Image0` and `Image1` each contain 1000 `.png` files named `00000.png` through `00999.png`.
* Sample images are 3840x2160 RGBA PNG.
* `Extrinsics0` and `Extrinsics1` each contain 1000 `.txt` files named `00000.txt` through `00999.txt`.
* Image stems and extrinsic stems match exactly for both sides.
* No separate intrinsics, camera, calibration, README, or metadata file was found under the local scene or nearby dataset root at max depth 3.
* Each local extrinsic txt starts with a 3x3 intrinsics row flattened on line 1, followed by a 3x4 extrinsic row flattened on line 2. The intrinsics are `fx=1920`, `fy=1920`, `cx=1920`, `cy=1080` for 3840x2160 images.
* The paired `Extrinsics0`/`Extrinsics1` files share rotation and differ by `-0.2` in the first translation component, which is consistent with the stored 3x4 matrix being world-to-camera for a stereo baseline in camera x coordinates.

### Current state

* The needed data format and current reader patterns are understood.
* No implementation has been applied yet.

### Next recommended step

* Add a scoped UnrealStereo4K reader in `scene/dataset_readers.py`, register it, and add detection/scene-root error handling in `scene/__init__.py`.

## 2026-06-07 13:51 - Add and verify UnrealStereo4K reader

### Goal
- Add UnrealStereo4K scene loading so training can use a single scene's `Image0` or `Image1` directory directly as `--source_path`.

### Files inspected
- `arguments/__init__.py`: checked default `images` argument after an initial UnrealStereo4K warning appeared during training.
- `scene/dataset_readers.py`: edited and verified UnrealStereo4K reader functions and callback registration.
- `scene/__init__.py`: edited and verified automatic scene type detection.

### Changes made
- `scene/dataset_readers.py`: added UnrealStereo4K path detection, scene resolution, natural image sorting, 100-frame uniform sampling with `np.unique`, robust txt/json/npy extrinsic parsing, embedded/local metadata intrinsics handling with fallback FOV, environment-variable extrinsic convention override, random point cloud generation under scene root, scene logging, and `"UnrealStereo4K"` callback registration.
- `scene/__init__.py`: added UnrealStereo4K `Image0`/`Image1` detection without changing existing Colmap/Blender/TartanAir/MapFree/BlendedMVS order, plus a clear error when a scene root is passed instead of an image-side directory.
- `/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/unreal4k_points3D_Image0.ply`: generated 50,000-point random initialization point cloud.
- `/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/unreal4k_points3D_Image1.ply`: generated 50,000-point random initialization point cloud.

### Commands run
```bash
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python -m py_compile scene/dataset_readers.py scene/__init__.py'
git diff -- scene/dataset_readers.py scene/__init__.py
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from scene.dataset_readers import readUnrealStereo4KSceneInfo
for side in ["Image0", "Image1"]:
    path = f"/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/{side}"
    info = readUnrealStereo4KSceneInfo(path, None, "", False, False)
    cam = info.train_cameras[0]
    print(side, "train", len(info.train_cameras), "test", len(info.test_cameras), "ply", info.ply_path)
    print(side, "first", cam.image_name, cam.width, cam.height, round(cam.FovX, 6), round(cam.FovY, 6), cam.is_test)
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from scene.dataset_readers import readUnrealStereo4KSceneInfo
path = "/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0"
info = readUnrealStereo4KSceneInfo(path, None, "", True, False, llffhold=8)
print("eval split", len(info.train_cameras), len(info.test_cameras), [c.uid for c in info.test_cameras[:5]])
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from scene.dataset_readers import resolveUnrealStereo4KScenePath
try:
    resolveUnrealStereo4KScenePath("/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000")
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
PY'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && CUDA_VISIBLE_DEVICES=1 python train.py -s /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0 -m /data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/output/unreal4k_00000_Image0_test --iterations 100 -r 4'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && CUDA_VISIBLE_DEVICES=1 python train.py -s /data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image1 -m /data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/output/unreal4k_00000_Image1_test --iterations 100 -r 4'
rg -n "images" arguments train.py
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python -m py_compile scene/dataset_readers.py scene/__init__.py'
bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate gaussian_splatting && python - <<"PY"
from scene.dataset_readers import readUnrealStereo4KSceneInfo
info = readUnrealStereo4KSceneInfo("/data/new_disk7/szd/worldmodel_dataset/unreal4k/00000/Image0", "images", "", False, False)
print("smoke", len(info.train_cameras), len(info.test_cameras), info.train_cameras[0].image_name, info.ply_path)
PY'
git diff -- scene/dataset_readers.py scene/__init__.py
git status --short
```

### Results

* `python -m py_compile scene/dataset_readers.py scene/__init__.py` passed.
* Reader smoke test passed for both sides:
  * `Image0`: 100 train cameras, 0 test cameras, first camera `00000_Image0_00000`, resolution 3840x2160, `FovX=1.570796`, `FovY=1.024779`.
  * `Image1`: 100 train cameras, 0 test cameras, first camera `00000_Image1_00000`, resolution 3840x2160, `FovX=1.570796`, `FovY=1.024779`.
* `eval=True,llffhold=8` smoke test produced 87 train cameras and 13 test cameras, with first test uids `[0, 8, 16, 24, 32]`.
* Scene-root error test raised a clear `RuntimeError` telling the user to pass `.../Image0` or `.../Image1`.
* Left short training completed successfully:
  * Command used `CUDA_VISIBLE_DEVICES=1`, `--iterations 100`, `-r 4`.
  * Output path: `output/unreal4k_00000_Image0_test`.
  * Loaded 100 sampled train cameras, 0 test cameras, `extrinsic_convention=w2c`, `intrinsics_source=extrinsics_txt`, `point_cloud=existing`.
  * Saved Gaussians at iteration 100 and printed `Training complete.`
* Right short training completed successfully:
  * Command used `CUDA_VISIBLE_DEVICES=1`, `--iterations 100`, `-r 4`.
  * Output path: `output/unreal4k_00000_Image1_test`.
  * Loaded 100 sampled train cameras, 0 test cameras, `extrinsic_convention=w2c`, `intrinsics_source=extrinsics_txt`, `point_cloud=existing`.
  * Saved Gaussians at iteration 100 and printed `Training complete.`
* Local txt camera files were judged to contain a 3x3 intrinsic matrix followed by a 3x4 world-to-camera extrinsic matrix. The `Image0`/`Image1` pair has identical rotation and a `-0.2` shift in the first translation component, matching a stereo baseline in camera x for `w2c`.
* No fallback intrinsics were needed for the local sample because the first line of each extrinsics txt provides `fx=1920`, `fy=1920`, `cx=1920`, `cy=1080`.
* Final `git status --short` showed modified `scene/__init__.py`, `scene/dataset_readers.py`, and `worklog.md`.

### Current state

* UnrealStereo4K `Image0` and `Image1` source paths are recognized and train for a 100-iteration smoke run.
* The reader samples 100 frames by default from the 1000-frame input directory.
* The reader generates/reuses side-specific random point clouds at the scene root.
* Existing Colmap, Blender, TartanAir, MapFree, and BlendedMVS detection paths were left in place.

### Next recommended step

* Run a longer training job on the desired UnrealStereo4K side after reviewing whether the `w2c` convention should remain the default for all local scenes; override with `UNREAL4K_EXTRINSIC_CONVENTION=c2w` only if a different source layout is confirmed.
