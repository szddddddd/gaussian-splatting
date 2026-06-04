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

