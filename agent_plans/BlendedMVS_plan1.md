你是一个工程级 Codex CLI agent。请在我的 Gaussian Splatting 代码仓库中新增 BlendedMVS 数据集读取能力，用于 3D Gaussian Splatting 训练。

工作目录：

```bash
/data/new_disk7/szd/worldmodel_dataset/gaussian-splatting
```

优先使用环境：

```bash
conda activate gaussian_splatting
```

主要需要修改：

```bash
/data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/scene/dataset_readers.py
/data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/scene/__init__.py
```

数据集位置：

```bash
# 单个 BlendedMVS 场景
/data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb

# 整个 BlendedMVS 根目录，下面有很多 scene id
/data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS
```

目标：让 Gaussian Splatting 能够读取 BlendedMVS 格式数据训练。现在只需要支持原始 BlendedMVS，不需要支持 BlendedMVS+ / BlendedMVS++。

你必须先阅读并理解官方数据格式：

1. 阅读 BlendedMVS 官方 repo：
   https://github.com/yoyo000/blendedmvs

2. 阅读它引用的 MVSNet input format / camera format：
   https://github.com/YoYo000/MVSNet

你需要确认这些事实，并以代码实现为准：

BlendedMVS 每个 scene id 目录大致结构如下：

```text
SCENE_ID/
├── blended_images/
│   ├── 00000000.jpg
│   ├── 00000000_masked.jpg
│   ├── 00000001.jpg
│   ├── 00000001_masked.jpg
│   └── ...
├── cams/
│   ├── pair.txt
│   ├── 00000000_cam.txt
│   ├── 00000001_cam.txt
│   └── ...
└── rendered_depth_maps/
    ├── 00000000.pfm
    ├── 00000001.pfm
    └── ...
```

当前只读取 `blended_images/` 中不带 `_masked` 的 `.jpg/.png/jpeg` 图像，例如：

```text
00000000.jpg
00000001.jpg
```

不要读取：

```text
00000000_masked.jpg
```

相机参数在 `cams/*_cam.txt`。MVSNet 相机文件格式通常是：

```text
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22

DEPTH_MIN DEPTH_INTERVAL (DEPTH_NUM DEPTH_MAX)
```

请判断并验证 BlendedMVS 的 `extrinsic` 是 world-to-camera 矩阵还是 camera-to-world 矩阵。通常 MVSNet 的 extrinsic `E=[R|t]` 是 world-to-camera。Gaussian Splatting 的 `CameraInfo` 中 `R` 的存储方式需要和现有 Colmap reader 保持一致：现有代码中 `R = np.transpose(qvec2rotmat(...))`，`T = tvec`。所以如果 BlendedMVS `extrinsic` 是 w2c，则应使用：

```python
w2c = extrinsic
R = np.transpose(w2c[:3, :3])
T = w2c[:3, 3]
```

不要盲目反转矩阵，除非你通过官方说明或本地数据验证证明必须反转。

实现要求：

1. 在 `scene/dataset_readers.py` 中新增 BlendedMVS detection/helper/reader：

   * `isBlendedMVSScenePath(path)`
   * `resolveBlendedMVSScenePaths(path)`
   * `_readBlendedMVSCamera(cam_path)`
   * `_buildBlendedMVSCameraInfos(scene_root, eval, llffhold)`
   * `_loadBlendedMVSPointCloud(scene_root, nerf_normalization)`
   * `readBlendedMVSSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8)`

2. 支持两种输入路径：

   A. 单个 scene：

   ```bash
   --source_path /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb
   ```

   这时直接读取这个 scene 下的：

   ```text
   blended_images/
   cams/
   ```

   B. 整个 BlendedMVS 根目录：

   ```bash
   --source_path /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS
   ```

   这时根目录下有很多 scene id。Gaussian Splatting 原本一次训练通常对应一个 scene，所以不要试图把全部 scene 混成一个训练 scene。请实现合理行为：

   * 如果 path 本身是 scene root，则读取该 scene。
   * 如果 path 是 BlendedMVS dataset root，则：

     * 优先读取可选参数/环境变量指定的单个 scene id，例如环境变量 `BLENDEDMVS_SCENE_ID`。
     * 如果没有指定，自动选择第一个合法 scene，并打印 warning，提示用户最好传入具体 scene path 或设置 `BLENDEDMVS_SCENE_ID`。
     * 不要一次性混合所有 scene 训练，除非代码原框架明确支持 multi-scene batching。

3. 图像匹配规则：

   对每个不带 `_masked` 的图像，例如：

   ```text
   blended_images/00000012.jpg
   ```

   匹配相机：

   ```text
   cams/00000012_cam.txt
   ```

   只保留 image 和 camera 都存在的 frame。排序时按 stem 数字/字符串稳定排序。

4. 读取 intrinsics：

   从 K 中读取：

   ```python
   fx = K[0, 0]
   fy = K[1, 1]
   cx = K[0, 2]
   cy = K[1, 2]
   ```

   Gaussian Splatting 的 `CameraInfo` 只需要 FovX/FovY，不直接使用 cx/cy。请用实际图像宽高计算：

   ```python
   FovX = focal2fov(fx, width)
   FovY = focal2fov(fy, height)
   ```

   读取图像实际 width/height，不要只相信 cam 文件。

5. `CameraInfo` 字段：

   每个 frame 构造：

   ```python
   CameraInfo(
       uid=uid,
       R=R,
       T=T,
       FovY=FovY,
       FovX=FovX,
       depth_params=None,
       image_path=str(image_path),
       image_name=image_path.stem,
       depth_path="",
       width=width,
       height=height,
       is_test=is_test,
   )
   ```

   如果后续你想利用 `rendered_depth_maps/*.pfm`，可以保留 TODO，但本任务不要强行加入 depth 训练逻辑。GS 训练当前核心只需要 image 和相机内外参。

6. eval/test split：

   仿照现有 Colmap/MapFree 逻辑：

   ```python
   is_test = eval and llffhold and uid % llffhold == 0
   ```

   如果 `eval=False`，全部作为 train。

7. 点云初始化：

   BlendedMVS scene 通常没有 Gaussian Splatting 需要的 COLMAP `sparse/0/points3D.ply`。请实现稳健 fallback：

   * 如果 scene 下存在 `sparse/0/points3D.ply/bin/txt`，优先读取/转换。
   * 否则生成一个 deterministic random point cloud，例如 `blendedmvs_points3D.ply`。
   * 随机点云不要用固定 `[-1.3, 1.3]`，应基于训练相机中心的 `nerf_normalization` 生成，类似当前 MapFree reader。
   * 点数可设为 10_000 或 100_000。为了快速 smoke test，默认 10_000 更稳。
   * 使用固定 rng seed，保证可复现。
   * 使用已有 `storePly` / `fetchPly` / `SH2RGB` / `BasicPointCloud`。

8. 修改 `sceneLoadTypeCallbacks`：

   在 `dataset_readers.py` 底部加入：

   ```python
   "BlendedMVS": readBlendedMVSSceneInfo
   ```

9. 修改 `scene/__init__.py`：

   阅读当前 Scene 初始化逻辑。现在可能是通过判断：

   * `sparse/`
   * `transforms_train.json`
   * MapFree metadata

   来选择 reader。请新增 BlendedMVS 自动识别逻辑。

   识别条件：

   * 单 scene：存在 `blended_images/` 和 `cams/`，并且 `cams` 下有 `*_cam.txt`。
   * dataset root：子目录中至少一个目录满足上述条件。

   当识别为 BlendedMVS 时调用：

   ```python
   sceneLoadTypeCallbacks["BlendedMVS"](...)
   ```

10. 不要破坏已有 Colmap / Blender / MapFree reader 行为。

11. 加入必要的错误信息：

* 找不到 `blended_images`
* 找不到 `cams`
* 找不到任何不带 `_masked` 的图像
* 找不到匹配的 `*_cam.txt`
* cam 文件解析失败
* extrinsic/intrinsic shape 不对

错误信息里打印具体 path，方便定位。

12. 本地验证：

请实际检查以下路径结构：

```bash
find /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb -maxdepth 2 -type f | head -50
ls /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb/blended_images | head
ls /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb/cams | head
sed -n '1,20p' /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb/cams/00000000_cam.txt
```

13. 写一个轻量 smoke test，不要直接长时间训练：

可以使用 Python 直接 import reader 并读取 scene：

```bash
conda activate gaussian_splatting
cd /data/new_disk7/szd/worldmodel_dataset/gaussian-splatting

python - <<'PY'
from scene.dataset_readers import readBlendedMVSSceneInfo
path = "/data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb"
scene = readBlendedMVSSceneInfo(path, images=None, depths="", eval=False, train_test_exp=False)
print("train", len(scene.train_cameras), "test", len(scene.test_cameras))
print("ply", scene.ply_path)
c = scene.train_cameras[0]
print(c.image_name, c.width, c.height, c.FovX, c.FovY)
print("R", c.R)
print("T", c.T)
PY
```

如果项目 import 需要特殊依赖，先激活 `gaussian_splatting` conda 环境再运行。

14. 可选短训练验证：

如果 smoke test 通过，再只做非常短的训练命令，例如几十 iteration，确认不会在 scene loading 阶段崩溃。不要跑完整训练。

示例，具体参数请根据仓库当前 `train.py --help` 调整：

```bash
python train.py \
  -s /data/new_disk7/szd/worldmodel_dataset/BlendedMVS/data/BlendedMVS/5a3ca9cb270f0e3f14d0eddb \
  -m /tmp/gs_blendedmvs_smoke \
  --iterations 30
```

15. 最终输出：

请在回答中给出：

* 修改了哪些文件
* 新增了哪些函数
* BlendedMVS camera extrinsic/intrinsic 是如何转换到 GS `CameraInfo` 的
* 单 scene path 和 dataset root path 的行为
* smoke test 命令和结果
* 如果有失败，贴出失败原因和下一步建议

实现时要尽量小改动、可读、稳健。不要大规模重构已有 reader。不要改训练核心逻辑。
