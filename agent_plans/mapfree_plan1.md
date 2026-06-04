你现在在一个 Linux 集群 Docker 环境中工作。目标是在 Gaussian Splatting 官方代码中新增 Map-Free Reloc train 数据格式读取能力，使它可以直接读取 Map-Free Reloc 的一个 scene/sequence 进行 3DGS 训练。

项目路径：

/data/new_disk7/szd/worldmodel_dataset/gaussian-splatting

目标数据路径：

/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0

参考仓库：

https://github.com/nianticlabs/map-free-reloc

主要修改文件：

/data/new_disk7/szd/worldmodel_dataset/gaussian-splatting/scene/dataset_readers.py

你需要先阅读 Map-Free Reloc 仓库 README 中 train 集组织形式，并检查本地真实数据文件。Map-Free train scene 大致结构如下：

train/s00000/
intrinsics.txt
poses.txt
poses_device.txt
overlaps.npz
seq0/
frame_00000.jpg
...
frame_00579.jpg
seq1/
frame_00000.jpg
...
frame_00579.jpg

本任务只需要支持 train scene 的 seq0。seq0 是围绕一个中心物体拍摄的一段视频，大约 580 张图；对 3D Gaussian Splatting 来说太密集，因此只需要从 seq0 中按时间顺序均匀采样 80 张图像用于训练。

实现要求：

1. 在 scene/dataset_readers.py 中新增 Map-Free reader。

2. 支持 source_path 既可以传入 scene 根目录：

/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000

也可以传入 sequence 目录：

/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0

如果传入的是 seq0，则自动把 scene_root 识别为父目录 s00000，并把 sequence_name 识别为 seq0。

3. 读取 scene_root/intrinsics.txt。格式：

frame_path fx fy cx cy frame_width frame_height

其中 frame_path 类似：

seq0/frame_00000.jpg

只保留以 seq0/ 开头的帧。注意不要假设每一帧都一定连续存在，应以 intrinsics.txt 和 poses.txt 中实际存在且图片文件也存在的帧为准。

4. 读取 scene_root/poses.txt。格式：

frame_path qw qx qy qz tx ty tz

Map-Free 官方说明中 pose 是 world-to-camera，即：

x_cam = R(q) @ x_world + t

Gaussian Splatting 当前 dataset_readers.py 里 Colmap reader 的写法是：

R = np.transpose(qvec2rotmat(extr.qvec))
T = np.array(extr.tvec)

因此 Map-Free reader 中也应对 qvec2rotmat(qvec) 做 transpose 后填入 CameraInfo.R，T 直接使用 tvec。

5. 将 intrinsics 和 poses 按 frame_path 对齐，只保留同时具备 intrinsics、pose、image 的帧。

6. 按排序后的 frame_path 从 seq0 中均匀选择 80 张图。如果可用帧不足 80 张，则使用全部可用帧。采样应稳定、确定性，不要随机。推荐使用 np.linspace 在 [0, N-1] 上取 80 个 index，然后去重并保持顺序。

7. 为每张采样图像构造 CameraInfo：

uid: 从 0 开始递增
R: qvec2rotmat(qvec).T
T: tvec
FovX: focal2fov(fx, width)
FovY: focal2fov(fy, height)
image_path: scene_root / frame_path
image_name: Path(frame_path).stem 或包含 seq 信息的稳定名字，避免 seq0/seq1 重名冲突
depth_path: ""
depth_params: None
width, height: intrinsics.txt 中的 frame_width/frame_height；同时可以用 PIL 打开图片校验尺寸，如不一致则优先使用真实图片尺寸并打印 warning
is_test: 默认 False

8. 实现 readMapFreeSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8) 之类函数，保持签名尽量兼容现有 readColmapSceneInfo / readNerfSyntheticInfo，以便 scene/**init**.py 能用同一套 callback 调用。

9. eval 处理：

   * 默认不需要 test set，全部 80 张作为 train。
   * 如果 eval=True，可以从采样后的 80 张中按 llffhold 规则划分 test，例如 idx % llffhold == 0 为 test，其余为 train。
   * 不要因为没有 validation/test 文件而报错。

10. 点云初始化：

* 如果 scene_root 下存在可用 COLMAP sparse 点云，例如 sparse/0/points3D.ply、points3D.bin、points3D.txt，则优先复用现有 Colmap 逻辑加载/转换。
* 如果没有点云，不要失败。参考 Blender reader 的做法生成随机初始点云，写到 scene_root/mapfree_points3D.ply 或 scene_root/points3d_mapfree.ply。
* 随机点云数量可以用 100000。
* 点云空间范围应根据训练相机的 nerf_normalization 来设置，而不是固定 Blender 的 [-1.3, 1.3]。可用相机中心和 radius 生成一个包围相机轨迹附近的随机点云。
* 颜色可以随机初始化，normal 为 0。

11. 计算 nerf_normalization：

* 使用现有 getNerfppNorm(train_cam_infos)。
* 注意 train_cam_infos 不能为空；如果为空，应抛出清晰错误，提示检查 intrinsics.txt、poses.txt、seq0 图片是否对齐。

12. 注册 loader：

* 在 sceneLoadTypeCallbacks 中加入 "MapFree": readMapFreeSceneInfo。
* 检查 scene/**init**.py 或其他数据集类型自动判断逻辑。如果当前代码只识别 Colmap sparse/0 或 Blender transforms_train.json，需要增加 Map-Free 自动识别：

  * 如果 path 本身或 path.parent 中存在 intrinsics.txt 和 poses.txt，并且存在 seq0 或当前目录名是 seq0，则使用 MapFree loader。
* 确保用户运行 train.py 时可以直接用：
  python train.py -s /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0 --model_path output/mapfree_s00000_seq0
  或：
  python train.py -s /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000 --model_path output/mapfree_s00000_seq0

13. 保持修改最小化：

* 优先只改 scene/dataset_readers.py 和必要的 scene/**init**.py。
* 不要移动、复制、重命名 Map-Free 原始图片。
* 不要改训练核心逻辑。
* 不要引入大型新依赖。
* 保持代码风格接近原项目。

14. 测试要求：
    在 conda 环境 gaussian_splatting 中运行。所有测试/运行/安装都必须先执行：

conda activate gaussian_splatting

然后：

cd /data/new_disk7/szd/worldmodel_dataset/gaussian-splatting

至少做以下检查：

A. 纯 reader 测试：
写一个临时 python snippet，import readMapFreeSceneInfo，读取：

/data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0

断言：

* len(scene_info.train_cameras) == 80，除非本地可用帧少于 80
* 每个 CameraInfo.image_path 都存在
* FovX/FovY 为合理正数
* R shape 是 3x3
* T shape 是 3
* point_cloud 不为 None
* ply_path 存在

B. 训练入口 smoke test：
尝试运行很短的训练，例如：

python train.py 
-s /data/new_disk7/szd/worldmodel_dataset/mapfree/data/mapfree/train/s00000/seq0 
--model_path output/mapfree_s00000_seq0_smoke 
--iterations 30

如果训练脚本参数不支持 --iterations 30 或当前环境 GPU/依赖有问题，请不要硬改无关代码；记录具体报错，并至少保证 reader 单元测试通过。

15. 完成后输出：

* 修改了哪些文件
* Map-Free loader 如何识别 scene root / seq0
* 实际读取到多少帧、采样了多少帧
* 是否生成了随机点云或使用了已有 sparse 点云
* 测试命令和结果
* 如果 smoke train 失败，说明是 reader 问题还是环境/训练依赖问题

请先理解现有代码再动手，不要盲改。重点是让 Gaussian Splatting 可以使用 Map-Free train/s00000/seq0 的 image + camera intrinsics/extrinsics 完成训练入口读取。
