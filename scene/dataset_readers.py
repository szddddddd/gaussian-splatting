#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def _has_mapfree_metadata(path: Path):
    return (path / "intrinsics.txt").exists() and (path / "poses.txt").exists()

def _has_blendedmvs_scene_layout(path: Path):
    if not path.is_dir():
        return False
    images_dir = path / "blended_images"
    cams_dir = path / "cams"
    return images_dir.is_dir() and cams_dir.is_dir() and any(cams_dir.glob("*_cam.txt"))

def _blendedmvs_frame_sort_key(path: Path):
    stem = path.stem
    if stem.endswith("_cam"):
        stem = stem[:-4]
    if stem.endswith("_masked"):
        stem = stem[:-7]
    return (0, int(stem)) if stem.isdigit() else (1, stem)

def isMapFreeScenePath(path):
    source_path = Path(path)
    return (
        (source_path.name == "seq0" and _has_mapfree_metadata(source_path.parent)) or
        (_has_mapfree_metadata(source_path) and (source_path / "seq0").is_dir())
    )

def isBlendedMVSScenePath(path):
    source_path = Path(path)
    if _has_blendedmvs_scene_layout(source_path):
        return True
    if not source_path.is_dir():
        return False
    return any(_has_blendedmvs_scene_layout(child) for child in source_path.iterdir() if child.is_dir())

def resolveMapFreeScenePath(path):
    source_path = Path(path)
    if source_path.name == "seq0" and _has_mapfree_metadata(source_path.parent):
        return source_path.parent, source_path.name
    if _has_mapfree_metadata(source_path) and (source_path / "seq0").is_dir():
        return source_path, "seq0"
    raise RuntimeError(
        f"Could not resolve MapFree scene root from '{path}'. Expected a scene root with "
        "intrinsics.txt, poses.txt, and seq0/, or the seq0 directory itself."
    )

def resolveBlendedMVSScenePaths(path):
    source_path = Path(path)
    if _has_blendedmvs_scene_layout(source_path):
        return source_path, False
    if not source_path.is_dir():
        raise RuntimeError(
            f"Could not resolve BlendedMVS scene root from '{path}'. "
            "Expected a scene directory or dataset root directory."
        )

    candidate_scenes = sorted(
        [child for child in source_path.iterdir() if _has_blendedmvs_scene_layout(child)],
        key=lambda candidate: candidate.name
    )
    if not candidate_scenes:
        raise RuntimeError(
            f"Could not resolve BlendedMVS scene root from '{path}'. Expected either "
            "a scene containing blended_images/ and cams/, or a dataset root with child scene directories."
        )

    requested_scene_id = os.environ.get("BLENDEDMVS_SCENE_ID")
    if requested_scene_id:
        requested_scene = source_path / requested_scene_id
        if _has_blendedmvs_scene_layout(requested_scene):
            return requested_scene, True
        raise RuntimeError(
            f"BLENDEDMVS_SCENE_ID='{requested_scene_id}' does not resolve to a valid BlendedMVS scene under "
            f"'{source_path}'."
        )

    selected_scene = candidate_scenes[0]
    print(
        f"Warning: '{source_path}' looks like a BlendedMVS dataset root. Automatically selecting scene "
        f"'{selected_scene.name}'. Pass a specific scene path or set BLENDEDMVS_SCENE_ID to avoid ambiguity."
    )
    return selected_scene, True

def _readMapFreeIntrinsics(intrinsics_path, sequence_name):
    intrinsics_by_frame = {}
    prefix = f"{sequence_name}/"
    with open(intrinsics_path, "r") as intrinsics_file:
        for line in intrinsics_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            frame_path = elems[0]
            if not frame_path.startswith(prefix):
                continue
            intrinsics_by_frame[frame_path] = {
                "fx": float(elems[1]),
                "fy": float(elems[2]),
                "cx": float(elems[3]),
                "cy": float(elems[4]),
                "width": int(elems[5]),
                "height": int(elems[6]),
            }
    return intrinsics_by_frame

def _readMapFreePoses(poses_path, sequence_name):
    poses_by_frame = {}
    prefix = f"{sequence_name}/"
    with open(poses_path, "r") as poses_file:
        for line in poses_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            frame_path = elems[0]
            if not frame_path.startswith(prefix):
                continue
            poses_by_frame[frame_path] = {
                "qvec": np.array(tuple(map(float, elems[1:5]))),
                "tvec": np.array(tuple(map(float, elems[5:8]))),
            }
    return poses_by_frame

def _sampleMapFreeFrames(frame_paths, sample_count=80):
    if len(frame_paths) <= sample_count:
        return frame_paths
    sample_indices = np.unique(np.linspace(0, len(frame_paths) - 1, sample_count, dtype=int))
    return [frame_paths[idx] for idx in sample_indices]

def _buildMapFreeCameraInfos(scene_root, sequence_name, eval, llffhold):
    intrinsics_by_frame = _readMapFreeIntrinsics(scene_root / "intrinsics.txt", sequence_name)
    poses_by_frame = _readMapFreePoses(scene_root / "poses.txt", sequence_name)

    aligned_frame_paths = []
    for frame_path in sorted(set(intrinsics_by_frame.keys()) & set(poses_by_frame.keys())):
        if (scene_root / frame_path).is_file():
            aligned_frame_paths.append(frame_path)

    sampled_frame_paths = _sampleMapFreeFrames(aligned_frame_paths)

    cam_infos = []
    for uid, frame_path in enumerate(sampled_frame_paths):
        intr = intrinsics_by_frame[frame_path]
        pose = poses_by_frame[frame_path]
        image_path = scene_root / frame_path

        width = intr["width"]
        height = intr["height"]
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if (actual_width, actual_height) != (width, height):
            print(
                f"Warning: MapFree metadata size mismatch for {frame_path}: "
                f"metadata=({width}, {height}), image=({actual_width}, {actual_height}). "
                "Using actual image size."
            )
            width = actual_width
            height = actual_height

        is_test = eval and llffhold and uid % llffhold == 0
        image_name = Path(frame_path).with_suffix("").as_posix().replace("/", "_")

        cam_infos.append(CameraInfo(
            uid=uid,
            R=np.transpose(qvec2rotmat(pose["qvec"])),
            T=pose["tvec"],
            FovY=focal2fov(intr["fy"], height),
            FovX=focal2fov(intr["fx"], width),
            depth_params=None,
            image_path=str(image_path),
            image_name=image_name,
            depth_path="",
            width=width,
            height=height,
            is_test=is_test,
        ))

    return cam_infos, len(aligned_frame_paths), len(sampled_frame_paths)

def _readBlendedMVSCamera(cam_path):
    cam_path = Path(cam_path)
    try:
        lines = [line.strip() for line in cam_path.read_text().splitlines() if line.strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to read BlendedMVS camera file '{cam_path}': {exc}") from exc

    if len(lines) < 10:
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': expected at least 10 non-empty lines, got {len(lines)}."
        )
    if lines[0].lower() != "extrinsic" or lines[5].lower() != "intrinsic":
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': expected 'extrinsic' and 'intrinsic' sections."
        )

    try:
        extrinsic = np.array([[float(value) for value in lines[idx].split()] for idx in range(1, 5)], dtype=np.float32)
        intrinsic = np.array([[float(value) for value in lines[idx].split()] for idx in range(6, 9)], dtype=np.float32)
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse numeric values from BlendedMVS camera file '{cam_path}': {exc}") from exc

    if extrinsic.shape != (4, 4):
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': extrinsic has shape {extrinsic.shape}, expected (4, 4)."
        )
    if intrinsic.shape != (3, 3):
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': intrinsic has shape {intrinsic.shape}, expected (3, 3)."
        )

    return extrinsic, intrinsic

def _buildBlendedMVSCameraInfos(scene_root, eval, llffhold):
    scene_root = Path(scene_root)
    images_dir = scene_root / "blended_images"
    cams_dir = scene_root / "cams"

    if not images_dir.is_dir():
        raise RuntimeError(f"BlendedMVS scene '{scene_root}' is missing blended_images directory at '{images_dir}'.")
    if not cams_dir.is_dir():
        raise RuntimeError(f"BlendedMVS scene '{scene_root}' is missing cams directory at '{cams_dir}'.")

    image_paths = sorted(
        [
            image_path for image_path in images_dir.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not image_path.stem.endswith("_masked")
        ],
        key=_blendedmvs_frame_sort_key
    )
    if not image_paths:
        raise RuntimeError(
            f"No unmasked BlendedMVS images found in '{images_dir}'. Expected .jpg/.jpeg/.png files without '_masked'."
        )

    cam_paths_by_stem = {}
    for cam_path in sorted(cams_dir.glob("*_cam.txt"), key=_blendedmvs_frame_sort_key):
        stem = cam_path.stem[:-4] if cam_path.stem.endswith("_cam") else cam_path.stem
        cam_paths_by_stem[stem] = cam_path

    matched_image_paths = [image_path for image_path in image_paths if image_path.stem in cam_paths_by_stem]
    if not matched_image_paths:
        raise RuntimeError(
            f"No BlendedMVS image/camera pairs found under '{scene_root}'. "
            f"Checked images in '{images_dir}' against camera files in '{cams_dir}'."
        )

    cam_infos = []
    for uid, image_path in enumerate(matched_image_paths):
        cam_path = cam_paths_by_stem[image_path.stem]
        w2c, intrinsic = _readBlendedMVSCamera(cam_path)

        with Image.open(image_path) as image:
            width, height = image.size

        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])

        cam_infos.append(CameraInfo(
            uid=uid,
            R=np.transpose(w2c[:3, :3]),
            T=w2c[:3, 3],
            FovY=focal2fov(fy, height),
            FovX=focal2fov(fx, width),
            depth_params=None,
            image_path=str(image_path),
            image_name=image_path.stem,
            depth_path="",
            width=width,
            height=height,
            is_test=eval and llffhold and uid % llffhold == 0,
        ))

    return cam_infos, len(image_paths), len(matched_image_paths)

def _loadMapFreePointCloud(scene_root, nerf_normalization):
    sparse_root = scene_root / "sparse" / "0"
    ply_path = sparse_root / "points3D.ply"
    bin_path = sparse_root / "points3D.bin"
    txt_path = sparse_root / "points3D.txt"

    if ply_path.exists() or bin_path.exists() or txt_path.exists():
        if not ply_path.exists():
            print("Converting MapFree sparse point cloud to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        return fetchPly(ply_path), str(ply_path), "colmap_sparse"

    ply_path = scene_root / "mapfree_points3D.ply"
    if not ply_path.exists():
        num_pts = 10_000
        print(f"Generating random MapFree point cloud ({num_pts})...")
        box_scale = 0.1
        radius = max(float(nerf_normalization["radius"]), 1e-3)*box_scale
        center = -np.asarray(nerf_normalization["translate"])
        rng = np.random.default_rng(0)
        xyz = center + (rng.random((num_pts, 3)) * 2.0 - 1.0) * radius
        shs = rng.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return fetchPly(ply_path), str(ply_path), "random"

def _loadBlendedMVSPointCloud(scene_root, nerf_normalization):
    scene_root = Path(scene_root)
    sparse_root = scene_root / "sparse" / "0"
    ply_path = sparse_root / "points3D.ply"
    bin_path = sparse_root / "points3D.bin"
    txt_path = sparse_root / "points3D.txt"

    if ply_path.exists() or bin_path.exists() or txt_path.exists():
        if not ply_path.exists():
            print("Converting BlendedMVS sparse point cloud to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        return fetchPly(ply_path), str(ply_path), "colmap_sparse"

    ply_path = scene_root / "blendedmvs_points3D.ply"
    if not ply_path.exists():
        num_pts = 10_000
        print(f"Generating random BlendedMVS point cloud ({num_pts})...")
        box_scale = 0.1
        radius = max(float(nerf_normalization["radius"]), 1e-3) * box_scale
        center = -np.asarray(nerf_normalization["translate"])
        rng = np.random.default_rng(0)
        xyz = center + (rng.random((num_pts, 3)) * 2.0 - 1.0) * radius
        shs = rng.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return fetchPly(ply_path), str(ply_path), "random"

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def readMapFreeSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    scene_root, sequence_name = resolveMapFreeScenePath(path)
    cam_infos, total_aligned_frames, sampled_frames = _buildMapFreeCameraInfos(scene_root, sequence_name, eval, llffhold)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    if not train_cam_infos:
        raise RuntimeError(
            f"No MapFree training cameras found for '{path}'. "
            f"Aligned {total_aligned_frames} frames and sampled {sampled_frames}. "
            "Check that intrinsics.txt, poses.txt, and seq0 image files align."
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path, point_cloud_source = _loadMapFreePointCloud(scene_root, nerf_normalization)
    if pcd is None:
        raise RuntimeError(f"Failed to load or generate MapFree point cloud at '{ply_path}'.")

    print(
        f"Loaded MapFree scene '{scene_root.name}/{sequence_name}': "
        f"aligned {total_aligned_frames} frames, sampled {sampled_frames}, "
        f"train={len(train_cam_infos)}, test={len(test_cam_infos)}, point_cloud={point_cloud_source}."
    )

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readBlendedMVSSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    scene_root, resolved_from_dataset_root = resolveBlendedMVSScenePaths(path)
    cam_infos, total_images, matched_frames = _buildBlendedMVSCameraInfos(scene_root, eval, llffhold)

    train_cam_infos = [cam_info for cam_info in cam_infos if train_test_exp or not cam_info.is_test]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.is_test]

    if not train_cam_infos:
        raise RuntimeError(
            f"No BlendedMVS training cameras found for '{path}'. "
            f"Found {total_images} unmasked images and {matched_frames} matched camera pairs in '{scene_root}'."
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path, point_cloud_source = _loadBlendedMVSPointCloud(scene_root, nerf_normalization)
    if pcd is None:
        raise RuntimeError(f"Failed to load or generate BlendedMVS point cloud at '{ply_path}'.")

    scene_label = scene_root.name if not resolved_from_dataset_root else f"{scene_root.parent.name}/{scene_root.name}"
    print(
        f"Loaded BlendedMVS scene '{scene_label}': "
        f"images={total_images}, matched={matched_frames}, train={len(train_cam_infos)}, "
        f"test={len(test_cam_infos)}, point_cloud={point_cloud_source}."
    )

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "MapFree": readMapFreeSceneInfo,
    "BlendedMVS": readBlendedMVSSceneInfo
}
