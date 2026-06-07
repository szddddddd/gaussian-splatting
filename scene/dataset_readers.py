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
import re
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

def _is_blendedmvs_image_file(path: Path):
    return (
        path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not path.stem.endswith("_masked")
    )

def _get_blendedmvs_image_paths(images_dir: Path):
    return sorted(
        [image_path for image_path in images_dir.iterdir() if _is_blendedmvs_image_file(image_path)],
        key=_blendedmvs_frame_sort_key
    )

def _has_blendedmvs_scene_layout(path: Path):
    if not path.is_dir():
        return False
    images_dir = path / "blended_images"
    cams_dir = path / "cams"
    return (
        images_dir.is_dir()
        and cams_dir.is_dir()
        and any(cams_dir.glob("*_cam.txt"))
        and any(_is_blendedmvs_image_file(image_path) for image_path in images_dir.iterdir())
    )

def _blendedmvs_frame_sort_key(path: Path):
    stem = path.stem
    if stem.endswith("_cam"):
        stem = stem[:-4]
    if stem.endswith("_masked"):
        stem = stem[:-7]
    return (0, int(stem)) if stem.isdigit() else (1, stem)

def _is_tartanair_image_dir(path: Path):
    if not path.is_dir() or path.name not in {"image_left", "image_right"}:
        return False

    camera = "left" if path.name == "image_left" else "right"
    pose_file = path.parent / f"pose_{camera}.txt"
    if not pose_file.is_file():
        return False

    image_suffixes = {".png", ".jpg", ".jpeg"}
    return any(image_path.is_file() and image_path.suffix.lower() in image_suffixes for image_path in path.iterdir())

def _natural_frame_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.stem)
    natural_parts = [int(part) if part.isdigit() else part for part in parts]
    return natural_parts + [path.suffix.lower(), path.name]

def _is_unreal4k_image_dir(path: Path):
    if not path.is_dir() or path.name not in {"Image0", "Image1"}:
        return False

    camera_index = path.name[-1]
    extrinsic_dir = path.parent / f"Extrinsics{camera_index}"
    if not extrinsic_dir.is_dir():
        return False

    image_suffixes = {".png", ".jpg", ".jpeg"}
    return any(image_path.is_file() and image_path.suffix.lower() in image_suffixes for image_path in path.iterdir())

def isUnrealStereo4KScenePath(path):
    return _is_unreal4k_image_dir(Path(path))

def isUnrealStereo4KSceneRootPath(path):
    source_path = Path(path)
    return (
        source_path.is_dir()
        and (source_path / "Image0").is_dir()
        and (source_path / "Image1").is_dir()
        and (source_path / "Extrinsics0").is_dir()
        and (source_path / "Extrinsics1").is_dir()
    )

def resolveUnrealStereo4KScenePath(path):
    image_dir = Path(path)
    if isUnrealStereo4KSceneRootPath(image_dir):
        raise RuntimeError(
            f"UnrealStereo4K source_path '{path}' is a scene root. Pass a single stereo image directory instead, "
            f"for example '{image_dir / 'Image0'}' or '{image_dir / 'Image1'}'."
        )
    if image_dir.name not in {"Image0", "Image1"}:
        raise RuntimeError(
            f"Could not resolve UnrealStereo4K scene from '{path}'. Expected a single scene image directory named "
            "'Image0' or 'Image1', not the dataset root or another folder."
        )
    if not image_dir.is_dir():
        raise RuntimeError(f"UnrealStereo4K image directory does not exist: '{image_dir}'.")

    camera_index = image_dir.name[-1]
    scene_root = image_dir.parent
    extrinsic_dir = scene_root / f"Extrinsics{camera_index}"
    if not extrinsic_dir.is_dir():
        raise RuntimeError(
            f"UnrealStereo4K '{image_dir.name}' expects matching extrinsics at '{extrinsic_dir}', but that directory "
            "was not found."
        )

    return {
        "scene_root": scene_root,
        "image_dir": image_dir,
        "extrinsic_dir": extrinsic_dir,
        "camera_name": image_dir.name,
        "scene_label": f"{scene_root.name}/{image_dir.name}",
    }

def _get_unreal4k_image_paths(image_dir: Path):
    image_suffixes = {".png", ".jpg", ".jpeg"}
    raw_paths = [image_path for image_path in image_dir.iterdir() if image_path.is_file() and image_path.suffix.lower() == ".raw"]
    image_paths = sorted(
        [image_path for image_path in image_dir.iterdir() if image_path.is_file() and image_path.suffix.lower() in image_suffixes],
        key=_natural_frame_sort_key
    )
    if raw_paths and not image_paths:
        raise RuntimeError(
            f"UnrealStereo4K image directory '{image_dir}' contains .raw files, but this reader currently supports "
            "only .png, .jpg, and .jpeg."
        )
    if raw_paths:
        print(
            f"Warning: UnrealStereo4K image directory '{image_dir}' contains {len(raw_paths)} .raw files; "
            "ignoring them and using .png/.jpg/.jpeg files only."
        )
    return image_paths

def _sampleUnreal4KFrames(image_paths, sample_count=100):
    if len(image_paths) <= sample_count:
        return np.arange(len(image_paths), dtype=int)
    return np.unique(np.linspace(0, len(image_paths) - 1, sample_count, dtype=int))

def _matrix4x4_from_values(values, source_path):
    values = np.asarray(values, dtype=np.float32)
    if values.shape == (4, 4):
        return values
    if values.shape == (3, 4):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :4] = values
        return matrix
    flat = values.reshape(-1)
    if flat.size == 16:
        return flat.reshape(4, 4)
    if flat.size == 12:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :4] = flat.reshape(3, 4)
        return matrix
    raise RuntimeError(
        f"Failed to parse UnrealStereo4K extrinsic '{source_path}': expected a 4x4 matrix, 3x4 matrix, "
        f"or flat 16/12 values, got shape {values.shape} with {flat.size} values."
    )

def _intrinsic3x3_from_values(values, source_path):
    values = np.asarray(values, dtype=np.float32)
    if values.shape == (3, 3):
        return values
    flat = values.reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3)
    if flat.size >= 4:
        fx, fy, cx, cy = flat[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    raise RuntimeError(
        f"Failed to parse UnrealStereo4K intrinsics '{source_path}': expected a 3x3 matrix or fx fy cx cy values."
    )

def _readUnreal4KExtrinsic(extrinsic_path: Path):
    extrinsic_path = Path(extrinsic_path)
    suffix = extrinsic_path.suffix.lower()

    if suffix == ".npy":
        try:
            loaded = np.load(extrinsic_path, allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to read UnrealStereo4K npy extrinsic '{extrinsic_path}': {exc}") from exc
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        if isinstance(loaded, dict):
            extrinsic_values = loaded.get("extrinsic", loaded.get("extrinsics", loaded.get("matrix", loaded.get("pose"))))
            intrinsic_values = loaded.get("intrinsic", loaded.get("intrinsics", loaded.get("K", loaded.get("camera_matrix"))))
            if extrinsic_values is None:
                raise RuntimeError(f"UnrealStereo4K npy extrinsic '{extrinsic_path}' does not contain an extrinsic matrix key.")
            intrinsic = _intrinsic3x3_from_values(intrinsic_values, extrinsic_path) if intrinsic_values is not None else None
            return _matrix4x4_from_values(extrinsic_values, extrinsic_path), intrinsic
        return _matrix4x4_from_values(loaded, extrinsic_path), None

    if suffix == ".json":
        try:
            data = json.loads(extrinsic_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Failed to read UnrealStereo4K json extrinsic '{extrinsic_path}': {exc}") from exc
        extrinsic_values = data.get("extrinsic", data.get("extrinsics", data.get("matrix", data.get("pose", data.get("transform_matrix")))))
        intrinsic_values = data.get("intrinsic", data.get("intrinsics", data.get("K", data.get("camera_matrix"))))
        if extrinsic_values is None:
            raise RuntimeError(f"UnrealStereo4K json extrinsic '{extrinsic_path}' does not contain an extrinsic matrix key.")
        intrinsic = _intrinsic3x3_from_values(intrinsic_values, extrinsic_path) if intrinsic_values is not None else None
        return _matrix4x4_from_values(extrinsic_values, extrinsic_path), intrinsic

    try:
        rows = []
        for line in extrinsic_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(value) for value in line.replace(",", " ").split()])
    except Exception as exc:
        raise RuntimeError(f"Failed to read UnrealStereo4K text extrinsic '{extrinsic_path}': {exc}") from exc

    if not rows:
        raise RuntimeError(f"UnrealStereo4K extrinsic file '{extrinsic_path}' is empty.")

    intrinsic = None
    if len(rows) >= 2 and len(rows[0]) == 9 and len(rows[1]) in {12, 16}:
        intrinsic = _intrinsic3x3_from_values(rows[0], extrinsic_path)
        return _matrix4x4_from_values(rows[1], extrinsic_path), intrinsic

    row_lengths = {len(row) for row in rows}
    if len(row_lengths) == 1:
        matrix = np.array(rows, dtype=np.float32)
        return _matrix4x4_from_values(matrix, extrinsic_path), intrinsic

    flat = [value for row in rows for value in row]
    if len(flat) in {12, 16}:
        return _matrix4x4_from_values(flat, extrinsic_path), intrinsic
    if len(flat) in {21, 25}:
        intrinsic = _intrinsic3x3_from_values(flat[:9], extrinsic_path)
        return _matrix4x4_from_values(flat[9:], extrinsic_path), intrinsic

    raise RuntimeError(
        f"Failed to parse UnrealStereo4K extrinsic '{extrinsic_path}': unsupported row lengths "
        f"{[len(row) for row in rows]}."
    )

def _read_unreal4k_intrinsics_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        values = data.get("intrinsic", data.get("intrinsics", data.get("K", data.get("camera_matrix"))))
        if values is None and all(key in data for key in ("fx", "fy", "cx", "cy")):
            values = [data["fx"], data["fy"], data["cx"], data["cy"]]
        if values is None:
            raise RuntimeError(f"UnrealStereo4K intrinsics json '{path}' does not contain K or fx/fy/cx/cy.")
        return _intrinsic3x3_from_values(values, path)
    if suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        if isinstance(loaded, dict):
            values = loaded.get("intrinsic", loaded.get("intrinsics", loaded.get("K", loaded.get("camera_matrix"))))
            if values is None and all(key in loaded for key in ("fx", "fy", "cx", "cy")):
                values = [loaded["fx"], loaded["fy"], loaded["cx"], loaded["cy"]]
            if values is None:
                raise RuntimeError(f"UnrealStereo4K intrinsics npy '{path}' does not contain K or fx/fy/cx/cy.")
            return _intrinsic3x3_from_values(values, path)
        return _intrinsic3x3_from_values(loaded, path)

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.extend(float(value) for value in line.replace(",", " ").split())
    return _intrinsic3x3_from_values(rows, path)

def _intrinsics_result_from_matrix(intrinsic, width, height, source):
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if abs(cx - width * 0.5) > 1.0 or abs(cy - height * 0.5) > 1.0:
        print(
            f"Warning: UnrealStereo4K intrinsics source '{source}' has principal point ({cx:.3f}, {cy:.3f}), "
            f"but CameraInfo stores only FOV. Training will use FOV and ignore principal point offset."
        )
    return {
        "FovX": focal2fov(fx, width),
        "FovY": focal2fov(fy, height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "matrix": intrinsic,
        "source": source,
    }

def _read_or_infer_unreal4k_intrinsics(scene_root, image_dir, width, height, embedded_intrinsic=None):
    scene_root = Path(scene_root)
    image_dir = Path(image_dir)
    camera_index = image_dir.name[-1] if image_dir.name in {"Image0", "Image1"} else ""
    candidates = [
        scene_root / f"Intrinsics{camera_index}.txt",
        scene_root / f"Intrinsics{camera_index}.json",
        scene_root / f"Intrinsics{camera_index}.npy",
        scene_root / "intrinsics.txt",
        scene_root / "intrinsics.json",
        scene_root / "camera.txt",
        scene_root / "camera.json",
        scene_root / "cameras.json",
        scene_root / "calib.txt",
        scene_root / "metadata.json",
    ]
    intrinsics_dir = scene_root / f"Intrinsics{camera_index}"
    if intrinsics_dir.is_dir():
        candidates.extend(sorted(
            [path for path in intrinsics_dir.iterdir() if path.is_file() and path.suffix.lower() in {".txt", ".json", ".npy"}],
            key=_natural_frame_sort_key
        ))

    for candidate in candidates:
        if candidate.is_file():
            intrinsic = _read_unreal4k_intrinsics_file(candidate)
            return _intrinsics_result_from_matrix(intrinsic, width, height, f"metadata:{candidate.name}")

    if embedded_intrinsic is not None:
        return _intrinsics_result_from_matrix(embedded_intrinsic, width, height, "extrinsics_txt")

    fovx_deg = float(os.environ.get("UNREAL4K_FOVX_DEG", "90.0"))
    fovx = np.deg2rad(fovx_deg)
    fx = fov2focal(fovx, width)
    fy = fx
    fovy = focal2fov(fy, height)
    print(
        f"Warning: no UnrealStereo4K intrinsics metadata found under '{scene_root}' for {image_dir.name}. "
        f"Using fallback UNREAL4K_FOVX_DEG={fovx_deg:.3f}; principal point is assumed at image center because "
        "CameraInfo does not store cx/cy."
    )
    return {
        "FovX": fovx,
        "FovY": fovy,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
        "matrix": np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        "source": "fallback",
    }

def _unreal4k_extrinsic_convention():
    requested = os.environ.get("UNREAL4K_EXTRINSIC_CONVENTION", "w2c").strip().lower()
    if requested not in {"w2c", "c2w", "auto"}:
        raise RuntimeError(
            "UNREAL4K_EXTRINSIC_CONVENTION must be one of 'w2c', 'c2w', or 'auto', "
            f"got '{requested}'."
        )
    if requested == "auto":
        return "w2c", "auto->w2c"
    return requested, requested

def _unreal4k_to_w2c(extrinsic, convention):
    if convention == "w2c":
        return extrinsic
    if convention == "c2w":
        return np.linalg.inv(extrinsic)
    raise RuntimeError(f"Unsupported UnrealStereo4K extrinsic convention '{convention}'.")

def _buildUnrealStereo4KCameraInfos(scene_paths, eval, llffhold):
    image_dir = scene_paths["image_dir"]
    extrinsic_dir = scene_paths["extrinsic_dir"]
    scene_root = scene_paths["scene_root"]
    image_paths = _get_unreal4k_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(
            f"No UnrealStereo4K images found in '{image_dir}'. Expected .png/.jpg/.jpeg files in the selected "
            "Image0 or Image1 folder."
        )

    extrinsic_suffixes = {".txt", ".json", ".npy"}
    extrinsic_paths = sorted(
        [path for path in extrinsic_dir.iterdir() if path.is_file() and path.suffix.lower() in extrinsic_suffixes],
        key=_natural_frame_sort_key
    )
    if len(extrinsic_paths) != len(image_paths):
        raise RuntimeError(
            f"UnrealStereo4K image/extrinsic count mismatch for '{scene_paths['scene_label']}': "
            f"images={len(image_paths)} in '{image_dir}', extrinsics={len(extrinsic_paths)} in '{extrinsic_dir}'."
        )

    image_stems = {path.stem for path in image_paths}
    extrinsic_stems = {path.stem for path in extrinsic_paths}
    missing_extrinsics = sorted(image_stems - extrinsic_stems, key=lambda stem: _natural_frame_sort_key(Path(stem)))
    missing_images = sorted(extrinsic_stems - image_stems, key=lambda stem: _natural_frame_sort_key(Path(stem)))
    if missing_extrinsics or missing_images:
        raise RuntimeError(
            f"UnrealStereo4K image/extrinsic stem mismatch for '{scene_paths['scene_label']}': "
            f"missing extrinsics for images={missing_extrinsics[:5]}, missing images for extrinsics={missing_images[:5]}."
        )
    extrinsic_by_stem = {path.stem: path for path in extrinsic_paths}

    sampled_indices = _sampleUnreal4KFrames(image_paths)
    with Image.open(image_paths[int(sampled_indices[0])]) as first_image:
        first_width, first_height = first_image.size

    first_extrinsic_path = extrinsic_by_stem[image_paths[int(sampled_indices[0])].stem]
    _, first_intrinsic = _readUnreal4KExtrinsic(first_extrinsic_path)
    base_intrinsics = _read_or_infer_unreal4k_intrinsics(scene_root, image_dir, first_width, first_height, first_intrinsic)
    convention, convention_label = _unreal4k_extrinsic_convention()

    cam_infos = []
    warned_intrinsics_change = False
    for uid, frame_idx in enumerate(sampled_indices.tolist()):
        image_path = image_paths[frame_idx]
        extrinsic_path = extrinsic_by_stem[image_path.stem]
        extrinsic, embedded_intrinsic = _readUnreal4KExtrinsic(extrinsic_path)
        w2c = _unreal4k_to_w2c(extrinsic, convention)

        with Image.open(image_path) as image:
            width, height = image.size

        intrinsics = base_intrinsics
        if base_intrinsics["source"] == "extrinsics_txt" and embedded_intrinsic is not None:
            if not np.allclose(embedded_intrinsic, base_intrinsics["matrix"], rtol=1e-5, atol=1e-4) and not warned_intrinsics_change:
                print(
                    f"Warning: UnrealStereo4K embedded intrinsics vary across frames in '{scene_paths['scene_label']}'. "
                    "Using each frame's embedded intrinsics where available."
                )
                warned_intrinsics_change = True
            intrinsics = _intrinsics_result_from_matrix(embedded_intrinsic, width, height, "extrinsics_txt")

        cam_infos.append(CameraInfo(
            uid=uid,
            R=np.transpose(w2c[:3, :3]),
            T=w2c[:3, 3],
            FovY=intrinsics["FovY"],
            FovX=intrinsics["FovX"],
            depth_params=None,
            image_path=str(image_path.resolve()),
            image_name=f"{scene_root.name}_{image_dir.name}_{image_path.stem}",
            depth_path="",
            width=width,
            height=height,
            is_test=eval and llffhold and uid % llffhold == 0,
        ))

    return {
        "cam_infos": cam_infos,
        "total_images": len(image_paths),
        "sampled_images": len(sampled_indices),
        "sampled_indices": sampled_indices.tolist(),
        "resolution": (first_width, first_height),
        "extrinsic_convention": convention_label,
        "intrinsics_source": base_intrinsics["source"],
    }

def _loadUnrealStereo4KPointCloud(scene_root, camera_name, nerf_normalization):
    scene_root = Path(scene_root)
    ply_path = scene_root / f"unreal4k_points3D_{camera_name}.ply"
    if ply_path.exists():
        return fetchPly(ply_path), str(ply_path), "existing"

    num_pts = 50_000
    print(f"Generating random UnrealStereo4K point cloud ({num_pts})...")
    center = -np.asarray(nerf_normalization["translate"], dtype=np.float32)
    radius = max(float(nerf_normalization["radius"]), 1e-3)
    rng = np.random.default_rng(0)
    xyz = center + (rng.random((num_pts, 3), dtype=np.float32) * 2.0 - 1.0) * radius
    shs = rng.random((num_pts, 3), dtype=np.float32) / 255.0
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return fetchPly(ply_path), str(ply_path), "random"

def isTartanAirScenePath(path):
    return _is_tartanair_image_dir(Path(path))

def resolveTartanAirScenePath(path):
    image_dir = Path(path)
    if not _is_tartanair_image_dir(image_dir):
        raise RuntimeError(
            f"Could not resolve TartanAir scene from '{path}'. Expected an 'image_left' or 'image_right' directory "
            "whose parent contains the matching pose_left.txt or pose_right.txt file."
        )

    traj_dir = image_dir.parent
    level_dir = traj_dir.parent
    env_dir = level_dir.parent
    camera = "left" if image_dir.name == "image_left" else "right"

    return {
        "image_dir": image_dir,
        "traj_dir": traj_dir,
        "level": level_dir.name,
        "env_name": env_dir.name,
        "camera": camera,
        "pose_file": traj_dir / f"pose_{camera}.txt",
        "scene_label": f"{env_dir.name}/{level_dir.name}/{traj_dir.name}/{image_dir.name}",
    }

def _get_tartanair_image_paths(image_dir: Path):
    image_suffixes = {".png", ".jpg", ".jpeg"}

    def tartanair_sort_key(path: Path):
        frame_token = path.stem.split("_")[0]
        if frame_token.isdigit():
            return (0, int(frame_token), path.name)
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        if digits:
            return (1, int(digits), path.name)
        return (2, path.name)

    return sorted(
        [image_path for image_path in image_dir.iterdir() if image_path.is_file() and image_path.suffix.lower() in image_suffixes],
        key=tartanair_sort_key
    )

def _sampleTartanAirFrames(image_paths, sample_count=100):
    if len(image_paths) <= sample_count:
        return np.arange(len(image_paths), dtype=int)
    return np.linspace(0, len(image_paths) - 1, sample_count, dtype=int)

def _quat_xyzw_to_rotmat(qx, qy, qz, qw):
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise RuntimeError("Encountered near-zero TartanAir quaternion while building camera poses.")
    qx, qy, qz, qw = quat / norm

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float32)

def _buildTartanAirCameraInfos(scene_paths, eval, llffhold):
    image_dir = scene_paths["image_dir"]
    pose_file = scene_paths["pose_file"]
    image_paths = _get_tartanair_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(
            f"No TartanAir images found in '{image_dir}'. Expected .png/.jpg/.jpeg files in the selected image folder."
        )

    try:
        poses = np.loadtxt(pose_file, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TartanAir pose file '{pose_file}': {exc}") from exc

    poses = np.atleast_2d(poses)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise RuntimeError(
            f"Invalid TartanAir pose file '{pose_file}': expected Nx7 values 'tx ty tz qx qy qz qw', got shape {poses.shape}."
        )
    if poses.shape[0] != len(image_paths):
        raise RuntimeError(
            f"TartanAir image/pose count mismatch for '{scene_paths['scene_label']}': "
            f"images={len(image_paths)}, poses={poses.shape[0]}."
        )

    sampled_indices = _sampleTartanAirFrames(image_paths)
    A_cv_to_tartan = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    cam_infos = []
    for uid, frame_idx in enumerate(sampled_indices.tolist()):
        image_path = image_paths[frame_idx]
        tx, ty, tz, qx, qy, qz, qw = poses[frame_idx]
        camera_center = np.array([tx, ty, tz], dtype=np.float32)

        R_c2w_tartan = _quat_xyzw_to_rotmat(qx, qy, qz, qw)
        R_c2w_cv = R_c2w_tartan @ A_cv_to_tartan
        R_w2c_cv = R_c2w_cv.T
        T_w2c_cv = -R_w2c_cv @ camera_center

        with Image.open(image_path) as image:
            width, height = image.size

        fx = 320.0
        fy = 320.0
        if (width, height) != (640, 480):
            print(
                f"Warning: TartanAir image '{image_path}' has size ({width}, {height}) instead of (640, 480). "
                "Scaling intrinsics proportionally."
            )
            fx = fx * width / 640.0
            fy = fy * height / 480.0

        image_name = (
            f"{scene_paths['env_name']}_{scene_paths['level']}_{scene_paths['traj_dir'].name}_"
            f"{scene_paths['camera']}_{image_path.stem}"
        )
        cam_infos.append(CameraInfo(
            uid=uid,
            R=R_w2c_cv.T,
            T=T_w2c_cv.astype(np.float32),
            FovY=focal2fov(fy, height),
            FovX=focal2fov(fx, width),
            depth_params=None,
            image_path=str(image_path.resolve()),
            image_name=image_name,
            depth_path="",
            width=width,
            height=height,
            is_test=eval and llffhold and uid % llffhold == 0,
        ))

    return cam_infos, len(image_paths), len(sampled_indices), sampled_indices.tolist()

def _loadTartanAirPointCloud(traj_dir, camera, nerf_normalization):
    traj_dir = Path(traj_dir)
    ply_path = traj_dir / f"tartanair_points3D_{camera}.ply"
    if not ply_path.exists():
        num_pts = 50_000
        print(f"Generating random TartanAir point cloud ({num_pts})...")
        center = -np.asarray(nerf_normalization["translate"], dtype=np.float32)
        radius = max(float(nerf_normalization["radius"]), 1e-3)
        rng = np.random.default_rng(0)
        xyz = center + (rng.random((num_pts, 3), dtype=np.float32) * 2.0 - 1.0) * radius
        shs = rng.random((num_pts, 3), dtype=np.float32) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return fetchPly(ply_path), str(ply_path), "random"

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

def resolveBlendedMVSScenePath(path):
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
        f"Warning: source_path '{source_path}' looks like a BlendedMVS dataset root. "
        f"Automatically selecting scene '{selected_scene.name}'. For training, it is recommended "
        "to pass a specific scene path or set BLENDEDMVS_SCENE_ID."
    )
    return selected_scene, True

def resolveBlendedMVSScenePaths(path):
    return resolveBlendedMVSScenePath(path)

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

    try:
        extrinsic_index = next(idx for idx, line in enumerate(lines) if line.lower() == "extrinsic")
        intrinsic_index = next(idx for idx, line in enumerate(lines) if line.lower() == "intrinsic")
    except StopIteration as exc:
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': expected 'extrinsic' and 'intrinsic' sections."
        ) from exc

    if intrinsic_index <= extrinsic_index:
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': 'intrinsic' section appears before 'extrinsic'."
        )

    extrinsic_rows = lines[extrinsic_index + 1:extrinsic_index + 5]
    intrinsic_rows = lines[intrinsic_index + 1:intrinsic_index + 4]
    if len(extrinsic_rows) != 4 or len(intrinsic_rows) != 3:
        raise RuntimeError(
            f"Failed to parse BlendedMVS camera file '{cam_path}': expected 4 extrinsic rows and 3 intrinsic rows, "
            f"got {len(extrinsic_rows)} and {len(intrinsic_rows)}."
        )

    try:
        extrinsic = np.array([[float(value) for value in row.split()] for row in extrinsic_rows], dtype=np.float32)
        intrinsic = np.array([[float(value) for value in row.split()] for row in intrinsic_rows], dtype=np.float32)
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

    image_paths = _get_blendedmvs_image_paths(images_dir)
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

def readTartanAirSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    if depths != "":
        print(f"Warning: depths argument '{depths}' is ignored for TartanAir scene '{path}'.")

    scene_paths = resolveTartanAirScenePath(path)
    cam_infos, total_images, sampled_images, sampled_indices = _buildTartanAirCameraInfos(scene_paths, eval, llffhold)

    train_cam_infos = [cam_info for cam_info in cam_infos if train_test_exp or not cam_info.is_test]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.is_test]

    if not train_cam_infos:
        raise RuntimeError(
            f"No TartanAir training cameras found for '{path}'. "
            f"Found {total_images} images and sampled {sampled_images} frames."
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path, point_cloud_source = _loadTartanAirPointCloud(
        scene_paths["traj_dir"], scene_paths["camera"], nerf_normalization
    )
    if pcd is None:
        raise RuntimeError(f"Failed to load or generate TartanAir point cloud at '{ply_path}'.")

    print(
        f"TartanAir sample indices for '{scene_paths['scene_label']}': "
        f"first={sampled_indices[:5]}, last={sampled_indices[-5:]}."
    )
    print(
        f"Loaded TartanAir scene '{scene_paths['scene_label']}': "
        f"images={total_images}, sampled={sampled_images}, train={len(train_cam_infos)}, "
        f"test={len(test_cam_infos)}, point_cloud={point_cloud_source}."
    )

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readUnrealStereo4KSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    if images not in {None, "images"}:
        print(f"Warning: images argument '{images}' is ignored for UnrealStereo4K scene '{path}'. Pass Image0/Image1 as source_path.")
    if depths != "":
        print(
            f"Warning: depths argument '{depths}' is ignored for UnrealStereo4K scene '{path}'. "
            "Disp0/Disp1 are not connected to training yet."
        )

    scene_paths = resolveUnrealStereo4KScenePath(path)
    build_result = _buildUnrealStereo4KCameraInfos(scene_paths, eval, llffhold)
    cam_infos = build_result["cam_infos"]

    train_cam_infos = [cam_info for cam_info in cam_infos if train_test_exp or not cam_info.is_test]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.is_test]

    if not train_cam_infos:
        raise RuntimeError(
            f"No UnrealStereo4K training cameras found for '{path}'. "
            f"Found {build_result['total_images']} images and sampled {build_result['sampled_images']} frames."
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path, point_cloud_source = _loadUnrealStereo4KPointCloud(
        scene_paths["scene_root"], scene_paths["camera_name"], nerf_normalization
    )
    if pcd is None:
        raise RuntimeError(f"Failed to load or generate UnrealStereo4K point cloud at '{ply_path}'.")

    print(
        f"UnrealStereo4K sample indices for '{scene_paths['scene_label']}': "
        f"first={build_result['sampled_indices'][:5]}, last={build_result['sampled_indices'][-5:]}."
    )
    print(
        f"Loaded UnrealStereo4K scene '{scene_paths['scene_label']}': "
        f"camera_side={scene_paths['camera_name']}, images={build_result['total_images']}, "
        f"sampled={build_result['sampled_images']}, train={len(train_cam_infos)}, test={len(test_cam_infos)}, "
        f"resolution={build_result['resolution'][0]}x{build_result['resolution'][1]}, "
        f"extrinsic_convention={build_result['extrinsic_convention']}, "
        f"intrinsics_source={build_result['intrinsics_source']}, point_cloud={point_cloud_source}."
    )

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
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
    if depths != "":
        print(
            f"Warning: depths argument '{depths}' is ignored for BlendedMVS scene '{path}'. "
            "rendered_depth_maps are not connected to training yet."
        )

    scene_root, resolved_from_dataset_root = resolveBlendedMVSScenePath(path)
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
    "TartanAir": readTartanAirSceneInfo,
    "UnrealStereo4K": readUnrealStereo4KSceneInfo,
    "MapFree": readMapFreeSceneInfo,
    "BlendedMVS": readBlendedMVSSceneInfo
}
