import functools
import numpy as np
import open3d as o3d

import sys
sys.path.append('/root/heguanhua/Work/robosuite_jimu')
import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix,
    get_real_depth_map,
    get_camera_extrinsic_matrix,
)

macros.IMAGE_CONVENTION = "opencv"


def parse_intrinsics(K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy

def get_individual_pcd(cam_name, obs, cam_width, cam_height, env=None):
    # load rgb-d image and camera parameters
    fx, fy, cx, cy = parse_intrinsics(get_camera_intrinsic_matrix(
        env.sim, cam_name, cam_height, cam_width))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=cam_width, height=cam_height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsic = get_camera_extrinsic_matrix(env.sim, cam_name)

    rgb = obs[f"{cam_name}_image"]
    depth = get_real_depth_map(env.sim, obs[f"{cam_name}_depth"])
    rgb = o3d.geometry.Image((rgb))
    depth = o3d.geometry.Image(depth)

    # Convert RGB-D image to a point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Transferm the point cloud to the world frame
    pcd.transform(extrinsic)
    return pcd


def get_pcd_from_rgbd(obs, camera_names, env=None):
    pcd_list = []
    for cam_name, cam_height, cam_width in zip(camera_names, env.camera_heights, env.camera_widths):
        pcd = get_individual_pcd(cam_name, obs, cam_width, cam_height, env=env)
        pcd_list.append(pcd)
    return pcd_list


def merge_pcds(pcd_list):
    return functools.reduce(lambda x, y: x+y, pcd_list)

def merge_point_cloud(camera_names, obs, env=None):
    pcd_list = get_pcd_from_rgbd(obs, camera_names, env)
    ret = merge_pcds(pcd_list)
    return ret

