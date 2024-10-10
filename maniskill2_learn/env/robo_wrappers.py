import cv2
import copy
import numpy as np
from collections import deque

from maniskill2_learn.utils.data import (
    DictArray,
    GDict,
    deepcopy,
    encode_np,
    is_num,
    to_array,
    SLICE_ALL,
    to_np,
)
from maniskill2_learn.utils.meta import Registry, build_from_cfg
from .observation_process import pcd_uniform_downsample
from .robo_pc import merge_point_cloud

import sys
sys.path.append("/root/heguanhua/Work/robosuite_jimu")
import robosuite
from robosuite.wrappers import Wrapper, GymWrapper
from robosuite.utils import camera_utils as CU
from robosuite.utils import transform_utils as TU

ROBOWRAPPERS = Registry("wrappers of robosuite environments")


class MyGymWrapper(GymWrapper):
    def __init__(self, env, kwargs):
        super(MyGymWrapper, self).__init__(env)
        self.img_size = None
        self._max_step = env.horizon
        self._cur_step = 0

        self.obs_mode = kwargs["obs_mode"]
        self.obs_frame = kwargs["obs_frame"]
        self.n_points = kwargs["n_points"]
        self.n_goal_points = kwargs["n_goal_points"]
        self.env_name = kwargs["env_name"]

        self.env.obs_mode = self.obs_mode

    def _ob_dict_process(self, ob_dict):
        from mani_skill2.utils.common import flatten_state_dict
        from maniskill2_learn.utils.lib3d.mani_skill2_contrib import(
                apply_pose_to_points, apply_pose_to_point)
        from sapien.core import Pose

        obs = ob_dict
        if self.obs_mode == "rgbd":
            rgb, depth, segs = [], [], []
            for cam in self.env.camera_names:
                rgb.append(obs[cam+"_image"])
                depth.append(obs[cam+"_depth"])
                obs.pop(cam+"_image")
                obs.pop(cam+"_depth")
            rgb = np.concatenate(rgb, axis=2)
            assert rgb.dtype == np.uint8
            depth = np.concatenate(depth, axis=2)

            if self.img_size is not None and \
               self.img_size != (rgb.shape[0], rgb.shape[1]):
                rgb = cv2.resize(
                        rgb.astype(np.float32),
                        self.img_size,
                        interpolation=cv2.INTER_LINEAR,)
                depth = cv2.resize(
                        depth,
                        self.img_size,
                        interpolation=cv2.INTER_LINEAR,)

            # Original MS2-Learn need calculate relativate distance between TCP & Goal
            # However, robosutie observation dont have key "tcp_pose" and "goal_pos"
            # For trial, we input

            # If goal info is provided, calculate the relative position between the robot fingers' tool-center-point (tcp) and the goal
            # if "tcp_pose" in obs["extra"].keys() and "goal_pos" in obs["extra"].keys():
            #     assert obs["extra"]["tcp_pose"].ndim <= 2
            #     if obs["extra"]["tcp_pose"].ndim == 2:
            #         tcp_pose = obs["extra"]["tcp_pose"][0] # take the first hand's tcp pose
            #     else:
            #         tcp_pose = obs["extra"]["tcp_pose"]
            #     obs["extra"]["tcp_to_goal_pos"] = (
            #         obs["extra"]["goal_pos"] - tcp_pose[:3]
            #     )
            # if "tcp_pose" in obs["extra"].keys():
            #     obs["extra"]["tcp_pose"] = obs["extra"]["tcp_pose"].reshape(-1)

            # breakpoint()
            # state = flatten_state_dict(obs)

            out_dict = {
                "rgb": rgb.astype(np.uint8, copy=False).transpose(2, 0, 1),
                "depth": depth.astype(np.float16,copy=False).transpose(2, 0, 1),
                "state": obs['robot0_proprio-state'],
            }
        elif self.obs_mode == "pointcloud":
            # Get robote eef pos and quat for to_origin
            # Calculate coordinate transformations that transforms poses in the world to self.obs_frame
            # These "to_origin" coordinate transformations are formally T_{self.obs_frame -> world}^{self.obs_frame}
            assert self.obs_frame == 'ee', \
                    f"Robosuite onnly support obs_frame = ee, but get obs_frame = {self.obs_frame}"
            p = obs['robot0_eef_pos']
            q = obs['robot0_eef_quat']
            to_origin = Pose(p=p, q=q).inv()

            # Get point cloud in world coordinate
            cameras = self.env.camera_names
            point_cloud = merge_point_cloud(cameras, obs, self.env)
            xyz = np.asarray(point_cloud.points)    # float64, un normalize
            rgb = np.asarray(point_cloud.colors)    # float64 0-1

            # Preprocess for downsample
            mask = (xyz[:,0] > -0.7) & (xyz[:,0] < 0.5) & \
                   (xyz[:,1] > -0.5) & (xyz[:,1] < 0.5) & \
                   (xyz[:,2] > 0.82)

            # Initialize return dict
            out_dict = {
                "xyz": xyz[mask],                    # final_shape:(n_points + n_goal_points, 3)
                "rgb": rgb[mask],                    # final_shape:(n_points + n_goal_points, 3)
                "frame_related_states": np.zeros((4, 3)),      # (4,3)
                "to_frames": np.zeros((2,4,4)),                # (2,4,4)
                "state": np.zeros((30,)),                      # (30,)
            }

            # Process observation point cloud segmentations, is given
            #TODO: We have not implemented this feature on Robosuite

            # Downsample point cloud, transform point cloud coordinate to self.obs_frame
            # Origin is random uniform downsample, which is not applible to robosuite
            uniform_downsample_kwargs = {"env": self.env, "ground_eps": 1e-4, "num": self.n_points}
            if "PointCloudPreprocessObsWrapper" not in self.env.__str__():
                pcd_uniform_downsample(out_dict, **uniform_downsample_kwargs)
            out_dict["xyz"] = apply_pose_to_points(out_dict["xyz"], to_origin)

            # Sample from and append the goal point cloud to the observation point cloud,
            # if the goal point cloud is given
            #XXX: ManiSkill2 Pick/Stack tasks dont have "target_points"
            goal_pcd_xyz = obs.pop("target_points", None)
            if goal_pcd_xyz is not None:
                ret_goal = {}
                ret_goal["xyz"] = goal_pcd_xyz
                for k in out_dict.keys():
                    if k != "xyz":
                        ret_goal[k] = np.ones_like(out_dict[k]) * (-1)
                pcd_uniform_downsample(ret_goal, **uniform_downsample_kwargs)
                ret_goal["xyz"] = apply_pose_to_points(ret_goal["xyz", to_origin])
                for k in out_dict.keys():
                    out_dict[k] = np.concatenate([out_dict[k], ret_goal[k]], axis=0)

            # Get all kinds of position and 6D poses from the observation
            # These postion and poses are in world frame for now (transformer later)
            eef_pos, eef_quat = obs['robot0_eef_pos'], obs['robot0_eef_quat']
            tcp_poses = np.concatenate((eef_pos, eef_quat))
            tcp_poses = tcp_poses[None, :]
            tcp_poses = [Pose(p=pose[:3], q=pose[3:]) for pose in tcp_poses] # robust for multi robots, may unuseful
            goal_pos = None
            goal_pose = None
            tcp_to_goal_pos = None
            #TODO: Need to get goal pos and pose
            #HACK: "Stack"
            if self.env_name == "Stack":
                table_height = self.env.table_offset[2]
                cubeB_pos = self.env.sim.data.body_xpos[self.env.cubeB_body_id]
                half_cubeB_height = cubeB_pos[2] - table_height
                goal_pos_z = table_height + (half_cubeB_height) * 2 + 0.03 # 0.03 is half of cubeA height
                goal_pos = np.array([cubeB_pos[0], cubeB_pos[1], goal_pos_z])
            elif self.env_name == "Jimu":
                cubeB_pos_ranges = self.env.tgt_cube_poses[-1]
                goal_pos = np.array([
                    (cubeB_pos_ranges[0][0] + cubeB_pos_ranges[0][1]) / 2,
                    (cubeB_pos_ranges[1][0] + cubeB_pos_ranges[1][1]) / 2,
                    (cubeB_pos_ranges[2][0] + cubeB_pos_ranges[2][1]) / 2])

            if tcp_poses is not None and goal_pos is not None:
                tcp_to_goal_pos = goal_pos - tcp_poses[0].p

            # Sample green points near the goal and append them to the observation point cloud, which serve as visual gaol indicator
            # if self.n_goal_points is specified and the goal information if given in an environment
            # Also, transform these points to self.obs_frame
            if self.n_goal_points > 0:
                assert(goal_pos is not None), \
                        "n_goal_points should only be used if goal_pos(e) is contained in the environment observation"
                goal_pts_xyz = (np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3)) * 0.01)
                goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
                goal_pts_xyz = apply_pose_to_points(goal_pts_xyz, to_origin)
                goal_pts_rgb = np.zeros_like(goal_pts_xyz)
                goal_pts_rgb[:, 1] = 1
                out_dict["xyz"] = np.concatenate([out_dict["xyz"], goal_pts_xyz])
                out_dict["rgb"] = np.concatenate([out_dict["rgb"], goal_pts_rgb])

            # Transform all kinds of positions to self.obs_frame; these information are dependent on
            # the choice of self.obs_frame, so we name them "frame_related_states"
            frame_related_states = []
            #TODO: Robosuite has no "base_pose" (robot position and quaternion in the world frame)
            #HACK: Use "robot0_eef_pos" and "robot0_eef_quat" instead of "base_pose"
            base_pose = np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
            base_info = apply_pose_to_point(base_pose[:3], to_origin)
            frame_related_states.append(base_info)
            if tcp_poses is not None:
                for tcp_pose in tcp_poses:
                    tcp_info = apply_pose_to_point(tcp_pose.p, to_origin)
                    frame_related_states.append(tcp_info)
            if goal_pos is not None:
                goal_info = apply_pose_to_point(goal_pos, to_origin)
                frame_related_states.append(goal_info)
            if tcp_to_goal_pos is not None:
                tcp_to_goal_info = apply_pose_to_point(tcp_to_goal_pos, to_origin)
                frame_related_states.append(tcp_to_goal_info)
            # Source code contain "gripper_pose", "joint_axis", "link_pos"
            # MS2-learn dont use these parameters, so we do same process on Robosuite
            frame_related_states = np.stack(frame_related_states, axis=0)
            out_dict["frame_related_states"] = frame_related_states  # original is [4, 3]

            # Transform the goal pose and the pose from the end-effector (tool-center point, tcp)
            # to the goal into self.obs_frame; these info are also dependent on the choice of self.obs_frame,
            # so we name them "frame_goal_related_poses"
            frame_goal_related_poses = []
            if goal_pose is not None:
                pose_wrt_origin = to_origin * goal_pose
                frame_goal_realted_poses.append(np.hstack([pose_wrt_origin.p, pose_wrt_origin.q]))
                if tcp_poses is not None:
                    for tcp_pose in tcp_poses:
                        pose_wrt_origin = (goal_pose * tcp_pose.inv())
                        pose_wrt_origin = to_origin * pose_wrt_origin
                        frame_goal_related_poses.append(np.hstack([pose_wrt_origin.p, pose_wrt_origin.q]))
            if len(frame_goal_related_poses) > 0:
                frame_goal_related_poses = np.stack(frame_goal_related_poses, axis=0)
                out_dict["frame_goal_related_poses"] = frame_goal_related_poses

            # out_dict["to_frame"] returns frame transformation information, which is information that transforms
            # from self.obs_frame to other common frames (e.g. robot base frame, end-effector frame, goal frame)
            # Each transformation is formally T_{target_frame -> self.obs_frame}^{target_frame}
            out_dict["to_frames"] = []
            #TODO: Robosuite has no base_pose
            #HACK: use "robot0_eef_pos" and "robot0_eef_quat" instead
            base_pose_p, base_pose_q = copy.deepcopy(obs["robot0_eef_pos"]), copy.deepcopy(obs["robot0_eef_quat"])
            base_frame = ((to_origin * Pose(p=base_pose_p, q=base_pose_q)).inv().to_transformation_matrix())
            out_dict["to_frames"].append(base_frame)
            if tcp_poses is not None:
                for tcp_pose in tcp_poses:
                    hand_frame = (to_origin * tcp_pose).inv().to_transformation_matrix()
                    out_dict["to_frames"].append(hand_frame)
            if goal_pose is not None:
                goal_frame = (to_origin * goal_pose).inv().to_transformation_matrix()
                out_dict["to_frames"].append(goal_frame)
            out_dict["to_frames"] = np.stack(out_dict["to_frames"], axis=0)  # should be [Nframe, 4, 4]

            # Obatain final agent state vector, which contains robot proprioceptive information, frame-related states,
            # and other miscellaneous states (propbaly important) from the environment
            #TODO: Robosuite has no qpos, qvel
            #HACK: Use tan value instead of "qpos"
            pos_cos, pos_sin = obs["robot0_joint_pos_cos"], obs["robot0_joint_pos_sin"]
            qpos = np.divide(pos_sin, pos_cos, out=np.zeros_like(pos_sin), where=pos_cos!=0)
            qvel = obs["robot0_joint_vel"]
            agent_state = np.concatenate([qpos, qvel])
            if len(frame_related_states) > 0:
                agent_state = np.concatenate([agent_state, frame_related_states.flatten()])
            if len(frame_goal_related_poses) > 0:
                agent_state = np.concatenate([agent_state, from_goal_related_states.flatten()])
            out_dict["state"] = agent_state

        return out_dict


    def step(self, action):
        ob_dict, reward, episode_done, info = self.env.step(action)
        done = self._check_success()
        if info == {}:
            info["success"] = done
            info["reward"] = reward
        #TODO: Early stop has quesetion
        # truncated = self._cur_step >= self._max_step
        # # truncated = False
        # self._cur_step += 1
        # if self._cur_step >= self._max_step:
        #     self.reset()
        #     self._cur_step = 0
        return self._ob_dict_process(ob_dict), reward, done, episode_done, info


    def reset(self, seed=None, options=None):
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._ob_dict_process(ob_dict), {}


    def get_obs_img(self):
        obs = self.env.viewer._get_observations() \
                if self.env.viewer_get_obs \
                else self.env._get_observations()
        cam = self.env.camera_names[0]
        img = obs[cam + "_image"]
        # img = np.array(img)
        return img


class ExtendedWrapper(Wrapper):
    def __getattr_(self, name):
        return getattr(self.env, name)


class RoboExtendedEnv(ExtendedWrapper):
    def __init__(self, env, reward_scale, use_cost):
        super(RoboExtendedEnv, self).__init__(env)
        assert reward_scale > 0, "Reward scale should be positive!"
        # judge action space is discrete or not
        self.is_discrete = False
        self.is_cost = -1 if use_cost else 1
        self.reward_cale = reward_scale * self.is_cost

    # if `action` is not discrete, nothing happened
    # Robosuite should get origin `action` back
    def _process_action(self, action):
        #TODO: Robosuit support not into here
        if self.is_discrete:
            if is_num(action):
                action = int(action)
            else:
                assert(
                    action.size == 1
                ), f"Dim of discrete action should be 1, but we got {len(action)}"
                action = int(action.reshape(-1)[0])
        return action

    # tranfer origin `obs` from f64 to f32
    def reset(self, *args, **kwargs):
        kwargs = dict(kwargs)
        obs, _ = self.env.reset(*args, **kwargs)
        return GDict(obs).f64_to_f32(wrapper=False)

    #TODO: need to align the step output
    def step(self, action, *args, **kwargs):
        action = self._process_action(action)
        ## terminated is whether task is done
        ## truncated is whether stop by max steps
        ## use GymWrapper, robosuite env can has 5 output
        obs, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)
        # obs, reward, done, info = self.env.step(action, *args, **kwargs)
        obs, info = GDict([obs, info]).f64_to_f32(wrapper=False)
        return obs, np.float32(reward * self.reward_scale), np.bool_(terminated), np.bool_(truncated), info

    def step_random_actions(self, num):
        ret = None
        obs = GDict(self.reset()).copy(wrapper=False)
        for i in range(num):
            #TODO: do robosuite has `sample` funciton
            action = self.action_space.sample()
            next_obs, rewards, terminated, truncated, infos = self.step(aciton)

            info_i = dict(
                    obs=obs,
                    next_obs=next_obs,
                    actions=action,
                    rewards=rewards,
                    dones=terminated,
                    infos=GDict(infos).copy(wrapper=False),
                    episode_dones=terminated or truncated,)
            info_i = GDict(info_i).to_array(wrapper=False)
            obs = GDict(next_obs).copy(wrapper=False)

            if ret is None:
                ret = DictArray(info_i, capacity=num)
            ret.assign(i, info_i)
            if terminated or truncated:
                obs = GDict(self.reset()).copy(wrapper=False)
        return ret.to_two_dims(wrapper=False)


    #TODO: unknown usage
    def step_states_action(self, state=None, actions=None):
        assert actions.ndim == 3
        rewards = np.zeros_like(actions[..., :1], dtype=np.float32)
        for i in range(len(actions)):
            if hasattr(self, "set_state") and states is not None:
                self.set_state(states[i])
            for j in range(len(actions[i])):
                rewards[i, j] = self.step(actions[i, j])[1]
        return rewards


    def get_env_state(self):
        ret = {}
        if hasattr(self.env, "get_state"):
            ret["env_states"] = self.env.get_state()
        if hasattr(self.env, "level"):
            ret["env_levels"] = self.env.level
        return ret


class RoboBufferAugmentedEnv(ExtendedWrapper):
    """
    For multi-process envionments, modified for robosuite
    Use a buffer to transfer data from sub-process to main process
    """
    def __init__(self, env, buffers):
        super(RoboBufferAugmentedEnv, self).__init__(env)
        #TODO: need to confirm on robosuite
        self.reset_buffer = GDict(buffers[0])
        self.step_buffer = GDict(buffers[:5])
        if len(buffers) == 6:
            self.vis_img_buffer = GDict(buffers[5])

    def reset(self, *args, **kwargs):
        alls = self.env.reset(*args, **kwargs)
        self.reset_buffer.assign_all(alls)

    def step(self, *args, **kwargs):
        alls = self.env.step(*args, **kwargs)
        self.step_buffer.assign_all(alls)

    #TODO: unknown funciton
    def render(self, *args, **kwargs):
        ret = self.env.render(*args, **kwargs)
        if ret is not None:
            assert(self.vis_img_buffer is not None), "Robosuite dont confirm for vis_img_buffer"
            self.vis_img_buffer.assign_all(ret)




