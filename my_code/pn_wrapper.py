'''
File: /pn_extractor.py
Project: my_code
Created Date: Thursday June 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pn_arch import PointNet
import gym
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
import gym.spaces as spaces
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper
from mani_skill2.vector import VecEnv, make as make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch as th


# also see: https://github.com/haosulab/ManiSkill2-Learn/blob/83dfe26c73b6ce6b0388a0fa07493f340e36dd44/maniskill2_learn/env/wrappers.py#L401
# https://github.com/charlesq34/pointnet/issues/12#issuecomment-308684441


# NOTE: https://arxiv.org/pdf/2302.04659.pdf
# use 1200 point clouds (plus some other tricks like including 50 green points around goal position)
# https://www.chenbao.tech/dexart/static/paper/dexart.pdf
# use 6k point clouds per object, using tricks like "imagined points"
MAX_POINTCLOUD_SIZE = 1200


class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "pointcloud"
        self.observation_space = self.init_observation_space(
            env.observation_space)

    @staticmethod
    def init_observation_space(obs_space: spaces.Dict):
        # States include robot proprioception (agent) and task information (extra)
        # NOTE: SB3 does not support nested observation spaces, so we convert them to flat spaces
        state_spaces = []
        state_spaces.extend(flatten_dict_space_keys(
            obs_space["agent"]).spaces.values())
        state_spaces.extend(flatten_dict_space_keys(
            obs_space["extra"]).spaces.values())
        # Concatenate all the state spaces
        state_size = sum([space.shape[0] for space in state_spaces])
        state_space = spaces.Box(-np.inf, np.inf, shape=(state_size,))

        pointcloud_space = spaces.Box(-np.inf, np.inf, shape=(
            MAX_POINTCLOUD_SIZE, 3), dtype=np.float32)
        rgb_space = spaces.Box(0, 1, shape=(
            MAX_POINTCLOUD_SIZE, 3), dtype=np.float32)

        return spaces.Dict({'state': state_space, 'pointcloud': pointcloud_space,
                            'rgb': rgb_space})

    @staticmethod
    def convert_observation(observation):
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

        pointcloud = observation["pointcloud"]["xyzw"]
        rgb = observation["pointcloud"]["rgb"]
        rgb = rgb / 255.0

        # Process pointcloud and rgb for single and vectorized environments
        # TODO: refactor this ugly code
        if pointcloud.ndim == 2:  # single environment
            mask = pointcloud[:, -1] >= 0.5
            pointcloud = pointcloud[mask, :-1]
            rgb = rgb[mask]
            if isinstance(pointcloud, th.Tensor):
                pointcloud = pointcloud.to(
                    device="cpu", non_blocking=True)
            if isinstance(rgb, th.Tensor):
                rgb = rgb.to(
                    device="cpu", non_blocking=True)

            # downsample by uniform random sampling
            if pointcloud.shape[0] > MAX_POINTCLOUD_SIZE:
                idx = np.random.choice(
                    pointcloud.shape[0], MAX_POINTCLOUD_SIZE, replace=False
                )
                pointcloud = pointcloud[idx]
                rgb = rgb[idx]
        else:  # vectorized environments
            mask = pointcloud[..., -1] >= 0.5

            num_envs = pointcloud.shape[0]
            new_pointcloud = []
            new_rgb = []
            # process each environment independently
            for i in range(num_envs):
                current_pointcloud = pointcloud[i][mask[i], :-1]
                current_rgb = rgb[i][mask[i]]
                if isinstance(current_pointcloud, th.Tensor):
                    current_pointcloud = current_pointcloud.to(
                        device="cpu", non_blocking=True)
                if isinstance(current_rgb, th.Tensor):
                    current_rgb = current_rgb.to(
                        device="cpu", non_blocking=True)

                # downsample by uniform random sampling
                if current_pointcloud.shape[0] > MAX_POINTCLOUD_SIZE:
                    idx = np.random.choice(
                        current_pointcloud.shape[0], MAX_POINTCLOUD_SIZE, replace=False
                    )
                    current_pointcloud = current_pointcloud[idx]
                    current_rgb = current_rgb[idx]

                new_pointcloud.append(current_pointcloud)
                new_rgb.append(current_rgb)

            pointcloud = np.stack(new_pointcloud)
            rgb = np.stack(new_rgb)

        return dict(state=state, pointcloud=pointcloud, rgb=rgb)

    def observation(self, observation):
        return self.convert_observation(observation)


# # We separately define an VecEnv observation wrapper for the ManiSkill VecEnv
# # as the gpu optimization makes it incompatible with the SB3 wrapper


class ManiSkillRPointCloudVecEnvWrapper(VecEnvObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "pointcloud"
        self.observation_space = ManiSkillPointCloudWrapper.init_observation_space(
            env.observation_space
        )

    def observation(self, observation):
        return ManiSkillPointCloudWrapper.convert_observation(observation)


if __name__ == "__main__":
    env_id = "PickCube-v0"
    # @param can be one of ['pointcloud', 'rgbd', 'state_dict', 'state']
    obs_mode = "pointcloud"
    # @param can be one of ['pd_ee_delta_pose', 'pd_ee_delta_pos', 'pd_joint_delta_pos', 'arm_pd_joint_pos_vel']
    control_mode = "pd_joint_delta_pos"
    reward_mode = "dense"  # @param can be one of ['sparse', 'dense']
    env = gym.make(env_id, obs_mode=obs_mode,
                   reward_mode=reward_mode, control_mode=control_mode)
    obs = env.reset()
    # see: https://haosulab.github.io/ManiSkill2/concepts/observation.html
    print("Action Space:", env.action_space)
    print('obs:', obs.keys())  # [agent, extra, pointcloud]
    # [ee_pos, ee_quat, joint_pos, joint_vel]
    print('agent:', obs['agent'].keys())
    # task specific info like goal-position, end-effector pos
    print('extra:', obs['extra'].keys())
    # pointcloud in the world frame
    print('pointcloud:', obs['pointcloud'].keys())  # [xyzw, rbg]
    # xyzw: [N, 4] point cloud where w=0 for infinite points beyond
    # camera depth range, and w=1 for the rest.
    # We will only use points with w=1.
    print('rgb:', obs['pointcloud']['rgb'].shape)  # [N, 3]
    # indicate the color at each point location
    # test the wrapper
    env.close()

    num_envs = 4
    env: VecEnv = make_vec_env(
        env_id,
        num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode
    )

    env = ManiSkillRPointCloudVecEnvWrapper(env)
    env = SB3VecEnvWrapper(env)
    env = VecMonitor(env)

    env.seed(0)
    obs = env.reset()
    print(obs.keys())
    print("pointcloud shape", obs["pointcloud"].shape)
    print("state shape", obs["state"].shape)
    print('rgb shape', obs['rgb'].shape)
    env.close()

    print('end')
