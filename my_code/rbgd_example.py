'''
File: /rl_examples.py
Project: ManiSkill2
Created Date: Thursday June 8th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import gym
import gym.spaces as spaces
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th

from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper
from functools import partial


from mani_skill2.vector import VecEnv, make as make_vec_env
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from typing import Dict, List
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from time import time
import datetime
from dataclasses import dataclass, field
from dataclasses import asdict
import gym
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper, SuccessInfoWrapper
from mani_skill2.utils.wrappers import RecordEpisode

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 256

        for key, subspace in observation_space.spaces.items():
            # We go through all subspaces in the observation space.
            # We know there will only be "rgbd" and "state", so we handle those below
            if key == "rgbd":
                # here we use a NatureCNN architecture to process images, but any architecture is permissble here
                in_channels = subspace.shape[-1]
                cnn = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=32,
                              kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten()
                )

                # to easily figure out the dimensions after flattening, we pass a test tensor
                test_tensor = th.zeros(
                    [subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(
                    nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors["rgbd"] = nn.Sequential(cnn, fc)
                total_concat_size += feature_size
            elif key == "state":
                # for state data we simply pass it through a single linear layer
                state_size = subspace.shape[0]
                extractors["state"] = nn.Linear(state_size, 64)
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "rgbd":
                observations[key] = observations[key].permute((0, 3, 1, 2))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


class ManiSkillRGBDWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
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

        # Concatenate all the image spaces
        image_shapes = []
        for cam_uid in obs_space["image"]:
            cam_space = obs_space["image"][cam_uid]
            image_shapes.append(cam_space["rgb"].shape)
            image_shapes.append(cam_space["depth"].shape)
        image_shapes = np.array(image_shapes)
        assert np.all(image_shapes[0, :2] == image_shapes[:, :2]), image_shapes
        h, w = image_shapes[0, :2]
        c = image_shapes[:, 2].sum(0)
        rgbd_space = spaces.Box(0, np.inf, shape=(h, w, c))

        # Create the new observation space
        return spaces.Dict({"rgbd": rgbd_space, "state": state_space})

    @staticmethod
    def convert_observation(observation):
        # Process images. RGB is normalized to [0, 1].
        images = []
        for cam_uid, cam_obs in observation["image"].items():
            rgb = cam_obs["rgb"] / 255.0
            depth = cam_obs["depth"]

            # NOTE: SB3 does not support GPU tensors, so we transfer them to CPU.
            # For other RL frameworks that natively support GPU tensors, this step is not necessary.
            if isinstance(rgb, th.Tensor):
                rgb = rgb.to(device="cpu", non_blocking=True)
            if isinstance(depth, th.Tensor):
                depth = depth.to(device="cpu", non_blocking=True)
            # rgb.shape = [num_envs, h, w, 3]
            # depth.shape = [num_envs, h, w, 1]
            images.append(rgb)
            images.append(depth)

        # Concatenate all the images
        rgbd = np.concatenate(images, axis=-1)
        # rgbd.shape = [num_envs, h, w, (3 (rbg) +1 (depth)) * (num_cameras) ]

        # Concatenate all the states
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

        return dict(rgbd=rgbd, state=state)

    def observation(self, observation):
        return self.convert_observation(observation)


# We separately define an VecEnv observation wrapper for the ManiSkill VecEnv
# as the gpu optimization makes it incompatible with the SB3 wrapper
class ManiSkillRGBDVecEnvWrapper(VecEnvObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = ManiSkillRGBDWrapper.init_observation_space(
            env.observation_space
        )

    def observation(self, observation):
        return ManiSkillRGBDWrapper.convert_observation(observation)


@dataclass
class EnvParams:
    env_id: str = "LiftCube-v0"
    obs_mode: str = "rgbd"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "dense"
    max_episode_steps: int = 100  # time horizon


@dataclass
class TrainingParams:
    rollout_steps: int = 3200  # how many steps to run to collect buffer
# data before updating the policy (i.e. buffer size for on-policy PPO)
    num_envs: int = 24
    # num_envs: int = 3
    seed: int = 0
    batch_size: int = 400  # minibatch size for each update taken from
    # the buffer
    n_epochs: int = 15  # no. of inner epochs through the buffer
    # to optimize the PPO loss
    # total_timesteps: int = 25_000_000  # total number of steps to run
    total_timesteps: int = 1_000  # total number of steps to run


@dataclass
class LearnerParams:
    policy_kwargs: Dict[str, List[int]] = field(
        default_factory=lambda: dict(features_extractor_class=CustomExtractor, net_arch=[256, 128]))


@dataclass
class CallbackParams:
    save_freq: int = 32_000

# define an SB3 style make_env function for evaluation


# define a make_env function for Stable Baselines
def make_env(env_id: str, max_episode_steps=None, record_dir: str = None,
             obs_mode: str = "rgbd", control_mode: str = "pd_ee_delta_pose", reward_mode: str = "dense"):
    # NOTE: Import envs here so that they are registered with gym in subprocesses
    import mani_skill2.envs

    env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode)
    # For training, we regard the task as a continuous task with infinite horizon.
    # you can use the ContinuousTaskWrapper here for that
    if max_episode_steps is not None:
        env = ContinuousTaskWrapper(env, max_episode_steps)
    env = ManiSkillRGBDWrapper(env)
    # For evaluation, we record videos
    if record_dir is not None:
        env = SuccessInfoWrapper(env)
        env = RecordEpisode(env, record_dir, save_trajectory=False,
                            info_on_video=True, render_mode="cameras")
    return env


def create_if_not_exists(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_callbacks(env_params, training_params,
                      callback_params, log_dir):
    env_fn = partial(
        make_env,
        env_params.env_id,
        record_dir=f"{log_dir}/videos",)

    eval_env = SubprocVecEnv([env_fn for i in range(1)])
    # eval_env = SubprocVecEnv([make_env(**asdict(env_params),
    #                                    record_dir=f"{log_dir}/videos") for _ in range(1)])
    # attach this so SB3 can log reward metrics
    eval_env = VecMonitor(eval_env)
    eval_env.seed(training_params.seed)
    eval_env.reset()

    # periodically eval and save the best model, and also videos
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir}",
                                 log_path=f"{log_dir}", eval_freq=callback_params.save_freq // training_params.num_envs,
                                 deterministic=True, render=False)

    # periodically save models and current training progress
    checkpoint_callback = CheckpointCallback(
        save_freq=callback_params.save_freq // training_params.num_envs,
        save_path=f"{log_dir}",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    return [eval_callback, checkpoint_callback]


def prepare_model(env, learner_params, training_params, log_dir):
    model = PPO("MultiInputPolicy", env, policy_kwargs=learner_params.policy_kwargs,
                verbose=1,
                n_steps=training_params.rollout_steps // training_params.num_envs,
                batch_size=training_params.batch_size,
                n_epochs=training_params.n_epochs,
                tensorboard_log=f"{log_dir}",
                gamma=0.8,
                target_kl=0.2
                )
    return model


def eval_model(model, env_params, log_dir):
    # make a new one that saves to a different directory
    env_fn = partial(
        make_env,
        env_params.env_id,
        record_dir=f"{log_dir}/eval_videos",)

    eval_env = SubprocVecEnv([env_fn for i in range(1)])
    # eval_env = SubprocVecEnv([make_env(**asdict(env_params),
    #                                    record_dir=f"{log_dir}/videos") for _ in range(1)])
    # attach this so SB3 can log reward metrics
    eval_env = VecMonitor(eval_env)
    eval_env.seed(1)
    eval_env.reset()

    returns, ep_lens = evaluate_policy(
        model, eval_env, deterministic=True, render=False, return_episode_rewards=True, n_eval_episodes=10)
    # episode length < max_episode_steps means we solved the task before time ran out
    dummy_env = make_env(env_params.env_id)
    success = np.array(ep_lens) < dummy_env.spec.max_episode_steps
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained model')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--env_id', type=str, default="LiftCube-v0",
                    help='Environment ID')
args = parser.parse_args()


if __name__ == "__main__":
    start = time()

    env_params = EnvParams()
    env_params.env_id = args.env_id
    training_params = TrainingParams()
    training_params.seed = args.seed
    learner_params = LearnerParams()
    callback_params = CallbackParams()

    print(f"Training on {env_params.env_id} with seed {training_params.seed}")
    env = make_vec_env(
        env_id=env_params.env_id,
        num_envs=training_params.num_envs,
        obs_mode=env_params.obs_mode,
        control_mode=env_params.control_mode,
        wrappers=[
            partial(ContinuousTaskWrapper,
                    max_episode_steps=env_params.max_episode_steps)
        ]
    )

    # flatten nested observation spaces by concatenating them
    env = ManiSkillRGBDVecEnvWrapper(env)
    env = SB3VecEnvWrapper(env)
    env = VecMonitor(env)

    set_random_seed(training_params.seed)  # set SB3's global seed to 0
    env.seed(training_params.seed)
    env.reset()

    log_dir = f"logs/{env_params.env_id}/{env_params.obs_mode}/seed_{training_params.seed}"
    create_if_not_exists(log_dir)

    callbacks = prepare_callbacks(env_params, training_params,
                                  callback_params, log_dir)

    model = prepare_model(env, learner_params, training_params, log_dir)
    if args.pretrained:
        print("Loading pretrained model")
        model = model.load(f"{log_dir}/latest_model")
    else:
        print("Training from scratch")
        model.learn(training_params.total_timesteps,
                    callback=callbacks,)

        model.save(f"{log_dir}/latest_model")

    eval_model(model, env_params, log_dir)
    end = time()

    print(f"Takes: {datetime.timedelta(seconds=end-start)}")
