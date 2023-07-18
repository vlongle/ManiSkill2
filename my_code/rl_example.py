'''
File: /rl_examples.py
Project: ManiSkill2
Created Date: Thursday June 8th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''


import os
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


@dataclass
class EnvParams:
    env_id: str = "LiftCube-v0"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "dense"
    max_episode_steps: int = 100  # time horizon


@dataclass
class TrainingParams:
    rollout_steps: int = 3200  # how many steps to run to collect buffer
    # data before updating the policy (i.e. buffer size for on-policy PPO)
    num_envs: int = 24
    seed: int = 0
    batch_size: int = 400  # minibatch size for each update taken from
    # the buffer
    n_epochs: int = 15  # no. of inner epochs through the buffer
    # to optimize the PPO loss
    # total_timesteps: int = 25_000_000  # total number of steps to run
    # total_timesteps: int = 1_000  # total number of steps to run
    total_timesteps: int = 200_000  # total number of steps to run


@dataclass
class LearnerParams:
    policy_kwargs: Dict[str, List[int]] = field(
        default_factory=lambda: dict(net_arch=[256, 256]))


@dataclass
class CallbackParams:
    save_freq: int = 100_000


def make_env(env_id: str, max_episode_steps: int = None, record_dir: str = None,
             obs_mode: str = "state", control_mode: str = "pd_ee_delta_pose",
             reward_mode: str = "dense"):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs
        env = gym.make(env_id, obs_mode=obs_mode,
                       reward_mode=reward_mode, control_mode=control_mode,)
        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            # TimeLimit wrapper basically
            env = ContinuousTaskWrapper(env, max_episode_steps)

        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            # record video to hdf5 files
            env = RecordEpisode(
                env, record_dir, info_on_video=True, render_mode="cameras"
            )
        return env
    return _init


def make_vec_env(env_id, num_envs, obs_mode, reward_mode, control_mode, max_episode_steps):
    return VecMonitor(SubprocVecEnv([make_env(env_id, max_episode_steps,
                                              obs_mode=obs_mode,
                                              control_mode=control_mode,
                                              reward_mode=reward_mode) for _ in range(num_envs)]))


def eval_model(model, env_params, log_dir):
    # make a new one that saves to a different directory
    eval_env = SubprocVecEnv(
        [make_env(env_params.env_id, record_dir=f"{log_dir}/eval_videos") for i in range(1)])
    # attach this so SB3 can log reward metrics
    eval_env = VecMonitor(eval_env)
    eval_env.seed(1)
    eval_env.reset()

    returns, ep_lens = evaluate_policy(
        model, eval_env, deterministic=True, render=False, return_episode_rewards=True, n_eval_episodes=10)
    # episode length < max_episode_steps means we solved the task before time ran out
    dummy_env = make_env(env_params.env_id)()
    success = np.array(ep_lens) < dummy_env.spec.max_episode_steps
    success_rate = success.mean()
    print(f"Success Rate: {success_rate}")
    print(f"Episode Lengths: {ep_lens}")


def prepare_model(env, learner_params, training_params, log_dir):
    model = PPO("MlpPolicy", env,
                policy_kwargs=learner_params.policy_kwargs,
                verbose=1,
                n_steps=training_params.rollout_steps // training_params.num_envs,
                batch_size=training_params.batch_size,
                n_epochs=training_params.n_epochs,
                tensorboard_log=f"{log_dir}",
                gamma=0.85,
                target_kl=0.05
                )

    return model


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_callbacks(env_params, training_params,
                      callback_params, log_dir):
    eval_env = SubprocVecEnv([make_env(**asdict(env_params),
                                       record_dir=f"{log_dir}/videos") for _ in range(1)])
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
    log_dir = f"logs/{env_params.env_id}/{env_params.obs_mode}/seed_{training_params.seed}"
    create_if_not_exists(log_dir)
    print("log dir:", log_dir)
    # check if latest_model.zip exists, if yes, then skip training
    if os.path.exists(f"{log_dir}/latest_model.zip"):
        print("Skipping training as latest_model.zip exists")
        exit()
    else:
        print("Training from scratch")

    env = make_vec_env(
        **asdict(env_params),
        num_envs=training_params.num_envs,
    )

    set_random_seed(training_params.seed)  # set SB3's global seed to 0
    env.seed(training_params.seed)
    env.reset()

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
