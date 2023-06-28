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

# define an SB3 style make_env function for evaluation


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
    import os
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