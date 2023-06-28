'''
File: /load.py
Project: my_code
Created Date: Thursday June 8th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

import numpy as np


with open("logs/evaluations.npz", "rb") as f:
    data = np.load(f)
    print(list(data.keys()))
    print("timesteps:", data["timesteps"])
    print("results:", data["results"])
    print("episode_lengths:", data["ep_lengths"])
    print("success:", data["successes"])


# load models from best_models

from stable_baselines3.common.evaluation import evaluate_policy



# eval_env.close() # close the old eval env
# # make a new one that saves to a different directory
# eval_env = SubprocVecEnv([make_env(env_id, record_dir="logs/eval_videos") for i in range(1)])
# eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
# eval_env.seed(1)
# eval_env.reset()

# returns, ep_lens = evaluate_policy(model, eval_env, deterministic=True, render=False, return_episode_rewards=True, n_eval_episodes=10)
# success = np.array(ep_lens) < 200 # episode length < 200 means we solved the task before time ran out
# success_rate = success.mean()
# print(f"Success Rate: {success_rate}")
# print(f"Episode Lengths: {ep_lens}")

