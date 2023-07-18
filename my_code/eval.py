"""
Layout: logs/env_id/obs_mode/seed_{seed}/
folder contains evaluation.npz
"""

import numpy as np
import pandas as pd
import os


def load_eval(folder):
    with open(f"{folder}/evaluations.npz", "rb") as f:
        data = np.load(f)
        # print(list(data.keys()))
        # print("timesteps:", data["timesteps"])
        # print("results:", data["results"])
        # print("episode_lengths:", data["ep_lengths"])
        # print("success:", data["successes"])
    success_rate = data["successes"].mean()
    return success_rate


def load_all_evals(log_dir):
    """
    Load all the evaluation.npz and write their success_rate to df
    with keys: env_id, obs_mode, seed, success_rate
    """
    df = pd.DataFrame(columns=["env_id", "obs_mode", "seed", "success_rate"])
    for env_id in os.listdir(log_dir):
        for obs_mode in os.listdir(f"{log_dir}/{env_id}"):
            for seed in os.listdir(f"{log_dir}/{env_id}/{obs_mode}"):
                folder = f"{log_dir}/{env_id}/{obs_mode}/{seed}"
                success_rate = load_eval(folder)
                df = df.append({"env_id": env_id, "obs_mode": obs_mode, "seed": seed,
                                "success_rate": success_rate}, ignore_index=True)
    return df


load_all_evals("logs")
