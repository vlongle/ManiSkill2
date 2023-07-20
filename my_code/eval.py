"""
Layout: logs/env_id/obs_mode/seed_{seed}/
folder contains evaluation.npz
"""

import numpy as np
import pandas as pd
import os
import json

"""
EvalCallBack is kinda weird. The separate eval loop after training
sometimes just achieves much better success rate than the callback.
Also, the separate eval loop doubles the horizon
"""
# def load_eval(folder):
#     with open(f"{folder}/evaluations.npz", "rb") as f:
#         data = np.load(f)
#         print(list(data.keys()))
#         print("timesteps\n:", data["timesteps"])
#         print("results:\n", data["results"])
#         print("episode_lengths:\n", data["ep_lengths"])
#         print("success:\n", data["successes"])
#     # print("data", data, data.keys())
#     # success_rate = data["successes"].mean()
#     # return success_rate


def load_eval(folder):
    """
    folder/eval_videos/**.json
    contains a list of eval episodes
    with "episodes" key that contains "info" that contains "success" keys
    """
    print('folder:', folder)
    json_file = [f for f in os.listdir(
        f"{folder}/eval_videos") if "json" in f][0]
    with open(f"{folder}/eval_videos/{json_file}", "r") as f:
        data = json.load(f)
        # print("data", data)
        success_rate = np.mean([episode["info"]["success"]
                               for episode in data["episodes"]])
        return success_rate


def load_all_evals(log_dir):
    """
    Load all the evaluation.npz and write their success_rate to df
    with keys: env_id, obs_mode, seed, success_rate
    """
    df_ls = []
    for env_id in os.listdir(log_dir):
        for obs_mode in os.listdir(f"{log_dir}/{env_id}"):
            for seed in os.listdir(f"{log_dir}/{env_id}/{obs_mode}"):
                folder = f"{log_dir}/{env_id}/{obs_mode}/{seed}"
                # check folder/eval_videos exists, otherwise, skip
                if not os.path.exists(f"{folder}/eval_videos"):
                    continue
                success_rate = load_eval(folder)
                df_ls.append({"env_id": env_id, "obs_mode": obs_mode, "seed": seed,
                              "success_rate": success_rate})
    df = pd.DataFrame(
        df_ls, columns=["env_id", "obs_mode", "seed", "success_rate"])
    return df


# print(load_eval("test_onev1_logs/LiftCube-v0/state/seed_0"))
# print(load_eval("test_onev1_logs/LiftCube-v0/rgbd/seed_0"))
# print(load_eval("test_onev1_logs/PushChair-v1/state/seed_0"))
# print(load_all_evals("test_one_logs"))
print(load_all_evals("logs"))
