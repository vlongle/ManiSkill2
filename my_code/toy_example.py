import platform
import time
import datetime
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained model')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--env_id', type=str, default="LiftCube-v0",
                    help='Environment ID')
args = parser.parse_args()

obs_mode = "toy"


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    start = time.time()
    time.sleep(1)

    # print(
    #     f"Server {platform.node()} Training {args.env_id} with seed {args.seed}")

    log_dir = f"{args.env_id}_{obs_mode}_{args.seed}.txt"
    # create_if_not_exists(log_dir)
    # append platform.node() to the log file
    with open(log_dir, "a") as f:
        f.write(f"{platform.node()}\n")

    # end = time.time()
    # print(f"Takes: {datetime.timedelta(seconds=end-start)}")
