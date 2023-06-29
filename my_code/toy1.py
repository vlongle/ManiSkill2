'''
File: /toy1.py
Project: my_code
Created Date: Thursday June 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained model')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--env_id', type=str, default="LiftCube-v0",
                    help='Environment ID')
args = parser.parse_args()

if __name__ == "__main__":
    print(f"foo {args.env_id} {args.seed}")
