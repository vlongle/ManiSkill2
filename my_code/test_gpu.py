'''
File: /test_gpu.py
Project: my_code
Created Date: Thursday June 29th 2023
Author: Long Le (vlongle@seas.upenn.edu)

Copyright (c) 2023 Long Le
'''

# simple pytorch test that use cuda-1

import torch

print("torch.cuda.is_available()", torch.cuda.is_available())

print("torch.cuda.device_count()", torch.cuda.device_count())

a = torch.cuda.FloatTensor(2).zero_()

print("a", a)

# use cuda:1
a = a.to(device=torch.device('cuda:1'))
print("a", a)
