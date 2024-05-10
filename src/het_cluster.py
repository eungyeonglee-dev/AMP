# Script to reproduce homogeneous setting results

import math
import time
from collections import defaultdict
import operator
import random
import os
import copy

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sa import amp_no_placement_strategy
from cost_het_cluster import AMP
from amp_utils import simulate, to_float_torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Whether to run real trials")
parser.add_argument("--budget", type=int, default=-1, help="how many real trials to launch")
parser.add_argument("--comm_type", type=str, default="ib", help="which type")
parser.add_argument("--hetero_node_num", type=int, default=1, help="hetero_node_num")
parser.add_argument("--node_per_gpu", type=int, default=4, help="node_per_gpu")
parser.add_argument("--num_node", type=int, default=8, help="num_node")
parser.add_argument("--gbs", type=int, default=64, help="num_node")

args = parser.parse_args()
# cluster information

time_s = time.time()
# number of GPU per node, number of nodes
M = args.node_per_gpu
N = args.num_node
hetero_node_num=args.hetero_node_num

if hetero_node_num == M:
    assert False, "you have to assign num of nodes inequaly"

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

cluster_info = {}

# inter-node bandwidth, intra-node bandwidth
for i in range(N):
        # A10 ethernet / PCIe
        if args.comm_type == "ib":
            cluster_info[i] = [torch.tensor([120 * 1e9]).float(), torch.tensor([108 * 1e9]).float(), torch.tensor([67 * 1e9]).float(), torch.tensor([16 * 8 * 1e9]).float()]
        elif args.comm_type == "eth":
            cluster_info[i] = [torch.tensor([40 * 1e9]).float(), torch.tensor([16 * 8 * 1e9]).float()]
for i in range(hetero_node_num):
        # A100 ethernet / NVLink
        if args.comm_type == "ib":
            # cluster_info[i] = [torch.tensor([120 * 1e9]).float(), torch.tensor([108 * 1e9]).float(), torch.tensor([67 * 1e9]).float(), torch.tensor([16 * 8 * 1e9]).float()]
            cluster_info[i] = [torch.tensor([120 * 1e9]).float(), torch.tensor([108 * 1e9]).float(), torch.tensor([67 * 1e9]).float(), torch.tensor([230 * 8 * 1e9]).float()]
        elif args.comm_type == "eth":
            cluster_info[i] = [torch.tensor([40 * 1e9]).float(), torch.tensor([230 * 8 * 1e9]).float()]
            
# device placement A100 A10 A10 A10 A10 A10 A10 A10 / A100:A10 = 1:7

#GPT2XL
model_config = {"hidden_size": torch.tensor([1600]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([48]).float(), 
                "vocab_size":torch.tensor([52256]).float(),
                "type":"gpt2XL"}

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())

exp_name = f"GPT2XL_A100_{hetero_node_num}_A10_{N-hetero_node_num}_IB_AMP"
# record_file = f"{os.path.join(dir_path, exp_name)}.csv"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.csv"
# simulate_dir = os.path.join(home_path, "amp_simulate")
# if not os.path.exists(simulate_dir):
#     os.mkdir(simulate_dir)

# remove cache directory from last run
if os.path.exists(os.path.join(home_path, "tmp")):
    for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
        for f in files:
            os.unlink(os.path.join(root, f))

# save this name to env
os.environ["amp_log_path"] = record_file

global_bs = args.gbs
model = AMP(args, model_config, exp_name)
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

want_simulate = [] 
feasible = {}

# with open(record_file, "a") as fp:
#     fp.write(f"{model_config}\n")                
#     fp.write(f"gbs:{global_bs}\n")                
known = None
iter_count = 0

# Estimating best configurations
while True:
    ret = amp_no_placement_strategy(M=M, N=N, gbs=global_bs, known=known)
    if ret is None:
        break
    else:
        h, w, mbs, known = ret
        tp = torch.ones(1,)*h
        dp = torch.ones(1,)*w
        pp = torch.ones(1,)*(M*N/(h*w))
        oth = {"mp_deg": tp, "dp_deg": dp, "pp_deg": pp}
        fake_config = np.ones((M,N)) * (-1)
        model_args = (fake_config, global_bs, mbs, cluster_info, model_config, oth)    
        
        with torch.no_grad():
            pipeline_cost, dp_side_cost, cost, partition = model(model_args)
        
        want_simulate.append((mbs, int(tp.item()), int(pp.item()), int(dp.item()), partition, cost.item(), pipeline_cost.item(), dp_side_cost.item()))
    iter_count += 1
    if iter_count % 10 == 0:
        print(f"AMP finishes {iter_count} iterations")
time_e = time.time()
print(f"AMP finishes without placement in {iter_count} iterations in {time_e - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[5])
df = pd.DataFrame(sorted_settings, columns=['mbs','tp','pp','dp','partition','estimated time (s/step)', \
                                            'pipeline time','DP AR time'])
df.to_csv(record_file, index=False)
print(f"save file: {record_file}")

# with open(record_file, "a") as fp:
#     for item in sorted_settings:
#         fp.write(f"rank {sorted_settings.index(item)}: {item}")
#         fp.write("\n")

# Run real trials to get ground truth runtime
if args.full:
    if args.budget == -1:
        budget = len(sorted_settings)
    else:
        budget = args.budget
    simulate_start = time.time()
    for i in range(budget):
        can = sorted_settings[i][0]
        rmap = None
        mbs = can[0]
        oth = can[1]
        partition = can[3]
        gt_cost = simulate([rmap], [partition], torch.ones(1,)*global_bs, to_float_torch([mbs]), model_config, [oth], exp_name)
        gt_cost = gt_cost[0]
        with open(record_file, "a") as fp:
            fp.write(f"Simulating result: {rmap}, {partition}, {mbs}, {oth}, with p_cost: {sorted_settings[i][1]}, r_cost: {gt_cost} \n")
            fp.write(f"running real trials till iter {i} takes {time.time() - time_s} \n")
