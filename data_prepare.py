import os
from glob import glob
from tqdm import tqdm
import argparse
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, required=True)
parser.add_argument('--output_root', type=str, default='filelists/crema-d')

args = parser.parse_args()

os.makedirs(args.output_root, exist_ok=True)

filelists = []
folders = os.listdir(args.dataset_root)
for folder in tqdm(folders, total=len(folders)):
    filelists.append(folder)

random.seed(0)
random.shuffle(filelists)

trainlists = filelists[:int(len(filelists) * 0.8)]
testlists = filelists[int(len(filelists) * 0.8):]

with open(f'{args.output_root}/train.txt', 'w') as f:
    for line in trainlists:
        f.write(f"{line}\n")

with open(f'{args.output_root}/test.txt', 'w') as f:
    for line in testlists:
        f.write(f"{line}\n")
        
#python data_prepare.py --dataset_root=/root/datasets/CREMA-D/IPLAP_audio/VideoFPS/ --output_root=data