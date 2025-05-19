import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile = f"{dataroot}/epinions"

dataset = "epinions"
testlength = 3
vallength = 1
length = 11
num_nodes = 8519
