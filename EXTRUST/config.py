import argparse
import torch
from pathlib import Path

parser = argparse.ArgumentParser()

# 1. dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="bitcoinalpha",
    help="epinions, bitcoinalpha",
)
parser.add_argument("--num_nodes", type=int, default=-1, help="num of nodes")
parser.add_argument("--nfeat", type=int, default=128, help="dim of input feature")

# 2. experiments
parser.add_argument("--mode", type=str, default="train", help="train, eval")
parser.add_argument("--use_cfg", type=int, default=0, help="if use configs")
parser.add_argument(
    "--max_epoch", type=int, default=2000, help="number of epochs to train"
)
parser.add_argument("--testlength", type=int, default=3, help="length for test")
parser.add_argument("--device", type=str, default="gpu", help="training device")
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--patience", type=int, default=50, help="patience for early stop")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-7,
    help="weight for L2 loss on basic models",
)
parser.add_argument("--output_folder", type=str, default="", help="need to be modified")
parser.add_argument(
    "--sampling_times", type=int, default=1, help="negative sampling times"
)
parser.add_argument("--min_epoch", type=int, default=50, help="min epoch")
parser.add_argument("--log_dir", type=str, default="../logs/")
parser.add_argument(
    "--every_epoch", type=int, default=50, help="every n epochs to add cvae loss"
)
parser.add_argument(
    "--log_interval", type=int, default=10, help="every n epoches to log"
)
parser.add_argument("--nhid", type=int, default=8, help="dim of hidden embedding")
parser.add_argument(
    "--delta_d", type=int, default=16, help="dimension under each environment"
)
parser.add_argument("--d_for_cvae", type=int, default=8, help="hidden dim for cvae")
parser.add_argument("--n_layers", type=int, default=2, help="number of hidden layers")
parser.add_argument("--heads", type=int, default=4, help="attention heads")
parser.add_argument("--n_factors", type=int, default=8, help="latent factors")
parser.add_argument("--norm", type=int, default=1, help="normalization")
parser.add_argument("--nbsz", type=int, default=10, help="number of sampling neighbors")
parser.add_argument("--maxiter", type=int, default=4, help="number of iteration")
parser.add_argument("--skip", type=int, default=0, help="")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--use_RTE", type=int, default=1, help="Relative Time Encoding")
parser.add_argument(
    "--n_intervene", type=int, default=10, help="number of intervention times"
)
parser.add_argument(
    "--interv_size_ratio",
    type=float,
    default=0.02,
    help="the ratio of intervention size",
)
parser.add_argument(
    "--gen_ratio", type=float, default=0.02, help="the ratio of mixing samples"
)
parser.add_argument("--split", type=int, default=0, help="dataset split")
parser.add_argument(
    "--warm_epoch", type=int, default=0, help="the number of warm epoches"
)
parser.add_argument("--agg_param", type=float, default=0.15, help="aggregation weights")
parser.add_argument(
    "--alpha", type=float, default=0.01, help="parameter of intervention variance"
)
parser.add_argument("--beta", type=float, default=0.01, help="parameter of ecvae")
parser.add_argument("--threshold", type=float, default=-1, help="prediction threshold")
parser.add_argument("--use_node_feature", default=False, action='store_true')
parser.add_argument("--use_spacial_module", default=False, action='store_true')
parser.add_argument("--use_gru_module", default=False, action='store_true')
parser.add_argument("--use_interv_module", default=False, action='store_true')
parser.add_argument("--early_stopping_indicator", default='auc', help="early stopping indicator")

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")

Path(args.log_dir).mkdir(parents=True, exist_ok=True)

def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)
