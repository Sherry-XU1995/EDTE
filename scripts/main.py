import csv
import datetime
import sys
from pathlib import Path

sys.path.append(".")
sys.path.append("..")

from EDTE.config import args
from EDTE.utils.data_util import *
from EDTE.Frontdoor.HlDeconfounder import HLDeconfounder

import warnings

warnings.simplefilter("ignore")

time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
key_args = f"k{args.n_factors}_intv{args.interv_size_ratio}_mix{args.gen_ratio}_nl{args.n_layers}_nhid{args.nhid}_d{args.delta_d}_h{args.heads}_nb{args.nbsz}_rte{args.use_RTE}_dc{args.d_for_cvae}_ni{args.n_intervene}_agg{args.agg_param}"
experiment_name = f"{time}_{args.dataset}_{key_args}"

print('-' * 80)
print(experiment_name)
print('-' * 80)

# load data
args, data = load_data(args)

# Runner
from EDTE.runner import Runner
from EDTE.model import EADGNN
from EDTE.model import ECVAE
from EDTE.model import GRUEncoder

model = EADGNN(args=args).to(args.device)
cvae = ECVAE(args=args).to(args.device)
hldeconfounder = HLDeconfounder(args=args).to(args.device)
gru = GRUEncoder(args=args).to(args.device)
runner = Runner(args, model, gru, hldeconfounder, cvae, data)

result = None

if args.mode == "train":
    result = runner.run()
elif args.mode == "eval":
    result = runner.re_run()

# post-logs
train_train_scores = result['train_train_scores']
train_test_scores = result['train_test_scores']
test_train_scores = result['test_train_scores']
test_test_scores = result['test_test_scores']
print('-' * 40)
print(experiment_name)
print(f"train_train_auc: {train_train_scores['auc']:.4f}, train_train_ap: {train_train_scores['ap']:.4f}, train_train_f: {train_train_scores['f1']:.4f}, train_train_acc: {train_train_scores['acc']:.4f}")
print(f"train_test_auc: {train_test_scores['auc']:.4f}, train_test_ap: {train_test_scores['ap']:.4f}, train_test_f: {train_test_scores['f1']:.4f}, train_test_acc: {train_test_scores['acc']:.4f}")
print(f"test_train_auc: {test_train_scores['auc']:.4f}, test_train_ap: {test_train_scores['ap']:.4f}, test_train_f: {test_train_scores['f1']:.4f}, test_train_acc: {test_train_scores['acc']:.4f}")
print(f"test_test_auc: {test_test_scores['auc']:.4f}, test_test_ap: {test_test_scores['ap']:.4f}, test_test_f: {test_test_scores['f1']:.4f}, test_test_acc: {test_test_scores['acc']:.4f}")


# with open(Path(args.log_dir, f'{experiment_name}_{train_train_scores["auc"]:.4f}_{test_test_scores["auc"]:.4f}.pkl'), "wb") as f:
#     pickle.dump(result, f)
#
# with open(Path(args.log_dir, f'experiment.log'), "a", encoding="utf8") as f:
#     writer = csv.writer(f)
#     writer.writerow([time, args.dataset,
#            args.use_spacial_module, args.use_gru_module, args.use_interv_module,
#            args.n_factors, args.interv_size_ratio, args.gen_ratio,
#            f"{result['total_time']:.4f}", result['total_epoch'], f"{result['total_time'] / result['total_epoch']:.4f}",
#            f"{train_train_scores['auc']:.4f}", f"{train_train_scores['ap']:.4f}", f"{train_train_scores['f1']:.4f}", f"{train_train_scores['acc']:.4f}",
#            f"{train_test_scores['auc']:.4f}", f"{train_test_scores['ap']:.4f}", f"{train_test_scores['f1']:.4f}", f"{train_test_scores['acc']:.4f}",
#            f"{test_train_scores['auc']:.4f}", f"{test_train_scores['ap']:.4f}", f"{test_train_scores['f1']:.4f}", f"{test_train_scores['acc']:.4f}",
#            f"{test_test_scores['auc']:.4f}", f"{test_test_scores['ap']:.4f}", f"{test_test_scores['f1']:.4f}", f"{test_test_scores['acc']:.4f}",
#            args.n_layers, args.nhid, args.delta_d, args.heads, args.nbsz, args.use_RTE, args.d_for_cvae, args.n_intervene, args.agg_param
#     ])
