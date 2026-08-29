# %%
import json
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from models.MTGFLOW import MTGFLOW, MTGFLOWZL
from models.graph_divergence import column_kl
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
import pickle


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def train_cdf_percentile(sorted_train_ref, values):
    """Map values to percentiles of the TRAIN empirical CDF (inductive,
    per-sample, deployment-valid) — transductive test-set ranking is not."""
    n = max(len(sorted_train_ref), 1)
    return np.searchsorted(sorted_train_ref, values, side='right') / n


parser = argparse.ArgumentParser()

# parser.add_argument('--data_dir', type=str,
#                     default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str,
                    default='./checkpoint/')
parser.add_argument('--name', default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='MAF')

parser.add_argument('--n_blocks', type=int, default=1,
                    help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1,
                    help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')
parser.add_argument('--seed', type=int, default=15, help='Learning rate.')
parser.add_argument('--log_emb', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--log_loss', type=bool, default=False)

parser.add_argument('--v_latent', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--loss_weight_manifold_po', type=float, default=0.1)
parser.add_argument('--loss_weight_manifold_ne', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--use_spectral_graph', type=str2bool, default=True)
parser.add_argument('--spectral_representation', type=str, default='real_imag',
                    choices=['real_imag', 'magnitude', 'log_magnitude'])
parser.add_argument('--drop_dc', type=str2bool, default=False)
parser.add_argument('--freq_patch_size', type=int, default=4)
parser.add_argument('--freq_patch_stride', type=int, default=2)
parser.add_argument('--freq_embed_dim', type=int, default=0,
                    help='Frequency graph embedding size. 0 means use hidden_size.')
parser.add_argument('--freq_dropout', type=float, default=0.1)
parser.add_argument('--spectral_graph_topk', type=int, default=0)
parser.add_argument('--spectral_gating_mode', type=str, default='mlp',
                    choices=['uniform', 'energy', 'mlp', 'hybrid'],
                    help='Patch gating: uniform / energy (beta=1, no MLP) / mlp (original) / '
                         'hybrid (MLP + beta*log-energy, learnable beta).')
parser.add_argument('--graph_fusion_mode', type=str, default='gated',
                    choices=['time_only', 'freq_only', 'mean', 'fixed_alpha', 'learnable_alpha', 'gated'])
parser.add_argument('--graph_alpha', type=float, default=0.7,
                    help='Time-graph weight for graph_fusion_mode=fixed_alpha.')
parser.add_argument('--learnable_alpha_init', type=float, default=0.7,
                    help='Initial time-graph weight for graph_fusion_mode=learnable_alpha.')
parser.add_argument('--log_spectral_stats', type=str2bool, default=False)
parser.add_argument('--x3_align_lambda', type=float, default=0.0,
                    help='Weight of the time-frequency graph alignment loss (0 = off, original behaviour).')
parser.add_argument('--x3_align_direction', type=str, default='t2f', choices=['t2f', 'f2t'],
                    help='t2f: KL(sg[A_time] || A_freq); f2t: KL(A_freq || sg[A_time]). '
                         'Teacher is always the (detached) time graph.')
parser.add_argument('--x3_log_divergence', type=str2bool, default=False,
                    help='Log per-epoch alignment loss and D-alone AUROC (both KL directions) on the test set.')
parser.add_argument('--x3_fusion_lambda', type=float, default=0.1,
                    help='Fixed weight for the rank-fusion preview score rank(usd)+lam*rank(D). '
                         'Chosen a priori from the zero-training diagnosis; NOT tuned on test.')

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


import random
import numpy as np

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# %%
print("Loading dataset")
print(args.name)
from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

if args.name == 'SWaT':
    train_loader, test_loader, n_sensor = loader_SWat(
        args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

elif args.name == 'Wadi':
    train_loader, test_loader, n_sensor = loader_WADI(
        args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
    train_loader, test_loader, n_sensor = load_smd_smap_msl(
        args.name, args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

elif args.name == 'PSM':
    train_loader, test_loader, n_sensor = loader_PSM(
        args.name, args.batch_size, args.window_size, args.stride_size, args.train_split, k=args.k,
        alpha=args.alpha,
        seed=args.seed
    )

print("n_sensor:", n_sensor)

# %%
model = MTGFLOWZL(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size,
                  n_sensor,
                  dropout=0.0, model=args.model, batch_norm=args.batch_norm,
                  use_spectral_graph=args.use_spectral_graph,
                  spectral_representation=args.spectral_representation,
                  drop_dc=args.drop_dc,
                  freq_patch_size=args.freq_patch_size,
                  freq_patch_stride=args.freq_patch_stride,
                  freq_embed_dim=args.freq_embed_dim,
                  freq_dropout=args.freq_dropout,
                  spectral_graph_topk=args.spectral_graph_topk,
                  spectral_gating_mode=args.spectral_gating_mode,
                  graph_fusion_mode=args.graph_fusion_mode,
                  graph_alpha=args.graph_alpha,
                  learnable_alpha_init=args.learnable_alpha_init)

model = model.to(device)

save_path = os.path.join(args.output_dir, args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %%
from torch.nn.utils import clip_grad_value_

loss_best = 100
roc_max = 0
ap_max = 0
best_feature_data = {}
x3_d_best_t2f = float('-inf')
x3_d_best_f2t = float('-inf')
x3_d_at_rocmax_t2f = float('nan')
x3_d_at_rocmax_f2t = float('nan')
x3_fuse_at_rocmax_t2f = float('nan')
x3_fuse_at_rocmax_f2t = float('nan')

lr = args.lr
# Keep scalar control parameters out of weight decay. Decay would pull
# energy_beta toward 0 and alpha_param's logit toward 0 (alpha=0.5).
no_decay_names = ('energy_beta', 'alpha_param')
decay_params = [p for n, p in model.named_parameters() if not any(key in n for key in no_decay_names)]
no_decay_params = [p for n, p in model.named_parameters() if any(key in n for key in no_decay_names)]
param_groups = [{'params': decay_params, 'weight_decay': args.weight_decay}]
if no_decay_params:
    param_groups.append({'params': no_decay_params, 'weight_decay': 0.0})
optimizer = torch.optim.AdamW(param_groups, lr=lr)

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")

for epoch in range(args.epoch):
    test_input_data_list = []
    test_h_data_list = []
    test_h_label_list = []
    train_h_data_list = []
    train_h_label_list = []
    train_input_data_list = []
    train_log_prob_list = []
    train_idx_list = []
    test_idx_list = []
    test_log_prob_list = []

    loss_train = []
    align_epoch_values = []
    model.train()
    for x_ori, x_aug, label, idx in train_loader:
        x_ori = x_ori.to(device)
        x_aug = x_aug.to(device)
        x_all = torch.cat([x_ori, x_aug], dim=0)

        optimizer.zero_grad()
        if args.x3_align_lambda > 0:
            hid, loss, _, _, graph_aux = model(x_all, return_aux=True)
        else:
            hid, loss, _, _ = model(x_all, )

        loss_mani_po, loss_mani_ne = model.LossManifold(
            input_data=x_all.reshape(x_all.shape[0], -1),
            latent_data=hid.reshape(x_all.shape[0], -1),
            v_input=100,
            v_latent=args.v_latent,
        )

        total_loss = -loss + loss_mani_po * args.loss_weight_manifold_po + loss_mani_ne * args.loss_weight_manifold_ne

        if args.x3_align_lambda > 0:
            # Align only on the unaugmented half (x_ori). Augmented windows are
            # frequency-perturbed by design; aligning them too would train the
            # graphs to agree under perturbation and blunt the divergence
            # signal the x3 score relies on. Training slices may still contain
            # unknown anomalies (unsupervised protocol) — this is alignment on
            # "unaugmented training windows", not certified-normal windows.
            half = x_ori.shape[0]
            a_time = graph_aux["A_time"][:half]
            a_freq = graph_aux["A_freq"][:half]
            if args.x3_align_direction == 't2f':
                l_align = column_kl(a_time.detach(), a_freq).mean()
            else:
                l_align = column_kl(a_freq, a_time.detach()).mean()
            total_loss = total_loss + args.x3_align_lambda * l_align
            align_epoch_values.append(l_align.item())

        total_loss.backward()
        clip_grad_value_(model.parameters(), 1)
        optimizer.step()

    model.eval()
    loss_test = []
    x3_monitor = args.x3_align_lambda > 0 or args.x3_log_divergence
    d_t2f_list, d_f2t_list = [], []
    d_train_t2f_list, d_train_f2t_list = [], []
    device_test = torch.device("cpu")
    with torch.no_grad():
        for x, x_aug, label, idx in test_loader:
            x = x.to(device)
            if x3_monitor:
                h_vis, log_prob, h_gcn_test, test_aux = model.test(x, return_aux=True)
                d_t2f_list.append(column_kl(test_aux["A_time"], test_aux["A_freq"]).cpu().numpy())
                d_f2t_list.append(column_kl(test_aux["A_freq"], test_aux["A_time"]).cpu().numpy())
            else:
                h_vis, log_prob, h_gcn_test = model.test(x, )

            h_vis = h_vis.to(device_test)
            log_prob = log_prob.to(device_test)
            h_gcn_test = h_gcn_test.to(device_test)

            loss = -log_prob.cpu().numpy()
            loss_test.append(loss)

            if args.log_emb:
                # test_input_data_list.append(x.cpu().detach().numpy().reshape(x.shape[0], -1))
                # test_h_data_list.append(h_vis.cpu().detach().numpy().reshape(h_gcn_train.shape[0], -1))

                if args.name == 'SWaT':
                    test_h_label_list.append(label.numpy())
                    test_log_prob_list.append(loss)
                    test_idx_list.append(idx.numpy())

        for x, x_aug, label, idx in train_loader:
            x = x.to(device)
            if x3_monitor:
                h_vis, log_prob, h_gcn_train, train_aux = model.test(x, return_aux=True)
                d_train_t2f_list.append(column_kl(train_aux["A_time"], train_aux["A_freq"]).cpu().numpy())
                d_train_f2t_list.append(column_kl(train_aux["A_freq"], train_aux["A_time"]).cpu().numpy())
            else:
                h_vis, log_prob, h_gcn_train = model.test(x, )

            h_vis = h_vis.to(device_test)
            log_prob = log_prob.to(device_test)
            h_gcn_test = h_gcn_test.to(device_test)

            loss = -log_prob.cpu().numpy()
            loss_train.append(loss)

            if args.log_emb:
                train_input_data_list.append(x.cpu().detach().numpy().reshape(x.shape[0], -1))
                train_h_data_list.append(h_vis.cpu().detach().numpy().reshape(h_gcn_train.shape[0], -1))
                train_h_label_list.append(label.numpy())
                if args.name == 'SWaT':
                    train_log_prob_list.append(loss)
                    train_idx_list.append(idx.numpy())

    loss_test = np.concatenate(loss_test)
    loss_train = np.concatenate(loss_train)
    roc_test = roc_auc_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)
    ap_test = average_precision_score(np.asarray(test_loader.dataset.label, dtype=int), loss_test)

    current_d_t2f = None
    current_d_f2t = None
    if x3_monitor:
        x3_labels = np.asarray(test_loader.dataset.label, dtype=int)
        d_t2f = np.concatenate(d_t2f_list)
        d_f2t = np.concatenate(d_f2t_list)
        current_d_t2f = roc_auc_score(x3_labels, d_t2f)
        current_d_f2t = roc_auc_score(x3_labels, d_f2t)
        x3_d_best_t2f = max(x3_d_best_t2f, current_d_t2f)
        x3_d_best_f2t = max(x3_d_best_f2t, current_d_f2t)
        # Inductive rank-fusion: test scores mapped to percentiles of the
        # TRAIN empirical CDF (usd and D each), then combined with a fixed
        # a-priori weight. Heavy tails ruled out z-fusion; transductive
        # test-set ranking was ruled out in protocol review.
        usd_train_sorted = np.sort(loss_train)
        d_train_t2f_sorted = np.sort(np.concatenate(d_train_t2f_list))
        d_train_f2t_sorted = np.sort(np.concatenate(d_train_f2t_list))
        lam_s = args.x3_fusion_lambda
        usd_pct = train_cdf_percentile(usd_train_sorted, loss_test)
        current_fuse_t2f = roc_auc_score(
            x3_labels, usd_pct + lam_s * train_cdf_percentile(d_train_t2f_sorted, d_t2f))
        current_fuse_f2t = roc_auc_score(
            x3_labels, usd_pct + lam_s * train_cdf_percentile(d_train_f2t_sorted, d_f2t))

    if roc_max < roc_test:
        roc_max = roc_test
        ap_max = ap_test
        if x3_monitor:
            x3_d_at_rocmax_t2f = current_d_t2f
            x3_d_at_rocmax_f2t = current_d_f2t
            x3_fuse_at_rocmax_t2f = current_fuse_t2f
            x3_fuse_at_rocmax_f2t = current_fuse_f2t

        # save embedding
        if args.log_emb:
            train_input_data = np.concatenate(train_input_data_list, axis=0)
            train_h_data = np.concatenate(train_h_data_list, axis=0)
            train_h_label = np.concatenate(train_h_label_list, axis=0)

            # test_input_data = np.concatenate(test_input_data_list, axis=0)
            # test_h_data = np.concatenate(test_h_data_list, axis=0)
            # test_h_label = np.concatenate(test_h_label_list, axis=0)

            if args.name == 'SWaT':
                train_log_prob = np.concatenate(train_log_prob_list, axis=0)
                print("train_log_prob:", train_log_prob.shape)
                test_log_prob = np.concatenate(test_log_prob_list, axis=0)
                print("test_log_prob:", test_log_prob.shape)

                train_idx = np.concatenate(train_idx_list, axis=0)
                print("train_idx:", train_idx.shape)
                test_idx = np.concatenate(test_idx_list, axis=0)
                print("test_idx:", test_idx.shape)

                test_h_label = np.concatenate(test_h_label_list, axis=0)

                best_feature_data = {
                    'train': {
                        'original': train_input_data,
                        'features': train_h_data,
                        'labels': train_h_label,
                        'log_prob': train_log_prob,
                        'idx': train_idx
                    },
                    'test': {
                        # 'original': test_input_data,
                        # 'features': test_h_data,
                        'labels': test_h_label,
                        'log_prob': test_log_prob,
                        'idx': test_idx
                    }
                }
            else:
                best_feature_data = {
                    'train': {
                        'original': train_input_data,
                        'features': train_h_data,
                        'labels': train_h_label
                    },
                    # 'test': {
                    #     'original': test_input_data,
                    #     'features': test_h_data,
                    #     'labels': test_h_label
                }

            save_Embedding = os.path.join('save_Embedding', args.name)
            if not os.path.exists(save_Embedding):
                os.makedirs(save_Embedding)

            best_feature_data_file_name = f'{save_Embedding}/path_to_{args.name}_dataset.pkl'
            print('best_feature_data_file_name:', best_feature_data_file_name)
            with open(best_feature_data_file_name, 'wb') as file:
                pickle.dump(best_feature_data, file)

        if args.save_model:
            save_model_file_name = f"{save_path}/{args.name}_dataset_model.pth"
            torch.save({'model': model.state_dict()}, save_model_file_name)
            print("save_model_file_name:", save_model_file_name)

        if args.log_loss:
            saved_test_label = np.asarray(test_loader.dataset.label, dtype=int)
            saved_test_loss = {
                'label': saved_test_label,
                'loss_test': loss_test
            }

            saved_loss = os.path.join('saved_loss', args.name)
            if not os.path.exists(saved_loss):
                os.makedirs(saved_loss)

            save_loss_file = f'{saved_loss}/{args.name}_dataset_loss.pkl'
            print('save_loss_file:', save_loss_file)
            with open(save_loss_file, 'wb') as file:
                pickle.dump(saved_test_loss, file)

    print('epoch:', epoch, 'seed:', args.seed, 'roc_max:', roc_max, 'ap_max:', ap_max)
    if x3_monitor:
        align_mean = float(np.mean(align_epoch_values)) if align_epoch_values else 0.0
        print('x3:', 'l_align:%.6f' % align_mean,
              'd_auroc_t2f:%.6f' % current_d_t2f,
              'd_auroc_f2t:%.6f' % current_d_f2t,
              'd_best_t2f:%.6f' % x3_d_best_t2f,
              'd_best_f2t:%.6f' % x3_d_best_f2t,
              'd_at_rocmax_t2f:%.6f' % x3_d_at_rocmax_t2f,
              'd_at_rocmax_f2t:%.6f' % x3_d_at_rocmax_f2t,
              'fuse_t2f:%.6f' % current_fuse_t2f,
              'fuse_f2t:%.6f' % current_fuse_f2t,
              'fuse_at_rocmax_t2f:%.6f' % x3_fuse_at_rocmax_t2f,
              'fuse_at_rocmax_f2t:%.6f' % x3_fuse_at_rocmax_f2t)
    if args.log_spectral_stats:
        spectral_stats = model.get_spectral_stats()
        if spectral_stats:
            stats_text = ' '.join(f'{key}:{value:.6f}' for key, value in spectral_stats.items())
            print('spectral_stats:', stats_text)

    # wandb.log({
    #     'loss_mani_po': loss_mani_po,
    #     'loss_mani_ne': loss_mani_ne,
    #     'loss_train': np.mean(loss_train),
    #     'loss_test': np.mean(loss_test),
    #     'roc_test_max': roc_max,
    #     'roc_test': roc_test,
    #     'ap_test': ap_test,
    #     'ap_test_max': ap_max
    # })

# wandb.finish()
