import torch
import numpy as np
import random
import os
import yaml
import json
import itertools
import torch.nn.functional as F


from tools.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_norm_values(norm_family = 'imagenet'):
    '''
        Inputs
            norm_family: String of norm_family
        Returns
            mean, std : tuple of 3 channel values
    '''
    if norm_family == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError('Incorrect normalization family')
    return mean, std


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def write_json(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_optimizer(model, config):
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=1e-3)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def get_scheduler(optimizer, config, num_batches=-1):
    if not hasattr(config, 'scheduler'):
        return None
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'linear_w_warmup' or config.scheduler == 'cosine_w_warmup':
        assert num_batches != -1
        num_training_steps = num_batches * config.epochs
        num_warmup_steps = int(config.warmup_proportion * num_training_steps)
        if config.scheduler == 'linear_w_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
        if config.scheduler == 'cosine_w_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler


def step_scheduler(scheduler, config, bid, num_batches):
    if config.scheduler in ['StepLR']:
        if bid + 1 == num_batches:    # end of the epoch
            scheduler.step()
    elif config.scheduler in ['linear_w_warmup', 'cosine_w_warmup']:
        scheduler.step()

    return scheduler


def multi_evaluation(simi, mask, targets_, args):
    """
    simi: shape: [n, num_class], the similarity between visual features and all num_class semantic labels;
    mask: shape: [n_class, num_nodes], each row indicates the label is composed of which attributes and objects;
    targets: shape: [n, 1], ground truth, range form 0 to num_class-1;
    """

    scores_ = mask[np.argmax(simi, axis=1)]
    targets_ = mask[np.squeeze(targets_)]

    n_sample, n_class = scores_.shape
    nc_attr_hard, nc_attr_soft = np.zeros(n_sample), np.zeros(n_sample)
    nc_obj = np.zeros(n_sample)
    for k in range(n_sample):
        pos_count_hard = 0
        pos_count_soft = 0
        total_count = 0
        calibration_factor = 1.0
        pred, gt = scores_[k, :args.num_attrs], targets_[k, :args.num_attrs]
        pred_idx, gt_idx = np.where(pred == 1)[0], np.where(gt == 1)[0]  # e.g., [ 93  97 115] [ 78  97 115]
        com = [list(itertools.combinations(gt_idx, i+1)) for i in range(len(gt_idx))]  # e.g., [[(78,), (97,), (115,)], [(78, 97), (78, 115), (97, 115)], [(78, 97, 115)]]
        com = list(itertools.chain.from_iterable(com))  # e.g., [(78,), (97,), (115,), (78, 97), (78, 115), (97, 115), (78, 97, 115)]

        n_com = len(com)
        for j in range(n_com):
            if set(com[j]).issubset(set(pred_idx)):
                pos_count_hard += 1
            if not set(com[j]).isdisjoint(set(pred_idx)):
                pos_count_soft += 1
            if len(pred_idx) > len(gt_idx):
                calibration_factor = 1 / 1.1 ** (len(pred_idx) - len(gt_idx))
            total_count += 1
        nc_attr_hard[k] = pos_count_hard / total_count * calibration_factor
        nc_attr_soft[k] = pos_count_soft / total_count * calibration_factor
        nc_obj[k] = ((scores_[k, args.num_attrs:] - targets_[k, args.num_attrs:]) ** 2).sum() == 0

    # print(nc_attr_hard.mean())
    # print(nc_attr_soft.mean())
    # print(nc_obj.mean())

    return (nc_attr_hard * nc_obj).mean(), (nc_attr_soft * nc_obj).mean()

"""
The following functions are adapted from https://github.com/tfzhou/ProtoSeg
"""

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def sinkhorn_knopp(out, n_iterations=3, epsilon=0.05, use_gumbel=False):
    L = torch.exp(out / epsilon).t()  # shape: [K, B,]
    K, B = L.shape

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(n_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indices = torch.argmax(L, dim=1)
    if use_gumbel:
        L = F.gumbel_softmax(L, tau=0.5, hard=True)
    else:
        L = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return L, indices


def greedy_sinkhorn(out, sinkhorn_iterations=50, epsilon=0.05, use_gumbel=False):
    L = torch.exp(out / epsilon).t()
    K = L.shape[0]
    B = L.shape[1]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    r = torch.ones((K,), dtype=L.dtype).to(L.device) / K
    c = torch.ones((B,), dtype=L.dtype).to(L.device) / B

    r_sum = torch.sum(L, axis=1)
    c_sum = torch.sum(L, axis=0)

    r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
    c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    for _ in range(sinkhorn_iterations):
        i = torch.argmax(r_gain)
        j = torch.argmax(c_gain)
        r_gain_max = r_gain[i]
        c_gain_max = c_gain[j]

        if r_gain_max > c_gain_max:
            scaling = r[i] / r_sum[i]
            old_row = L[i, :]
            new_row = old_row * scaling
            L[i, :] = new_row

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)
        else:
            scaling = c[j] / c_sum[j]
            old_col = L[:, j]
            new_col = old_col * scaling
            L[:, j] = new_col

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    L = L.t()

    indices = torch.argmax(L, dim=1)
    # G = F.gumbel_softmax(L, tau=0.5, hard=True)
    if use_gumbel:
        L = F.gumbel_softmax(L, tau=0.5, hard=True)
    else:
        L = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return L, indices

def unbalanced_sinkhorn(out, reg=0.05, reg_m=0.5, n_iterations=20):
    """
    PyTorch implementation of Unbalanced Sinkhorn algorithm.
    out: 相似度矩阵 (logits) shape [N, K]
    reg: 熵正则化系数 (epsilon)
    reg_m: 边缘松弛系数 (kappa)，控制对质量守恒约束的放宽程度
    """
    N, K = out.shape
    # 初始化边缘分布
    a = torch.ones(N, dtype=out.dtype, device=out.device) / N
    b = torch.ones(K, dtype=out.dtype, device=out.device) / K

    # 缩放后的相似度矩阵
    K_matrix = torch.exp(out / reg)

    u = torch.ones(N, dtype=out.dtype, device=out.device)
    v = torch.ones(K, dtype=out.dtype, device=out.device)

    power = reg_m / (reg_m + reg)

    for _ in range(n_iterations):
        u = (a / (torch.matmul(K_matrix, v) + 1e-8)) ** power
        v = (b / (torch.matmul(K_matrix.t(), u) + 1e-8)) ** power

    # 计算最终的最优传输耦合矩阵
    T = u.unsqueeze(1) * K_matrix * v.unsqueeze(0)
    return T

def save_loss_for_split(state, filename="loss_for_split.pth.tar", prefix=""):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print("model save {} failed, remaining {} trials".format(filename, tries))
        if not tries:
            raise error

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"