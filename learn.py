import json
import pathlib as pl

import numpy as np
import torch as pt
import torch.cuda.amp as ptca
import torch.nn.functional as ptnf
import torch.optim as pto
import torch.utils.tensorboard as ptutb

from utils import register_module


#### optimizers


Adam = pto.Adam
GradScaler = ptca.grad_scaler.GradScaler


# class AmpAdam(mo.AmpOptimWrapper):
#     """"""
#     def __init__(
#         self,
#         loss_scale="dynamic",
#         dtype=None,
#         use_fsdp=False,
#         named_params: dict = ...,
#         lr=1e-4,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#         weight_decay=0,
#         accumulative_counts=1,
#         clip_grad=dict(max_norm=0.05),
#         custom={
#             "^encoder\.backbone\..*conv\d+\.weight$*": dict(lr_mult=1e-3, decay_mult=1),
#             "^encoder\.backbone\..*conv\d+\.bias$*": dict(lr_mult=1e-3, decay_mult=1),
#             "^encoder\.backbone\..*bn\d+.weight*": dict(lr_mult=1, decay_mult=1),
#             "^encoder\.backbone\..*bn\d+.bias*": dict(lr_mult=1, decay_mult=1),
#         },
#     ):
#         optim = pto.Adam(named_params.copy().values(), lr, betas, eps, weight_decay)
#         super().__init__(
#             loss_scale,
#             dtype,
#             use_fsdp,
#             optimizer=optim,
#             accumulative_counts=accumulative_counts,
#             clip_grad=clip_grad,
#         )
#         # self._split_param_groups()
#         # self._custom_hyperparams(list(named_params.items()), custom)
#     def _split_param_groups(self):
#         """
#         Split ``optimizer.param_groups`` into list of dicts of ``{param, lr, weight_decay,..}``.
#         """
#         assert (
#             isinstance(self.optimizer.param_groups, list)
#             and len(self.optimizer.param_groups) == 1
#         )
#         param_groups2 = []
#         common = self.optimizer.param_groups[0].copy()
#         for param in common.pop("params"):
#             param_group = {"params": param, **common}
#             param_groups2.append(param_group)
#         self.optimizer.param_groups = param_groups2
#     def _custom_hyperparams(self, named_params: list, custom: dict):
#         """
#         Modify training hyper-params of model parameters finely according to ``custom`` dict.
#         """
#         assert len(self.optimizer.param_groups) == len(named_params)
#         for param_group, (name, param) in zip(
#             self.optimizer.param_groups, named_params
#         ):
#             assert id(param_group["params"]) == id(param)
#             matches = []
#             key0 = None
#             for pattern in custom.keys():
#                 match = re.findall(pattern, name)
#                 if len(match) > 0:
#                     matches.append(match)
#                     key0 = pattern
#             if len(matches) == 0:
#                 continue
#             assert len(matches) == 1 and len(matches[0]) == 1
#             assert name == matches[0][0]
#             param_group["lr"] *= custom[key0].get("lr_mult", 1)
#             param_group["weight_decay"] *= custom[key0].get("decay_mult", 1)
#             print(name, key0, custom[key0])


#### callbacks


LinearLR = pto.lr_scheduler.LinearLR


CosineAnnealingLR = pto.lr_scheduler.CosineAnnealingLR


SequentialLR = pto.lr_scheduler.SequentialLR


class LambdaLR(pto.lr_scheduler.LambdaLR):
    """"""

    def __init__(self, optimizer, lr_lambda):
        lr_lambda = self.map_fn(lr_lambda)
        super().__init__(optimizer, lr_lambda)

    def map_fn(self, fn_dict):
        fn_dict = fn_dict.copy()
        fn0 = getattr(self, fn_dict.pop("fn"))

        def new_fn(lr0, cnt):
            return fn0(lr0, cnt, **fn_dict)

        return new_fn

    def get_lr(self):
        return [
            lmbd(lr0, self._step_count)
            for lmbd, lr0 in zip(self.lr_lambdas, self.base_lrs)
        ]

    @staticmethod
    def linear_cosine(
        base_lr, last_iter, warmup_step, total_step, start_factor=0, eta_min=0
    ):
        if last_iter < warmup_step:
            return LambdaLR.linear(
                base_lr, last_iter, warmup_step, start_factor, end_factor=1
            )
        # last_iter starts from 1 and ends at total_step+1
        elif warmup_step <= last_iter <= total_step + 1:
            return LambdaLR.cosine(
                base_lr, last_iter - warmup_step, total_step - warmup_step, eta_min
            )
        else:
            raise "ValueError"

    @staticmethod
    def linear(base_lr, last_iter, max_iter, start_factor, end_factor=1):
        return base_lr * (
            start_factor + (end_factor - start_factor) / max_iter * last_iter
        )

    @staticmethod
    def cosine(base_lr, last_iter, max_iter, eta_min):
        return (
            eta_min
            + (base_lr - eta_min) * (1 + np.cos(np.pi * last_iter / max_iter)) / 2
        )


class AverageLog:
    """"""

    def __init__(self, log_file):
        self.log_file = log_file
        self._info_dict = {}
        self._idx = None
        self._current_dict = None

    def index(self, epoch):
        assert epoch not in self._info_dict
        self._idx = epoch
        self._current_dict = {}

    def append(self, kwds):
        for key, value in kwds.items():
            value = value.detach().cpu()
            if key in self._info_dict:
                self._current_dict[key].append(value)
            else:
                self._current_dict[key] = [value]

    def step(self, **kwargs):
        avg_dict = {}
        for k, v in self._current_dict.items():
            avg_dict[k] = np.mean(v).item()
        self._info_dict[self._idx] = avg_dict.copy()
        if kwargs:
            avg_dict.update(kwargs)
        line = json.dumps({self._idx: avg_dict})
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
        return avg_dict


SummaryWriter = ptutb.SummaryWriter


class SaveBestModel:
    """"""

    def __init__(self, save_dir, mode="min"):
        self.save_dir = save_dir
        self.mode = mode
        self.best_score = np.inf if mode == "min" else 0

    def save(self, epoch, score, model):
        if self.mode == "min":
            if self.best_score <= score:
                return
        elif self.mode == "max":
            if self.best_score >= score:
                return
        else:
            raise "ValueError"
        pt.save(
            {"model": model, "score": score},
            pl.Path(self.save_dir) / f"{epoch:04d}.pth",
        )
        self.best_score = score


#### metrics


class MetricDict:
    """"""

    def __init__(self, grad: bool, **metrics):
        self.metric = {k: self.map_fn(v) for k, v in metrics.items()}
        self.grad = grad

    def map_fn(self, fn_dict):
        fn_dict = fn_dict.copy()
        fn0 = getattr(self, fn_dict.pop("fn"))
        w = fn_dict.pop("weight")
        map1 = fn_dict.pop("output")
        map2 = fn_dict.pop("batch")

        def new_fn(output, batch):
            if self.grad:
                input1 = {k: output[v] for k, v in map1.items()}
                device = list(input1.values())[0].device
                input2 = {k: batch[v].to(device) for k, v in map2.items()}
            else:
                input2 = {k: batch[v] for k, v in map2.items()}
                device = list(input2.values())[0].device
                input1 = {
                    k: output[v].detach().to(device) for k, v in map1.items()
                }  # .clone().detach()
            return w * fn0(**input1, **input2, **fn_dict)

        return new_fn

    def __call__(self, output: dict, batch: dict) -> dict:
        if self.grad:
            return {k: v(output, batch) for k, v in self.metric.items()}
        else:
            with pt.no_grad():
                return {k: v(output, batch) for k, v in self.metric.items()}

    @staticmethod
    def huber(pred, true, reduction="mean", delta=1.0):
        return ptnf.huber_loss(pred, true, reduction, delta=delta)

    @staticmethod
    def mse(pred, true, reduction="mean"):  # mean all -- space & time, as well as batch
        return ptnf.mse_loss(pred, true, reduction=reduction)

    @staticmethod
    def ari(idx_pd, idx_gt, num_pd, num_gt, mask=None, fg=False, reduction="mean"):
        """
        Computes the adjusted Rand index (ARI), a clustering similarity score.
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

        Arguments
        ---
        - idx_pd: predicted cluster assignment, encoded as integer indexes, in shape (b,t,h,w)
        - idx_gt: ground-truth cluster assignment, encoded as integer indexes, in shape (b,t,h,w)
        - num_pd: maximum number of predicted instances, default 24
        - num_gt: maximum number of ground-truth instances, default 24
        - mask: ones/zeros hinting where are valid/invalid, in shape (b,t,h,w)
        - fg: remove the background or not

        Returns
        ---
        - ari value in shape ()
        """
        assert len(idx_pd.shape) == len(idx_gt.shape) == 4
        pred_oh = ptnf.one_hot(idx_pd, num_pd).char()  # bthwc
        true_oh = ptnf.one_hot(idx_gt, num_gt).char()  # bthwd

        if mask is not None:
            assert mask.dtype == pt.uint8
            true_oh = true_oh * mask[..., None]
        if fg:
            true_oh = true_oh[..., 1:]  # remove background

        N = pt.einsum("bthwc,bthwd->bcd", true_oh.double(), pred_oh.double())
        A = pt.sum(N, dim=-1)  # (b,c)
        B = pt.sum(N, dim=-2)  # (b,d)
        num_point = pt.sum(A, dim=1)  # (b,)

        idx_r = pt.sum(N * (N - 1), dim=[1, 2])
        idx_a = pt.sum(A * (A - 1), dim=1)
        idx_b = pt.sum(B * (B - 1), dim=1)

        idx_r_exp = idx_a * idx_b / pt.clip(num_point * (num_point - 1), 1)
        idx_r_max = (idx_a + idx_b) / 2
        denominat = idx_r_max - idx_r_exp
        ari = (idx_r - idx_r_exp) / denominat

        # the denominator can be zero:
        # - both pred & true idxs assign all pixels to one cluster
        # - both pred & true idxs assign max 1 point to each cluster
        ari.masked_fill_(denominat == 0, 1)
        assert reduction == "mean"
        return ari.mean(dim=0)

    @staticmethod
    def distinct(slots, mask, margin=0.5, reduction="mean"):
        """
        slots: (bt,n,c)
        mask: (bt,n)
        """
        simi = ptnf.cosine_similarity(slots[:, :, None], slots[:, None, :], dim=3)
        mask = pt.sub(
            mask[:, :, None] * mask[:, None, :],
            pt.stack([pt.diag(pt.ones(mask.size(1)))] * mask.size(0), dim=0).to(mask),
        ).to(slots)
        mean = (simi * mask).sum([1, 2]) / mask.sum([1, 2])
        loss = pt.maximum(mean, pt.tensor(margin).to(slots))
        assert reduction == "mean"
        return loss.mean()


[register_module(_) for _ in locals().values() if isinstance(_, type)]
