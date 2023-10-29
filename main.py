import argparse as ap
import pathlib as pl

import numpy as np
import torch as pt
import tqdm

from datum import DataLoader
from learn import GradScaler, AverageLog, SaveBestModel
from model import ModelHelper
from utils import Config, build_from_config


def train_loop(epoch, model, dataset, losses_fn, metrics_fn, optim, gscale, callbacks):
    model.train()
    (sched_lr, avg_metr) = callbacks
    avg_metr.index(epoch)

    for batch in tqdm.tqdm(dataset):
        with pt.autocast("cuda", pt.float16, enabled=True):
            output = model(batch)
            losses = losses_fn(output, batch)

        gscale.scale(sum(losses.values())).backward()
        gscale.unscale_(optim)
        pt.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.05)  # TODO XXX
        gscale.step(optim)
        gscale.update()
        optim.zero_grad()

        metrics = metrics_fn(output, batch)

        sched_lr.step()
        avg_metr.append(losses)
        avg_metr.append(metrics)

    lr = sched_lr.get_last_lr()[0]
    log_info = avg_metr.step(lr=lr)
    print("#" * 10, epoch, log_info)


@pt.no_grad()
def eval_loop(epoch, model, dataset, losses_fn, metrics_fn, callbacks):
    model.eval()
    (avg_metr, save_mdl) = callbacks
    avg_metr.index(f"{epoch}/val")

    for batch in tqdm.tqdm(dataset):
        with pt.autocast("cuda", pt.float16, enabled=True):
            output = model(batch)
            losses = losses_fn(output, batch)

        metrics = metrics_fn(output, batch)

        avg_metr.append(losses)
        avg_metr.append(metrics)

    log_info = avg_metr.step()
    save_mdl.save(epoch, log_info["ari_fg"], model)
    print(" #" * 5, epoch, log_info)


def main(args):
    print(args)
    cfg = Config.fromfile(args.config)

    pt.manual_seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf

    tf.random.set_seed(args.seed)

    save_dir = pl.Path(args.save_dir) / str(args.seed)
    save_dir.mkdir(parents=True)

    ## datum

    cfg.dataset_train.data_file = pl.Path(args.data_dir) / "train"
    cfg.dataset_val.data_file = pl.Path(args.data_dir) / "val"
    dataset_t = build_from_config(cfg.dataset_train)
    dataset_v = build_from_config(cfg.dataset_val)

    dload_t = DataLoader(
        dataset_t,
        batch_size=cfg.batch_size_train,
        shuffle=True,
        num_workers=cfg.num_worker_train,
        collate_fn=dataset_t.collate_fn,
    )
    dload_v = DataLoader(
        dataset_v,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_worker_val,
        collate_fn=dataset_v.collate_fn,
    )

    ## model

    model = build_from_config(cfg.model)
    # model = pt.compile(model)  # got strange error
    model = model.cuda()
    print(model)

    model = ModelHelper(model, cfg.model_in_keys, cfg.model_out_keys, "cuda")

    ## learn

    losses = build_from_config(cfg.losses)
    metrics = build_from_config(cfg.metrics)

    cfg.optimizer.params = model.parameters()
    optim = build_from_config(cfg.optimizer)
    gscale = GradScaler(enabled=True)

    cfg.sched_lr.lr_lambda.total_step = len(dload_t) * cfg.num_epoch
    print(f"total step: {cfg.sched_lr.lr_lambda.total_step}")
    cfg.sched_lr.optimizer = optim

    sched_lr = build_from_config(cfg.sched_lr)
    avg_metr = AverageLog(f"{save_dir}.txt")
    save_mdl = SaveBestModel(save_dir, mode="max")

    ## train

    for epoch in range(cfg.num_epoch):
        train_loop(
            epoch, model, dload_t, losses, metrics, optim, gscale, [sched_lr, avg_metr]
        )
        if (epoch + 1) % cfg.val_num_epoch == 0:
            eval_loop(epoch, model, dload_v, losses, metrics, [avg_metr, save_mdl])
        # if epoch >= 10:
        #     exit()


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/savi_small-movi_a.py")
    parser.add_argument("--seed", type=int, default=np.random.randint(2**16))
    parser.add_argument(
        "--data_dir", type=str, default="/media/GeneralZ/Storage/Static/datasets/movi_a"
    )
    parser.add_argument("--save_dir", type=str, default="./output")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
