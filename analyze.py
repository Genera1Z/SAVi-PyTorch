from glob import glob
import json

import numpy as np
import matplotlib.pyplot as plt


def read_column(log_lines, col_key, is_val=False):
    out_dict = {}
    for log_line in log_lines:
        log_dict = json.loads(log_line.strip())
        assert len(log_dict.keys()) == 1
        idx_key = list(log_dict.keys())[0]
        if is_val and "val" in idx_key:
            idx_key2 = int(idx_key.split("/")[0])
            out_dict[idx_key2] = log_dict[idx_key][col_key]
        elif not is_val and "val" not in idx_key:
            idx_key2 = int(idx_key)
            out_dict[idx_key2] = log_dict[idx_key][col_key]
    return out_dict


def main():
    log_files = glob("./output/*.txt")
    modes = [False, True]
    col_keys = ["mse", "ari", "ari_fg"]
    # groups = ["mse", "ari"]

    # read data points
    curves = {}
    for mode in modes:
        for col_key in col_keys:
            records = []
            for log_file in log_files:
                with open(log_file, "r") as f:
                    log_content = f.readlines()
                log_dict = read_column(log_content, col_key, mode)
                records.append(log_dict)
            suffix = "train" if not mode else "val"
            curves[f"{col_key}-{suffix}"] = records

    # calculate mean and std
    for key, item in curves.items():
        xs = [list(_.keys()) for _ in item]
        assert all(xs[0] == _ for _ in xs[1:])
        trials = [list(_.values()) for _ in item]
        mean = np.mean(trials, axis=0)
        std = np.std(trials, axis=0)
        curves[key] = [xs[0], mean, std]

    # plot the curves
    # _, axs = plt.subplots(len(groups))
    # axs = axs.flatten()
    # for group, ax in zip(groups, axs):
    #     ax.set_title(group)
    #     for label, data in curves.items():
    #         if group in label:
    #             ax.plot(data[0], data[1], label=label)
    #             ax.fill_between(data[0], -data[2], data[2], alpha=0.3)
    #     ax.legend()
    # plt.show()
    _, axs = plt.subplots(len(modes), len(col_keys))
    axs = axs.flatten()
    for ax, (title, (xs, mean, std)) in zip(axs, curves.items()):
        ax.set_title(title)
        ax.plot(xs, mean)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.show()


def main_viz_lr():
    log_file = "./output/23989.txt"
    col_key = "lr"

    # read data points
    with open(log_file, "r") as f:
        log_content = f.readlines()
    log_dict = read_column(log_content, col_key, False)

    xs = list(log_dict.keys())
    ys = list(log_dict.values())
    plt.plot(xs, ys)
    plt.show()


if __name__ == "__main__":
    main()
    # main_viz_lr()
