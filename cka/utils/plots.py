import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict


def heatmap(ax, data, metadata, cmap="inferno"):
    """
    data is in a [N, M] grid shape, where for each value of n (y-axis), we
    have data for all values of m (x-axis). We invert the y-axis to start plotting
    from bottom left (imshow by default plots from top left)
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 2
    fig = ax.imshow(data, cmap=cmap, vmin=metadata.vmin, vmax=metadata.vmax)
    ax.set_xlabel(metadata.xlabel, fontsize=15)
    ax.set_ylabel(metadata.ylabel, fontsize=15)
    ax.set_xticks(metadata.xticks)
    ax.set_yticks(metadata.yticks)
    ax.set_title(metadata.title, fontsize=15)
    ax.invert_yaxis()
    ax.grid(False)
    return fig


def plot_helper(cka_dir, graphs, title=None):
    fig, axs = plt.subplots(1, 1, figsize=(3 * 3, 3), sharey=True)
    titles = ["NonRobust-Robust", "Robust-Robust(Adv)", "Robust-Robust(Ben)"]

    for i in range(graphs):
        if i == 0:
            cka = torch.load(
                os.path.join(
                    cka_dir,
                    "Model-adv-False_True-identicalTrue-layers_all_all-.pt",
                )
            )["cka"]
        elif i == 1:
            cka = torch.load(
                os.path.join(
                    cka_dir,
                    "Model-adv-True_True-layers_all_all-.pt",
                )
            )["cka"]
        elif i == 2:
            cka = torch.load(
                os.path.join(
                    cka_dir,
                    "Model-adv-False_False-layers_all_all-.pt",
                )
            )["cka"]
        else:
            raise ValueError

        metadata = EasyDict(
            {
                "xlabel": "Layer",
                "ylabel": "Layer " if i == 0 else "",
                "xticks": np.arange(0, cka.shape[0],10),
                "yticks": np.arange(0, cka.shape[1],10),
                "title": titles[i],
                "vmin": 0.0,
                "vmax": 1.0,
            }
        )

        # metadata = EasyDict(
        #     {
        #         "xlabel": "",
        #         "ylabel": "",
        #         "xticks": np.arange(0, cka.shape[0],10),
        #         "yticks": np.arange(0, cka.shape[1],10),
        #         "title": "",
        #         "vmin": 0.0,
        #         "vmax": 1.0,
        #     }
        # )
        f = heatmap(axs, cka, metadata)

    fig.colorbar(f, ax=axs, shrink=0.72, format="%.1f")
    # fig.supxlabel("Layer ", x=0.45, y=0.06)
    fig.savefig(
        os.path.join(cka_dir, "./cka_plot_single.jpg"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":

    # exp_path = "/data/output-ai/shruthi.gowda/reinit/reinit_cifar10/madry/nobn"
    # lst_models = os.listdir(exp_path)
    #
    # for model in lst_models:
    #     model_file = os.path.join(exp_path, model, 'cka')
    #     title = model.split('-')[-3].replace('f','B-')
    #     plot_helper(cka_dir=model_file,
    #                 graphs=1, title=title)

    plot_helper(cka_dir="/output/cka", graphs=1)
