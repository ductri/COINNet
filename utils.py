from pathlib import Path
import torch
from torch import Tensor
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ray
import wandb
from omegaconf import OmegaConf


def get_loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def count_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


def plot_confusion_matrix(A):
    """
    A: (M, K, K)
    """
    assert A.ndim == 3
    M = A.shape[0]
    fig, axs = plt.subplots(1, M, figsize=(20, 6), dpi=80)
    if M == 1:
        im = axs.imshow(A[0])
        for (j, i), label in np.ndenumerate(A[0]):
            axs.text(i,j,f'{label:.3f}',ha='center',va='center')
        plt.colorbar(im, ax=axs)
    else:
        for m in range(M):
            im = axs[m].imshow(A[m])
            # Loop over data dimensions and create text annotations.
            for (j, i), label in np.ndenumerate(A[m]):
                axs[m].text(i,j,f'{label:.3f}',ha='center',va='center')
            plt.colorbar(im, ax=axs[m])
            # for i in range(A.shape[0]):
            #     for j in range(A.shape[1]):
            #         text = axs[m].text(j, i, A[m, i, j], ha="center", va="center", color="w")
    plt.tight_layout()
    return fig


def plot_distribution(x: Tensor, title: str) -> plt.Figure:
    x = x.detach().cpu()
    batch_size = 1

    fig, axes = plt.subplots(2, max(batch_size // 2, 1), constrained_layout=True)
    axes = axes.flatten()

    for b in range(batch_size):
        hist, edges = torch.histogram(x, density=True)
        axes[b].plot(edges[:-1], hist)

    mean, std = x.mean(), x.std()
    fig.suptitle(f"{title} | Mean: {mean:.4f} Std: {std:.4f}")
    fig.supxlabel("X")
    fig.supylabel("Density")

    return fig


def turn_off_grad(module):
    for j, p in enumerate(module.parameters()):
        p.requires_grad_(False)


def save_model(model, path):
    print(f'Saved model to {path}')
    torch.save(model.state_dict(), path)

def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(f'Restored model from {path}')

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f'Created path {path}')


def create_ray_wrapper(main_func):
    """
    main_func(conf)
    """
    project_name = 'shahana_outlier'
    @ray.remote(num_cpus=12, num_gpus=1)
    def ray_wrapper(conf):
            main_func(conf)

    def my_wrapper(conf):
        ray.init(address='auto')
        ray.get(ray_wrapper.remote(conf))

    return my_wrapper


def create_wandb_wrapper(main_func):
    """
    main_func(conf, unique_name)
    """
    project_name = 'shahana_outlier'
    def wrapper(conf):
        with wandb.init(entity='narutox', project=project_name, tags=conf.tags, config=OmegaConf.to_container(conf, resolve=True)) as run:
            main_func(conf, run.name)
    return wrapper

