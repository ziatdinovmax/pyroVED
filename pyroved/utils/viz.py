from typing import Union, List
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def plot_img_grid(imgdata: torch.Tensor, d: int,
                  **kwargs: Union[str, int, List[float]]) -> None:
    """
    Plots a square grid of images
    """
    grid = make_grid(imgdata[:, None], nrow=d,
                     padding=kwargs.get("padding", 2),
                     pad_value=kwargs.get("pad_value", 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(grid[0], cmap=kwargs.get("cmap", "gnuplot"),
               origin=kwargs.get("origin", "upper"),
               extent=kwargs.get("extent"))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.show()


def plot_spect_grid(spectra: torch.Tensor, d: int):  # TODO: Add 'axes' and 'extent'
    _, axes = plt.subplots(d, d, figsize=(8, 8),
                           subplot_kw={'xticks': [], 'yticks': []},
                           gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for ax, y in zip(axes.flat, spectra):
        ax.plot(y)
    plt.show()
