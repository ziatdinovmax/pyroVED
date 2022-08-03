from typing import Union, List, Tuple
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def plot_img_grid(
    imgdata: torch.Tensor, d: int, figsize: Tuple[float] = (8.0, 8.0),
    **kwargs: Union[str, int, List[float]]
) -> None:
    """Plots a *d*-by-*d* square grid of 2D images."""

    if imgdata.ndim < 3:
        raise AssertionError("Images must be passed as a 3D or 4D tensor")
    imgdata = imgdata[:, None] if imgdata.ndim == 3 else imgdata
    grid = make_grid(
        imgdata, nrow=d, padding=kwargs.get("padding", 2),
        pad_value=kwargs.get("pad_value", 0)
    )

    plt.figure(figsize=figsize)
    plt.imshow(grid[0].squeeze(), cmap=kwargs.get("cmap", "gnuplot"),
               origin=kwargs.get("origin", "upper"),
               extent=kwargs.get("extent"))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.show()


def plot_spect_grid(
    spectra: torch.Tensor, d: int, figsize: Tuple[float] = (8.0, 8.0),
    **kwargs: List[float]
) -> None:
    """Plots a *d*-by-*d* square grid with 1D spectral plots.

    TODO: Add 'axes' and 'extent'
    """

    _, axes = plt.subplots(
        d, d, figsize=figsize,
        subplot_kw={'xticks': [], 'yticks': []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1)
    )
    ylim = kwargs.get("ylim")
    for ax, y in zip(axes.flat, spectra):
        ax.plot(y.squeeze())
        if ylim:
            ax.set_ylim(*ylim)
    plt.show()


def plot_grid_traversal(imgdata: torch.Tensor, d: int,
                        data_dim: Tuple[int], disc_dim: int,
                        **kwargs: Union[str, int, List[float]]
                        ) -> None:
    """
    Plots a *disc_dim*-by-*d* grid of 2D images
    """
    if imgdata.ndim < 3:
        raise AssertionError("Images must be passed as a 3D or 4D tensor")
    imgdata = imgdata[:, None] if imgdata.ndim == 3 else imgdata
    grid = make_grid(imgdata, nrow=d,
                     padding=kwargs.get("padding", 2),
                     pad_value=kwargs.get("pad_value", 0))
    grid = grid[0][:(data_dim[0]+kwargs.get("padding", 2)) * disc_dim]
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=kwargs.get("cmap", "gnuplot"),
               origin=kwargs.get("origin", "upper"),
               extent=kwargs.get("extent"))
    plt.xlabel("$z_{cont}$", fontsize=18)
    plt.ylabel("$z_{disc}$", fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.show()
