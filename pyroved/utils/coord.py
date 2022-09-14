from typing import Union, Tuple
import torch
import pyro.distributions as dist
tt = torch.tensor


def grid2xy(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    X = torch.cat((X1[None], X2[None]), 0)
    d0, d1 = X.shape[0], X.shape[1] * X.shape[2]
    X = X.reshape(d0, d1).T
    return X


def imcoordgrid(im_dim: Tuple[int]) -> torch.Tensor:
    xx = torch.linspace(-1, 1, im_dim[0])
    yy = torch.linspace(1, -1, im_dim[1])
    x0, x1 = torch.meshgrid(xx, yy, indexing='ij')
    return grid2xy(x0, x1)


def generate_grid(data_dim: Tuple[int]) -> torch.Tensor:
    """Generates 1D or 2D grid of coordinates. Returns a torch tensor with two
    axes. If the input data_dim indicates only one dimensional data, then the
    output will be a 2d torch tensor artificially augmented along the last
    dimension, of shape [N, 1].

    Args:
        data_dim:
            Dimensions of the input data.

    Raises:
        NotImplementedError:
            If the dimension (length) of the provided data_dim is not equal to
            1 or 2.

    Returns:
        The grid (always 2d).
    """

    if len(data_dim) not in [1, 2]:
        raise NotImplementedError("Currently supports only 1D and 2D data")
    if len(data_dim) == 1:
        return torch.linspace(1, -1, data_dim[0])[:, None]
    return imcoordgrid(data_dim)


def transform_coordinates(coord: torch.Tensor,
                          phi: Union[torch.Tensor, float] = 0,
                          coord_dx: Union[torch.Tensor, float] = 0,
                          scale: Union[torch.Tensor, float] = 1.,
                          ) -> torch.Tensor:
    """
    Rotation of 2D coordinates followed by scaling and translation.
    For 1D grid, there is only transaltion. Operates on batches.
    """
    if coord.shape[-1] == 1:
        return coord + coord_dx
    coord = rotate_coordinates(coord, phi)
    coord = scale_coordinates(coord, scale)
    return coord + coord_dx


def rotate_coordinates(coord: torch.Tensor,
                       phi: Union[torch.Tensor, float] = 0
                       ) -> torch.Tensor:
    """
    Rotation of 2D coordinates. Operates on batches
    """
    if torch.sum(phi) == 0:
        phi = coord.new_zeros(coord.shape[0])
    rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
    rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
    coord = torch.bmm(coord, rotmat)
    return coord


def scale_coordinates(coord: torch.Tensor,
                      scale: torch.Tensor
                      ) -> torch.Tensor:
    """
    Scaling of 2D coordinates. Operates on batches
    """
    scalemat = coord.new_zeros(coord.shape[0], 2, 2)
    scalemat[:, 0, 0] = scale
    scalemat[:, 1, 1] = scale
    coord = torch.bmm(coord, scalemat)
    return coord


def generate_latent_grid(d: int, **kwargs) -> torch.Tensor:
    """
    Generates a grid of latent space coordinates
    """
    if isinstance(d, int):
        d = [d, d]
    z_coord = kwargs.get("z_coord")
    if z_coord:
        z1, z2, z3, z4 = z_coord
        grid_x = torch.linspace(z2, z1, d[0])
        grid_y = torch.linspace(z3, z4, d[1])
    else:
        grid_x = dist.Normal(0, 1).icdf(torch.linspace(0.95, 0.05, d[0]))
        grid_y = dist.Normal(0, 1).icdf(torch.linspace(0.05, 0.95, d[1]))
    z = []
    for xi in grid_x:
        for yi in grid_y:
            z.append(tt([xi, yi]).float().unsqueeze(0))
    return torch.cat(z), (grid_x, grid_y)


def generate_latent_grid_traversal(d: int, cont_dim: int, disc_dim,
                                   cont_idx: int, cont_idx_fixed: int,
                                   num_samples: int) -> Tuple[torch.Tensor]:
    """
    Generates continuous and discrete grids for latent space traversal
    """
    # Get continuous latent coordinates
    samples_cont = torch.zeros(size=(num_samples, cont_dim)) + cont_idx_fixed
    cont_traversal = dist.Normal(0, 1).icdf(torch.linspace(0.95, 0.05, d))
    for i in range(d):
        for j in range(d):
            samples_cont[i * d + j, cont_idx] = cont_traversal[j]
    # Get discrete latent coordinates
    n = torch.arange(0, disc_dim)
    n = n.tile(d // disc_dim + 1)[:d]
    samples_disc = []
    for i in range(d):
        samples_disc_i = torch.zeros((d, disc_dim))
        samples_disc_i[:, n[i]] = 1
        samples_disc.append(samples_disc_i)
    samples_disc = torch.cat(samples_disc)
    return samples_cont, samples_disc
