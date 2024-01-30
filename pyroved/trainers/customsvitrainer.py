######### This standard svi trainner class is extended to incorporate physics defined loss functions along with VAE loss
import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyroved as pv
import skimage
from skimage import filters
from skimage import color
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, chan_vese, clear_border
from skimage.segmentation import mark_boundaries
from skimage.restoration import denoise_bilateral
import weakref
import torch
import pyro
import pyro.infer as infer
import pyro.optim as optim
import warnings
import pyro.ops.jit
from pyro import poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.abstract_infer import TracePosterior
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import (
    MultiFrameTensor,
    get_plate_stacks,
    is_validation_enabled,
    torch_item,
)
from pyro.util import check_if_enumerated, warn_if_nan
from torch.cuda import random
from typing import Type, Optional, Union
from pyroved.utils.nn import set_deterministic_mode


class custom_SVItrainer:
    """
    Stochastic variational inference (SVI) trainer for
    unsupervised and class-conditioned VED models consisting
    of one encoder and one decoder.
    Args:
        model:
            Initialized model. Must be a subclass of torch.nn.Module
            and have self.model, self.encode (new modification) and self.guide methods
        data:
            Torch.tensor Dataset to encode into latent space (new modification)
        optimizer:
            Pyro optimizer (Defaults to Adam with learning rate 1e-3)
        loss:
            ELBO objective (Defaults to pyro.infer.Trace_ELBO)
        enumerate_parallel:
            Exact discrete enumeration for discrete latent variables
        seed:
            Enforces reproducibility
        weight: 
            float to give slack to loss function, recommdended range 0<w<= 0.5 (new modification)
    
    Keyword Args:
        lr: learning rate (Default: 1e-3)
        device:
            Sets device to which model and data will be moved.
            Defaults to 'cuda:0' if a GPU is available and to CPU otherwise.
    Examples (new modifications):
    Train a model with customed loss SVI trainer using default settings
    >>> # Initialize model
    >>> data_dim = (28, 28)
    >>> trvae = pyroved.models.iVAE(data_dim, latent_dim=2, invariances=['r', 't'])
    >>> # Initialize SVI trainer
    >>> trainer = custom_SVItrainer(trvae, data = train_data) 
    >>> # Train for 200 epochs:
    >>> for _ in range(200):
    >>>     trainer.step(train_loader)
    >>>     trainer.print_statistics()
    Train a model with SVI trainer with a "time"-dependent KL scaling factor
    >>> # Initialize model
    >>> data_dim = (28, 28)
    >>> rvae = pyroved.models.iVAE(data_dim, latent_dim=2, invariances=['r'])
    >>> # Initialize SVI trainer
    >>> trainer = custom_SVItrainer(rvae, data = train_data, weight = 0.1)
    >>> kl_scale = torch.linspace(1, 4, 50) # ramp-up KL scale factor from 1 to 4 during first 50 epochs
    >>> # Train
    >>> for e in range(100):
    >>>     sc = kl_scale[e] if e < len(kl_scale) else kl_scale[-1]
    >>>     trainer.step(train_loader, scale_factor=sc)
    >>>     trainer.print_statistics()
    >>>     trainer.visualize_boundaries(custom_sh_vae)
    >>>     trainer.visualize_manifolds(custom_sh_vae)
    """
    ### New modification: additional input arguments as "data = None(default)"
    # Here data will be the training data (torch.tensor) to encode
    def __init__(self,
                 model: Type[torch.nn.Module],
                 data= None,
                 optimizer: Type[optim.PyroOptim] = None,
                 loss: Type[infer.ELBO] = None,
                 enumerate_parallel: bool = False,
                 seed: int = 1,
                 weight: float =0.5,
                 **kwargs: Union[str, float]
                 ) -> None:
        """
        Initializes the trainer's parameters
        """
        pyro.clear_param_store()
        set_deterministic_mode(seed)
        self.device = kwargs.get(
            "device", 'cuda' if torch.cuda.is_available() else 'cpu')
        ########## Start New lines #########
        if data is None:
            print("Full training data missing. Will work with small batch data")
        else:
            self.data = data
        ########## End New lines #########
        if optimizer is None:
            lr = kwargs.get("lr", 1e-3)
            optimizer = optim.Adam({"lr": lr})
        if loss is None:
            if enumerate_parallel:
                loss = infer.TraceEnum_ELBO(
                    max_plate_nesting=1, strict_enumeration_warning=False)
                #Need to customize for the TraceEnum_ELBO.
            else:
              ########## Modified below line to call custom ELBO class #########
                loss = CustomTrace_ELBO()
                #loss = simple_elbo

        guide = model.guide
        if enumerate_parallel:
            guide = infer.config_enumerate(guide, "parallel", expand=True)
        ########## Modified below lines to define custom SVI class, passing additional arguments as model.encode within model, and data, and initializing two training losses #########  
        self.svi = custom_SVI(model, guide, optimizer, weight, loss=loss, data = self.data)
        self.loss_history = {"training_ELBOloss": [], "training_customloss": [], "training_totalloss": [], "test_loss": []}
        self.current_epoch = 0

 ########## Modified below function to output ELBO and custom loss individually for plot #########
    def train(self,
              train_loader: Type[torch.utils.data.DataLoader],
              **kwargs: float) -> float:
        """
        Trains a single epoch
        """
        
        epoch_ELBOloss = 0.
        epoch_customloss = 0
        epoch_totalloss = 0
        
        
        for data in train_loader:
            if len(data) == 1:  # VAE mode
                x = data[0]
                #print("Passing a batch:")
                #print(x)
                total_loss, loss1, loss2 = self.svi.step(x.to(self.device), **kwargs)
            else:  # VED or cVAE mode
                x, y = data
                loss = self.svi.step(
                    x.to(self.device), y.to(self.device), **kwargs)
            
            epoch_ELBOloss += loss1
            epoch_customloss += loss2
            epoch_totalloss += total_loss
            #print(epoch_ELBOloss, epoch_customloss, epoch_totalloss)

        epoch_ELBOloss_mean = epoch_ELBOloss / len(train_loader.dataset)
        epoch_customloss_mean = epoch_customloss / len(train_loader.dataset)
        epoch_totalloss_mean = epoch_totalloss / len(train_loader.dataset)

        return epoch_totalloss_mean, epoch_ELBOloss_mean, epoch_customloss_mean

    def evaluate(self,
                 test_loader: Type[torch.utils.data.DataLoader],
                 **kwargs: float) -> float:
        """
        Evaluates current models state on a single epoch
        """
        
        test_loss = 0.
        
        with torch.no_grad():
            for data in test_loader:
                if len(data) == 1:  # VAE mode
                    x = data[0]
                    loss = self.svi.step(x.to(self.device), **kwargs)
                else:  # VED or cVAE mode
                    x, y = data
                    loss = self.svi.step(
                        x.to(self.device), y.to(self.device), **kwargs)
                test_loss += loss
        
        return test_loss / len(test_loader.dataset)

 ########## Modified below function to output ELBO and custom loss individually for plot #########
    def step(self,
             train_loader: Type[torch.utils.data.DataLoader],
             test_loader: Optional[Type[torch.utils.data.DataLoader]] = None,
             **kwargs: float) -> None:
        """
        Single training and (optionally) evaluation step
        Args:
            train_loader:
                Pytorch’s dataloader object with training data
            test_loader:
                (Optional) Pytorch’s dataloader object with test data
        
        Keyword Args:
            scale_factor:
                Scale factor for KL divergence. See e.g. https://arxiv.org/abs/1804.03599
                Default value is 1 (i.e. no scaling)
        """
        
        train_totalloss, train_ELBOloss, train_customloss = self.train(train_loader, **kwargs)
        
        self.loss_history["training_ELBOloss"].append(train_ELBOloss)
        self.loss_history["training_customloss"].append(train_customloss)
        self.loss_history["training_totalloss"].append(train_totalloss)
        if test_loader is not None:
            test_loss = self.evaluate(test_loader, **kwargs)
            self.loss_history["test_loss"].append(test_loss)
        self.current_epoch += 1

 ########## Modified below function to print and plot ELBO and custom loss individually #########
    def print_statistics(self) -> None:
        """
        Prints training and test (if any) losses for current epoch
        """
        e = self.current_epoch
        if len(self.loss_history["test_loss"]) > 0:
            template = 'Epoch: {} Training loss: {:.4f}, Test loss: {:.4f}'
            print(template.format(e, self.loss_history["training_loss"][-1],
                                  self.loss_history["test_loss"][-1]))
        else:
            template = 'Epoch: {} Training ELBO loss: {:.4f}, Training custom loss: {:.4f}, Total loss: {:.4f}'
            print(template.format(e, self.loss_history["training_ELBOloss"][-1],
                                  self.loss_history["training_customloss"][-1],
                                  self.loss_history["training_totalloss"][-1]))
            
        #PLotting losses
        fig, ax = plt.subplots(ncols=1, nrows= 1, figsize=(5,5))
        ax.plot(self.loss_history["training_ELBOloss"], 'ro-', linewidth =2, label="ELBO")
        ax.plot(self.loss_history["training_customloss"], 'go-', linewidth =2, label="Customed")
        #ax.plot(self.loss_history["training_totalloss"], 'bo-', linewidth =2, label="AugELBO")
        ax.legend(loc="best")
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.set_title('Loss comparison')
        plt.show()

 ############ New function added to vislualize boundaries of latent images###########
    def visualize_boundaries(self, model, threshold=1e-3) -> None:
        """
        Plot latent manifold for current epoch
        """
        e = self.current_epoch
        data = self.data
        z_mean, z_sd = model.encode(data)
        d1 = int(data.shape[0]**(1/2))
        d2 = d1
        
        z1_org = z_mean[:, 2].reshape(d1,d2)
        z2_org = z_mean[:, 3].reshape(d1,d2)
        z1_org = (z1_org-torch.min(z1_org))/(torch.max(z1_org)-torch.min(z1_org))
        z2_org = (z2_org-torch.min(z2_org))/(torch.max(z2_org)-torch.min(z2_org))

        # Ignoring if the max-min latent image values <= threshold (1e-3)

        if (torch.absolute(torch.max(z_mean[:, 2])-torch.min(z_mean[:, 2])) <= 1e-3):
            z_mean[:, 2] = torch.mean(z_mean[:, 2])

        if (torch.absolute(torch.max(z_mean[:, 3])-torch.min(z_mean[:, 3])) <= 1e-3):
            z_mean[:, 3] = torch.mean(z_mean[:, 3])

        z1 = z_mean[:, 2].reshape(d1,d2)
        z2 = z_mean[:, 3].reshape(d1,d2)
        z1 = (z1-torch.min(z1))/(torch.max(z1)-torch.min(z1))
        z2 = (z2-torch.min(z2))/(torch.max(z2)-torch.min(z2))
        #Denoising the image using bilateral filter

        z1 = denoise_bilateral(z1.detach().numpy(), sigma_color=0.3, sigma_spatial=5)
        z2 = denoise_bilateral(z2.detach().numpy(), sigma_color=0.3, sigma_spatial=5)

        #Multi label segmentation with Multi Otsu threshold
        val1 = filters.threshold_multiotsu(z1)
        emag_z1 = np.digitize(z1, bins=val1)
        val2 = filters.threshold_multiotsu(z2)
        emag_z2 = np.digitize(z2, bins=val2)


        print("Mark Edges of latent images at current epoch")
        #print(np.mean(emag_z1[2]), np.mean(emag_z2[2]))
        plt.figure()
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)

        ax[0, 0].imshow(emag_z1, origin="lower", cmap="jet")
        ax[0, 0].set_title("Edge detection, $z_1$")
        ax[0, 1].imshow(z1_org, origin="lower", cmap="viridis")
        ax[0, 1].set_title("$z_1$")
        ax[1, 0].imshow(emag_z2, origin="lower", cmap="jet")
        ax[1, 0].set_title('Edge detection, $z_2$')
        ax[1, 1].imshow(z2_org, origin="lower", cmap="viridis")
        ax[1, 1].set_title("$z_2$")
        
        #plt.savefig(f"{images_dir}/"+str(e-1)+".png")
        plt.show()


 ############ New function added to vislualize latent manifolds###########
    def visualize_manifolds(self, model) -> None:
        """
        Plot latent manifold for current epoch
        """
        data = self.data
        z_mean, z_sd = model.encode(data)
        d1 = int(data.shape[0]**(1/2))
        d2 = d1

        print("Latent manifolds and images at current epoch")
        custom_sh_vae.manifold2d(d=10)

        fig = plt.figure(constrained_layout=True, figsize=(25, 12))
        gs = fig.add_gridspec(2, 3, wspace=0.8)

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.scatter(z_mean[:, -2], z_mean[:, -1], s=10, alpha=.3)
        ax1 = sns.kdeplot(z_mean[:, -2], z_mean[:, -1], cmap="Reds", shade=True, shade_lowest=False, cut = 0, alpha = 0.6, cbar = True)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.set_xlabel('$z_1$')
        ax1.set_ylabel('$z_2$')


        ax2 = fig.add_subplot(gs[0, 2])
        z_img = z_mean[:, 0].reshape(d1,d2)
        z = ax2.imshow(z_img, origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 = fig.colorbar(z, cax=cax2)
        ax2.axis('off')
        ax2.set_title('$s_1$')

        ax3 = fig.add_subplot(gs[1, 2])
        z_img = z_mean[:, 1].reshape(d1,d2)
        z = ax3.imshow(z_img, origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.1)
        cbar3 = fig.colorbar(z, cax=cax3)
        ax3.axis('off')
        ax3.set_title('$s_2$')

        ax4 = fig.add_subplot(gs[0, 1])
        z_img = z_mean[:, 2].reshape(d1,d2)
        z = ax4.imshow(z_img, origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes("right", size="5%", pad=0.1)
        cbar4 = fig.colorbar(z, cax=cax4)
        ax4.axis('off')
        ax4.set_title('$z_1$')

        ax5 = fig.add_subplot(gs[1, 1])
        z_img = z_mean[:, 3].reshape(d1,d2)
        z = ax5.imshow(z_img, origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes("right", size="5%", pad=0.1)
        cbar5 = fig.colorbar(z, cax=cax5)
        ax5.axis('off')
        ax5.set_title('$z_2$')
        plt.show()


