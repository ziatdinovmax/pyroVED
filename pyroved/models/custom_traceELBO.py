######### We extend the standard trace ELBO class to incorporate the inclusion of the physics driven loss functions 
###########and augmenting the physics driven loss with VAE loss during training process

import math
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyroved as pv
import atomai as aoi
import skimage
from skimage import filters
from skimage import color
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, chan_vese, clear_border
from skimage.segmentation import mark_boundaries
from skimage.restoration import denoise_bilateral
from scipy import fftpack
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


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_plate_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r


class CustomTrace_ELBO(ELBO):


    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(
                guide_trace.log_prob_sum()
            )
            elbo += elbo_particle / self.num_particles
        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

########################################################################################################
# This below function is modified to augment custom loss with ELBO loss -- to generate augmented loss
    def _differentiable_loss_particle(self, model_trace, guide_trace, encode, data, w1, w2):
        elbo_particle = 0
        aug_particle = 0
        surrogate_aug_particle = 0
        log_r = None
        custom_loss = self.add_loss(encode, data) # Call any stated loss functions -Here we have 2 loss functions

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                #custom_loss = self.add_loss(encode, data)
                elbo_particle = elbo_particle + (torch_item(site["log_prob_sum"]))
                aug_particle = aug_particle + ((torch_item(site["log_prob_sum"])) * (w1+custom_loss))
                surrogate_aug_particle = surrogate_aug_particle + ((site["log_prob_sum"]) * (w1+custom_loss))
            #i=i+1

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                #custom_loss = self.add_loss(encode, data)
                log_prob, score_function_term, entropy_term = site["score_parts"]
                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])
                aug_particle = aug_particle - ((torch_item(site["log_prob_sum"])) * (w1+custom_loss))

                if not is_identically_zero(entropy_term):
                    surrogate_aug_particle = (
                        surrogate_aug_particle - ((entropy_term.sum()) * (w1+custom_loss))
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_aug_particle = (
                        surrogate_aug_particle + (site * score_function_term).sum()
                    )
            #j = j+1

        return -aug_particle, -surrogate_aug_particle, -elbo_particle, (custom_loss)*10**4

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.0
        surrogate_loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")

        return loss + (surrogate_loss - torch_item(surrogate_loss))

########################################################################################################
# This below functions are added which calculate any additional loss -- other than ELBO
# Here the loss function is edge magnitude of latent images using scharr filter, after denoising and otsu threshold of original latent images
    def add_loss(self, encode, data):
        """
        Computes the additional loss term to augment with ELBO loss
        Input training data (only if we run expensive encoding with full data) as ndarray and model encoder of VAE
        Note: The setup is currently with encoding only small batch data as defined train_loader 
        Output scalar value of loss
        """
        z_mean, z_sd = encode(data)
        d1 = int(data.shape[0]**(1/2))
        d2 = d1
        loss = 0
        if (d1*d2 == data.shape[0]): #Check if the batch data is squared
            
            z1 = z_mean[:, 2].reshape(d1,d2)
            z2 = z_mean[:, 3].reshape(d1,d2)
            z1 = (z1-torch.min(z1))/(torch.max(z1)-torch.min(z1))
            z2 = (z2-torch.min(z2))/(torch.max(z2)-torch.min(z2))
            #Denoising the image using bilateral filter

            z1 = denoise_bilateral(z1.detach().numpy(), sigma_color=None, sigma_spatial=5)
            z2 = denoise_bilateral(z2.detach().numpy(), sigma_color=None, sigma_spatial=5)

            #Edge magnitudeof latent image using Scharr transform
            emag_z1 = filters.scharr(z1)
            emag_z2 = filters.scharr(z2)
            #emag_z1 = slic(z1_rgb, start_label=1)
            #emag_z2 = slic(z2_rgb, start_label=1)
            #Cal additional loss as the mean of the sum of edge magnitude of latent images (We want to minimize the magnitude)

            loss = (np.sum(emag_z1) + np.sum(emag_z2))/2
            #loss = max(np.sum(emag_z1), np.sum(emag_z2))/total_emag
            #loss = max(np.sum(emag_z1), np.sum(emag_z2)) - min(np.sum(emag_z1), np.sum(emag_z2))
            #loss = (len(np.unique(emag_z1)) + len(np.unique(emag_z2)))/2
            loss = loss #Scaled with ELBO
            #print(loss)
            
        return loss

# Here the loss function is mean ratio of intensity of outside central peak and total intensity, after FFT of  original latent images
    def add_loss2(self, encode, data):
        z_mean, z_sd = encode(data)
        d1 = int(data.shape[0]**(1/2))
        d2 = d1
        loss = 0
        if (d1*d2 == data.shape[0]): #Check if the batch data is squared
            
            z1 = z_mean[:, 2].reshape(d1,d2)
            z2 = z_mean[:, 3].reshape(d1,d2)
            
            image_FFTz1 = np.array(z1)
            image_FFTz2 = np.array(z2)
            Fz1 = fftpack.fft2(image_FFTz1)
            Fz2 = fftpack.fft2(image_FFTz2)
            F_shiftz1 = fftpack.fftshift(Fz1)
            F_shiftz2 = fftpack.fftshift(Fz2)
            F_shiftz1 = np.log(np.abs(F_shiftz1)+ 1)
            F_shiftz2 = np.log(np.abs(F_shiftz2) + 1)


            total_int = (np.sum(F_shiftz1) + np.sum(F_shiftz2))/2
            lim_kx_min, lim_kx_max = int(d1/2 - 1), int(d1/2 + 1)
            lim_ky_min, lim_ky_max = int(d2/2 - 1), int(d2/2 + 1)

            # Central peak intensity
            central_peak_intz1 = np.sum(F_shiftz1[lim_kx_min : lim_kx_max, lim_ky_min : lim_ky_max])
            central_peak_intz2 = np.sum(F_shiftz2[lim_kx_min : lim_kx_max, lim_ky_min : lim_ky_max])

            # Intensity outside central peak
            outcentral_peak_intz1 = np.sum(F_shiftz1) - central_peak_intz1
            outcentral_peak_intz2 = np.sum(F_shiftz2) - central_peak_intz2
            #print(central_peak_intz1, central_peak_intz2, F_shiftz1.mean(), F_shiftz2.mean())
            outcentral_peak_int_mean = (outcentral_peak_intz1 + outcentral_peak_intz2)/2
            #Cal additional loss as the mean ratio of the sum of intensity of outside central peak of FFT of latent images (We want to minimize the magnitude)
            loss = outcentral_peak_int_mean/total_int
            loss = loss

        return loss

########################################################################################################
# This below function is modified to call function of custom loss
    def loss_and_grads(self, model, guide, data, weight, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO+additional elements
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        elbo_loss = 0
        surrogate_loss_particle =0.0
        ########## Start New lines #########
        encode = model.encode
        M = model.model
        # Here we use the subset data in batch to build latent image-- Fast to execute
        # To encode the full data each eval, use the data params but this seems too slow to run
        data= args[0].data
        data = torch.reshape(data,(data.shape[0],data.shape[2], data.shape[2]))
        w1 = weight

        ########## End New lines ##########


        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(M, guide, args, kwargs):
            
            augloss_particle, surrogate_loss_particle, loss_particle, custom_loss = self._differentiable_loss_particle(
                model_trace, guide_trace, encode, data, w1, w2
            )
            
            ########### Modified below two lines to add custom_loss to ELBO loss- Weighted total loss
            loss += ((augloss_particle)) / self.num_particles
            elbo_loss += ((loss_particle)) / self.num_particles

            #surrogate_loss_particle = surrogate_loss_particle
            ########## End New lines ##########

 
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        #print("eval")
        ########## return ELBO and custom loss individually ##########

        return loss, elbo_loss, custom_loss 
