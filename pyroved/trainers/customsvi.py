######### This class is modified to pass additional arguments in "model", "data", "weight" as stated in comments under (new modification)
## SVI class is extended to incorporate custom physics driven loss with existing VAE loss during training process 

import warnings

import torch

import pyro
import pyro.optim
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item



class custom_SVI(TracePosterior):
    """
    :param model: Initialized model. Must be a subclass of torch.nn.Module
            and have self.model, self.encode (new modification) methods
    :param guide: the guide (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: ~pyro.optim.optim.PyroOptim
    :param weight: float or list to give weightage to loss function (new modification)
    :param loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
        Pyro provides three built-in losses:
        :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
        :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`, and
        :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.
        See the :class:`~pyro.infer.elbo.ELBO` docs to learn how to implement
        a custom loss.
    :type loss: pyro.infer.elbo.ELBO
    :param data: torch.tensor for Data to encode (new modification)

    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 weight,
                 loss,
                 data= None,
                 loss_and_grads=None,
                 num_samples=0,
                 num_steps=0,
                 **kwargs
                 ):
        """
        Initializes the trainer's parameters
        """
        if num_steps:
            warnings.warn(
                "The `num_steps` argument to SVI is deprecated and will be removed in "
                "a future release. Use `SVI.step` directly to control the "
                "number of iterations.",
                FutureWarning,
            )
        if num_samples:
            warnings.warn(
                "The `num_samples` argument to SVI is deprecated and will be removed in "
                "a future release. Use `pyro.infer.Predictive` class to draw "
                "samples from the posterior.",
                FutureWarning,
            )

        self.model = model
        self.guide = guide
        self.optim = optim
        self.weight = weight
        self.num_steps = num_steps
        self.num_samples = num_samples
        ########## Start New lines #########
        if data is None:
            print("Full training data missing. Working on small batch data")
        else:
            self.data = data
        ########## End New lines #########  
        super().__init__(**kwargs)

        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError(
                "Optimizer should be an instance of pyro.optim.PyroOptim class."
            )

        if isinstance(loss, ELBO):
            self.loss = loss.loss
            self.loss_and_grads = loss.loss_and_grads
        else:
            if loss_and_grads is None:

                def _loss_and_grads(*args, **kwargs):
                    loss_val = loss(*args, **kwargs)
                    if getattr(loss_val, "requires_grad", False):
                        loss_val.backward(retain_graph=True)
                    return loss_val

                loss_and_grads = _loss_and_grads
            self.loss = loss
            self.loss_and_grads = loss_and_grads

    def run(self, *args, **kwargs):
        """
        .. warning::
            This method is deprecated, and will be removed in a future release.
            For inference, use :meth:`step` directly, and for predictions,
            use the :class:`~pyro.infer.predictive.Predictive` class.
        """
        warnings.warn(
            "The `SVI.run` method is deprecated and will be removed in a "
            "future release. For inference, use `SVI.step` directly, "
            "and for predictions, use the `pyro.infer.Predictive` class.",
            FutureWarning,
        )
        if self.num_steps > 0:
            with poutine.block():
                for i in range(self.num_steps):
                    self.step(*args, **kwargs)
        return super().run(*args, **kwargs)


    def _traces(self, *args, **kwargs):
        for i in range(self.num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, trace=guide_trace)
            ).get_trace(*args, **kwargs)
            yield model_trace, 1.0

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        with torch.no_grad():
            loss = self.loss(self.model, self.guide, *args, **kwargs)
            if isinstance(loss, tuple):
               
                return type(loss)(map(torch_item, loss))
            else:
                return torch_item(loss)

########################################################################################################
# This below function is modified to add argument to input training data for encoding to cal loss in latent image space
    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # Modified below line to pass additional arguments: self.data, self.weight, and output ELBO and custom loss individually
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss, loss1, loss2 = self.loss_and_grads(self.model, self.guide, self.data, self.weight, *args, **kwargs)

        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        return loss, loss1, loss2

        #if isinstance(loss1, tuple):
        #    return type(loss1)(map(torch_item, loss1))
        #else:
        #    return torch_item(loss1)
