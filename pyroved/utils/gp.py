import pyro
import pyro.contrib.gp as gp
import torch

def gp_model(input_dim: int = None, encoded_X: torch.Tensor = None, y: torch.Tensor = None):
        """
        Gaussian Process model.
        Args:
            input_dim: Dimensionality of the input data.
            encoded_X: Encoded data.
            y: Target data.
        Returns:
            gpr: GP regression model.
            optimizer: Optimizer used to train the GP model.
            loss_fn: Loss function used to train the GP model.
        """
        # Define and train the GP model
        print("Training GP model...")
        kernel = gp.kernels.RBF(input_dim=encoded_X.shape[1])
        gpr = gp.models.GPRegression(encoded_X, y, kernel)
        optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

            
        return gpr, optimizer, loss_fn