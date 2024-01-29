import pyro
import pyro.contrib.gp as gp
import torch
from tqdm import tqdm
def gp_model(input_dim: int = None, encoded_X: torch.Tensor = None, y: torch.Tensor = None, gp_iterations: int = 1):
        """
        Returns a GP model trained on the encoded data.
        Args:
            input_dim: Dimensionality of the input data.
            encoded_X: Encoded data.
            y: Target data.
        Returns:
            gpr: GP regression model.
        """
        # Define and train the GP model
        print("Training GP model...")
        kernel = gp.kernels.RBF(input_dim=encoded_X.shape[1])
        gpr = gp.models.GPRegression(encoded_X, y, kernel)
        optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        loss = loss_fn(gpr.model, gpr.guide)
        for _ in tqdm(range(gp_iterations)):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("GP model trained.")  
        
        return gpr