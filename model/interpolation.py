import torch

class Interpolation(torch.nn.Module):
    def __init__(self, conv_name):
        """
        Initialization.

        Args:
            conv_name (str): Name of the convolution type or strategy (currently unused).
        """
        super(Interpolation, self).__init__()
        self.S = 1        # Number of interpolation passes (fixed to 1)
        self.V = 3072     # Number of target directions or interpolation points

    def forward(self, x, sampling2sampling):
        """
        Forward pass for interpolating the input signal.

        Args:
            x (torch.Tensor): Input diffusion signal.
            sampling2sampling (torch.Tensor): Interpolation matrix mapping input to target directions.

        Returns:
            torch.Tensor: Interpolated signal.
        """
        y = x.new_zeros(x.size(0), x.size(1) * self.S, self.V, *x.shape[3:])

        for i in range(self.S):
            # Perform directional interpolation via matrix projection
            y[:, i * x.size(1):(i + 1) * x.size(1)] = torch.einsum('ijklmn,ikp->ijplmn', x, sampling2sampling)

        return y
