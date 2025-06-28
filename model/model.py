import torch
from .deconvolution import Deconvolution
from .reconstruction import Reconstruction

class Model(torch.nn.Module):
    def __init__(self, filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in=1):
        super(Model, self).__init__()
        
        n_fodf_coff = 1  # Number of spherical harmonic FODF coefficients
        n_extra_trapped = 2  # Number of extra isotropic compartments (e.g., extra-axonal and trapped water)
        
        self.deconvolution = Deconvolution(
            filter_start, kernel_sizeSph, kernel_sizeSpa,
            n_fodf_coff, n_extra_trapped, normalize, conv_name, isoSpa, feature_in
        )
        
        self.reconstruction = Reconstruction(8, 45)

    def forward(self, x, sampling2sampling, table, A):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input diffusion signals.
            sampling2sampling (torch.Tensor): Projection matrix for basis conversion.
            table (torch.Tensor): Lookup table for b-values and directions.
            A (torch.Tensor): Gradient direction matrix or encoding matrix.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed diffusion signals.
            x_deconvolved_fodf_coff_shc (torch.Tensor): FODF coefficients in spherical harmonic basis.
            x_deconvolved_extra_trapped_shc (torch.Tensor): Extra isotropic compartment coefficients.
        """

        # Step 1: Deconvolve the input signal into SH coefficients for FODF and isotropic compartments
        x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc, iso = self.deconvolution(x, sampling2sampling)

        # Step 2: Prepare the b-value table for reconstruction
        shape = [x.shape[0], x.shape[2], 1, 1, 1]
        table = table[:, :-1, -1].reshape(shape)

        # Step 3: Concatenate all SH coefficients for signal reconstruction
        x_deconvolved_shc_norm = torch.cat(
            (x_deconvolved_fodf_coff_shc.squeeze(1), x_deconvolved_extra_trapped_shc.squeeze(2)),
            dim=1
        )

        # Step 4: Reconstruct the diffusion signal from SH coefficients
        x_reconstructed = self.reconstruction(x_deconvolved_shc_norm, iso, table, A)

        return x_reconstructed, x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc
