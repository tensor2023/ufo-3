import torch

class Reconstruction(torch.nn.Module):
    """
    Reconstructs the diffusion signal from spherical harmonic coefficients and isotropic compartments.
    """

    def __init__(self, max_order, num_features):
        """
        Initialization.

        Args:
            max_order (int): Maximum spherical harmonic order.
            num_features (int): Number of gradient directions or signal channels to reconstruct.
        """
        super(Reconstruction, self).__init__()
        self.max_order = max_order
        self.num_features = num_features

    def forward(self, x_fodf_coff_shc, iso, table, A):
        """
        Forward pass for signal reconstruction.

        Args:
            x_fodf_coff_shc (torch.Tensor): Concatenated spherical harmonic coefficients,
                                            including FODF and isotropic compartments.
            iso (torch.Tensor): Isotropic diffusivity values for each voxel (e.g., λ_iso).
            table (torch.Tensor): b-values corresponding to each diffusion direction.
            A (torch.Tensor): Spherical harmonic basis evaluated at each diffusion direction.

        Returns:
            x_reconstructed (torch.Tensor): Reconstructed diffusion signal.
        """
        bs = x_fodf_coff_shc.shape[0]
        G_shape_1 = A.shape[1]

        # Compute signal attenuation for the extra-axonal compartment
        beta = torch.exp(-iso * table / 1000).view(
            bs, G_shape_1, 1,
            x_fodf_coff_shc.shape[-1],
            x_fodf_coff_shc.shape[-1],
            x_fodf_coff_shc.shape[-1]
        )

        # Intra-axonal (FODF) component reconstruction using SH basis
        x_reconstructed_intra = torch.einsum('agb,abijk->agijk', A, x_fodf_coff_shc[:, :-2])

        # Extra-axonal component reconstruction using exponential decay
        x_reconstructed_extra = torch.einsum('agbijk,abijk->agijk', beta, x_fodf_coff_shc[:, 45:46])

        # Trapped water component reconstruction (constant term)
        x_reconstructed_trap = torch.einsum(
            'agb,abijk->agijk',
            torch.ones(bs, G_shape_1, 1, device=x_fodf_coff_shc.device),
            x_fodf_coff_shc[:, 46:47]
        )

        # Total signal reconstruction by summing all compartments
        x_reconstructed = x_reconstructed_intra + x_reconstructed_extra + x_reconstructed_trap

        return x_reconstructed
