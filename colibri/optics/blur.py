import torch
from colibri.optics.functional import psf_single_doe_spectral, convolutional_sensing, wiener_filter, ideal_panchromatic_sensor
from colibri.optics.sota_does import conventional_lens, nbk7_refractive_index
from .utils import BaseOpticsLayer

class Convolution(BaseOpticsLayer):
    def __init__(self, kernel: torch.Tensor):
        # Store kernel as a buffer (not trainable, but moves with model to device)
        self.kernel = kernel
        # self.learnable_optics = kernel
        super(Convolution, self).__init__(learnable_optics=kernel, sensing=self.convolution, backward=self.deconvolution)

    def convolution(self, x, kernel):
        r"""
        Forward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, 1, M, N) 
        """

        # psf = self.kernel
        field = convolutional_sensing(x, kernel, domain='fourier')
        return field


    def deconvolution(self, x, kernel, alpha=1e-3):
        r"""
        Backward operator of the SingleDOESpectral layer.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 1, M, N)
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        """

        # psf = self.kernel
        # print(kernel.shape)
        # out = torch.nn.functional.conv_transpose2d(x, kernel,padding=((kernel.shape[2]-1)//2))
        # # print(out.shape)
        # out = out[:,:,:x.shape[2],:x.shape[3]]
        # # print(out.shape)
        out = wiener_filter(x, kernel, alpha)
        return out
    
    
    def forward(self, x, type_calculation="forward"):
        r"""
        Performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N) 
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        return super(Convolution, self).forward(x, type_calculation)


