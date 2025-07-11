import numpy as np
import torch 
import random
from torch.autograd import Function
import math
from hadamard_spc.libs.ordering import hadamard, sequency, cake_cutting
from torch import Tensor
import os



def create_low_pass_kernel(kernel_size=5, sigma=1.0, factor=1.0):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    gaussian = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    gaussian = gaussian / torch.sum(gaussian)

    return  gaussian.view(1, 1, kernel_size, kernel_size)

def psnr_fun(preds: Tensor, target: Tensor) -> Tensor:
    """
    Calcula el PSNR de cada muestra en el batch y su media.
    Si mse_per_sample == 0 → PSNR = +∞
    Si peak == 0     → PSNR = -∞

    Args:
        preds: Tensor de predicciones, forma (N, C, H, W).
        target: Tensor de ground truth, forma (N, C, H, W).
        use_dynamic_range: si True, usa (max-min) por muestra; 
                           si False, usa solo max (asumiendo min=0).

    Returns:
        psnr_per_sample: Tensor (N,) con el PSNR de cada muestra.
        psnr_mean: escalar con el PSNR medio sobre el batch.
    """
    N = preds.shape[0]
    flat_preds  = preds.view(N, -1)
    flat_target = target.view(N, -1)

    # 1) MSE por muestra
    mse_per_sample = torch.mean((flat_preds - flat_target) ** 2, dim=1)  # (N,)

    # 2) Pico por muestra
    max_val = flat_target.max(dim=1).values
    min_val = flat_target.min(dim=1).values
    peak = max_val - min_val

    # 3) PSNR por muestra, sin eps para permitir ±∞
    psnr_per_sample = 10 * torch.log10(peak**2 / mse_per_sample)

    # 4) Media sobre el batch (ignorando infinities en la media de PyTorch)
    psnr_mean = psnr_per_sample.mean()

    return psnr_mean
class BinaryQuantize(Function):

    @staticmethod
    def forward(ctx, input):
        # out = torch.sign(input) * 0.5 + 0.5
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


class ExponentialScheduler:
    def __init__(self, initial_value: float, final_value: float, num_iter: int):
        """
        Exponential scheduler for an arbitrary hyperparameter.

        Args:
            initial_value (float): Initial value of the parameter.
            final_value (float): Final value of the parameter.
            num_iter (int): Number of iterations over which to decay.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_iter = num_iter
        
        # Compute the decay rate (lambda) for the exponential function
        self.decay_rate = math.log(final_value / initial_value) / num_iter

    def step(self,step_count):
        """Update the parameter value using an exponential decay function."""
        # if self.step_count < self.num_iter:
        #     self.step_count += 1  # Increase step count
        return self.initial_value * math.exp(self.decay_rate * step_count)

    def get_value(self,step_count):
        """Return the current value of the parameter."""
        return self.initial_value * math.exp(self.decay_rate * step_count)



class LinearScheduler:
    def __init__(self, initial_value: float, final_value: float, num_iter: int):
        """
        Linear scheduler for an arbitrary hyperparameter.

        Args:
            initial_value (float): Initial value of the parameter.
            final_value (float): Final value of the parameter.
            num_iter (int): Number of iterations over which to decay.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_iter = num_iter
        self.delta = (final_value - initial_value) / num_iter

    def step(self, step_count):
        """Update the parameter value using a linear decay function."""
        value = self.initial_value + self.delta * step_count
        # Clamp to [min, max] to avoid overshooting
        if self.delta < 0:
            value = max(self.final_value, value)
        else:
            value = min(self.final_value, value)
        return value

    def get_value(self, step_count):
        """Return the current value of the parameter."""
        return self.step(step_count)

class CosineScheduler:
    def __init__(self, initial_value: float, final_value: float, num_iter: int):
        """
        Cosine scheduler for an arbitrary hyperparameter.

        Args:
            initial_value (float): Initial value of the parameter.
            final_value (float): Final value of the parameter.
            num_iter (int): Number of iterations over which to decay.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_iter = num_iter

    def step(self, step_count):
        """Update the parameter value using a cosine annealing schedule."""
        if step_count >= self.num_iter:
            return self.final_value
        cos_inner = math.pi * step_count / self.num_iter
        value = self.final_value + 0.5 * (self.initial_value - self.final_value) * (1 + math.cos(cos_inner))
        return value

    def get_value(self, step_count):
        """Return the current value of the parameter."""
        return self.step(step_count)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch
from typing import Union

def hf_loss(recon, target, cutoff=0.4):
    # recon, target: tensor of shape (B,C,H,W)
    F_recon = torch.fft.rfft2(recon, norm='ortho')
    F_target = torch.fft.rfft2(target, norm='ortho')
    # Build radial frequency mask: frequencies above cutoff*Nyquist
    B,C,H,W2 = F_recon.shape
    fy = torch.fft.fftfreq(H).unsqueeze(1)  # H×1
    fx = torch.fft.rfftfreq(W2*2 - 1).unsqueeze(0)  # 1×W2
    freq_rad = torch.sqrt(fx**2 + fy**2).to(recon.device)
    mask = (freq_rad >= cutoff * 0.5).float()  # assuming Nyquist = 0.5
    diff_amp = F_recon.abs() - F_target.abs()
    l2 = (mask * diff_amp.pow(2)).mean()
    return l2

def add_awgn(measurements: torch.Tensor,
             snr_db: Union[float, torch.Tensor],
             dims_to_average: tuple = None,
             *,
             inplace: bool = False,
             eps: float = 1e-12) -> torch.Tensor:
    """
    Add AWGN to `measurements` to obtain the requested SNR.

    Parameters
    ----------
    measurements : torch.Tensor
        The clean measurement tensor (any shape, real or complex¹).
    snr_db : float or torch.Tensor
        Desired SNR in decibels (scalar or one value per sample).
    dims_to_average : tuple, optional
        Dimensions over which to compute the signal power.  
        • None (default)   → use *all* dims  
        • (1, 2, …)        → e.g. average over H×W but keep batch dim
    inplace : bool, default False
        If True, corrupt `measurements` in place; otherwise return a copy.
    eps : float, default 1e-12
        Numerical floor to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Noisy measurements with the requested SNR.
    """
    # Ensure inputs are tensors on the same device
    if not torch.is_tensor(measurements):
        raise TypeError("`measurements` must be a torch.Tensor")
    snr_db = torch.as_tensor(snr_db, dtype=measurements.dtype,
                             device=measurements.device)

    # 1) Compute signal power ⟨|x|²⟩
    if dims_to_average is None:
        sig_power = measurements.abs().pow(2).mean()
    else:
        # keepdim=True so subsequent broadcast works if snr_db is per-sample
        sig_power = measurements.abs().pow(2).mean(dims_to_average, keepdim=True)

    # 2) Convert SNR from dB to linear scale
    snr_linear = 10.0 ** (snr_db / 10.0)

    # 3) Derive noise variance σ² = P_signal / SNR
    noise_var = (sig_power / (snr_linear + eps)).clamp_min(eps)
    noise_std = noise_var.sqrt()

    # 4) Draw iid Gaussian noise (real or complex)
    if torch.is_complex(measurements):
        noise = (torch.randn_like(measurements.real) +
                 1j * torch.randn_like(measurements.imag)) * noise_std / torch.sqrt(torch.tensor(2.0, device=measurements.device))
    else:
        noise = torch.randn_like(measurements) * noise_std

    # 5) Add noise
    if inplace:
        measurements.add_(noise)
        return measurements
    else:
        return measurements + noise

def get_A_Ag(m,mb,n,C,dimension_ag=2,type_A='gaussian', type_Ag='well_conditioned_gaussian'):

    print(f'm: {m}, mb: {mb}')

    
    if type_A == 'gaussian':
        A = torch.randn(m, n).to('cuda')
        A = A / torch.norm(A, 2, 0)
    elif type_A == 'bernoulli':
        # A = torch.bernoulli(0.5*torch.ones(m, n)).to('cuda')
        H_path = f'H_{n}.pt'
        if os.path.exists(H_path):
            A = torch.load(H_path).to('cuda')
        else:
            A = torch.bernoulli(0.5*torch.ones(m, n)).to('cuda')
            torch.save(A, H_path)
        # A = np.load(f'H.npy').astype(np.float32)
        # A = torch.from_numpy(A[:m,:]).to('cuda')

    elif type_A == 'hadamard':
        H = hadamard.get_matrix(n)
        A = torch.from_numpy(H[:m,:]).to('cuda')
        A = A.type(torch.float32)
    


    if type_Ag == 'well_conditioned_from_A':

        Ag = A
        Ag = Ag / torch.norm(Ag, 2, 0)
        U, S, V = torch.svd(Ag, compute_uv=True)
        Sr = S[0]*( 1-((C-1)/C)*(S[0]-S)/(S[0]-S[-1]))
        Ag = U @ torch.diag(Sr) @ V.T
        Ag = Ag.to('cuda')
        Ag =Ag / torch.norm(Ag, 2, 0)
    elif type_Ag == 'gaussian':
        Ag = torch.randn(mb, n).to('cuda')
        Ag = Ag / torch.norm(Ag, 2, 0)
    
    elif type_Ag == 'bernoulli':
        if type_A == 'bernoulli':
           
           Ag = torch.load(H_path).to('cuda')
        
        Ag = Ag[:mb,:]
    elif type_Ag == 'hadamard':
        # ORDERINGS = {'hadamard': hadamard, 'sequency': sequency, 'cake_cutting': cake_cutting}

        H = cake_cutting.get_matrix(n)
        Ag = torch.from_numpy(H[:mb,:]).to('cuda')
        Ag = Ag.type(torch.float32)


    

    print(f'Condition number of A: {torch.linalg.cond(A@A.T)}')
    print(f'Condition number of Ag: {torch.linalg.cond(Ag@Ag.T)}')

    return A, Ag

def hf_loss(recon, target, cutoff=0.4):
    # recon, target: tensor of shape (B,C,H,W)
    F_recon = torch.fft.rfft2(recon, norm='ortho')
    F_target = torch.fft.rfft2(target, norm='ortho')
    # Build radial frequency mask: frequencies above cutoff*Nyquist
    B,C,H,W2 = F_recon.shape
    fy = torch.fft.fftfreq(H).unsqueeze(1)  # H×1
    fx = torch.fft.rfftfreq(W2*2 - 1).unsqueeze(0)  # 1×W2
    freq_rad = torch.sqrt(fx**2 + fy**2).to(recon.device)
    mask = (freq_rad >= cutoff * 0.5).float()  # assuming Nyquist = 0.5
    diff_amp = F_recon.abs() - F_target.abs()
    l2 = (mask * diff_amp.pow(2)).mean()
    return l2

