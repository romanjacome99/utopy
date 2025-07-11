
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch_dct as dct
from torchmetrics.image import PeakSignalNoiseRatio
import os
from tqdm import tqdm
from utils import *
from colibri.models import Unet

from torch.nn import functional as F

class LearnedCostPGDMixed(nn.Module):
    def __init__(self, algo_params=None):
        super(LearnedCostPGDMixed, self).__init__()
        self.alpha = algo_params['alpha']
        self.alpha_1 = algo_params['alpha_1']
        self.lambd = algo_params['lambd']
        self.max_iter = algo_params['max_iter']
        self.n = algo_params['n']

    def forward(self, y, A, gt=None, g=None,get_metrics=False):
        theta = self.init_val(A, y)
        metrics = {'psnr':[], 'cost_l':[], 'cost_g':[]}
        for i in range(self.max_iter):
            if g is not None:
                x = dct.idct_2d(theta.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                x = torch.Tensor(x).requires_grad_(True)
                cost_l = torch.norm(g(x, y, A))

                grad = torch.autograd.grad(cost_l, x,retain_graph=True,create_graph=True)[0]

                grad_l = dct.dct_2d(grad.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                
                psi_theta = dct.idct_2d(theta.view(-1, 1, self.n, self.n))
                psi_theta = psi_theta.reshape(-1,self.n**2)
                grad =(psi_theta @A.T- y)

                cost_g = torch.norm(grad)

                theta = theta - self.alpha_1*grad_l - self.alpha*self.compute_gradient(theta, y, A)
            else:
                psi_theta = dct.idct_2d(theta.view(-1, 1, self.n, self.n))
                psi_theta = psi_theta.reshape(-1,self.n**2)
                grad =(psi_theta @A.T- y)@A
                cost_l = torch.norm(grad)
                theta = theta - self.alpha*self.compute_gradient(theta, y, A)
            
            # theta = theta - self.alpha*self.compute_gradient(theta, y, A)
            theta = soft_thresholding(theta, self.lambd)
            # if gt is not None:
            #     if i % 10 == 0:
            #         x = dct.idct_2d(theta.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
            #         psnr = PeakSignalNoiseRatio()(x.view(-1,self.n,self.n),gt.view(-1,self.n,self.n))
            #         print(f'iter {i} PSNR: {psnr}')
            if get_metrics:
                x = dct.idct_2d(theta.reshape(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                x = x/x.max()
                psnr = PeakSignalNoiseRatio()(x.view(-1,self.n,self.n),gt.view(-1,self.n,self.n))
                metrics['psnr'].append(psnr.item())
                metrics['cost_l'].append(cost_l.item())
                metrics['cost_g'].append(cost_g.item())


        x = dct.idct_2d(theta.reshape(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
            
        if get_metrics:
            return x/x.max(), metrics
        return x
            
        
    
    def compute_gradient(self, theta_old,y,A):
        psi_theta = dct.idct_2d(theta_old.view(-1, 1, self.n, self.n))
        psi_theta = psi_theta.reshape(-1,self.n**2)
        grad =(psi_theta @A.T- y)@A
        grad = grad.reshape(-1, 1, self.n, self.n)
        grad = dct.dct_2d(grad)
        grad = grad.reshape(-1, self.n**2)

        return grad
    
    def init_val(self, A, y):
        x = y@A
        x = x.view(-1, 1, self.n, self.n)
        x = dct.dct_2d(x)
        x = x.reshape(-1,self.n*self.n)
        return x/x.max()

    def get_gradient_from_x(self, x, y, A):
        grad = (x@A.T - y)@A
        grad = grad.view(-1, 1, self.n, self.n)
        grad = dct.dct_2d(grad)
        grad = grad.reshape(-1,self.n*self.n)
        return grad



class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)






class LearnedCostPGD(nn.Module):
    def __init__(self, algo_params=None):
        super(LearnedCostPGD, self).__init__()
        self.alpha = algo_params['alpha']
        self.lambd = algo_params['lambd']
        self.max_iter = algo_params['max_iter']
        self.n = algo_params['n']

    def forward(self, y, A, gt=None, g=None,get_metrics=False):
        theta = self.init_val(A, y)
        metrics = {'psnr':[], 'cost':[]}
        for i in range(self.max_iter):
            if g is not None:
                x = dct.idct_2d(theta.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                x = torch.Tensor(x).requires_grad_(True)
                cost = torch.norm(g(x, y, A))

                grad = torch.autograd.grad(cost, x,retain_graph=True,create_graph=True)[0]

                grad = dct.dct_2d(grad.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                
                theta = theta - self.alpha*grad
            else:
                psi_theta = dct.idct_2d(theta.view(-1, 1, self.n, self.n))
                psi_theta = psi_theta.reshape(-1,self.n**2)
                grad =(psi_theta @A.T- y)@A
                cost = torch.norm(grad)
                theta = theta - self.alpha*self.compute_gradient(theta, y, A)
            
            # theta = theta - self.alpha*self.compute_gradient(theta, y, A)
            theta = soft_thresholding(theta, self.lambd)
            # if gt is not None:
            #     if i % 10 == 0:
            #         x = dct.idct_2d(theta.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
            #         psnr = PeakSignalNoiseRatio()(x.view(-1,self.n,self.n),gt.view(-1,self.n,self.n))
            #         print(f'iter {i} PSNR: {psnr}')
            if get_metrics:
                x = dct.idct_2d(theta.reshape(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                x = x/x.max()
                psnr = PeakSignalNoiseRatio()(x.view(-1,self.n,self.n),gt.view(-1,self.n,self.n))
                metrics['psnr'].append(psnr.item())
                metrics['cost'].append(cost.item())


        x = dct.idct_2d(theta.reshape(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
            
        if get_metrics:
            return x/x.max(), metrics
        return x
            
        
    
    def compute_gradient(self, theta_old,y,A):
        psi_theta = dct.idct_2d(theta_old.view(-1, 1, self.n, self.n))
        psi_theta = psi_theta.reshape(-1,self.n**2)
        grad =(psi_theta @A.T- y)@A
        grad = grad.reshape(-1, 1, self.n, self.n)
        grad = dct.dct_2d(grad)
        grad = grad.reshape(-1, self.n**2)

        return grad
    
    def init_val(self, A, y):
        x = y@A
        x = x.view(-1, 1, self.n, self.n)
        x = dct.dct_2d(x)
        x = x.reshape(-1,self.n*self.n)
        return x/x.max()

    def get_gradient_from_x(self, x, y, A):
        grad = (x@A.T - y)@A
        grad = grad.view(-1, 1, self.n, self.n)
        grad = dct.dct_2d(grad)
        grad = grad.reshape(-1,self.n*self.n)
        return grad



class PGD(nn.Module):
    def __init__(self, algo_params=None):
        super(PGD, self).__init__()
        self.alpha = algo_params['alpha']
        self.lambd = algo_params['lambd']
        self.max_iter = algo_params['max_iter']
        self.n = algo_params['n']

    def forward(self, y, A, gt=None):
        theta = self.init_val(A, y)
        for i in range(self.max_iter):
            
            theta = theta - self.alpha*self.compute_gradient(theta, y, A)
            theta = soft_thresholding(theta, self.lambd)
            if gt is not None:
                if i % 10 == 0:
                    x = dct.idct_2d(theta.view(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
                    psnr = PeakSignalNoiseRatio()(x.view(-1,self.n,self.n),gt.view(-1,self.n,self.n))
                    print(f'iter {i} PSNR: {psnr}')
        x = dct.idct_2d(theta.reshape(-1, 1, self.n, self.n)).reshape(-1,self.n**2)
       
        return x/x.max()
            
        
    
    def compute_gradient(self, theta_old,y,A):
        psi_theta = dct.idct_2d(theta_old.view(-1, 1, self.n, self.n))
        psi_theta = psi_theta.reshape(-1,self.n**2)
        grad =(psi_theta @A.T- y)@A
        grad = grad.reshape(-1, 1, self.n, self.n)
        grad = dct.dct_2d(grad)
        grad = grad.reshape(-1, self.n**2)
        return grad
    
    def init_val(self, A, y):
        x = y@A
        x = x.view(-1, 1, self.n, self.n)
        x = dct.dct_2d(x)
        x = x.reshape(-1,self.n*self.n)
        return x/x.max()

    def get_gradient_from_x(self, x, y, A):
        grad = (x@A.T - y)@A
        grad = grad.view(-1, 1, self.n, self.n)
        grad = dct.dct_2d(grad)
        grad = grad.reshape(-1,self.n*self.n)
        return grad
    



class LearnedCost(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation_fn=nn.ReLU):
        """
        Initializes the MLP model.
        
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_sizes (list): List of integers specifying the number of neurons in each hidden layer.
            activation_fn (class): Activation function class (default: nn.ReLU).
        """
        super(LearnedCost, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class LearnedCost(nn.Module):
#     def __init__(self, n, m, l, spectral_norm=False):
#         super(LearnedCost, self).__init__()
#         self.n = n
#         self.m = m
#         self.l = l
#         if spectral_norm:
#             self.fc1 = nn.utils.spectral_norm(nn.Linear(m, l*2))
#             self.fc2 = nn.utils.spectral_norm(nn.Linear(l*2, l))

#         else:
#             self.fc1 = nn.Linear(m, l*2)
#             self.fc2 = nn.Linear(l*2, l)
        
        

#         self.relu = nn.ReLU()
        

#     def forward(self, x, y, H):
#         ys = H(x)
#         ys = ys.view(-1, self.m)
#         y = y.view(-1, self.m)
#         w1 = self.relu(self.fc1(y))
#         w2 = self.relu(self.fc1(ys))
#         w3 = w1+w2
#         w4 = self.relu(self.fc2(w3))
#         return w4



class LearnedCost_big(nn.Module):
    def __init__(self, n, m, hidden_sizes_w, hidden_sizes_out, spectral_norm=False, activation_fn=nn.ReLU):
        """
        Initializes the LearnedCost model.
        
        Args:
            n (int): Input size parameter (not directly used but kept for compatibility).
            m (int): Size of the input/output feature vector for `H`.
            hidden_sizes_w (list): List of integers specifying the number of neurons in each hidden layer for the first model.
            hidden_sizes_out (list): List of integers specifying the number of neurons in each hidden layer for the second model.
            spectral_norm (bool): If True, applies spectral normalization to layers.
            activation_fn (class): Activation function class (default: nn.ReLU).
        """
        super(LearnedCost_big, self).__init__()
        self.n = n
        self.m = m
        
        # First model for processing y and H(x)
        self.model_w = self._build_model(m, hidden_sizes_w, spectral_norm, activation_fn)
        
        # Second model for processing w3
        self.model_out = self._build_model(hidden_sizes_w[-1], hidden_sizes_out, spectral_norm, activation_fn)

    def _build_model(self, input_size, hidden_sizes, spectral_norm, activation_fn):
        """
        Helper function to build a customizable MLP model.
        
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List of integers specifying the number of neurons in each hidden layer.
            spectral_norm (bool): If True, applies spectral normalization to layers.
            activation_fn (class): Activation function class (default: nn.ReLU).
        
        Returns:
            nn.Sequential: The constructed MLP model.
        """
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            if spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(prev_size, size)))
            else:
                layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fn())
            prev_size = size
        
        # Final layer without activation
        if spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(prev_size, hidden_sizes[-1])))
        else:
            layers.append(nn.Linear(prev_size, hidden_sizes[-1]))
        
        return nn.Sequential(*layers)

    def forward(self, x, y, H):
        """
        Forward pass for the LearnedCost model.
        
        Args:
            x (Tensor): Input tensor for function H.
            y (Tensor): Input tensor to process.
            H (function): A function that transforms x into a tensor of shape (-1, m).
        
        Returns:
            Tensor: The output after passing through the two customizable models.
        """
        # Process input with function H
        ys = H(x)
        ys = ys.view(-1, self.m)
        y = y.view(-1, self.m)
        
        # Process y and H(x) through the first model
        w1 = self.model_w(y)
        w2 = self.model_w(ys)
        w3 = w1 + w2

        # Process w3 through the second model
        output = self.model_out(w3)
        return output
    

class LearnedCost_Taylor(nn.Module):

    def __init__(self, degree: int):
        """
        Taylor Expansion Layer
        Args:
            degree (int): Degree of the Taylor expansion
        """
        super(LearnedCost_Taylor, self).__init__()
        self.degree = degree
        # Learnable coefficients for the Taylor expansion
        self.coefficients = nn.Parameter(torch.randn(degree + 1))  # Initialize randomly

    def forward(self,  x, y, H) -> torch.Tensor:
        """
        Forward pass for the Taylor expansion
        Args:
            W (torch.Tensor): Input vector W (batch_size, d)
            V (torch.Tensor): Input vector V (batch_size, d)
        Returns:
            torch.Tensor: Result of the Taylor expansion
        """
        W = H(x)
        W = W.view(x.shape[0], -1)
        V = y.view(x.shape[0], -1)

        assert W.shape == V.shape, "W and V must have the same shape"
        # normalize the input vectors
        # W = W / torch.norm(W, dim=-1, keepdim=True)
        # V = V / torch.norm(V, dim=-1, keepdim=True)
        dot_product = torch.mean(W * V, dim=-1)/W.shape[-1]  # Shape: (batch_size,)
        
        result = self.coefficients[0].expand_as(dot_product)
        
        # Add terms of the Taylor series
        term = dot_product  # First-order term
        for i in range(1, self.degree + 1):
            result = result + self.coefficients[i] * term
            term = term * dot_product  # Update term for the next power
        
        return result
    

class Unfolding(nn.Module):
    def __init__(self,n_class,n_stages):
        super(Unfolding, self).__init__()
        self.n_stages = n_stages
        self.proximals =nn.ModuleList(
             [Proximal_Mapping(channel=n_class).to('cuda')
              for _ in range(n_stages)
              ])
        self.alphas =  nn.ParameterList(
            [
            nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda') 
            for _ in range(n_stages)]
            )

        self.rhos =  nn.ParameterList(
            [
            nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda') 
            for _ in range(n_stages)]
            )

        self.betas =  nn.ParameterList(
            [
            nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda') 
            for _ in range(n_stages)]
            )

    def forward(self, y,spc):
        x = spc(y,'backward')
        x = x/x.max()
        u = torch.zeros_like(x)
        for i in range(self.n_stages):
            h = self.proximals[i](x+u)
            x = x - self.alphas[i]*(spc(spc(x,'forward')-y,'backward') + self.rhos[i]*(x-h+u) )
            u = u + self.betas[i]*(x-h)
        return x

class ProgressiveLossUnfolding(nn.Module):
    def __init__(self,n_class,n_stages):
        super(ProgressiveLossUnfolding, self).__init__()
        self.n_stages = n_stages
        self.proximals =nn.ModuleList(
             [Proximal_Mapping(channel=n_class).to('cuda')
              for _ in range(n_stages)
              ])
        self.alphas =  nn.ParameterList(
            [
            nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda') 
            for _ in range(n_stages)]
            )

        self.rhos =  nn.ParameterList(
            [
            nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda') 
            for _ in range(n_stages)]
            )

        self.betas =  nn.ParameterList(
            [
            nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda') 
            for _ in range(n_stages)]
            )

    def forward(self, y,spc, yt, spc_t,alpha):
        x = (1-alpha)*spc(y,'backward')+alpha*spc_t(yt,'backward')
        x = x/x.max()
        u = torch.zeros_like(x)
        for i in range(self.n_stages):
            h = self.proximals[i](x+u)
            x = x - self.alphas[i]*((1-alpha)*spc(spc(x,'forward')-y,'backward') +alpha*spc_t(spc_t(x,'forward')-yt,'backward')+ self.rhos[i]*(x-h+u) )
            u = u + self.betas[i]*(x-h)
        return x

from colibri.recovery.terms.fidelity import L2

class ProgressiveLossUnfoldingPGD(nn.Module):
    def __init__(self,n_class,n_stages):
        super(ProgressiveLossUnfoldingPGD, self).__init__()
        self.n_stages = n_stages
        self.proximals = nn.ModuleList(
            [Unet(features=[16,32,64,128]).to('cuda')
             for _ in range(n_stages)
            ])
        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1, requires_grad=True) * 0.01).to('cuda')
                for _ in range(n_stages)
            ]
        )
        self.fidelity = L2()

    def forward(self, y, spc, yt, spc_t, alpha):
        x = (1 - alpha) * spc(y, 'backward') + alpha * spc_t(yt, 'backward')
        x = x / x.max()
        y_k = x.clone()
        t_k = 1.0
        for i in range(self.n_stages):
            # Gradient step
            grad = (1 - alpha) * spc(spc(y_k, 'forward') - y, 'backward') + \
                   alpha * spc_t(spc_t(y_k, 'forward') - yt, 'backward')
            x_new = self.proximals[i](y_k - self.alphas[i] * grad)

            # FISTA acceleration
            t_kp1 = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
            y_k = x_new + ((t_k - 1) / t_kp1) * (x_new - x)
            x = x_new
            t_k = t_kp1

        return x




class Proximal_Mapping(nn.Module):
    def __init__(self,channel):
        super(Proximal_Mapping, self).__init__()


        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.theta = nn.Parameter(torch.ones(1,requires_grad=True)*0.01).to('cuda')

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(32,channel, kernel_size=3, padding=1)


        self.Sp = nn.Softplus()


        
    def forward(self,x):

        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Softhreshold
        # xs = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.theta))

        # Decode
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        return x



class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation_fn=nn.ReLU):
        """
        Initializes the MLP model.
        
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_sizes (list): List of integers specifying the number of neurons in each hidden layer.
            activation_fn (class): Activation function class (default: nn.ReLU).
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)