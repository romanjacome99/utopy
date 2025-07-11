import torch
from torch import nn

from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity

class Fista(nn.Module):
    r"""
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    The FISTA algorithm solves the optimization problem:

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{arg min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The FISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        t_{k+1} &= \frac{1 + (1 + 4t_k^2)^{0.5}}{2} \\
        \mathbf{z}_{k+1} &=  \mathbf{x}_{k+1} + \frac{t_k-1}{t_{k+1}}( \mathbf{x}_{k} - \mathbf{x}_{k-1})
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.

    Implementation based on the formulation of authors in https://doi.org/10.1137/080716542
    """

    def __init__(self, acquistion_model, fidelity=L2(), prior=Sparsity("dct"), max_iters=5, alpha=1e-3, _lambda=0.1):
        r"""
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the FISTA algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.
            _lambda (float): The regularization parameter for the prior term. Defaults to 0.1.

        Returns:
            None
        """
        super(Fista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior

        self.H = lambda x: self.acquistion_model.forward(x)

        self.max_iters = max_iters
        self.alpha = alpha if  type(alpha) is list else [alpha]* max_iters
        self._lambda = _lambda
        t = 1
        
        self.t_squence = []
        for _ in range(self.max_iters):
            t_old = t
            t = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            
            self.t_squence.append((t_old -1)/t)

    def forward(self, y, x0=None, verbose=False, freq=0):
        r"""Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """ 
        self.prior.reset()

        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        z = x.clone()
        xs = [] 

        for i in range(self.max_iters):
            x_old = x.clone()

            # gradient step
            x = z - self.alpha[i] * self.fidelity.grad(z, y, self.H) 

            # proximal step
            x = self.prior.prox(x, self._lambda)

            # FISTA step
           
            z = x + self.t_squence[i] * (x - x_old)

            # if xgt is not None:
            #     psnr_ = colibri2.metrics.psnr(x, xgt, 1)
            #     print(f"PSNR: {psnr_}")
            if freq != 0:
                if i % freq == 0 and i != 0:
                    xs.append(x)
            
            if verbose:
                error = self.fidelity.forward(x, y, self.H).item()
                print("Iter: ", i, "fidelity: ", error)

        return x,xs 




class ProgressiveFista(nn.Module):
    r"""
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    The FISTA algorithm solves the optimization problem:

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{arg min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The FISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        t_{k+1} &= \frac{1 + (1 + 4t_k^2)^{0.5}}{2} \\
        \mathbf{z}_{k+1} &=  \mathbf{x}_{k+1} + \frac{t_k-1}{t_{k+1}}( \mathbf{x}_{k} - \mathbf{x}_{k-1})
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.

    Implementation based on the formulation of authors in https://doi.org/10.1137/080716542
    """

    def __init__(self, acquistion_model, fidelity=L2(), prior=Sparsity("dct"), max_iters=5, alpha=1e-3, _lambda=0.1):
        r"""
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the FISTA algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.
            _lambda (float): The regularization parameter for the prior term. Defaults to 0.1.

        Returns:
            None
        """
        super(ProgressiveFista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior

        self.H = [lambda x: acc.forward(x) for acc in acquistion_model]

        self.max_iters = max_iters
        self.alpha = alpha if  type(alpha) is list else [alpha]* max_iters
        self._lambda = _lambda
        t = 1
        
        self.t_squence = []
        for _ in range(self.max_iters):
            t_old = t
            t = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            
            self.t_squence.append((t_old -1)/t)

    def forward(self, y, x0=None, verbose=False, freq=0,gamma=0):
        r"""Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """ 
        self.prior.reset()

        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        z = x.clone()
        xs = [] 

        for i in range(self.max_iters):
            x_old = x.clone()

            # gradient step
            x = z - self.alpha[i] * (gamma*self.fidelity.grad(z, y[0], self.H[0]) + (1-gamma)*self.fidelity.grad(x, y[1], self.H[1]))

            # proximal step
            # normalization
            # x = x / torch.norm(x, p=2, dim=[1, 2, 3], keepdim=True)
            x = self.prior.prox(x, self._lambda)

            # FISTA step
           
            z = x + self.t_squence[i] * (x - x_old)

            # if xgt is not None:
            #     psnr_ = colibri2.metrics.psnr(x, xgt, 1)
            #     print(f"PSNR: {psnr_}")
            if freq != 0:
                if i % freq == 0 and i != 0:
                    xs.append(x)
            
            if verbose:
                error = self.fidelity.forward(x, y, self.H).item()
                print("Iter: ", i, "fidelity: ", error)

        
        return x,xs 




class MultiRegFista(nn.Module):
    r"""
    MultReg Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    The FISTA algorithm solves the optimization problem:

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{arg min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The FISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        t_{k+1} &= \frac{1 + (1 + 4t_k^2)^{0.5}}{2} \\
        \mathbf{z}_{k+1} &=  \mathbf{x}_{k+1} + \frac{t_k-1}{t_{k+1}}( \mathbf{x}_{k} - \mathbf{x}_{k-1})
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.

    Implementation based on the formulation of authors in https://doi.org/10.1137/080716542
    """

    def __init__(self, forwards, fidelity=list, prior=list, max_iters=5, alpha=[1e-3], _lambda=0.1):
        r"""
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the FISTA algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.
            _lambda (float): The regularization parameter for the prior term. Defaults to 0.1.

        Returns:
            None
        """
        super(MultiRegFista, self).__init__()

        self.fidelity = fidelity
        # self.acquistion_model = acquistion_model
        self.prior = prior

        self.H =forwards

        self.max_iters = max_iters
        self.alpha = alpha
        self._lambda = _lambda


    def forward(self, y, x0=None, verbose=False, xgt =None, freq=0):
        r"""Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        if x0 is None:
            x0 = torch.zeros_like(y[0])

        x = x0
        t = 1
        z = x.clone()
        xs = [] 
        for i in range(self.max_iters):
            x_old = x.clone()

            # gradient step
            
            grad = 0
            for j in range(len(self.alpha)):
                grad += self.fidelity[j].grad(z, y[j], self.H[j])


            x = z - self.alpha[0]*grad

            # proximal step
            x = self.prior.prox(x, self._lambda)

            # FISTA step
            t_old = t
            t = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)
            if xgt is not None:
              pass

            if freq != 0:
                if i % freq == 0 and i != 0:
                    xs.append(x)
        return x,xs    
        
