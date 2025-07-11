import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse 
import os
import wandb   
from tqdm import tqdm
from models import *
from utils import *

from colibri.optics.spc import SPC
from colibri.recovery.terms.fidelity import L2
from data_loaders import *
from colibri.models.unrolling import UnrollingProgressiveFISTA
from deepinv.models import DnCNN, Restormer, SCUNet, DiffUNet, GSDRUNet # Add this import
from colibri.recovery.terms.prior import Denoiser, Sparsity
from colibri.recovery.fista import Fista, MultiRegFista

def set_matplotlib_latex(fs=12):
    """
    Configure matplotlib to use LaTeX rendering with a given global font size.
    
    Parameters:
        fs (int): Global font size for text (default is 12).
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": fs,
        "font.family": "serif",
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

def compute_norm_per_batch(x):
    x = x.view(x.shape[0],-1)
    return torch.norm(x,dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)

def main(args):

    set_seed(args.seed)

    set_matplotlib_latex(20)
    
    _, _, testloader = get_dataloaders(args)





    
    
    # Sensing matrix
    m = int(args.n**2*args.cr)
    mt = m+int((args.n**2-m)*args.dimension_ag)

    A, At = get_A_Ag(m,mt,args.n**2,C=args.cond, dimension_ag=args.dimension_ag, type_A=args.type_A, type_Ag=args.type_Ag)
    
    spc = SPC((1,args.n,args.n), m,trainable=False,initial_ca=A,binary=True).to(args.device)
    


    spc_t = SPC((1,args.n,args.n), mt,trainable=False,initial_ca=At).to(args.device)
    
    # Only define one prior model (Unet or DnCNN) based on an argument

    prior_args = {'in_channels': 1, 'out_channels': 1, 'feautures': [32, 64, 128,256]}
    unet = [Unet(**prior_args) for _ in range(args.max_iter)]
    models = nn.Sequential(*unet)


    algo_params = {"max_iters": args.max_iter}
    l2 = L2().to(args.device)
    plu = UnrollingProgressiveFISTA([spc_t, spc], l2, **algo_params, models=models).to(args.device)
    print(f"Parameters of PLU Model: {sum(p.numel() for p in plu.parameters() if p.requires_grad)}")
    # Scheduler for alpha parameter

    prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download_lipschitz", 'device': args.device}).to(args.device)

    fista_base = Fista(spc, fidelity=l2, prior=prior, max_iters=100, alpha=1/torch.norm(A.T @A), _lambda=0.01).eval()


    # Learning rate scheduler: decay LR by 0.5 every 1/3 of epochs

    SSIM = StructuralSimilarityIndexMeasure().to(args.device)
    mse = nn.MSELoss()

    # test 
    loss_x_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    psnr_pnp = AverageMeter()
    ssim_pnp = AverageMeter()

    path = r'results\FINAL_MODELS\baseline'
    plu.load_state_dict(torch.load(f'{path}\model.pth', map_location=args.device))
    data_loop_test = tqdm(testloader, desc='Testing', colour='green')

    for i, data in enumerate(data_loop_test):

        if args.dataset in ['cifar10', 'stl', 'fashion_mnist']:
            data = data[0]
        x = data.to(args.device)
        y = spc(x, type_calculation="forward")
        if args.add_noise:
            y = add_awgn(y,args.snr)

        x0 = spc(y, type_calculation="backward")    
        den = x0.view(x0.size(0), -1).amax(dim=1, keepdim=True)   # shape (B, 1)
        x0 = x0 / den.view(x0.size(0), 1, 1, 1)   
        yt = spc_t(x, type_calculation="forward")             
        x_hat,xs = plu(x0=x0, y=[yt,y], gamma=0.0,freq=1)

        loss_x = mse(x_hat, x)
        psnr = psnr_fun(x_hat, x)
        ssim = SSIM(x_hat, x)

        loss_x_meter.update(loss_x.item(), x.size(0))
        psnr_meter.update(psnr.item(), x.size(0))
        ssim_meter.update(ssim.item(), x.size(0))

        x_hat_pnp,_ = fista_base(y, x0=x0)
        psnr_pnp.update(psnr_fun(x_hat_pnp, x).item(), x.size(0))
        ssim_pnp.update(SSIM(x_hat_pnp, x).item(), x.size(0))

        data_loop_test.set_postfix(
            loss=loss_x_meter.avg,
            psnr=psnr_meter.avg,
            ssim=ssim_meter.avg,
            psnr_pnp=psnr_pnp.avg,
            ssim_pnp=ssim_pnp.avg
        )   
    psnr_per_stage = np.array([psnr_fun(x, xss).item() for xss in xs])
    print(psnr_meter.avg, ssim_meter.avg, loss_x_meter.avg)
    print(f"PSNR PnP: {psnr_pnp.avg}, SSIM PnP: {ssim_pnp.avg}, Loss PnP: {loss_x_meter.avg}")

    # print(f"PSNR per stage: {psnr_per_stage}")
    # np.save(f'{path}/psnr_per_stage.npy', psnr_per_stage)
    psnr_forimage_0_0 = psnr_fun(x_hat[0, 0].cpu(), x[0, 0].cpu()).item()
    SSIM_forimage_0_0 = SSIM(x_hat[0, 0].unsqueeze(0).unsqueeze(0).cpu(), x[0, 0].unsqueeze(0).unsqueeze(0).cpu()).item()
    psnr_forx0_0 = psnr_fun(x0[0, 0].cpu(), x[0, 0].cpu()).item()
    SSIM_forx0_0 = SSIM(x0[0, 0].unsqueeze(0).unsqueeze(0).cpu(), x[0, 0].unsqueeze(0).unsqueeze(0).cpu()).item()
    x = x.cpu().detach().numpy()
    x_hat = x_hat.cpu().detach().numpy()
    x0 = x0.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(x[0, 0], cmap='gray')
    ax[0].set_title('GT \n PSNR/SSIM')
    ax[0].axis('off')
    ax[1].imshow(x0[0, 0], cmap='gray')
    text = r'$\mathbf{H}^\top\mathbf{y}$'
    ax[1].set_title(f'{text} \n {psnr_forx0_0:.2f} / {SSIM_forx0_0:.2f}')
    ax[1].axis('off')
    ax[2].imshow(x_hat[0, 0],cmap='gray')
    ax[2].set_title(f'Baseline \n {psnr_forimage_0_0:.2f} / {SSIM_forimage_0_0:.2f}')
    ax[2].axis('off')
    # plt.savefig(f'{path}/reconstruction.svg')

   


    plu = UnrollingProgressiveFISTA([spc_t, spc], l2, **algo_params, models=models).to(args.device)

    # print(psnr_meter.avg, ssim_meter.avg, loss_x_meter.avg)
            

    schedulers = ['linear']

    dim_augs = [0.1]

    for dim_aug in dim_augs:
        for scheduler in schedulers:
            loss_x_meter = AverageMeter()
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            path = f'results\FINAL_MODELS\{scheduler}\dim_ag={dim_aug}'
            plu.load_state_dict(torch.load(f'{path}\model.pth', map_location=args.device))
            data_loop_test = tqdm(testloader, desc='Testing', colour='green')

            for i, data in enumerate(data_loop_test):

                if args.dataset in ['cifar10', 'stl', 'fashion_mnist']:
                    data = data[0]
                x = data.to(args.device)
                y = spc(x, type_calculation="forward")
                if args.add_noise:
                    y = add_awgn(y,args.snr)

                x0 = spc(y, type_calculation="backward")    
                den = x0.view(x0.size(0), -1).amax(dim=1, keepdim=True)   # shape (B, 1)
                x0 = x0 / den.view(x0.size(0), 1, 1, 1)   
                yt = spc_t(x, type_calculation="forward")             
                x_hat,xs = plu(x0=x0, y=[yt,y], gamma=0.0,freq=1)

                loss_x = mse(x_hat, x)
                psnr = psnr_fun(x_hat, x)
                ssim = SSIM(x_hat, x)

                loss_x_meter.update(loss_x.item(), x.size(0))
                psnr_meter.update(psnr.item(), x.size(0))
                ssim_meter.update(ssim.item(), x.size(0))
                data_loop_test.set_postfix(
                    loss=loss_x_meter.avg,
                    psnr=psnr_meter.avg,
                    ssim=ssim_meter.avg,
                    dim_aug=dim_aug,
                    scheduler=scheduler
                )   

            print(psnr_meter.avg, ssim_meter.avg, loss_x_meter.avg)

            psnr_per_stage = np.array([psnr_fun(x, xss).item() for xss in xs])
            # print(f"PSNR per stage for {scheduler} with dim_aug={dim_aug}: {psnr_per_stage}")
            np.save(f'{path}/psnr_per_stage.npy', psnr_per_stage)

            psnr_forimage_0_0 = psnr_fun(x_hat[0, 0].cpu(), x[0, 0].cpu()).item()
            SSIM_forimage_0_0 = SSIM(x_hat[0, 0].unsqueeze(0).unsqueeze(0).cpu(), x[0, 0].unsqueeze(0).unsqueeze(0).cpu()).item()
            psnr_forx0_0 = psnr_fun(x0[0, 0].cpu(), x[0, 0].cpu()).item()
            SSIM_forx0_0 = SSIM(x0[0, 0].unsqueeze(0).unsqueeze(0).cpu(), x[0, 0].unsqueeze(0).unsqueeze(0).cpu()).item()
            x = x.cpu().detach().numpy()
            x_hat = x_hat.cpu().detach().numpy()
            x0 = x0.cpu().detach().numpy()
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(x[0, 0], cmap='gray')
            ax[0].set_title('GT')
            ax[0].axis('off')
            ax[1].imshow(x0[0, 0], cmap='gray')
            ax[1].axis('off')
            ax[1].set_title(f'Initial guess PSNR : {psnr_forx0_0:.2f}, SSIM: {SSIM_forx0_0:.2f}')
            ax[2].imshow(x_hat[0, 0],cmap='gray')
            text =f'$\eta = {dim_aug}$'
            ax[2].set_title(f'{text} \n {psnr_forimage_0_0:.2f} / {SSIM_forimage_0_0:.2f}')
            ax[2].axis('off')
            # plt.savefig(f'{path}/reconstruction.svg')
            




if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(description='Learning cost')
    #----------------------------------- Dataset parameters ---------------------------------
    parser.add_argument('--n', type=int, default=64, help='Size of the image')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--dataset', type=str, default='CelebA', help='Dataset')  # OPTIONS 
    parser.add_argument('--augment', type=bool, default=False, help='Use augmentation')
    #----------------------------------- Sensing parameters ---------------------------------
    parser.add_argument('--cr', type=float, default=0.35, help='Compression ratio')
    parser.add_argument('--hadamard_ordering', type=str, default='cake_cutting', help='Ordering of the matrix')
    #----------------------------------- FISTA parameters -----------------------------------
    parser.add_argument('--algo', type=str, default='fista', help='Algorithm')
    parser.add_argument('--max_iter', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--_lambda', type=float, default=0.1, help='Regularization parameter')
    #----------------------------------- Other parameters -----------------------------------
    parser.add_argument('--seed', type=int, default=5, help='Seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    # ---------------------------------- Unfolding parameters --------------------------------   
    parser.add_argument('--freq', type=int, default=2, help='Frequency of the scheduler')
    # ---------------------------------- Training Parameters --------------------------------
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--number_batches', type=int, default=0, help='Number of batches')
    parser.add_argument('--dimension_ag', type=float, default=0.3, help='Dimension of the aggregation matrix')
    parser.add_argument('--alpha_scheduler', type=str, default='exponential', choices=['exponential', 'linear','cosine'], help='Scheduler type for alpha parameter (exponential or linear)')

    parser.add_argument('--type_run', type=str, default='proposed', help='Type of run: baseline or progressive')
    parser.add_argument('--cond', type=float, default=1.0)

    parser.add_argument('--type_A', type=str, default='hadamard', help='Type of A matrix')
    parser.add_argument('--type_Ag', type=str, default='hadamard', help='Type of Ag matrix')
    parser.add_argument('--add_noise',type=bool,default=True)
    parser.add_argument('--snr',type=float,default=35)

    parser.add_argument('--param_continuation', type=float, default=0.7, help='Parameter for continuation path')
    
    args = parser.parse_args()
    for cr in [0.25, 0.30, 0.35]:
        args.cr = cr
        main(args)

    
    # main(args)
