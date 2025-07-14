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

def validate_continuation_path(spc, spc_t, plu, x, alpha):
    yt = spc_t(x, type_calculation="forward")
    y = spc(x, type_calculation="forward")
    x0 = alpha * spc_t(yt, type_calculation="backward") + (1 - alpha) * spc(y, type_calculation="backward")
    den = x0.view(x0.size(0), -1).amax(dim=1, keepdim=True)   # shape (B, 1)
    x0 = x0 / den.view(x0.size(0), 1, 1, 1)
    x_hat, _ = plu(x0=x0, y=[yt, y], gamma=alpha)
    return x_hat
    
def compute_norm_per_batch(x):
    x = x.view(x.shape[0],-1)
    return torch.norm(x,dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)

def main(args):

    wandb.login(key='key-wandb')
    set_seed(args.seed)

    
    
    trainloader, valloader, testloader = get_dataloaders(args)

    exp_name = f"CAMSAP_dim_ag={args.dimension_ag}_type_A_{args.type_A}_type_Ag_{args.type_Ag}_iters={args.max_iter}_cr={args.cr}_dataset={args.dataset}_freq={args.freq}_type_run={args.type_run}_n={args.n}_batch_size={args.batch_size}_lr={args.lr}_epochs={args.epochs}_lr={args.lr}_cond={args.cond}_param_cont={args.param_continuation}_scheduler={args.alpha_scheduler}"       
    
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
    if args.alpha_scheduler == 'exponential':
        scheduler_alpha = ExponentialScheduler(initial_value=1.0, final_value=1e-7, num_iter=int(args.epochs*args.param_continuation))
    elif args.alpha_scheduler == 'linear':
        scheduler_alpha = LinearScheduler(initial_value=1.0, final_value=1e-7, num_iter=int(args.epochs*args.param_continuation))
    
    elif args.alpha_scheduler == 'cosine':  
        scheduler_alpha = CosineScheduler(initial_value=1.0, final_value=1e-7, num_iter=int(args.epochs*args.param_continuation))
    else:

        raise ValueError(f"Unknown alpha_scheduler: {args.alpha_scheduler}")

    # Optimizer selection
    if args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(plu.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(plu.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(plu.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Learning rate scheduler: decay LR by 0.5 every 1/3 of epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.5, verbose=True)

    SSIM = StructuralSimilarityIndexMeasure().to(args.device)
    mse = nn.MSELoss()
    l1_loss = nn.L1Loss()
    ssim_loss = lambda x, y: 1 - SSIM(x, y)
    
    def combined_loss(x_hat, x):
        return 0.8*l1_loss(x_hat, x) + 0.2*ssim_loss(x_hat,x) + 0.02*hf_loss(x_hat,x)

    wandb.init(project="CAMSAP_UTOPY", name=exp_name, config=args, entity='entity')
    path = f"./results/{exp_name}"
    os.makedirs(path, exist_ok=True)
    if args.number_batches == 0:
        number_batches = len(trainloader)
    else:
        number_batches = args.number_batches
    alpha = 1.0
    continuation_path = 0
    for image in valloader:
        if args.dataset in ['cifar10', 'stl', 'fashion_mnist']:
            image = image[0]
        x_val_cont = image.to(args.device)
        break
    for iter in range(args.epochs):
        if args.type_run == 'baseline':
            alpha = 0.0
        else:
            if iter % args.freq == 0 and iter != 0 and iter < int(args.epochs*args.param_continuation):
                alpha = scheduler_alpha.step(iter)
            elif iter >= int(args.epochs*args.param_continuation):
                alpha = 0.0

        # Training loop
        data_loop = tqdm(trainloader, total=number_batches, colour='green')
        loss_x_meter = AverageMeter()
        psnr_train_s = AverageMeter()
        ssim_train_s = AverageMeter()
        plu.train()
        for i, data in enumerate(data_loop):
            optimizer.zero_grad()
            if args.dataset in ['cifar10', 'stl', 'fashion_mnist']:
                data = data[0]
            x = data.to(args.device)
            y = spc(x, type_calculation="forward")
            if args.add_noise:
                y = add_awgn(y,args.snr)
            yt = spc_t(x, type_calculation="forward")
            x0 = alpha * spc_t(yt, type_calculation="backward") + (1 - alpha) * spc(y, type_calculation="backward")
            den = x0.view(x0.size(0), -1).amax(dim=1, keepdim=True)   # shape (B, 1)
            x0 = x0 / den.view(x0.size(0), 1, 1, 1)
            x_hat, _ = plu(x0=x0, y=[yt, y], gamma=alpha)
            loss = combined_loss(x_hat, x)
            loss.backward()
            optimizer.step()
            loss_x_meter.update(loss.item(), y.shape[0])
            psnr_train_s.update(psnr_fun(x_hat, x).item(), y.shape[0])
            ssim_train_s.update(SSIM(x_hat, x).item(), y.shape[0])
            data_loop.set_description(f'Training [iter {iter}/{args.epochs}]')
            data_loop.set_postfix({'Loss X': loss_x_meter.avg, 'PSNR S': psnr_train_s.avg, 'SSIM S': ssim_train_s.avg, 'alpha': alpha})
            if i >= number_batches:
                break
        print('LR: ', optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        # Validation loop
        loss_x_val_meter = AverageMeter()
        psnr_val_s = AverageMeter()
        ssim_val_s = AverageMeter()
        plu.eval()
        with torch.no_grad():
            data_loop = tqdm(valloader, total=len(valloader), colour='magenta')
            for i, data in enumerate(data_loop):
                if args.dataset in ['cifar10', 'stl', 'fashion_mnist']:
                    data = data[0]
                x = data.to(args.device)
                y = spc(x, type_calculation="forward")
                if args.add_noise:
                    y = add_awgn(y, args.snr)
                yt = spc_t(x, type_calculation="forward")
                x0 = spc(y, type_calculation="backward")
                den = x0.view(x0.size(0), -1).amax(dim=1, keepdim=True)   # shape (B, 1)
                x0 = x0 / den.view(x0.size(0), 1, 1, 1)
                x_hat, _ = plu(x0=x0, y=[yt, y], gamma=0.0)
                loss = combined_loss(x_hat, x)
                loss_x_val_meter.update(loss.item(), y.shape[0])
                psnr_val_s.update(psnr_fun(x_hat, x).item(), y.shape[0])
                ssim_val_s.update(SSIM(x_hat, x).item(), y.shape[0])
                data_loop.set_description(f'Validation [iter {iter}/{args.epochs}]')
                data_loop.set_postfix({'Loss X': loss_x_val_meter.avg, 'PSNR S': psnr_val_s.avg, 'SSIM S': ssim_val_s.avg})
        # testing loop
        loss_x_test_meter = AverageMeter()
        psnr_test_s = AverageMeter()
        ssim_test_s = AverageMeter()
        plu.eval()
        
        with torch.no_grad():
            data_loop = tqdm(testloader, total=len(testloader), colour='blue')
            for i, data in enumerate(data_loop):
                if args.dataset in ['cifar10', 'stl', 'fashion_mnist']:
                    data = data[0]
                x = data.to(args.device)
                y = spc(x, type_calculation="forward")
                if args.add_noise:
                    y = add_awgn(y,args.snr)
                yt = spc_t(x, type_calculation="forward")
                x0 = spc(y, type_calculation="backward")
                den = x0.view(x0.size(0), -1).amax(dim=1, keepdim=True)   # shape (B, 1)
                x0 = x0 / den.view(x0.size(0), 1, 1, 1)                
                x_hat, _ = plu(x0=x0, y= [yt,y], gamma=0.0)
                loss = combined_loss(x_hat, x)
                loss_x_test_meter.update(loss.item(), y.shape[0])
                psnr_test_s.update(psnr_fun(x_hat, x).item(), y.shape[0])
                ssim_test_s.update(SSIM(x_hat, x).item(), y.shape[0])
                data_loop.set_description(f'Testing [iter {iter}/{args.epochs}]')
                data_loop.set_postfix({'Loss X': loss_x_test_meter.avg, 'PSNR S': psnr_test_s.avg, 'SSIM S': ssim_test_s.avg})

        # plot some reconstructions
        

        # compute continuation path

        x_hat_k = validate_continuation_path(spc, spc_t, plu, x_val_cont, alpha)
        if iter == 0:
            x_old = x_hat_k
        else:
            continuation_path = mse(x_hat_k, x_old).item()
            x_old = x_hat_k


        psnr_forimage_0_0 = psnr_fun(x_hat[0, 0].cpu(), x[0, 0].cpu()).item()
        SSIM_forimage_0_0 = SSIM(x_hat[0, 0].unsqueeze(0).unsqueeze(0).cpu(), x[0, 0].unsqueeze(0).unsqueeze(0).cpu()).item()
        psnr_forx0_0 = psnr_fun(x0[0, 0].cpu(), x[0, 0].cpu()).item()
        SSIM_forx0_0 = SSIM(x0[0, 0].unsqueeze(0).unsqueeze(0).cpu(), x[0, 0].unsqueeze(0).unsqueeze(0).cpu()).item()
        x = x.cpu().detach().numpy()
        x_hat = x_hat.cpu().detach().numpy()
        x0 = x0.cpu().detach().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(x[0, 0], cmap='gray')
        ax[0].set_title('Original')
        ax[1].imshow(x0[0, 0], cmap='gray')
        ax[1].set_title(f'Initial guess PSNR : {psnr_forx0_0:.2f}, SSIM: {SSIM_forx0_0:.2f}')
        ax[2].imshow(x_hat[0, 0],cmap='gray')
        ax[2].set_title(f'Reconstructed PSNR : {psnr_forimage_0_0:.2f}, SSIM: {SSIM_forimage_0_0:.2f}')
        plt.savefig(f'{path}/reconstruction.svg')
        wandb.log({'Loss X': loss_x_meter.avg, 'PSNR S': psnr_train_s.avg, 'SSIM S': ssim_train_s.avg,
                   'Loss X Val': loss_x_val_meter.avg, 'PSNR S Val': psnr_val_s.avg, 'SSIM S Val': ssim_val_s.avg,
                   'reconstruction': wandb.Image(fig),
                   'Loss X Test': loss_x_test_meter.avg, 'PSNR S Test': psnr_test_s.avg, 'SSIM S Test': ssim_test_s.avg,
                   'alpha': alpha, 'continuation_path': continuation_path})
        plt.close(fig)
        torch.save(plu.state_dict(), f'{path}/model.pth')
    wandb.finish()
        


if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(description='Learning cost')
    #----------------------------------- Dataset parameters ---------------------------------
    parser.add_argument('--n', type=int, default=64, help='Size of the image')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--dataset', type=str, default='CelebA', help='Dataset')  # OPTIONS 
    parser.add_argument('--augment', type=bool, default=False, help='Use augmentation')
    #----------------------------------- Sensing parameters ---------------------------------
    parser.add_argument('--cr', type=float, default=0.1, help='Compression ratio')
    parser.add_argument('--hadamard_ordering', type=str, default='cake_cutting', help='Ordering of the matrix')
    #----------------------------------- FISTA parameters -----------------------------------
    parser.add_argument('--algo', type=str, default='fista', help='Algorithm')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
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

    parser.add_argument('--type_A', type=str, default='bernoulli', help='Type of A matrix')
    parser.add_argument('--type_Ag', type=str, default='well_conditioned_from_A', help='Type of Ag matrix')
    parser.add_argument('--add_noise',type=bool,default=True)
    parser.add_argument('--snr',type=float,default=35)

    parser.add_argument('--param_continuation', type=float, default=0.7, help='Parameter for continuation path')
    
    args = parser.parse_args()
    
    main(args)
    
    # main(args)
