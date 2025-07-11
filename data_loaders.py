import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch_dct as dct
from torchmetrics.image import PeakSignalNoiseRatio
import os
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset
# import deepinv as dinv
# from deepinv.utils.demo import load_dataset, load_degradation
from PIL import Image
import cv2 
from concurrent.futures import ThreadPoolExecutor
 
class SimpleImageLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith('.png')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('L')  # Convert image to grayscale

        if self.transform:
            image = self.transform(image)

        return image, 0  # Returns a dummy label since there are no classes



class PNGImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [img for img in os.listdir(image_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image =  Image.fromarray(np.array(cv2.imread(img_path))).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Assuming labels are encoded in the file name or directory structure, modify as needed
        

        return image
    
class JPGImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [img for img in os.listdir(image_dir) if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')]
        for img_path in self.image_names:
            with Image.open(img_path) as image:
                image = image.convert('L')  # or 'RGB'
                if self.transform:
                    image = self.transform(image)
                self.images.append(image)
                
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image =  Image.fromarray(np.array(cv2.imread(img_path))).convert('L')
        if self.transform:
            image = self.transform(image)
        
        # Assuming labels are encoded in the file name or directory structure, modify as needed
        

        return image

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png') or img.endswith('.jpg')]#[:2993]
        # Preload all images into RAM as tensors
        self.images = []
        # Use multithreading to speed up image loading

        def load_image(img_path):
            with Image.open(img_path) as image:
                image = image.convert('L')  # or 'RGB'
                if self.transform:
                    image = self.transform(image)
            return image

        with ThreadPoolExecutor() as executor:
            self.images = list(tqdm(executor.map(load_image, self.image_paths), total=len(self.image_paths), desc="Loading images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
def get_dataloaders(args):
    # Data augmentation transforms
    augmentation_transforms = []
    if args.augment:
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
        ]
    # Training transform (with augmentation)
    train_transform = transforms.Compose([
        *augmentation_transforms,
        transforms.ToTensor(),
        transforms.Resize((args.n, args.n), antialias=True),
        transforms.Grayscale()  # Remove if you want RGB
    ])
    # Test/val transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.n, args.n), antialias=True),
        transforms.Grayscale()  # Remove if you want RGB
    ])

    if args.dataset == 'mnist':
        full_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        total_len = len(full_trainset)
        val_size = int(0.1 * total_len)
        train_size = total_len - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'cifar10':
        full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        total_len = len(full_trainset)
        val_size = int(0.1 * total_len)
        train_size = total_len - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'stl':
        full_trainset = datasets.STL10(root=r"C:\\Roman\\LearningCost\\data\\", split='train', download=True, transform=train_transform)
        testset = datasets.STL10(root='C:\\Roman\\LearningCost\\data\\', split='test', download=True, transform=test_transform)
        total_len = len(full_trainset)
        val_size = int(0.1 * total_len)
        train_size = total_len - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'BSDS500':
        dataset = PNGImageDataset(r'./data/BSDS500', transform=None)
        total_len = len(dataset)
        val_size = int(0.1 * total_len)
        train_size = int(0.8 * total_len) - val_size
        test_size = total_len - train_size - val_size
        trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])
        trainset.dataset.transform = train_transform
        valset.dataset.transform = test_transform
        testset.dataset.transform = test_transform
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'CelebA':
        train_dir = './CelebA2/train'
        valid_dir = './CelebA2/test'
        full_trainset = ImageDataset(train_dir, transform=train_transform)
        total_len = len(full_trainset)
        val_size = int(0.1 * total_len)
        train_size = total_len - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])
        testset = ImageDataset(valid_dir, transform=test_transform)
        trainloader =DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset =='ct':
        dataset = PNGImageDataset(args.dataset_path, transform=None)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])
        trainset.dataset.transform = train_transform
        valset.dataset.transform = test_transform
        testset.dataset.transform = test_transform
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'DIV2K':
        train_dir = './data/DIV2K/DIV2K_train_HR'
        valid_dir = './data/DIV2K/DIV2K_valid_HR'
        fullset = ImageDataset(train_dir, transform=train_transform)
        total_len = len(fullset)
        val_size = int(0.1 * total_len)
        train_size = total_len - val_size
        trainset, valset = random_split(fullset, [train_size, val_size])
        testset = ImageDataset(valid_dir, transform=test_transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    else:
        # Generic fallback: try to load as PNGImageDataset and split
        dataset = PNGImageDataset(args.dataset_path, transform=None)
        total_len = len(dataset)
        val_size = int(0.1 * total_len)
        train_size = int(0.8 * total_len) - val_size
        test_size = total_len - train_size - val_size
        trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])
        trainset.dataset.transform = train_transform
        valset.dataset.transform = test_transform
        testset.dataset.transform = test_transform
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    return trainloader, valloader, testloader
    
import scipy

def get_data_training(x,ns,m,args):

    vars = torch.linspace(args.min_var, args.max_var, x.shape[0]).to(args.device)
    vars = vars[torch.randperm(vars.shape[0])]
    vars = vars.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    xn = x + torch.randn_like(x)*vars

    w = torch.rand(x.size(0),args.n**2-m)*0.1
    w = w.to(args.device)
    xm = x.view(x.shape[0],-1) + w@ns.t()
    xm = xm.view(x.shape)
    
    return x, xn, xm


