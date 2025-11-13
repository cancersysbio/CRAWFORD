import os
import h5py
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import tqdm
from PIL import Image
from utils.unet import BaseUnet64, SRUnet256
from utils.crawford import CRAWFORD
from utils.cdm_trainer import CRAWFORDTrainer
from kornia.filters import gaussian_blur2d
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LazyH5Dataset(Dataset):
    def __init__(self, h5_files, image_size=(256, 256)):
        self.h5_files = h5_files
        self.num_samples_per_file = [self._get_num_samples(f) for f in h5_files]
        self.cumulative_sizes = np.cumsum([0] + self.num_samples_per_file)
        self.resize_transform = transforms.Resize(image_size)
        self.image_size = image_size

    def _get_num_samples(self, h5_file):
        with h5py.File(h5_file, 'r') as f:
            return f['images'].shape[0]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _find_file_idx(self, idx):
        return np.searchsorted(self.cumulative_sizes, idx, side='right') - 1

    def __getitem__(self, idx):
        file_idx = self._find_file_idx(idx)
        sample_idx = idx - self.cumulative_sizes[file_idx]

        with h5py.File(self.h5_files[file_idx], 'r') as f:
            image = torch.tensor(f['images'][sample_idx], dtype=torch.float32).permute(2, 0, 1) / 255.0  # [0, 1]
            feature = torch.tensor(f['features'][sample_idx], dtype=torch.float32)

        image = self.resize_transform(image)
        return image, feature


def create_dataloader(h5_files, batch_size=64, num_workers=8, image_size=(256, 256)):
    dataset = LazyH5Dataset(h5_files, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader

if __name__ == '__main__':
    # Set up argument parsing (no config file and no wandb)
    import argparse
    parser = argparse.ArgumentParser(description='DDPM training on histology data using BaseUnet64')
    parser.add_argument('--checkpoint', type=str, default=None, help='File with the checkpoint to start with (if continuing)')
    parser.add_argument('--seed', type=int, default=11, help='Seed for random generation')
    parser.add_argument('--first_train', action='store_true', default=False, help='First training run on this checkpoint')
    parser.add_argument('--save_dir', type=str, default='results/', help='Directory to save model checkpoints')
    parser.add_argument('--train_data_dir', type=str, default='data/', help='Directory containing the training .h5 files')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--gpus', type=str, default="0", help='GPUs list')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name')
    parser.add_argument('--num_steps', type=int, default=1000000, help='Number of training steps for the model')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for the diffusion model')
    parser.add_argument('--num_samples_save', type=int, default=1000, help='Number of samples to save')
    parser.add_argument('--max_batch_size', type=int, default=4, help='Max batch size for gradient accumulation')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    parser.add_argument('--unet_number_to_train', type=int, default=2, help='Unet number to train, set to 2 for super-resolution')
    args = parser.parse_args()

    # Set up seed
    set_seed(args.seed)

    # Set up transformations and dataset
    transforms_ = nn.Sequential(
        transforms.ConvertImageDtype(torch.float)
    )

    # Create DataLoader
    filenames_samples = os.listdir(args.train_data_dir)
    h5_files = [os.path.join(args.train_data_dir, filename) for filename in filenames_samples]
    
    # Sample images for testing
    with h5py.File(h5_files[0], 'r') as f:
        feats = f['features'][:]
        f0 = feats[3]
        
    with h5py.File(h5_files[1], 'r') as f:
        feats = f['features'][:]
        f1 = feats[3]
        
    uni_embeds_to_gen = torch.tensor(f0, dtype=torch.float32)
    uni_embeds_to_gen = uni_embeds_to_gen.unsqueeze(0)
    uni_embeds_to_gen_f1 = torch.tensor(f1, dtype=torch.float32)
    uni_embeds_to_gen_f1 = uni_embeds_to_gen_f1.unsqueeze(0)
    
    train_dl = create_dataloader(h5_files, batch_size=args.batch_size)
    total_batches = len(train_dl)

    # Initialize the BaseUnet64 with 1024-dimensional conditioning
    unet_base = BaseUnet64(
        cond_dim=512,
        cond_on_uni=True,
        max_uni_len=512
    )
    
    unet_sr = SRUnet256(
        cond_dim=512,  
        cond_on_uni=True,
        max_uni_len=512
    )
    
    print(unet_base)
    print(unet_sr)

    # Set up CRAWFORD
    diffusion_model = CRAWFORD(
        unets=[unet_base, unet_sr],
        image_sizes=[64, 256],
        timesteps=args.timesteps,
        cond_drop_prob=0.1,
        condition_on_uni=True,
        uni_embed_dim=1536
    )

    # Move the model to GPU
    trainer = CRAWFORDTrainer(diffusion_model, only_train_unet_number=args.unet_number_to_train).cuda()
        
    start_epoch = 0
    start_batch_idx = 0
    total_steps_manual = 0
    base_loss_tracker = 0
    sr_loss_tracker = 0
    
    if args.checkpoint and args.first_train == True:
        trainer.load(args.checkpoint)
    elif args.checkpoint and args.first_train == False:
        checkpoint = trainer.load(args.checkpoint)
        start_epoch = checkpoint.get('epoch', 0)
        start_batch_idx = checkpoint.get('batch_idx', 0)
        total_steps_manual = checkpoint.get('total_steps_manual', 0)
        
    print("start_batch_idx", start_batch_idx, flush=True)
    print("start_epoch", start_epoch, flush=True)

    # Start training
    print('Starting training...')
    #start_time = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        
        for batch_idx, batch in enumerate(tqdm.tqdm(train_dl)):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue
                
            total_steps_manual += 1
            progress_percent = (batch_idx) / total_batches * 100
            images, features = batch
            images = images.cuda()
            features = features.cuda()  # Assuming your dataloader provides a 'features' key
            
            # Perform training step
            loss_sr = trainer(images, uni_embeds=features, unet_number=2, max_batch_size=args.max_batch_size)
            trainer.update(unet_number=2)
            sr_loss_tracker += loss_sr
            
            # Save model checkpoints and generated samples at intervals
            if total_steps_manual % args.num_samples_save == 0:
                print()
                print("progress_percent: " + str(progress_percent) + "%")
                print("Current loss:", loss_sr)
                #print("Average base loss:", str(base_loss_tracker / args.num_samples_save))
                print("Average sr loss:", str(sr_loss_tracker / args.num_samples_save))
                sr_loss_tracker = 0
                
                # Generate samples using the trained model
                generated_images_sr = trainer.sample(batch_size=1, uni_embeds=uni_embeds_to_gen, cond_scale=1.0, stop_at_unet_number=2)
                generated_images1_sr = trainer.sample(batch_size=1, uni_embeds=uni_embeds_to_gen_f1, cond_scale=1.0, stop_at_unet_number=2)
                
                # Save generated samples to disk
                    
                for i, img in enumerate(generated_images_sr):
                    img = img.detach().cpu().numpy()
                    img = img * 255.0
                    img = img.astype(np.uint8)
                
                    # Handle the case of 3-channel RGB images
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                
                    # Convert to a PIL Image and save it
                    pil_img = Image.fromarray(img)
                    pil_img.save(f"{args.save_dir}/sr_generated_image1_{total_steps_manual}_{i}.png")
                    
                for i, img in enumerate(generated_images1_sr):
                    img = img.detach().cpu().numpy()
                    img = img * 255.0
                    img = img.astype(np.uint8)
                
                    # Handle the case of 3-channel RGB images
                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                        
                    # Convert to a PIL Image and save it
                    pil_img = Image.fromarray(img)
                    pil_img.save(f"{args.save_dir}/sr_generated_image2_{total_steps_manual}_{i}.png")
                    
                # Save model checkpoint
                trainer.save(f'{args.save_dir}/model.pt', epoch=epoch, batch_idx=batch_idx, total_steps_manual=total_steps_manual)
            
    print('Training complete!')