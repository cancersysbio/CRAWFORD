"""
This script is adapted from sequoia-pub available at:
https://https://github.com/gevaertlab/sequoia-pub

Original author(s): Marija Pizurica, Yuanning Zheng, Francisco Carrillo Perez
Modifications made by: Brennan Simon
Summary of changes: Modifed to accept UNI2 embeddings instead of bulk RNAseq embeddings and 
                    minor changes to workflow.

Licensed under the MIT license.
"""

import os
import h5py
import torch
import numpy as np
from PIL import Image
import argparse
from utils.unet import BaseUnet64, SRUnet256
from utils.crawford import CRAWFORD
from utils.cdm_trainer import CRAWFORDTrainer


def load_trained_model(checkpoint_path, device):
    unet_base = BaseUnet64(cond_dim=512, cond_on_uni=True, max_uni_len=512)
    unet_sr = SRUnet256(cond_dim=512, cond_on_uni=True, max_uni_len=512)
    
    model = CRAWFORD(
        unets=[unet_base, unet_sr],
        image_sizes=[64, 256],
        timesteps=100,
        cond_drop_prob=0.1,
        condition_on_uni=True,
        uni_embed_dim=1536
    )
    
    trainer = CRAWFORDTrainer(model, only_train_unet_number=2).to(device)
    trainer.load(checkpoint_path)
    return trainer


def denormalize_image(img_tensor):
    img_np = img_tensor.detach().cpu().numpy() * 255.0
    img_np = img_np.astype(np.uint8)
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    return Image.fromarray(img_np)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained diffusion model
    trainer = load_trained_model(args.checkpoint, device)

    # Load UNI embeddings from HDF5
    with h5py.File(args.h5_path, "r") as f:
        embeddings = f["features"][:]  # shape: (N, 1536)

    basename = os.path.splitext(os.path.basename(args.h5_path))[0]

    for i, embedding in enumerate(embeddings):
        emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1536)

        for j in range(args.num_imgs_per_emb): # Generate 5 images per embedding
            gen_img = trainer.sample(batch_size=1, uni_embeds=emb_tensor, cond_scale=1.0, stop_at_unet_number=2)[0]
            gen_img_pil = denormalize_image(gen_img)
            gen_img_pil.save(os.path.join(args.output_dir, f"{basename}_{i}_gen{j}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from multiple UNI embeddings in HDF5")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to .h5 file containing 'features' dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--num_imgs_per_emb", type=int, default=5, help="Number of tile images to generate per embedding")
    
    args = parser.parse_args()
    main(args)
