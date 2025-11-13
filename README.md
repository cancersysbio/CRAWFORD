# CRAWFORD: Cascaded Diffusion Model for UNI2 (or train your own!) Embedding -> H&E Tile Image

**CRAWFORD** is a cascaded diffusion model for the generation of images based on an imput UNI2 embedding. CRAWFORD was trained
on H&E of 61 tissue types across healthy and cancerous tissue. CRAWFORD is designed to generate high quality images from
embeddings for 1) synthetic embedding visualization for validation and 2) perturbational embedding analysis for model interpretation.

> **Companion tool:**  
> - **BERGERON** — class conditional variational autoencoder for synthetic H&E tile embedding generation  

## Installation

```bash
# Conda
conda env create -f environment.yml
conda activate CRAWFORD
```

## Generating from Pretrained Checkpoint
Our pretrained CRAWFORD model for generating UNI2 images is available on HuggingFace (UPDATE WITH LINK)

Data Format
Each .h5 file contains all embeddings to generate images from.

Datasets inside each file

features: [N_tiles, D] — embedding vectors (D=embedding dimension of foundation model)

Example directory

/data/h5_files/
  TCGA-XX-0001.h5
  TCGA-XX-0002.h5
  
Quickstart
Generate from Pretrained Checkpoint

```bash
python CRAWFORD_generate.py \
--h5_path /path/to/h5_file.h5 \
--checkpoint /path/to/model.pt \
--output_dir /path/to/output_directory \
--num_imgs_per_emb 5
```

## Training a New CRAWFORD Model (for a different foundation model)
We recommend using the publicly available H&E datasets TCGA and GTEx for a wide varieyt of tissue types and disease statuses

Data Format
Each .h5 file contains all embeddings to generate images from.

Datasets inside each file

features: [N_tiles, D] — embedding vectors (D=embedding dimension of foundation model)

images: [N_tiles, img_size, img_size, 3] - H&E tile images associated with each embedding

Example directory

/data/h5_files/
TCGA-XX-0001.h5
TCGA-XX-0002.h5
  
Quickstart
Train a new CRAWFORD model

```bash
python CRAWFORD_train.py \
--save_dir /path/to/save_directory \
--train_data_dir /path/to/training_directory \
--num_epochs 10 \
--batch_size 64 \
--img_size 256 \
--unet_number_to_train 2 \
--num_samples_save 600
```

Training Notes
Train until loss stabilizes and generated image quality is satisfactory. We recommend generating a few images at regular intervals
(see # Sample images for testing section in CRAWFORD_train.py) to identify generated image quality consistently

Citation
If you use CRAWFORD in your work:

To-update

License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Acknowledgments
Developed in the Curtis Lab at Stanford University.
