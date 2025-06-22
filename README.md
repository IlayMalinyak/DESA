# DESA - Dual Embedding model for Stellar Astrophysics

![Project Logo](images/high_level_arch_blue.png)
*DESA architecture*

DESA is a multimodal model for stellar data. DESA consists of pretrianed spectra and light curve encoders and a novel modula, DualFormer, that effectively aligns the induvidual embeddings. DESA outperforms other well-known self supervised and multi modality framework on a veriaty of tasks including zero-shot, few-shot and fine-tuning on challenging astronomical datasets.  


## Setup Environment

You can download a copy of all the files in this repository by cloning the
repository:

    git clone https://github.com/IlayMalinyak/DESA.git
    cd DESA
    pip install -r requirements.txt

## Run Experiments
training was done in a multi-gpu slurm-based environemnt with. To run experiments on other environemtns, some small modifications are needed.
all experiments can be found in [src](https://github.com/IlayMalinyak/DESA/tree/main/src)
folder.

## Pretrained Models

We provide three pretrained models on the Hugging Face model hub for easy access. Each model includes weights (.pth), configuration (yaml), and additional resources. Model details and downloads are below.

| Model Name | Description | # Params. | Download |
|------------|-------------|-----------|----------|
| DESA | full model | [545]M | [🤗 Hub](https://huggingface.co/Ilayk\DESA) |
| spectra encoder | pre trained spectra encoder | [522]M | [🤗 Hub](https://huggingface.co/Ilayk/spectra_encoder) |
| light curve encoder | pre trained light curve encoder | [11]M | [🤗 Hub](https://huggingface.co/Ilayk/lc_encoder) |

## Results

![umap](images/compare_umaps_flag_CMD_numeric.png)
*final features UMAP*

![umap](images/experiments_comparison.png)
*binary detection experiment results*