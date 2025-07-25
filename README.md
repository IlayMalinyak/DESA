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
| DESA | full model | [545]M | [🤗 HF Hub](https://huggingface.co/Ilayk/DESA) |
| spectra encoder | pre trained spectra encoder | [522]M | [🤗 HF Hub](https://huggingface.co/Ilayk/spectra_encoder) |
| light curve encoder | pre trained light curve encoder | [11]M | [🤗 HF Hub](https://huggingface.co/Ilayk/lc_encoder) |

### Loading the Pretrained Models
After downloading the .pth files, you will need to update the [full config](https://github.com/IlayMalinyak/DESA/tree/main/nn/full_config.yaml) with the correct paths and hyperparameters. Then you can load dataloaders and models using the get_model and get_data functions:

    import generator
    from util.utils import Container
    local_rank, world_size, gpus_per_node = setup()
    args_dir = 'DESA/nn/full_config.yaml'
    data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
    def create_train_test_dfs(params):
        pass
        # this should be a function that creates a dataframe of data samples

    
    train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                    data_generation_fn=create_train_test_dfs,
                                                    dataset_name='LightSpec')
    desa_model, optim_args, tuner_args, complete_config, lc_model, spec_model = generator.get_model(data_args,
                                                                                        args_dir,
                                                                                        complete_config,
                                                                                        local_rank,
                                                                                        )
## Results

![umap](images/compare_umaps_flag_CMD_numeric.png)
*final features UMAP*

![umap](images/experiments_comparison.png)
*binary detection experiment results*