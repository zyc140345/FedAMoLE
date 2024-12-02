# Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures
This repository contains the official implementation for the work 
“Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures”. 
See more details in our [paper](https://arxiv.org/abs/2411.19128).
> A large amount of instructional text data is essential to enhance the performance of 
> pre-trained large language models (LLMs) for downstream tasks. This data can contain 
> sensitive information and therefore cannot be shared in practice, resulting in data 
> silos that limit the effectiveness of LLMs on various tasks. Federated learning (FL) 
> enables collaborative fine-tuning across different clients without sharing their data. 
> Nonetheless, in practice, this instructional text data is highly heterogeneous in both 
> quantity and distribution across clients, necessitating distinct model structures to 
> best accommodate the variations. However, existing federated fine-tuning approaches 
> either enforce the same model structure or rely on predefined ad-hoc architectures 
> unaware of data distribution, resulting in suboptimal performance. To address this 
> challenge, we propose FedAMoLE, a lightweight personalized federated fine-tuning framework 
> that leverages data-driven heterogeneous model architectures. FedAMoLE introduces the 
> Adaptive Mixture of LoRA Experts (AMoLE) module, which facilitates model heterogeneity 
> with minimal communication overhead by allocating varying numbers of LoRA-based domain 
> experts to each client. Furthermore, we develop a reverse selection-based expert 
> assignment (RSEA) strategy, which enables data-driven model architecture adjustment 
> during fine-tuning by allowing domain experts to select clients that best align with 
> their knowledge domains. Extensive experiments across six different scenarios of data 
> heterogeneity demonstrate that FedAMoLE significantly outperforms existing methods for 
> federated LLM fine-tuning, achieving superior accuracy while maintaining good scalability.

## Project Structure
```
.
├── model
|   ├── common.py  // loading pre-trained LLMs and tokenizer
|   ├── loss.py  // loss function
|   └── mole
|       ├── layer.py  // implementation of the AMoLE module
|       └── util.py  // utilities for the AMoLE module
├── data
|   ├── data_loader.py  // entrance to get dataloaders
|   ├── dataset.py  // utilities to load LLM datasets
|   ├── hete_partition.py  // utilities to simulate heterogeneous data distributions
|   ├── preprocess.py  // utilities to preprocess LLM datasets
|   └── prompt.py  // prompt templates
├── baselines  // implementation of baselines
|   ├── node.py  // server and client nodes
|   ├── trainer.py
|   ├── fdlora.py  // implementation of FDLoRA
|   ├── evaluator.py  // utilities to perform evaluation
|   └── recorder.py  // utilities to record experiment results
├── scripts
|   ├── run.py  // entrance to run batch experiments for accuracy comparison
|   ├── tune_client_num.py  // entrance to run batch experiments with different client numbers
|   ├── tune_hyper_params.py  // entrance to run batch experiments for hyper-parameters tuning
|   ├── train_moe.sh  // a script to reproduce the FedAMoLE results shown in TABLE 1 and 2
|   ├── train_baselines.sh  // a script to reproduce the baseline results shown in TABLE 1 and 2
|   ├── tune_hyper_params.sh  // a script to reproduce Fig. 9
|   └── tune_client_num.sh  // a script to reproduce Fig. 6
├── node.py  // inplementation of FedAMoLE's server and client nodes
├── optimize.py  // implementation of Problem IV.1 optimization
├── util.py  // general utilities
├── main.py  // entrance to run FedAMoLE
├── train_baselines.py  // entrance to run baselines
├── evaluation.py  // evaluation metrics
├── extract_result.py  // utilities to extract experiment results from logs 
└── plot.py  // utilities to plot figures
```

## Environment Setup
1. Create a conda environment and install the dependencies
```shell
git clone https://github.com/zyc140345/FedAMoLE.git && cd FedAMoLE
conda create -n fed_amole python==3.9.18
conda activate fed_amole
pip install -r requirements.txt
```
2. Download the pre-trained LLMs and datasets
```shell
# Llama-3.2-1B (you may request access on the Hugging Face Hub first)
huggingface-cli download meta-llama/Llama-3.2-3B --exclude "original/*"
# SNLI
huggingface-cli download stanfordnlp/snli --repo-type dataset
# Dolly-15K
huggingface-cli download databricks/databricks-dolly-15k --repo-type dataset
# Natural Instructions
mkdir ~/.dataset && cd ~/.dataset
wget https://github.com/allenai/natural-instructions/archive/refs/tags/v2.8.zip && unzip v2.8.zip
```

## Reproduce the Main Results
Run the following commands to generate experimental data:
1. FedAMoLE on SNLI, Dolly-15K and Natural Instructions
```shell
./scripts/train_moe.sh
```
2. Baselines on SNLI, Dolly-15K and Natural Instructions
```shell
./scripts/train_baselines.sh
```
3. Scalability
```shell
./scripts/tune_client_num.sh
```
4. Hyper-parameter sensitivity
```shell
./scripts/tune_hyper_params.sh
```
To generate figures, modify `plot.py` to specify the figure type and the path to the experimental data. Then, run the following commands:
```shell
mkdir figures
python plot.py
```

## License
This project adopts the MIT License. If the implementations and/or our paper were useful to you, please consider citing this [work](https://arxiv.org/abs/2411.19128):
```bibtex
@misc{zhang2024personalized,
    title = {Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures},
    author = {Yicheng Zhang and Zhen Qin and Zhaomin Wu and Shuiguang Deng},
    year = {2024},
    eprint = {2411.19128},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    doi = {10.48550/arXiv.2411.19128}
}
```