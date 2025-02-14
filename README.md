# Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures
This repository contains the official implementation for the work 
“Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures”.
> Large-scale instruction data is essential for aligning pre-trained Large Language Models (LLMs) 
> with human instructions through instruction tuning. However, such data is often sourced from distinct clients, 
> characterized by privacy concerns and heterogeneity. Federated Learning (FL) addresses privacy issues by 
> enabling collaborative fine-tuning without data sharing. However, traditional FL assumes a uniform model structure 
> and data distribution, making it ineffective for training on highly heterogeneous data with varying amounts and 
> formats. To address this, we propose FedAMoLE—a lightweight, personalized FL framework enabling data-driven 
> heterogeneous model architectures based on mixture of LoRA experts (MoLE). It features an adaptive MoLE module 
> for aggregating heterogeneous models and a reverse selection-based expert assignment strategy to optimize model 
> architectures based on local and global data distributions. Experiments across five scenarios show that FedAMoLE 
> improves accuracy by an average of 5.14% over existing approaches while obtaining good scalability.

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
├── optimize.py  // implementation of Problem C.1 optimization
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
