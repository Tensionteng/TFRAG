# SRRF: Self-adaptive Retrieval-augmented Reinforcement learning for Time Series Forecasting

**Please note: This codebase is under active development, and features/documentation will be progressively updated and refined.**

## Project Introduction

This project implements SRRF (Self-adaptive Retrieval-augmented Reinforcement learning for time series Forecasting), a novel plug-and-play training enhancement module designed to address the common spectral bias issue in deep learning models for time series forecasting. By combining Retrieval-Augmented Generation (RAG) and Reinforcement Learning (RL), SRRF enables base models to internalize high-frequency dynamic modeling capabilities during the training phase, without incurring additional computational costs or requiring architectural changes to the base model during inference.

## Usage

1.  Install Python 3.8. For convenience, execute the following command:

    ```bash
    pip install -r requirements.txt
    ```

2.  Prepare Data. You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). Then place the downloaded data in the folder `./dataset`.(We use `ETT`, `ECL`, `Traffic`, `Weather`, `Exchange` datasets for our experiments.
)

    A summary of supported datasets (from original TSLib):
    <p align="center">
    <img src=".\pic\dataset.png" height="200" alt="Overview of supported datasets" align=center />
    </p>
    
3.  Train and evaluate the model. We provide experiment scripts for benchmarks under the `./scripts/` folder. You can reproduce experiment results using examples like the following:
    *(You will need to provide or modify script examples here to reflect how to run SRRF-enhanced models.)*

    ```bash
    # Example for long-term forecast with a base model
    # bash ./scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh

    # To run with SRRF enhancement, see the section below on --use_rag and adapt your scripts.
    ```

4.  Develop your own model.
    * Add your model file to the `./models` folder. You can refer to `./models/Transformer.py` as an example.
    * Include the newly added model in the `Exp_Basic.model_dict` within `./exp/exp_basic.py`.
    * Create corresponding experiment scripts under the `./scripts` folder.

## Enabling SRRF Enhancement: The `--use_rag` Parameter

The `--use_rag` parameter is a key command-line argument in this codebase used to activate the Retrieval-Augmented Generation (RAG) mechanism, which is a core component of our proposed SRRF (Self-adaptive Retrieval-augmented Reinforcement learning) framework.

**What it does:**
* When you include `--use_rag` (typically set as `--use_rag 1`) in your training script, your base forecasting model will leverage an external database of similar historical samples.
* During training, relevant historical sequences are retrieved to provide contextual grounding for the base model.
* This mechanism works in conjunction with the Reinforcement Learning agent (also typically activated by relevant parameters) to help the model better capture complex temporal patterns, especially high-frequency dynamics, and mitigate the spectral bias common in traditional MSE-optimized models.

**Potential Benefits:**
* Improved forecasting accuracy, particularly for volatile time series and fine-grained patterns.
* Enhanced model robustness through historical context.
* Better generalization to unseen patterns by learning from diverse retrieved samples.

**How to use (Example):**
To train a model (e.g., iTransformer) on the ETTh1 dataset using the SRRF framework, you would typically add the `--use_rag 1` argument when executing your training script:

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_SRRF \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp_with_SRRF' \
  --itr 1 \
  --use_rag 1 # Add this argument to enable RAG/SRRF
  # You may also need to add other SRRF-specific parameters, 
  # e.g.:
  # --num_samples [Ns_value] \ # Number of RL samples (Ns)
  # --num_retrieval [k_value] # Number of retrieved exemplars (k)
```
# Thanks
This repo is fork from [TS-lib](https://github.com/thuml/Time-Series-Library)
