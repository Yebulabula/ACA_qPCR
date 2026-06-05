<h1 align="center">
Programmable Amplification Kinetics Enable AI-Driven High-Level Multiplexing in Single-Channel TaqMan Real-Time PCR
</h1>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#data-preparation">Data</a> |
  <a href="#byol-pretraining">BYOL Pretraining</a> |
  <a href="#t-cdan-training">T-CDAN Training</a> |
  <a href="#evaluation">Evaluation</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-required-ee4c2c">
  <img alt="Task" src="https://img.shields.io/badge/Task-qPCR%20classification-green">
</p>

<p align="center">
  Official implementation for BYOL pretraining and T-CDAN fine-tuning for qPCR curve classification.
</p>

## Authors

Louis Kreitmann*, Kenny Malpartida-Cardenas*, Ye Mao*, Zexuan Zhao, San Chun Hin,
Anirudhha Hazarika, Ke Xu, Luca Miglietta, Zara Breese, Alison H. Holmes,
Karen Brengel-Pesce, Laurent Drazek, and Jesus Rodriguez-Manzano.

*Equal contribution. Correspondence: `j.rodriguez-manzano@imperial.ac.uk`

**Affiliations.** Imperial College London; The Fleming Initiative, Imperial
College London and Imperial College Healthcare NHS Trust; bioMerieux.

## Contents

- [News](#news)
- [Key Takeaways](#key-takeaways)
- [Expected Repository Structure](#expected-repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [BYOL Pretraining](#byol-pretraining)
- [T-CDAN Training](#t-cdan-training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## News

- `2026-06-05`: Repository reorganized with a compact `src/` layout and a
  BYOL -> T-CDAN workflow.

## Key Takeaways

- **Core problem.** Single-channel TaqMan real-time PCR limits multiplexing
  because overlapping fluorescence signals are difficult to separate reliably.
- **Model idea.** Programmable amplification kinetics provide curve-level
  signatures that can be learned by a Transformer-based qPCR classifier.
- **Training recipe.** BYOL contrastive pretraining learns curve
  representations before T-CDAN adapts the model for target-domain
  classification.
- **Supported settings.** The repository includes 7-plex and 8-plex entry
  points with matching dataset and output folders.

## Expected Repository Structure

Place downloaded data inside `ACA_qPCR/`:

```text
ACA_qPCR/
├── BYOL_CDAN_7_plex.py        # 7-plex entry point
├── BYOL_CDAN_8_plex.py        # 8-plex entry point
├── requirements.txt
├── src/                       # implementation code
│   ├── byol_cdan_common.py
│   ├── byol-pytorch/
│   ├── tools/
│   └── tst/
├── 7_plex_data/               # downloaded 7-plex data
├── 8_plex_data/               # downloaded 8-plex data
├── 7_plex_output/             # generated 7-plex checkpoints/results
└── 8_plex_output/             # generated 8-plex checkpoints/results
```

## Installation

### 1. Create an environment

```bash
conda create -n byol-cdan python=3.10 -y
conda activate byol-cdan
```

### 2. Install PyTorch

Install the PyTorch build matching your CUDA setup. For CPU-only use, the default
pip package is sufficient; for GPU training, follow the official PyTorch install
selector for your CUDA version.

### 3. Install remaining dependencies

```bash
cd ACA_qPCR
pip install -r requirements.txt
```

## Data Preparation

Download the dataset from the temporary Google Drive link:

```text
https://drive.google.com/drive/folders/<temporary-dataset-link>
```

Extract the downloaded folders into `ACA_qPCR/`. The expected files include:

- `7_plex_data/df_qPCR_GB_2025.csv`
- `7_plex_data/df_dPCR_GB_2025.csv`
- `7_plex_data/df_dPCR_SP_2025.csv`
- `7_plex_data/param_df_5_20250305_2248.csv`
- `8_plex_data/df_8plex_CNS_qPCR_GB.csv`
- `8_plex_data/df_8plex_CNS_qPCR_SP.csv`
- `8_plex_data/df_8plex_CNS_dPCR_total.csv`
- `8_plex_data/params_df_5_spline_total.csv`

Each curve CSV should contain numeric qPCR curve columns and a `Target_cat`
label column.

## BYOL Pretraining

Run all commands from `ACA_qPCR/`.

### 8-plex

```bash
python BYOL_CDAN_8_plex.py --run_cl true
```

### 7-plex

```bash
python BYOL_CDAN_7_plex.py --run_cl true
```

The final BYOL checkpoints are saved as:

- `8_plex_output/pretrained_model_CL_final.pth`
- `7_plex_output/pretrained_model_CL_final.pth`

## T-CDAN Training

### 8-plex

```bash
python BYOL_CDAN_8_plex.py \
  --run_byol_cdan true \
  --pretrained_cl_checkpoint 8_plex_output/pretrained_model_CL_final.pth
```

### 7-plex

```bash
python BYOL_CDAN_7_plex.py \
  --run_byol_cdan true \
  --pretrained_cl_checkpoint 7_plex_output/pretrained_model_CL_final.pth
```

Useful options include `--num_iterations`, `--test_interval`, `--batch_size`,
`--lr`, `--weight_decay`, `--trade_off`, `--dir_data`, `--dir_out`, and
`--wandb`.

## Evaluation

### 8-plex

```bash
python BYOL_CDAN_8_plex.py \
  --eval_checkpoint 8_plex_output/byol_cdan.pth \
  --eval_split test
```

### 7-plex

```bash
python BYOL_CDAN_7_plex.py \
  --eval_checkpoint 7_plex_output/byol_cdan.pth \
  --eval_split test
```

Outputs are written to `7_plex_output/` or `8_plex_output/`, including
checkpoints, logs, confusion matrices, `run_config.json`, and
`final_results.json`.

## Citation

If you find this repository useful, please cite the paper:

```bibtex
@article{kreitmann2026programmable,
  title   = {Programmable Amplification Kinetics Enable AI-Driven High-Level Multiplexing in Single-Channel TaqMan Real-Time PCR},
  author  = {Kreitmann, Louis and Malpartida-Cardenas, Kenny and Mao, Ye and Zhao, Zexuan and San Chun Hin and Hazarika, Anirudhha and Xu, Ke and Miglietta, Luca and Breese, Zara and Holmes, Alison H. and Brengel-Pesce, Karen and Drazek, Laurent and Rodriguez-Manzano, Jesus},
  year    = {2026}
}
```

## License

Please refer to the project license before using the code or dataset.
