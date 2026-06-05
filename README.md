# Programmable Amplification Kinetics Enable AI-Driven High-Level Multiplexing in Single-Channel TaqMan Real-Time PCR

Official implementation for BYOL pretraining and T-CDAN training for qPCR curve
classification.

## Authors

Louis Kreitmann*, Kenny Malpartida-Cardenas*, Ye Mao*, Zexuan Zhao, San Chun Hin,
Anirudhha Hazarika, Ke Xu, Luca Miglietta, Zara Breese, Alison H. Holmes,
Karen Brengel-Pesce, Laurent Drazek, and Jesus Rodriguez-Manzano.

*Equal contribution.

## Affiliations

1. Department of Infectious Disease, Imperial College London, London, UK
2. Open Innovation & Partnerships, bioMerieux, Marcy-l'Etoile, France
3. Department of Electrical and Electronic Engineering, Imperial College London, London, UK
4. The Fleming Initiative, Imperial College London and Imperial College Healthcare NHS Trust, London, UK
5. Molecular Biology, Research & Development, bioMerieux, Grenoble, France
6. Data Science, Research & Development, bioMerieux, Grenoble, France

Correspondence: `j.rodriguez-manzano@imperial.ac.uk`

## Repository Layout

```text
ACA_qPCR/
  BYOL_CDAN_7_plex.py        # 7-plex entry point
  BYOL_CDAN_8_plex.py        # 8-plex entry point
  requirements.txt
  src/                       # implementation code
  7_plex_data/               # downloaded 7-plex data
  8_plex_data/               # downloaded 8-plex data
  7_plex_output/             # generated checkpoints/results
  8_plex_output/             # generated checkpoints/results
```

## Dataset

Download the dataset from the temporary Google Drive link:

```text
https://drive.google.com/drive/folders/<temporary-dataset-link>
```

Place the downloaded data folders directly inside `ACA_qPCR/`. The expected CSVs
include:

- `7_plex_data/df_qPCR_GB_2025.csv`
- `7_plex_data/df_dPCR_GB_2025.csv`
- `7_plex_data/df_dPCR_SP_2025.csv`
- `7_plex_data/param_df_5_20250305_2248.csv`
- `8_plex_data/df_8plex_CNS_qPCR_GB.csv`
- `8_plex_data/df_8plex_CNS_qPCR_SP.csv`
- `8_plex_data/df_8plex_CNS_dPCR_total.csv`
- `8_plex_data/params_df_5_spline_total.csv`

Each curve CSV should contain numeric qPCR curve columns and a `Target_cat` label
column.

## Environment

```bash
conda create -n byol-cdan python=3.10 -y
conda activate byol-cdan
cd ACA_qPCR
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version before
installing the remaining packages.

## Train BYOL Then T-CDAN

Run all commands from `ACA_qPCR/`.

For 8-plex:

```bash
# 1. BYOL contrastive pretraining
python BYOL_CDAN_8_plex.py --run_cl true

# 2. T-CDAN training initialized from the BYOL checkpoint
python BYOL_CDAN_8_plex.py \
  --run_byol_cdan true \
  --pretrained_cl_checkpoint 8_plex_output/pretrained_model_CL_final.pth
```

For 7-plex:

```bash
# 1. BYOL contrastive pretraining
python BYOL_CDAN_7_plex.py --run_cl true

# 2. T-CDAN training initialized from the BYOL checkpoint
python BYOL_CDAN_7_plex.py \
  --run_byol_cdan true \
  --pretrained_cl_checkpoint 7_plex_output/pretrained_model_CL_final.pth
```

Useful options include `--num_iterations`, `--test_interval`, `--batch_size`,
`--lr`, `--weight_decay`, `--trade_off`, `--dir_data`, `--dir_out`, and
`--wandb`.

## Evaluation

Evaluate a saved 8-plex T-CDAN checkpoint:

```bash
python BYOL_CDAN_8_plex.py \
  --eval_checkpoint 8_plex_output/byol_cdan.pth \
  --eval_split test
```

Evaluate a saved 7-plex T-CDAN checkpoint:

```bash
python BYOL_CDAN_7_plex.py \
  --eval_checkpoint 7_plex_output/byol_cdan.pth \
  --eval_split test
```

Training and evaluation outputs are written to `7_plex_output/` or
`8_plex_output/`, including checkpoints, logs, confusion matrices,
`run_config.json`, and `final_results.json`.
