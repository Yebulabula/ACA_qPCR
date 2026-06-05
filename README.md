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
  <a href="https://arxiv.org/abs/ARXIV_ID">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Paper-b31b1b">
  </a>
  <a href=".">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-black">
  </a>
  <a href="https://drive.google.com/drive/folders/1oxxWH3mHM2xiN-eWj-X_3C-GE3sa8dJe?usp=drive_link">
    <img alt="Model" src="https://img.shields.io/badge/Model-Google_Drive-green">
  </a>
  <a href="https://drive.google.com/drive/folders/1oxxWH3mHM2xiN-eWj-X_3C-GE3sa8dJe">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-Google_Drive-orange">
  </a>
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
- [Download Models](#download-models)
- [BYOL Pretraining](#byol-pretraining)
- [T-CDAN Training](#t-cdan-training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## News

- `2026-06-05`: All code, models, and datasets are released!


## Expected Repository Structure

Place downloaded data inside `ACA_qPCR/`:

```text
ACA_qPCR/
|-- BYOL_CDAN_7_plex.py        # 7-plex entry point
|-- BYOL_CDAN_8_plex.py        # 8-plex entry point
|-- requirements.txt
|-- src/                       # implementation code
|   |-- byol_cdan_common.py
|   |-- byol-pytorch/
|   |-- tools/
|   `-- tst/
|-- 7_plex/               # downloaded 7-plex data
|-- 8_plex/               # downloaded 8-plex data
|-- 7_plex_output/             # generated 7-plex models/results
`-- 8_plex_output/             # generated 8-plex models/results
```

## Installation

### 1. Create an environment

```bash
conda create -n aca_pcr python=3.10 -y
conda activate aca_pcr
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

Download the dataset from Google Drive:

```text
https://drive.google.com/drive/folders/1oxxWH3mHM2xiN-eWj-X_3C-GE3sa8dJe
```

Extract the downloaded folders into `ACA_qPCR/`. 

## Download Models

Download pretrained BYOL and T-CDAN models from Google Drive:

```text
https://drive.google.com/drive/folders/1oxxWH3mHM2xiN-eWj-X_3C-GE3sa8dJe?usp=drive_link
```

Place the model files under the corresponding output folders:

```text
ACA_qPCR/
|-- 7_plex_output/
|   |-- pretrained_model_CL_final.pth
|   `-- byol_cdan.pth
`-- 8_plex_output/
    |-- pretrained_model_CL_final.pth
    `-- byol_cdan.pth
```

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

The final BYOL models are saved as:

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
`--lr`, `--weight_decay`, `--trade_off`, `--dir`, `--dir_out`, and
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
models, logs, confusion matrices, `run_config.json`, and
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

This project is released under the [MIT License](LICENSE).
