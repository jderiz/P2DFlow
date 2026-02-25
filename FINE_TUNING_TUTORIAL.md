# Fine-Tuning P2DFlow with a Custom MD Dataset

This tutorial walks you through the complete workflow of fine-tuning P2DFlow on your own molecular dynamics (MD) dataset.

## Overview

The fine-tuning process consists of five main stages:

1. **Prepare raw MD data** – organise your `.pdb` topology and `.xtc` trajectory files.
2. **Calculate approximate energy and select representative structures** – use `gaussian_kde` to estimate conformational free energy and sample a diverse set of structures.
3. **Process the selected structures** – convert `.pdb` files to `.pkl`, compute ESM-2 node/pair representations, and predict a static structure with ESMFold.
4. **Configure and run fine-tuning** – edit `configs/base.yaml` to point to your data and set `warm_start` to the pretrained checkpoint.
5. **Run inference** – generate ensembles with the fine-tuned model.

---

## Prerequisites

Follow the [Installation](README.md#Installation) section of the main README to set up the conda environment before starting:

```bash
conda env create -f environment.yml
conda activate P2DFlow
```

Download the pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/11mdVfMi2rpVn7nNG2mQAGA5sNXCKePZj?usp=sharing) (filename `pretrained.ckpt`) and place it in the `./weights/` directory:

```bash
mkdir -p weights
# copy pretrained.ckpt into ./weights/
```

---

## Step 1 – Prepare Raw MD Data

P2DFlow expects one sub-directory per protein, each containing:

| File | Description |
|------|-------------|
| `<NAME>.pdb` | Reference / topology structure |
| `<NAME>_R1.xtc` or `<NAME>_R1.dcd` | Trajectory replica 1 |
| `<NAME>_R2.xtc` or `<NAME>_R2.dcd` | Trajectory replica 2 |
| `<NAME>_R3.xtc` or `<NAME>_R3.dcd` | Trajectory replica 3 |

Both **XTC** (GROMACS) and **DCD** (NAMD / CHARMM / AMBER) formats are supported via the `--traj_format` flag (default: `xtc`).

> **Note:** The scripts assume exactly three replicas (`_R1`, `_R2`, `_R3`). If your dataset has a different number of replicas, edit the loop in `dataset/traj_analyse_select.py` accordingly.

Example directory layout for two proteins `PROT1` and `PROT2`:

```
dataset/
└── my_md_data/
    ├── PROT1/
    │   ├── PROT1.pdb
    │   ├── PROT1_R1.xtc   # or PROT1_R1.dcd
    │   ├── PROT1_R2.xtc
    │   └── PROT1_R3.xtc
    └── PROT2/
        ├── PROT2.pdb
        ├── PROT2_R1.xtc   # or PROT2_R1.dcd
        ├── PROT2_R2.xtc
        └── PROT2_R3.xtc
```

Create a plain-text file that lists all protein names (one per line):

```bash
ls dataset/my_md_data/ > dataset/my_md_data/my_filenames.txt
```

---

## Step 2 – Calculate Approximate Energy and Select Representative Structures

`dataset/traj_analyse_select.py` loops over every protein in the filename list, computes a 2-D Gaussian KDE over (radius-of-gyration, RMSD-to-reference) to estimate the free energy of each frame, and then samples `--select_num` frames spanning the conformational landscape.

```bash
# XTC trajectories (GROMACS) – default
python dataset/traj_analyse_select.py \
    --dir_path   dataset/my_md_data \
    --filename   my_filenames.txt \
    --select_num 100 \
    --select_dir dataset/my_md_data/select

# DCD trajectories (NAMD / CHARMM / AMBER)
python dataset/traj_analyse_select.py \
    --dir_path    dataset/my_md_data \
    --filename    my_filenames.txt \
    --select_num  100 \
    --select_dir  dataset/my_md_data/select \
    --traj_format dcd
```

**Key outputs**

| Path | Content |
|------|---------|
| `dataset/my_md_data/<NAME>/traj_info.csv` | Per-frame energy, Rg, RMSD |
| `dataset/my_md_data/<NAME>/md.png` | KDE free-energy landscape plot |
| `dataset/my_md_data/select/` | Selected `.pdb` frames |
| `dataset/my_md_data/select/traj_info_select.csv` | Energy values for the selected frames |

> **Tip:** Use `--select_num 10` for a quick test run, or `--select_num 100` (default) for more diversity. The sampling function applies a cube-root mapping to favour higher-energy (less-populated) regions of the landscape.

---

## Step 3 – Create a Validation Split

Before processing the data, decide which proteins to hold out as a validation set.  
Create a CSV file in the same format as `inference/valid_seq.csv`:

```
file,seq_len,seq
PROT1,123,MSEQSTRING...
```

You can fill `seq_len` and `seq` manually, or extract them programmatically:

```python
import pandas as pd
from Bio import SeqIO

records = []
for pdb_path in your_val_pdb_files:
    seq = "".join(r.resname for r in ...)   # extract one-letter sequence
    records.append({"file": name, "seq_len": len(seq), "seq": seq})

pd.DataFrame(records).to_csv("inference/my_valid_seq.csv", index=False)
```

> Only the `file` column (the six-character protein identifier, e.g. `PROT1`) is used by `process_pdb_files.py` to mark which entries belong to the validation set.

---

## Step 4 – Process Selected Structures

`data/process_pdb_files.py` performs three operations in sequence:

1. Parses each `.pdb` file and writes a preliminary `.pkl` with atom coordinates.
2. Runs ESM-2 to compute per-residue node embeddings (1280-d) and pairwise representations (20-d), storing the result back into each `.pkl`.
3. Runs ESMFold to predict a static structure (the "prior" for the flow), and stores the resulting backbone frames inside each `.pkl`.
4. Merges the structural metadata CSV with the energy CSV to produce `metadata_merged.csv`.

```bash
python data/process_pdb_files.py \
    --pdb_dir           dataset/my_md_data/select \
    --write_dir         dataset/my_md_data/select/pkl \
    --traj_info_file    dataset/my_md_data/select/traj_info_select.csv \
    --valid_seq_file    inference/my_valid_seq.csv \
    --merged_output_file dataset/my_md_data/select/pkl/metadata_merged.csv \
    --esm_device        cuda \
    --num_processes     8
```

| Argument | Description |
|----------|-------------|
| `--pdb_dir` | Directory containing the selected `.pdb` files from Step 2 |
| `--write_dir` | Output directory for `.pkl` files |
| `--traj_info_file` | The `traj_info_select.csv` produced in Step 2 |
| `--valid_seq_file` | Validation split CSV created in Step 3 |
| `--merged_output_file` | Path for the final merged metadata CSV |
| `--esm_device` | `cuda` (recommended) or `cpu` |
| `--num_processes` | Number of parallel workers for the initial PDB parsing stage |

After this step, `dataset/my_md_data/select/pkl/` should contain:
- One `.pkl` per structure, each storing backbone frames, ESM-2 representations, and ESMFold prediction.
- `metadata.csv` – per-structure metadata.
- `metadata_merged.csv` – metadata enriched with approximate energy values and train/validation flags.

> **Memory note:** ESMFold and ESM-2 (650 M) both require GPU memory. Running on a GPU with ≥ 24 GB is recommended; use `--esm_device cpu` if no GPU is available (significantly slower).

---

## Step 5 – Configure Fine-Tuning

Open `configs/base.yaml` and make the following changes:

### 5a – Point to your dataset

```yaml
data:
  dataset:
    csv_path: dataset/my_md_data/select/pkl/metadata_merged.csv
```

### 5b – Enable warm-starting from the pretrained checkpoint

```yaml
experiment:
  warm_start: ./weights/pretrained.ckpt
  warm_start_cfg_override: True   # merges the pretrained model config with base.yaml
```

### 5c – (Optional) Reduce the learning rate for fine-tuning

A lower learning rate is typically better when fine-tuning from a pretrained checkpoint:

```yaml
experiment:
  optimizer:
    lr: 1e-5   # reduced from the default 1e-4
```

### 5d – (Optional) Adjust the W&B run name

```yaml
experiment:
  wandb:
    name: finetune_my_dataset
    project: P2DFlow
```

### 5e – (Optional) Limit the number of epochs

```yaml
experiment:
  trainer:
    max_epochs: 200
```

---

## Step 6 – Run Fine-Tuning

```bash
python experiments/train_se3_flows.py
```

Checkpoints are saved to `ckpt/P2DFlow/finetune_my_dataset/<date_time>/`.  
Set `WANDB_MODE=offline` (already set in the script) if you do not have a W&B account.

### Monitoring training

```
ckpt/
└── P2DFlow/
    └── finetune_my_dataset/
        └── 2025-06-01_12-00-00/
            ├── config.yaml       # full run config
            ├── last.ckpt         # always kept
            └── epoch=XX-step=YY.ckpt
```

The monitored metric is `valid/rmsd_loss` (lower is better).

---

## Step 7 – Run Inference with the Fine-Tuned Model

Edit `configs/inference.yaml`:

```yaml
inference:
  ckpt_path: ./ckpt/P2DFlow/finetune_my_dataset/<date_time>/last.ckpt

  samples:
    validset_path: ./inference/my_valid_seq.csv
    sample_num: 250
    sample_batch: 5
```

Then run:

```bash
python experiments/inference_se3_flows.py
```

Generated ensembles are written to `inference_outputs/` under a sub-directory named after the checkpoint.

---

## Step 8 – Evaluate Results (Optional)

```bash
python analysis/eval_result.py \
    --pred_org_dir   inference_outputs/... \
    --valid_csv_file inference/my_valid_seq.csv \
    --pred_merge_dir /tmp/pred_merge \
    --target_dir     dataset/my_md_data/select \
    --crystal_dir    dataset/my_md_data/select
```

For PCA-based comparison of the generated ensemble against the MD reference:

```bash
python analysis/pca_analyse.py \
    --pred_pdb_dir  inference_outputs/... \
    --target_dir    dataset/my_md_data/select \
    --crystal_dir   dataset/my_md_data/select
```

---

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| `KeyError: 'energy'` in `merge_pdb` | `traj_info_select.csv` filename column does not match the `.pdb` filenames | Make sure the `traj_filename` values in the CSV exactly match the files in `--pdb_dir` |
| OOM during ESMFold | GPU memory too small | Use `--esm_device cpu`, or process in smaller batches |
| No valid proteins after filtering | `min_num_res` / `max_num_res` mismatch | Adjust `data.dataset.min_num_res` and `data.dataset.max_num_res` in `configs/base.yaml` |
| `warm_start_cfg_path` not found | Missing `config.yaml` next to the pretrained `.ckpt` | Set `experiment.warm_start_cfg_override: False` in `base.yaml` |

---

## Summary of Key Files

| File | Purpose |
|------|---------|
| `dataset/traj_analyse_select.py` | Energy estimation and frame selection |
| `data/process_pdb_files.py` | PDB → PKL + ESM-2 + ESMFold pipeline |
| `configs/base.yaml` | Training configuration |
| `configs/inference.yaml` | Inference configuration |
| `experiments/train_se3_flows.py` | Training entry point |
| `experiments/inference_se3_flows.py` | Inference entry point |
| `analysis/eval_result.py` | Validity / fidelity / dynamics metrics |
| `analysis/pca_analyse.py` | PCA comparison |
