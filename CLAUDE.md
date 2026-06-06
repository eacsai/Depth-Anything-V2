# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) extended with KITTI- and satellite-specific tooling that produces the per-image depth maps consumed by [BevSplat](https://github.com/wangqww/BevSplat). The upstream Depth-Anything-V2 stack (encoder + DPT head + train/eval scripts + the gradio demo) is left as-is; the additions live almost entirely under `metric_depth/`.

## Upstream provenance

- Forked from `DepthAnything/Depth-Anything-V2`. Original `README.md`, `LICENSE`, `requirements.txt`, `run.py`, `run_video.py`, `app.py`, the `depth_anything_v2/` package, and the `metric_depth/` codebase remain unchanged in spirit; only the `metric_depth/predict_*`, `metric_depth/run_*`, `metric_depth/forward_mapping_KITTI.py`, and `metric_depth/dataset/kitti.py` files plus the `metric_depth/dataset/splits/kitti/` lists are project-local.

## What was added on top of upstream

All custom code lives in `metric_depth/`. The relevant commits are `f7de18b` through `fb58ced` (commits where the prefix is `feat:` or `fix:` rather than `Release …` / `Update README …`).

### Entry points (BevSplat-facing)

| File | Purpose |
|---|---|
| `metric_depth/predict_KITTI_depth.py` | **The script the BevSplat README points at.** Walks `<KITTI_ROOT>/depth_data/<date>/<drive>/image_02/data/`, runs Depth-Anything V2 metric (vKITTI-finetuned, vitl encoder) over every PNG, masks the sky using a Depth-Anything V1 forward pass, then writes each result to `<drive>/image_02/grd_depth/{name}_grd_depth.pt` — exactly the path and tensor format BevSplat's `dataLoader/KITTI_dataset.py` consumes. Also writes per-pixel ground heights to `image_02/grd_height/{name}_grd_height.pt` (used by `predict_KITTI_sat_height.py`). |
| `metric_depth/predict_KITTI_sat_height.py` | Uses the `grd_height` tensors above and the KITTI satellite tiles to produce a per-tile height map (`image_02/sat_height/{name}_sat_height.pt`). |
| `metric_depth/forward_mapping_KITTI.py` | Differentiable forward warp from ground perspective image to BEV, ported from OrienterNet. Used by some research-only paths; **not required for BevSplat reproduction.** |

### Satellite ↔ ground warping (research-only)

These files came in commits `f7de18b … e20ed5d`. They explore ground-to-satellite (`run_grd_to_sky_forward.py`, `run_grd_to_sky_inverse.py`) and reverse (`run_sky_to_grd.py`) projections. **None of them are needed to reproduce the depth files BevSplat consumes** — they're left in the repo for later research use.

### Dataset additions

- `metric_depth/dataset/kitti.py` — KITTI Dataset class for finetuning / eval (val mode only).
- `metric_depth/dataset/vkitti2.py` — Virtual KITTI 2 dataset (upstream had this; we just extend it).
- `metric_depth/dataset/splits/kitti/` — text file lists.

## Pipeline used to generate BevSplat's KITTI depth files

```
<KITTI_ROOT>/depth_data/<date>/<drive>/image_02/data/*.png
                                  │
                                  ▼
        Depth-Anything V1 (vitl)  →  binary "sky" mask
                                  │ (zero where sky)
                                  ▼
       Depth-Anything V2 metric   →  metric depth in meters
       (vKITTI-finetuned, vitl)
                                  │ (× sky_mask)
                                  ▼
                  torch.save  →  image_02/grd_depth/{name}_grd_depth.pt   ← BevSplat reads this
                                  │
                                  ▼ (back-project via camera_k_inv, take Y-axis component)
                  torch.save  →  image_02/grd_height/{name}_grd_height.pt
```

Both saved tensors are at 256×1024 (= BevSplat's `GrdImg_H × GrdImg_W`); the input images are KITTI raw left-color frames at their original resolution.

## Hardcoded paths to change before running

`metric_depth/predict_KITTI_depth.py` has three filesystem strings that almost certainly point at the original developer's tree:

| Line | What | Change to |
|---|---|---|
| ~45 | `depth_v2_load_from = '/home/wangqw/.../depth_anything_v2_metric_vkitti_vitl.pth'` | `metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth` (relative to repo root) |
| ~50 | `depth_anything_v1.load_state_dict(torch.load('/home/wangqw/.../depth_anything_vitl14.pth'))` | `metric_depth/checkpoints/depth_anything_vitl14.pth` |
| ~100 | `root_directory = '/home/wangqw/video_dataset/KITTI/depth_data'` | Your `${KITTI_ROOT}/depth_data` |

`predict_KITTI_sat_height.py` and the `forward_mapping_KITTI.py` warpers carry similar hardcoded paths — only edit those if you actually run them.

## Required checkpoints

Download both into `metric_depth/checkpoints/`:

| File | Source | Used by |
|---|---|---|
| `depth_anything_v2_metric_vkitti_vitl.pth` | [HuggingFace · Depth-Anything-V2-Metric-VKITTI-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large) | metric depth regression |
| `depth_anything_vitl14.pth` | [HuggingFace · LiheYoung/Depth-Anything (vitl14)](https://huggingface.co/LiheYoung/Depth-Anything) | sky masking (V1 forward pass) |

## Setup

```bash
git clone https://github.com/eacsai/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r metric_depth/requirements.txt   # matplotlib, opencv-python, open3d, torch, torchvision
mkdir -p metric_depth/checkpoints
# Drop the two .pth files listed above into metric_depth/checkpoints/
```

## Running the BevSplat-KITTI depth pass

```bash
cd metric_depth
# Edit predict_KITTI_depth.py — fix the three hardcoded paths (above).
python predict_KITTI_depth.py
```

The script iterates the five raw-KITTI dates hardcoded in `dates = ['2011_09_29', '2011_09_26', '2011_10_03', '2011_09_30', '2011_09_28']`. For every drive under `${KITTI_ROOT}/depth_data/<date>/<drive>/image_02/data/`, it writes:
- `<drive>/image_02/grd_depth/<name>_grd_depth.pt`  ← consumed by BevSplat
- `<drive>/image_02/grd_height/<name>_grd_height.pt`

Total runtime on a single RTX 4090: ~2–3 hours for the full KITTI train + test1 + test2 frame set, ~6 GB of `.pt` output.

## What NOT to expect

- **No `requirements.txt` install of the BevSplat algorithm.** This repo only generates input data; the actual BevSplat package lives at https://github.com/wangqww/BevSplat.
- **Other than `predict_KITTI_depth.py` and `predict_KITTI_sat_height.py`, no script here is strictly required for BevSplat reproduction.** The `forward_mapping_KITTI.py` and `run_*sky*.py` files are research detritus; ignore unless you're doing your own cross-view experiments.
- **Sky-masking via V1 is an implementation choice, not a quality requirement.** If you only have one DA-V2 checkpoint, you can comment out the V1 load and the `depth = depth * mask` line — the resulting `.pt` files still load fine in BevSplat but will include depth values inside the sky region.

## Where this fits in the BevSplat reproduction stack

```
vita-epfl/Loc2          (KITTI raw drives, satellite tiles, train/test splits)
        │
        ▼
eacsai/Depth-Anything-V2  ← THIS REPO   (per-frame *_grd_depth.pt via predict_KITTI_depth.py)
        │
        ▼
wangqww/BevSplat        (loads .pt files via dataLoader/KITTI_dataset.py:130-135;
                         runs Stage-1 training/eval through kitti_main/)
```

Companion repo for VIGOR depth: [eacsai/UniK3Dnew](https://github.com/eacsai/UniK3Dnew).
