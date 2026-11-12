# Cross-Modal Alignment with Gradient Modulation for Image-Text Retrieval

This repository contains the implementation for the paper **"Cross-Modal Alignment with Gradient Modulation for Image-Text Retrieval"**
(WACV 2025).

The released code was written by the authors and is intended to reproduce the
main results reported in the paper.

## Repository layout

- `src/model.py` - dual-encoder retrieval model
- `src/losses.py` - gradient-modulated contrastive loss
- `src/train.py` - training entry point

## Environment

```
python>=3.9
torch>=2.1
numpy>=1.24
```

Install the dependencies with `pip install -r requirements.txt`.

## Training

```
python -m src.train --config configs/main.yaml
```

## Citation

```
@inproceedings{cross-modal2025,
  title={Cross-Modal Alignment with Gradient Modulation for Image-Text Retrieval},
  author={Scott, Jeffrey and Raman, Priya},
  booktitle={WACV 2025},
  year={2025}
}
}
