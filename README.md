# COAST Model Code

COAST: Consensus-Gated Test-Time Adaptation for Compositional Zero-Shot Learning

## Contents

- `models/coast.py`: the COAST three-branch base model, TextKAM/VisualKAM modules, cache utilities, and COAST test-time adaptation functions.
- `models/common.py`: neural network building blocks required by the COAST model.
- `models/__init__.py`: public COAST model exports.
- `clip_modules/`: the CLIP wrapper, tokenizer, text encoder, and BPE vocabulary required by the model.
- `train.py`: training entry point for the three-branch base model.
- `test.py`: standard CZSL evaluation utilities and base evaluator.
- `test_tta.py`: COAST test-time adaptation evaluation entry point.
- `config/`: main dataset configuration files for COAST.
- `tools/`: general utility scripts used by training or dataset preparation.


## Basic Import

```python
from models import (
    ThreeBranchesPretrain,
    TextKAM,
    VisualKAM,
    predict_logits_text_first_with_coast,
)
```

## Training and Evaluation

```bash
python train.py --yml_path config/ut-zappos.yml
python test_tta.py --yml_path config/ut-zappos.yml
```

Before running these commands, place datasets, CLIP weights, and trained checkpoints at the relative paths specified in `config/*.yml`, for example:

```text
./datasets/ut-zap50k
./pretrained_clip/ViT-L-14.pt
./checkpoints/ThreeBranchesPretrain_UTZap/test_best.pt
```

## Dependencies

```bash
pip install -r requirements.txt
```

The CLIP checkpoint and trained COAST checkpoint are not included. Load them from your own training or release assets when integrating this model code into a full training/evaluation pipeline.
