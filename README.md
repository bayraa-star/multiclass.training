## Vehicle Brand/Color/Side Multi-Label Training

This repository contains the training and inference scripts I use to build a multi-label classifier that predicts several vehicle attributes at once (brand, body color, and camera view). The model is based on `timm`'s EfficientNet-B0 backbone trained with a sigmoid head so that each attribute is treated as an independent binary label.

### Repository layout

- `train.py` – full training loop with logging, validation, and final test evaluation. Saves the production checkpoint to `multilabel_vehicle_model_production.pth`.
- `test.py` – lightweight inference helper that loads a saved checkpoint and reports only the labels whose probability exceeds 0.5.
- `_classes.csv` – example label file showing the expected schema (`filename` column followed by one column per attribute).
- `requirements.txt` – python dependencies used by both the training and inference scripts.
- `*.pth` – example checkpoints already trained on my Roboflow dataset (70k epochs and production versions).
- `test.png` – sample image for quickly sanity-checking inference.

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code expects Python 3.9+ with CUDA available for faster training, but it will fall back to CPU if no GPU is detected.

### Dataset expectations

1. Split images into `train/`, `valid/`, and `test/` directories.
2. Inside each split, create an `_classes.csv` file with the following structure:
   - Column 0: `filename` (relative path to the image inside the split directory).
   - Columns 1..N: binary indicators for each attribute (e.g., `Toyota_Land_C`, `blue`, `front_side`).
3. Update the `base_dir` variables in `train.py` and `test.py` so they point to the parent directory that contains the split folders.

Each CSV should contain exactly the same label columns so that the dynamically computed `num_classes` stays consistent between training and inference.

### Training

```bash
python train.py
```

Key training defaults:

- Backbone: `efficientnet_b0` (loaded via `timm.create_model`).
- Loss: binary cross-entropy (`nn.BCELoss`) with a sigmoid applied to each output.
- Batch size: 128.
- Epochs: 70.

You can safely tweak `BATCH_SIZE`, `num_epochs`, optimizer settings, or even swap out the model name; the label count is computed automatically from the CSV. Training logs are emitted via `logging` so you can track per-batch progress and per-epoch train/validation losses. At the end of training the script evaluates on the held-out test split and saves `multilabel_vehicle_model_production.pth` in the project root.

### Inference / evaluation

1. Place the checkpoint you want to evaluate (e.g., `multilabel_vehicle_model_epch_70K.pth`) in the project root.
2. Open `test.py` and ensure both `base_dir` (for label names) and `img_path` (the image you want to classify) are set correctly.
3. Run `python test.py`.

The script prints only the labels whose probability is greater than 0.5, grouped by the helper `get_category` function so you can quickly see the predicted color, vehicle type, and viewing side. Adjust the threshold or categories as needed for your deployment.

### Next steps

- Swap in different `timm` models (e.g., EfficientNetV2, ConvNeXt) or freeze the backbone to speed up training.
- Replace the CSV loader with a custom dataset if you add bounding boxes or per-object crops.
- Add metrics such as F1-score per label to catch imbalances across brand/color classes.
