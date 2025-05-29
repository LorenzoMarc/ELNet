# ELNet: Knee MRI Classification

This repository contains a PyTorch-based implementation for classifying knee MRI scans, inspired by the MRNet architecture described in ["Deep-learning-assisted diagnosis for knee magnetic resonance imaging"](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).

## Project Structure

- `models/elnet.py`: Defines the ELNet neural network architecture.
- `dataloader.py`: Implements the custom PyTorch dataset and data augmentation utilities.
- `utils.py`: Contains helper functions for data augmentation and evaluation metrics.
- `train.py`: Main training and validation script.
- `train_elnet.sh`: Example shell script to launch training for different tasks and planes.

## Data Preparation

1. Download the MRNet dataset from the [official website](https://stanfordmlgroup.github.io/competitions/mrnet/).
2. Set the `DATA_PATH` variable in `train_elnet.sh` to the directory containing the MRNet data.

## Training

To train the models, run:
```
bash train_elnet.sh
```
This will train separate models for each combination of task (`abnormal`, `acl`, `meniscus`) and plane (`sagittal`, `coronal`, `axial`). Training logs, checkpoints, and results will be saved in the `experiments` directory.

## Custom Training

You can run custom experiments using:
```
python train.py -t <task> -p <plane> --experiment <experiment_name> --data-path <path_to_data> --prefix_name <prefix>
```
See `train.py` for all available arguments.

## Requirements

- Python 3.6+
- PyTorch
- scikit-learn
- pandas
- numpy
- tensorboardX
- tqdm
- scikit-image
- torchvision

## Notes

- Data augmentation and normalization are handled in `dataloader.py` and `utils.py`.
- Model checkpoints and logs are organized by experiment name and configuration.
- Evaluation metrics include AUC, MCC, accuracy, sensitivity, and specificity.

---
For more details, refer to the code and comments in each file.