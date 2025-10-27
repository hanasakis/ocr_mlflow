

# 2. ml/README_data.md (Data Versioning)

```markdown
# Data Versioning Documentation (DVC)

This document tracks the version history of datasets used in the OCR digit recognition project, managed via **DVC (Data Version Control)**. DVC ensures reproducibility by linking data versions to code commits and model experiments.


## Dataset Overview
- **Source**: MNIST handwritten digit dataset (publicly available at [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/))
- **Content**: 60,000 training images and 10,000 test images of handwritten digits (0-9), each as 28x28 grayscale pixels
- **Purpose**: Train and evaluate CNN/SVM models for digit recognition


## DVC Configuration
- **Remote Storage**: DAGsHub DVC Remote  
  `dvc remote add origin https://dagshub.com/[your-username]/cnn-ocr-app.dvc`
- **Tracked Directories**:
  - `data/raw/`: Raw MNIST files (`.gz` archives, unprocessed)
  - `data/processed/`: Preprocessed data (normalized, reshaped, reshaped for model input)
- **Ignored Files**: Configured in `.dvcignore` (logs, temporary files, OS-specific metadata)


## Version History

### Version 1.0 (2025-10-20)
- **Commit Hash**: `a3b7c9d` (linked Git commit)
- **Description**: Initial dataset import
- **Content**:
  - Raw files: `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`
  - No preprocessing applied
- **Changes**: First dataset version imported from MNIST source
- **DVC Command**:
  ```bash
  dvc add data/raw
  dvc push data/raw.dvc
  git add data/raw.dvc
  git commit -m "feat: add raw mnist data v1.0"

  Version 2.0 (2025-10-22)
Commit Hash: f2e4d6a (linked Git commit)
Description: Preprocessed dataset with normalization
Content:
Raw data unchanged (same as v1.0)
Processed data:
Images normalized to [0, 1] range (pixel values scaled from 0-255 to 0.0-1.0)
Labels converted to integer format
Split preserved (60k train / 10k test)
Changes: Added preprocessing step via data_preprocessing.py
DVC Command:
bash
python data_preprocessing.py  # Generates processed data
dvc add data/processed
dvc push data/processed.dvc
git add data/processed.dvc
git commit -m "feat: add processed data v2.0 (normalized)"
Version 3.0 (2025-10-25)
Commit Hash: e5f8g1h (linked Git commit)
Description: Augmented dataset for improved model generalization
Content:
Raw data unchanged
Processed data updates:
Added 10,000 augmented training samples (rotations ±10°, zoom ±10%)
Class balance ensured (equal distribution of digits 0-9)
Validation split added (10% of training data: 5,400 samples)
Changes: Added data augmentation to reduce overfitting
DVC Command:
bash
python data_preprocessing.py --augment  # Regenerates processed data with augmentation
dvc commit data/processed.dvc  # Updates existing DVC file
dvc push data/processed.dvc
git commit data/processed.dvc -m "feat: update processed data to v3.0 (augmented)"
Data Workflow
Retrieve Latest Data:
bash
dvc pull data/raw.dvc data/processed.dvc  # Pulls current versions
Check Data Versions:
bash
dvc log data/raw.dvc  # Shows history of raw data
dvc log data/processed.dvc  # Shows history of processed data
Revert to Previous Version:
bash
git checkout <old-commit-hash> data/processed.dvc
dvc checkout data/processed.dvc  # Restores data to that version
Link to Experiments: Each data version is linked to MLflow experiments via Git commit hash (see ml/README_experiments.md).