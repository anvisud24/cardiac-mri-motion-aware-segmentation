# Motion-Aware Foundation Model for Cardiac MRI Segmentation with Self-Supervised Pretraining

This repository contains the code and notebooks for a cardiac MRI segmentation project focused on motion-aware representation learning and self-supervised pretraining for downstream segmentation.

## Repository Contents

- `model.ipynb` - main project notebook for model development, training, and evaluation
- `acdc_test_visualization.png` - example qualitative visualization
- `README.md` - project overview, setup, and dataset attribution
- `.gitignore` - excludes datasets, checkpoints, outputs, and local artifacts from version control


## Overview

This project explores motion-aware learning for cardiac MRI segmentation with self-supervised pretraining. The goal is to learn robust temporal and anatomical representations that improve segmentation quality when labeled medical imaging data is limited.

The implementation is notebook-based and uses PyTorch and medical imaging utilities for loading, preprocessing, training, and visualization.

## Dependencies

Based on the current notebooks, the project uses:

- Python 3.10+
- PyTorch
- torchvision
- numpy
- matplotlib
- opencv-python
- nibabel
- tqdm
- pandas
- python-dotenv
- sqlalchemy
- psycopg2-binary

Install them with:

```bash
pip install torch torchvision numpy matplotlib opencv-python nibabel tqdm pandas python-dotenv sqlalchemy psycopg2-binary
```

## Usage

Launch Jupyter and open the main notebook:

```bash
jupyter notebook model.ipynb
```

Update any local dataset paths in the notebook before running cells.

## Data

This repository does not include the raw cardiac MRI datasets, manual contours, checkpoints, or training outputs.

The GitHub repo is intended to contain only code, notebooks, lightweight figures, and documentation. Large local folders such as `training_ACDC/`, `testing_ACDC/`, `SCD_IMAGES_*`, `SCD_CAPModels/`, `SCD_ManualContours/`, `checkpoints/`, and `outputs/` should remain excluded from version control.

## ACDC Dataset Description

The Automated Cardiac Diagnosis Challenge (ACDC) dataset was created from anonymized clinical cardiac MRI exams acquired at the University Hospital of Dijon. According to the official ACDC database page, the full dataset contains 150 exams from different patients divided into five evenly distributed subgroups: normal subjects, previous myocardial infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, and abnormal right ventricle cases. The official challenge pages also describe the released split as 100 training patients and 50 testing patients.

Source: [ACDC Challenge Database](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

## Citation

The official ACDC website states that any use of the ACDC database should cite:

Bernard O, Lalande A, Zotti C, Cervenansky F, Yang X, Heng PA, Cetin I, Lekadir K, Camara O, Ballester MAG, Sanroma G, Napel S, Petersen S, Tziritas G, Grinias E, Khened V, Kollerathu VA, Krishnamurthi G, Rohe MM, Pennec X, Sermesant M, Isensee F, Jager P, Maier-Hein KH, Full PM, Wolf I, Engelhardt S, Baumgartner CF, Koch LM, Wolterink JM, Išgum I, Jang Y, Hong Y, Patravali J, Jain S, Humbert O, and Jodoin PM. *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?* IEEE Transactions on Medical Imaging. 2018;37(11):2514-2525. doi:10.1109/TMI.2018.2837502.

Official citation source: [ACDC Challenge Database](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

## License

This repository's original code is licensed under the MIT License. External datasets, annotations, and third-party materials are not covered by this license and remain subject to their own terms, citation requirements, and usage restrictions.

## Disclaimer

This repository is for research and educational use. It does not provide clinical guidance or medical advice.
