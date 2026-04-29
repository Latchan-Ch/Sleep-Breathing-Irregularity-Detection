# AI for Health
Detecting breathing irregularities during sleep using physiological signals.

## What this does
- Visualizes Nasal Airflow, Thoracic Movement and SpO2 signals across 8 hour sleep sessions
- Preprocesses signals and creates labeled windows for classification
- Trains a 1D CNN to classify Normal vs Apnea/Hypopnea events

## How to run
pip install -r requirements.txt
python scripts/vis.py -name "Data/AP01"
python scripts/create_dataset.py -in_dir Data -out_dir Dataset
python scripts/train_model.py --dataset_dir Dataset --epochs 20

## Results (Leave-One-Participant-Out CV)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 91.2%  |
| Precision | 83.2%  |
| Recall    | 91.2%  |

## Thank You 
