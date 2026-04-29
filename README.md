# AI for Health
Detecting breathing irregularities during sleep using physiological signals.

## Code available at :
https://colab.research.google.com/drive/1JTq9dU0emRwb8N04nTVksgyS2Ul122ZD?usp=sharing

## What this does
- Visualizes Nasal Airflow, Thoracic Movement and SpO2 signals across 8 hour sleep sessions
- Preprocesses signals and creates labeled windows for classification
- Trains a 1D CNN to classify Normal vs Apnea/Hypopnea events
  
## Results (Leave-One-Participant-Out CV)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 91.2%  |
| Precision | 83.2%  |
| Recall    | 91.2%  |

## Thank You 
