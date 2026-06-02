
# Wafer Map Defect Detection using CNN

Semiconductor wafer defect classification using Convolutional Neural Network on WM-811K dataset.

## Results
- **Accuracy:** 90%+ 
- **Dataset:** WM-811K, 25,519 defect wafers
- **Classes:** 8 defect types (Center, Edge-Ring, Scratch, etc.)

## Files
1. `01_eda.py` - Exploratory Data Analysis
2. `02_train.py` - CNN Training with Class Weights
3. `03_predict.py` - Predict new wafer defects

## How to Run
```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn
py -3.11 01_eda.py
py -3.11 02_train.py
py -3.11 03_predict.py
## Dataset Setup
1. Download `LSWMD.pkl` from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. Place `LSWMD.pkl` in the same folder as the `.py` files
## Results & Graphs

### 1. Training History
![Training History](training_history.png)
Model 90%+ accuracy tak pahucha with Early Stopping.

### 2. Confusion Matrix  
![Confusion Matrix](confusion_matrix.png)
Scratch class ka recall 11% se 62% improve hua.

### 3. Defect Samples
![Wafer Samples](wafer_samples.png)
8 types ke defects: Center, Donut, Edge-Loc, etc.

### 4. Class Distribution
![Distribution](defect_distribution.png)
Dataset me class imbalance clearly visible hai.
### 5. Sample Prediction
![Prediction](prediction_result.png)
Model ne 'Scratch' defect correctly detect kiya with 94% confidence.
