# README.txt

## Overview
This repository demonstrates a comprehensive approach to bankruptcy prediction using classical and quantum machine learning techniques. The project uses various machine learning algorithms and quantum-enhanced models to classify companies based on financial data, optimizing for performance metrics like F1 score.

## Features
1. **Data Preprocessing**:
   - Feature selection based on correlation with the target variable (`Bankrupt`).
   - Removal of highly correlated features to avoid multicollinearity.
   - Balancing the dataset using undersampling and SMOTE techniques.

2. **Classical Machine Learning**:
   - Models: KNN, Logistic Regression, Random Forest, and SVC.
   - Hyperparameter tuning using `GridSearchCV`.
   - Performance evaluation using F1 score on training and test sets.

3. **Quantum Machine Learning**:
   - Implementation of Variational Quantum Classifier (VQC) using Qiskit.
   - Feature mapping with `ZZFeatureMap` and ansatz with `RealAmplitudes` or `EfficientSU2`.
   - Optimization using `COBYLA`.

4. **Visualization**:
   - Heatmaps for correlation analysis.
   - Iterative loss visualization for quantum model training.

## Requirements
- Python 3.8+
- Libraries: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`
  - `qiskit`, `qiskit-machine-learning`, `scipy`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Data
The dataset contains financial indicators for companies with a `Bankrupt` label (0 or 1). Preprocessed to reduce noise and improve model accuracy.

- **Input Data**: `data/data.csv`
- **Processed Features**: Selected based on correlation and feature reduction.

## Usage
### Data Preprocessing
Run the script to clean and balance the dataset:
```python
python preprocess.py
```

### Model Training
Train and evaluate classical models:
```python
python train_classical.py
```

Train and evaluate quantum models:
```python
python train_quantum.py
```

### Visualize Results
Generate visualizations of feature correlations and quantum model training loss:
```python
python visualize.py
```

## Results
- **Classical Models**:
  - Best model: Random Forest with `F1 (Validation) ≈ 0.82`.

- **Quantum Models**:
  - VQC with `EfficientSU2` achieved `F1 (Validation) ≈ 0.80`.

## Folder Structure
```
.
├── data/
│   └── data.csv        # Raw dataset
├── preprocess.py       # Data preprocessing script
├── train_classical.py  # Classical ML training script
├── train_quantum.py    # Quantum ML training script
├── visualize.py        # Visualization scripts
└── README.txt          # Project overview
```

## Acknowledgments
This project leverages the power of Qiskit for quantum ML and Scikit-learn for classical ML. Special thanks to open-source contributors for these robust libraries.

## License
This project is open-sourced under the MIT License. See `LICENSE` for details.
