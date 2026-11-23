# EnsembleLab — Interactive Ensemble Learning Studio

A Streamlit project comparing four classification algorithms on the Breast Cancer Wisconsin Diagnostic dataset:
- Decision Tree (baseline)
- Bagging
- AdaBoost
- Random Forest

## Features
- Dark dashboard theme
- Dataset exploration and interactive 3D PCA projection
- Algorithm explorer with tree-member visualization
- Model Arena: metrics, radar chart, confusion matrix, ROC curves and feature importance
- Prediction Studio: test sample or manual feature entry, four model predictions and majority vote
- Experiment Lab: adjustable test split, estimator count, tree depth and AdaBoost learning rate
- Optional small parameter search using five stratified training folds; held-out test data is used only for final scoring
- Voting Simulator: reveal individual trees and weighted AdaBoost learner contributions, using the shared default models
- Grouped manual prediction inputs with training-range hints, explicit tied votes, and malignant-class recall

## Presentation improvements

The original dataset, four algorithms, default settings, and page structure are preserved.
Scores are displayed as percentages. Model Arena explains each metric, including malignant-class recall.
Prediction Studio groups the 30 measurements into Mean, Standard Error, and Worst inputs.
A 2–2 vote is shown as a tie rather than assigned a class.

To try modest tuning, open Experiment Lab, enable **Try small parameter search**, and click **Run experiment**.
This searches tree depths 3, 5, and unlimited, ensemble sizes 100 and 200, and AdaBoost learning rates 0.1, 0.5, and 1.0.
Settings are chosen by five-fold training accuracy; the resulting test scores are compared with the baseline on the same split.
Tuning may take a few minutes and does not guarantee higher test accuracy. Other pages continue to use the default models.

## Requirements
- Python 3.9 or newer recommended
- Internet connection for initial package installation (and Google Fonts in the theme; app still works without the font)

## Windows setup (PowerShell)
```powershell
cd path\to\EnsembleLab
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

## Run
Run `streamlit run app.py` from the project root folder. Streamlit opens the app at `http://localhost:8501`.

## Dataset notes
The dataset is loaded from Scikit-learn; no manual CSV download is required.
Scikit-learn target encoding:
- `0 = malignant`
- `1 = benign`

Precision, recall and F1 in this starter app use benign (class 1) as the positive class. The 3D PCA chart is a visualization only; model training uses all 30 original features.

## Important
This is an educational machine-learning project, not a medical diagnostic system. Do not use its predictions for healthcare decisions.
