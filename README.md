# Wine Quality Classification (Neural Network)

A deep learning project that classifies wine quality into three categories — **Low**, **Medium**, and **High** — using a Neural Network built with Keras/TensorFlow.

## Project Overview

- **Dataset:** WineQT.csv (wine chemical properties + quality score)
- **Model:** Feedforward Neural Network (Dense layers with Dropout)
- **Task:** Multi-class classification (3 classes)
- **Result:** ~87% validation accuracy

## How It Works

1. The original wine quality score (0–10) is grouped into 3 simpler classes:
   - Low (score ≤ 4)
   - Medium (score 5–6)
   - High (score ≥ 7)
2. Features are scaled using `StandardScaler` so the neural network trains effectively.
3. A Neural Network with two hidden layers (64 and 32 neurons) and Dropout (to prevent overfitting) is trained.
4. The model is evaluated using accuracy, a classification report, and a confusion matrix.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (preprocessing, metrics)
- TensorFlow / Keras (model building)
- Matplotlib (visualizations)

## Files

- `wine_quality_classification.ipynb` — full notebook (data loading, model building, training, evaluation)

## Author

Asma — NAVTTC Data Science & Machine Learning Program
