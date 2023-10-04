# Disease_Prediction_Using_ML
This repository contains Python code for building and evaluating machine learning models to predict three different diseases: diabetes, hypertension, and stroke. The code is organized into sections, each corresponding to a specific disease prediction model. 


---


## Table of Contents

- [Getting Started](#getting-started)
- [Diabetes Prediction](#diabetes-prediction)
- [Hypertension Prediction](#hypertension-prediction)
- [Stroke Prediction](#stroke-prediction)
- [Making Predictions](#making-predictions)
- [Usage](#usage)

## Getting Started

To use this code, you'll need Python and the following libraries installed:

- pandas
- scikit-learn

You can install these libraries using `pip`:

```
pip install pandas scikit-learn
```

Clone this repository to your local machine:

```
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

## Diabetes Prediction

In the "diabetes" section of the code, a random forest classifier is trained to predict diabetes. The dataset used is loaded from a CSV file. Model performance metrics such as accuracy, F1 score, precision, and recall are calculated for both the training and test datasets.

## Hypertension Prediction

In the "hypertension" section of the code, a decision tree classifier is trained to predict hypertension. Similar to the diabetes model, this section also calculates performance metrics for the training and test datasets.

## Stroke Prediction

The "stroke" section of the code trains a K-Nearest Neighbors (KNN) classifier to predict strokes. As in the previous sections, performance metrics are computed for the training and test datasets.

## Making Predictions

The code includes a function `predict_disease` that allows users to make predictions for each disease based on input data. You can provide a list of input features, and the function will return the predicted disease status (e.g., "Diabetes," "Hypertension," or "Stroke").

## Usage

To use the disease prediction models, follow these steps:

1. Ensure you have Python and the required libraries installed.

2. Clone this repository to your local machine.

3. Navigate to the repository's directory.

4. Open the Jupyter Notebook or Python script where the code is located.

5. Modify the input data in the code to make predictions for specific cases.

6. Run the code to see the predicted disease status for the given input.

