# ML Assignment 2: Breast Cancer Classification

Streamlit App Link: <PASTE STREAMLIT APP LINK HERE>

## a. Problem Statement

The aim is to predict whether a breast tumour is Malignant or Benign using 30 features calculated from a digitized image of a fine needle aspirate (FNA) of a breast mass. It is a binary classification problem. Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) are trained on the same dataset and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are then used in a Streamlit web app where test data can be uploaded and the results can be checked.

## b. Dataset Description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository (also available on Kaggle)
- Number of instances: 569
- Number of features: 30 (all numeric, no missing values)
- Target column: diagnosis (1 = Malignant, 0 = Benign)
- Class count: 357 Benign (63%), 212 Malignant (37%)
- Train/test split: 80:20 stratified split with random_state 2611 (455 rows for training, 114 rows for testing)

The features are computed for the cell nuclei in the image. There are 10 base measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each of these the mean, standard error and worst (largest) value is given, so total 30 features. Example column names: mean_radius, radius_error, worst_radius.

## c. GitHub Repository Link

https://github.com/sree9578/2026AJ05777-ML-Assignment2

Repository structure:

```
project-folder/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- metrics.json
```

- train_models.py: loads the data, splits it, trains and tunes the 5 models, calculates the metrics and saves the models
- app.py: Streamlit app which loads the saved models and evaluates them on the uploaded CSV (no retraining)
- test_data.csv: test data (114 rows) with all 30 features and the diagnosis column
- model/metrics.json: feature list, best parameters and test metrics

Steps to run:

```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Preprocessing and training: StandardScaler is applied for Logistic Regression, KNN, Naive Bayes and SVM using a sklearn Pipeline, so the scaler is saved inside the model file. Decision Tree is used without scaling. Hyperparameters were selected using GridSearchCV with 5 fold stratified cross validation on the training data (scoring = F1). For SVM, CalibratedClassifierCV is used to get probabilities for the AUC score.

## d. Models Used

Comparison table (on test data, 114 rows):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9974 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Decision Tree | 0.9737 | 0.9745 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| KNN | 0.9737 | 0.9853 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9841 | 0.9268 | 0.9048 | 0.9157 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9960 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |

Best parameters found:
- Logistic Regression: C = 1
- Decision Tree: criterion = entropy, max_depth = 5, min_samples_leaf = 5
- KNN: n_neighbors = 9, weights = uniform
- Naive Bayes (Gaussian): var_smoothing = 1e-9
- SVM: kernel = rbf, C = 1, gamma = scale

Observations:

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy is 97.37% and it has the highest AUC (0.9974). There were no false positives, so precision is 1.0, but 3 out of 42 malignant cases were predicted as benign (recall 0.9286). The data is close to linearly separable after scaling, so a simple linear model works well here. |
| Decision Tree | Accuracy is the same as Logistic Regression (97.37%). Recall is better (only 2 malignant cases missed) but there is 1 false positive. AUC is the lowest (0.9745) because the tree gives less smooth probability values. Without limiting max_depth and min_samples_leaf the tree was overfitting. |
| KNN | Got the same confusion matrix as Logistic Regression (3 false negatives, 0 false positives) with k = 9. Scaling is important for KNN because features like area have much larger values than others. AUC is 0.9853. |
| Naive Bayes | Lowest performance among all models with accuracy 93.86% and MCC 0.8676. It missed 4 malignant cases and gave 3 false positives. The reason could be that Naive Bayes assumes features are independent, but here radius, perimeter and area are highly correlated (correlation above 0.99). |
| Support Vector Machine | Best results with accuracy 98.25%, F1 0.9756 and MCC 0.9626. Only 2 malignant cases were missed and there were no false positives. AUC (0.9960) is almost the same as Logistic Regression. |
| Overall, Winner for your dataset? | Support Vector Machine (RBF kernel) is the best model for this dataset as it has the highest accuracy, recall, F1 and MCC. Logistic Regression is very close and has the best AUC, so it is also a good choice if a simpler model is needed. |

## Streamlit App Features

- Upload test data as CSV (sample test_data.csv can also be downloaded from the app)
- Dropdown to select any of the 5 models
- Shows Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model
- Confusion matrix, ROC curve and classification report
- Comparison of all 5 models on the uploaded data
- Option to download the predictions

The app only loads the saved models from the model folder. It does not retrain on the uploaded data.
