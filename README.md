# ML Assignment 2 — Census Income Classification

## a. Problem Statement
Predict whether an individual's annual income exceeds $50,000 based on demographic
and employment attributes (age, education, occupation, hours worked, marital
status, etc.). This is a **binary classification** problem. Five classical ML
models are trained and compared on the same dataset, and an interactive
Streamlit app is provided to score new test data against any of the trained
models.

## b. Dataset Description
- **Source:** [UCI Machine Learning Repository — Adult / Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Instances:** 32,561 (after dropping rows with a missing target)
- **Features:** 14 input features (age, workclass, fnlwgt, education,
  education-num, marital-status, occupation, relationship, race, sex,
  capital-gain, capital-loss, hours-per-week, native-country) — a mix of
  numeric and categorical columns
- **Target:** `income` — binarized to `0` (`<=50K`) / `1` (`>50K`); class
  balance is roughly 76% / 24%
- **Preprocessing:** missing values (`?`) imputed (most-frequent for
  categorical columns, median for numeric columns); categorical columns
  one-hot encoded; numeric columns standardized. All preprocessing is bundled
  into each model's `sklearn.Pipeline`, so saved models can score raw,
  un-preprocessed CSV input directly.

## c. GitHub Repository Link
`<ADD YOUR GITHUB REPO URL HERE AFTER PUSHING>`

## d. Models Used

### Comparison Table
Evaluated on an 80/20 stratified train/test split (`random_state=42`),
6,513 test rows:

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression  | 0.8558   | 0.9078 | 0.7406    | 0.6173 | 0.6734 | 0.5858 |
| Decision Tree        | 0.8142   | 0.7537 | 0.6091    | 0.6371 | 0.6228 | 0.4999 |
| KNN                  | 0.8366   | 0.8679 | 0.6775    | 0.6135 | 0.6439 | 0.5394 |
| Naive Bayes          | 0.5382   | 0.7495 | 0.3374    | 0.9528 | 0.4983 | 0.3294 |
| Support Vector Machine | 0.8597 | 0.9056 | 0.7654    | 0.6014 | 0.6736 | 0.5927 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced performer (AUC 0.908, MCC 0.586). A linear decision boundary over the one-hot/scaled features already separates the two income classes well, suggesting the relationship between features and income is largely additive/linear. |
| Decision Tree | Weakest of the non-Naive-Bayes models (AUC 0.754, MCC 0.500) without pruning/depth limits — a single unconstrained tree overfits the training data and generalizes worse than the other models, despite reasonable accuracy. |
| KNN | Solid but a step behind Logistic Regression/SVM (AUC 0.868, MCC 0.539). Performance is sensitive to the high-dimensional one-hot encoded space (curse of dimensionality) and the implicit Euclidean-distance assumption across mixed numeric/categorical features. |
| Naive Bayes | Lowest accuracy (0.538) but highest recall (0.953) — the conditional-independence assumption between features (clearly violated here, e.g. education and education-num are highly correlated) causes it to over-predict the positive class, trading precision for recall. |
| Support Vector Machine | Best overall (highest accuracy 0.860 and MCC 0.593, AUC comparable to Logistic Regression). The RBF-kernel SVM captures the same near-linear structure as Logistic Regression but with slightly better margin separation, at a higher computational/training cost. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** — highest Accuracy and MCC, and AUC essentially tied with Logistic Regression, making it the best single choice for this dataset. Logistic Regression is a close second and is much cheaper to train/serve. |

## Project Structure
```
Assignment2/
├── app.py              # Streamlit app
├── train_models.py     # data loading, preprocessing, training, evaluation, saving
├── requirements.txt
├── README.md
├── adult.data           # raw UCI Adult dataset (source data)
├── test_data.csv         # held-out raw test sample used by the Streamlit app
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json      # metrics computed during training
```

## Running Locally
```bash
pip install -r requirements.txt
python train_models.py   # retrains all 5 models (optional — already included)
streamlit run app.py
```

## Streamlit App Features
- CSV upload (test data only, with the `income` target column included)
- Model selection dropdown (5 trained classifiers)
- Display of all 6 evaluation metrics for the selected model
- Confusion matrix and classification report
- Side-by-side comparison table across all 5 models on the uploaded data

## Live Streamlit App Link
`<ADD YOUR DEPLOYED STREAMLIT APP URL HERE>`
