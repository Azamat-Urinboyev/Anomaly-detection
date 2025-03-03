# Anomaly Detection in Time-Series Data

This project focuses on detecting anomalies in time-series data using various machine learning models. The main code is implemented in the Jupyter Notebook `main.ipynb`.

## Table of Contents
- [Installation](link)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)


## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow

You can install the required libraries using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

## Dataset
The dataset used in this project is the KDD Cup 1999 Data, specifically the 10% subset of the data ```(kddcup.data_10_percent.csv)```. This dataset is widely used for network intrusion detection research. It contains a wide variety of intrusions simulated in a military network environment.

## Data Preprocessing

The data is loaded from the file ***kddcup.data_10_percent.csv***. The preprocessing steps include:

1. ***Loading the Data:*** The data is loaded into a pandas DataFrame.
2. ***Encoding Categorical Features:*** Categorical features are encoded using OneHotEncoder, because the categorical features are not ordinal data.
3. ***Sliding Window*** Beacause the KDD Cup data not originally time series data, I used sliding window to get usefull info from the order of traffic 
4. ***Scaling Numerical Features:*** Numerical features are scaled using MinMaxScaler.
5. ***Applying PCA:*** Principal Component Analysis (PCA) is applied to reduce the dimensionality of the data.

## Model Training
### XGBoost
An XGBoost classifier is trained on the preprocessed data. XGBoost is a powerful gradient boosting algorithm that is widely used for classification tasks.

### Random Forest
Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs for better accuracy and robustness. It reduces overfitting by averaging predictions (regression) or using majority voting (classification). It works well with high-dimensional data and is resistant to noise.

## One-Class SVM
One-Class SVM is an unsupervised anomaly detection algorithm that learns the boundary of normal data and classifies outliers as anomalies. It uses a kernel trick to map data into a higher-dimensional space, where it separates normal and abnormal points. It's effective for high-dimensional datasets but can be sensitive to hyperparameters like nu (controls the fraction of outliers) and gamma (kernel coefficient).

## Artificial Neural Network
Binary classification neural network using TensorFlow/Keras. It has three fully connected layers (128, 64, 32 neurons) with ReLU activation, dropout for regularization, and a final sigmoid layer for output. It uses binary cross-entropy loss and Adam optimizer. Early stopping prevents overfitting by monitoring validation loss.

## Evaluation
The models are evaluated using F1-score and classification reports. The F1-score is a measure of a model's accuracy that considers both precision and recall.

## Results
The results of the models are visualized using matplotlib and seaborn.

| Models | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| XGBoost| 1.00      | 1.00   | 1.00     |
|Random Forest| 1.00      | 1.00   | 1.00     |
|One-Class SVM| 95   | 95     | 0.99     |
|Artificial Neural Network|1|1|1|


### Key Findings:
- **XGBoost, Random Forest, and Artificial Neural Network (ANN)** models achieved **perfect scores** (F1 = 1.00), indicating they can accurately distinguish between normal and anomalous network traffic.
- **One-Class SVM** performed slightly lower, with an **F1-score of 0.99**, which still reflects strong anomaly detection capabilities but suggests a few misclassifications.
- The **high precision and recall** across all models indicate that false positives and false negatives are minimal, making them reliable choices for intrusion detection.

### Final Thoughts:
The **tree-based models (XGBoost, Random Forest) and ANN** showed the best performance, making them excellent candidates for real-world network anomaly detection tasks. However, given the **imbalanced nature of the dataset**, further testing on unseen or real-time network data is recommended to validate robustness. Additionally, **anomaly detection models should be regularly updated** to adapt to evolving cyber threats.