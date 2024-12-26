# Bank-Customer-Churn-Model

**Overview**

The Bank Customer Churn Model is a machine learning project designed to predict whether a customer is likely to leave a bank (churn) based on various features. This model uses advanced preprocessing techniques and a Support Vector Machine (SVM) classifier for prediction. The project also focuses on optimizing the model's performance through hyperparameter tuning and handling class imbalance.


## Project Objectives

**Data Encoding:**

Transform categorical variables into numerical representations to make the dataset suitable for machine learning models.

**Feature Scaling:**

Normalize the data to ensure all features contribute equally to the model’s performance.

**Handling Imbalanced Data:**

*Address class imbalance in the dataset using the following techniques:*

Random Under Sampling (RUS): Reduce the majority class to balance the dataset.

Random Over Sampling (ROS): Increase the minority class to balance the dataset.

**Support Vector Machine Classifier:**

Use SVM, a powerful classification algorithm, to predict customer churn.

**Grid Search for Hyperparameter Tuning:**

Optimize the SVM model’s hyperparameters for better accuracy and performance


## Project Workflow


**1. Data Preprocessing**

Data Cleaning: Handle missing or inconsistent data.

Encoding: Apply encoding techniques like one-hot or label encoding for categorical variables.

Scaling: Use techniques like MinMaxScaler or StandardScaler to normalize numerical features.

**2. Imbalanced Data Handling**

Apply RUS and ROS to ensure balanced class distribution.

**3. Model Building**

Train the SVM classifier on the processed data.

**4. Model Optimization**

Perform grid search to identify the best combination of hyperparameters for the SVM model.

**5. Evaluation**

Evaluate the model’s performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.


## Tech Stack

**Technologies and Tools Used**

*Programming Language:*

Python

*Libraries:*

Pandas

NumPy

Scikit-learn

Matplotlib/Seaborn (for visualizations)


## Result

The project delivers a well-optimized churn prediction model with balanced performance metrics, addressing class imbalance and improving accuracy through hyperparameter tuning.


## Future Scope

Incorporate additional algorithms to compare performance.

Experiment with advanced techniques like SMOTE for handling imbalanced data.

Deploy the model as a web application for real-time predictions.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

