# Credit-Risk-Classification

In this analysis we will utilize a [dataset](lending_data.csv) of historical lending activity from a peer-to-peer lending services company. Using this data we will train and evaluate a machine learning model that can identify the credit of borrowers based on loan risk.

# Overview of the Analysis

In this analysis, a dataset comprising 77,536 data points was investigated with the goal of building a logistic regression model to assess loan risk. The dataset was divided into training and testing sets to facilitate model development and evaluation.

Initially, a logistic regression model (referred to as Logistic Regression Model 1) was constructed using the LogisticRegression module from the scikit-learn library. The objective was to predict whether a loan given to a borrower in the testing set would be categorized as low-risk or high-risk. The dataset used for this model contained 75,036 low-risk loan data points and 2,500 high-risk data points.

To ensure the model's effectiveness and unbiased performance, a resampling technique was employed on the training data. The RandomOverSampler module from the imbalanced-learn library was utilized to address the class imbalance, generating a balanced dataset with 56,277 data points for both low-risk (0) and high-risk (1) loans. This balanced data was then used to create a new logistic regression model, referred to as Logistic Regression Model 2.

The purpose of Logistic Regression Model 2 remained the same as the initial model: to assess whether a loan given to a borrower in the testing set would be classified as low-risk or high-risk. The results of this model are outlined below, providing valuable insights into the model's predictive performance and its ability to distinguish between low- and high-risk loans in the testing dataset.

The analysis highlights the significance of addressing class imbalance in the training data, as it can substantially impact the model's ability to accurately identify high-risk instances. Logistic Regression Model 2, built on the balanced dataset, presents a robust and more reliable solution for predicting loan risk, with implications for making informed lending decisions and mitigating potential financial risks.

# Results

## Logistic Regression Model 1 - Original Data
- Precision: 92% average. 100% precision for low-risk (0) loans and 85% for high-risk (1) loans

- Accuracy: 99%

- Recall: (Sensitivity or true positive rate) 95% average. 99% precision for low-risk (0) loans and 91% for high-risk (1) loans

## Logistic Regression Model 2 - Resampled Training Data
- Precision: 92% average. 100% precision for low-risk (0) loans and 85% for high-risk (1) loans

- Accuracy: 99%

- Recall: (Sensitivity or true positive rate) 95% average. 99% precision for low-risk (0) loans and 91% for high-risk (1) loans

# Summary
In this analysis, two models were developed to address a classification task, both of which yielded identical predictions. The initial model (Logistic Regression Model 1) was built using the original dataset without any resampling. Despite obtaining satisfactory results, the preference leans towards utilizing this initial model due to its simplicity and reliance on the original data.

However, recognizing the importance of exploring various scenarios and striving for robustness in model performance, the intention to experiment with resampled data using different random states assigned to the RandomOverSampler module was expressed. The motivation behind this lies in assessing how model outcomes might differ based on varied resampled datasets while maintaining the focus on achieving better class balance in the training set.

By employing diverse random states, the RandomOverSampler can generate multiple resampled datasets with different class distributions. Subsequently, separate models can be trained based on these datasets, allowing for an investigation of potential variations in predictive outcomes. This strategy aims to enhance model understanding, address potential overfitting concerns, and ensure the model's adaptability to different data scenarios.

Considering the importance of model generalization and robustness, the exploration of resampled datasets with distinct random states is a prudent approach to validate and reinforce the reliability of the chosen model for practical applications. By embracing this data-driven methodology, one can attain deeper insights into the model's performance and its sensitivity to class imbalance, ultimately facilitating the selection of an optimal and well-suited solution for the classification task at hand.
