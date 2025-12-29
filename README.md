# Bank Account Fraud Detection Under Extreme Class Imbalance

## Overview

This repository contains an end-to-end machine learning pipeline for detecting fraudulent bank account applications using the NeurIPS 2022 Bank Account Fraud Dataset. The project focuses on modeling under extreme class imbalance and emphasizes ranking quality, recall optimization, decision threshold tuning, and model interpretability rather than naive accuracy maximization.

The workflow reflects real-world fraud and risk analytics systems where decisions must balance operational cost, customer friction, and fraud capture rate.


## Problem Definition

Fraud detection in financial applications is a rare-event classification problem characterized by a highly skewed target distribution. Fraudulent cases represent less than one percent of all observations, rendering traditional evaluation metrics such as accuracy misleading and ineffective.

The objective of this project is to identify high-risk applications as early as possible while minimizing unnecessary intervention on legitimate customers. This reframes fraud detection as a decision-optimization and ranking problem rather than a standard binary classification task.

## Dataset Description

The analysis is based on the NeurIPS 2022 Bank Account Fraud Dataset, Variant II.

Key Properties

Target variable: fraud_bool

Positive class prevalence: approximately 0.9 percent

Feature types: numeric, categorical, behavioral, temporal

Presence of encoded missing values and domain-specific anomalies


## Key Challenges Addressed

Extreme class imbalance

Encoded missing values treated as informative signals

Heavy-tailed behavioral distributions

Non-linear feature interactions

Business-driven decision thresholds

Interpretability and auditability requirements

# Feature Engineering Strategy
## Missingness as Information

Several variables encode missing or unavailable values using sentinel values such as -1. Instead of discarding or blindly imputing these values, the pipeline explicitly converts them to missing values and introduces binary missing-indicator features.

This allows the model to learn whether the absence of information itself is predictive, which is common in real-world fraud and risk assessment systems.

## Domain-Specific Feature Flags

Additional features are constructed to capture anomalous or suspicious patterns, including:

Negative intended balance indicators

Missing credit risk score flags

Session duration and address history absence indicators

These features enable the model to detect behavioral irregularities beyond raw numeric magnitudes.

# Preprocessing Pipeline

All preprocessing steps are implemented using a unified ColumnTransformer to ensure reproducibility and prevent data leakage.

##Transformations Applied

Median imputation for numeric features

Log transformation for highly skewed variables

Standard scaling for continuous features

One-hot encoding for categorical variables

Safe log transformations to handle zero and negative values

All transformations are fit exclusively on the training data.

# Models Evaluated

Three models are evaluated using the same preprocessing pipeline to ensure fair comparison.

Logistic Regression

A class-weighted logistic regression model serves as a transparent baseline. It provides interpretability and establishes a lower bound for performance under imbalance-aware training.

## Random Forest

A random forest classifier with balanced subsampling is used to capture non-linear relationships and feature interactions while remaining robust to noisy variables.

## XGBoost

XGBoost is used as the primary model due to its strong ranking performance under extreme imbalance. Class imbalance is explicitly addressed using the scale_pos_weight parameter.

### Evaluation Metrics

Accuracy is intentionally excluded due to its misleading nature in highly imbalanced datasets.

The primary evaluation metrics are:

Precision–Recall AUC (PR-AUC)

Receiver Operating Characteristic AUC (ROC-AUC)

Recall

F1-score

PR-AUC is emphasized as it directly measures performance on the minority class.

# Model Selection Results

Across all evaluated models, XGBoost consistently achieved the highest PR-AUC and ROC-AUC values. Logistic regression provided a strong interpretable baseline, while random forest demonstrated moderate improvement but did not outperform gradient boosting.

XGBoost was therefore selected for downstream optimization and analysis.

# Decision Threshold Optimization

Instead of using a fixed probability threshold of 0.5, the model’s decision threshold is optimized using a validation set.

Thresholds are evaluated across a predefined range, and the optimal threshold is selected based on F1-score and business-oriented performance considerations. This converts probabilistic predictions into actionable operational decisions.

# Business-Oriented Performance Analysis

Using the optimized decision threshold, the model captures approximately 81 percent of fraudulent applications while flagging less than 20 percent of the total population for review.

This represents more than a fourfold improvement over random targeting and demonstrates substantial gains in operational efficiency and cost control.

# Funnel and Gain Analysis

Cumulative gain analysis shows that fraudulent cases are highly concentrated among the top-ranked predictions. A relatively small fraction of applications contains the majority of fraud risk, enabling targeted investigation strategies rather than blanket screening.

This behavior closely mirrors real-world fraud detection pipelines used in financial institutions.

# Model Explainability

To ensure transparency and regulatory compatibility, both global and local explainability techniques are applied.

## SHAP Analysis

SHAP values are used to identify the most influential features globally and locally. Behavioral velocity features and recent activity indicators dominate the model’s decision process, highlighting the importance of temporal and interaction-aware modeling.

## LIME Analysis

LIME is applied for instance-level explanations, providing localized feature contributions for individual predictions. This is particularly useful for manual review and audit workflows.

# Calibration Analysis

Model calibration is evaluated to ensure that predicted probabilities correspond to observed fraud rates. Well-calibrated probabilities are essential for threshold tuning, cost modeling, and risk-based decision systems.
