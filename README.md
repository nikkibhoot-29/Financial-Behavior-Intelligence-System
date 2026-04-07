# Financial Behavior Intelligence System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikkibhoot-29/Financial-Behavior-Intelligence-System/blob/main/notebooks/Financial_Behavior_Intelligence_System.ipynb)

Data Science project for customer segmentation, anomaly detection, and risk prediction using financial transaction data.

## Author

**Nikki Bhoot**  : Undergraduate Student | Aspiring Data Scientist  

**Project Type:** : Customer Segmentation, Anomaly Detection & Classification

---

## Overview
This project presents an end-to-end data science pipeline for analyzing financial transaction data to extract behavioral insights, identify anomalous activity, and predict high-value customers.

It integrates multiple datasets and applies both unsupervised and supervised learning techniques to support data-driven decision-making in financial systems.

---

## Problem Statement
Financial institutions generate large volumes of transactional data but often lack structured approaches to:
- Understand customer behavior
- Identify high-value segments
- Detect unusual or risky financial activity

---

## Objectives
- Perform customer segmentation using clustering techniques
- Detect anomalies in transaction behavior
- Build predictive models for customer classification
- Derive actionable business insights from transaction data

---

## Dataset
The analysis is based on multiple relational datasets:
- **FactTransaction** – transaction-level data
- **DimAccount** – account-related information
- **DimCustomer** – customer demographics

These datasets are merged to create a unified analytical dataset.

---

## Methodology

### Data Preparation
- Data cleaning and validation
- Handling missing values
- Feature engineering at customer level
- Time-based feature extraction

### Exploratory Data Analysis
- Channel-wise transaction patterns
- Account type behavior
- Customer status analysis
- Temporal trends (year, month, day)

### Customer Segmentation
- Algorithm: KMeans Clustering
- Identified distinct customer groups:
  - High-value customers
  - Regular customers
  - Low-activity customers

### Anomaly Detection
- Algorithm: Isolation Forest
- Identified ~5% of customers with unusual transaction behavior

### Predictive Modeling

#### Multiclass Classification
- Target: Customer segments (clusters)
- Model: XGBoost
- Result: Moderate accuracy (~55%), indicating realistic overlap between segments

#### Binary Classification
- Target: High-value vs others
- Models:
  - Logistic Regression
  - XGBoost
- Result: High accuracy (~95%) with better class separability

### Model Interpretation
- Feature importance analysis performed
- Key drivers:
  - Transaction frequency
  - Channel usage (Web, Mobile)
  - Age and ATM usage

---

## Model Performance Summary

| Model                  | Task                      | Accuracy |
|----------------------|---------------------------|----------|
| XGBoost              | Multiclass Classification | ~55%     |
| Logistic Regression  | Binary Classification     | ~95%     |
| XGBoost              | Binary Classification     | ~95%     |

---

## Key Insights
- High-value customers are more digitally active (especially Web)
- ATM usage is lower among high-value customers
- Younger customers show higher financial engagement
- Transaction behavior shows seasonal trends (higher during festive periods)
- Suspended accounts show irregular high-value transactions, indicating risk patterns

---

## Tech Stack

- Programming Language: Python
- Data Analysis: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost

---

## Requirements

- Python >= 3.8  
- Tested on Python 3.10  

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
---

## Business Applications
- Customer segmentation for targeted marketing
- Fraud detection and risk monitoring
- Behavioral analysis for financial decision-making
- Identification of high-value customers for prioritization

---

## Conclusion
This project demonstrates a complete data science workflow, from raw data integration to advanced modeling and interpretation. It highlights the importance of combining domain understanding with machine learning techniques to generate meaningful and actionable insights.

---
