# Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning

![Alt Link](pictures/cover.png)

# Overview
A company can grow rapidly by understanding its customers' personality behaviors, enabling it to provide better services and benefits to customers who have the potential to become loyal customers. By processing historical marketing campaign data to improve performance and target the right customers to transact on the company’s platform, the focus from this data insight is to create a predictive clustering model, making it easier for the company to make decisions.

## Goals
The main goal of customer segmentation based on characteristics using machine learning is to identify groups of customers with similar needs, preferences, and behaviors, enabling the company to develop more relevant and effective marketing strategies, offers, and services.

## Objective
1. Extracting in-depth and valuable insights through EDA  
2. Performing data cleaning and preparation for modeling  
3. Building an unsupervised learning machine learning model using K-Means  
4. Interpreting the clusters obtained  
5. Making business recommendations based on the findings  

# Tools
1. Python Programming Language
2. Jupyterlab / Jupyter Notebook

# Model and Metric Evaluation
1. K-Means Clustering.
2. Silhouette Score.

# [Dataset](https://github.com/Yogaaprila/Predict-Customer-Personality-to-Boost-Marketing-Campaign-by-Using-Machine-Learning/blob/main/marketing_campaign_data.csv)
This dataset contain 2240 rows and 30 columns

| Column                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Unnamed: 0           | Index column, likely from the original dataset                              |
| ID                   | Unique identifier for each customer                                         |
| Year_Birth           | Year of birth of the customer                                              |
| Education            | Educational level of the customer                                          |
| Marital_Status       | Marital status of the customer                                             |
| Income               | Annual household income of the customer                                    |
| Kidhome              | Number of small children in the customer’s household                      |
| Teenhome             | Number of teenagers in the customer’s household                           |
| Dt_Customer          | Date when the customer enrolled with the company                          |
| Recency              | Number of days since the last purchase                                    |
| MntCoke              | Amount spent on Coke products                                             | 
| MntFruits            | Amount spent on fruits                                                    |
| MntMeatProducts      | Amount spent on meat products                                             |
| MntFishProducts      | Amount spent on fish products                                             |
| MntSweetProducts     | Amount spent on sweet products                                            |
| MntGoldProds         | Amount spent on gold products                                             |
| NumDealsPurchases    | Number of purchases made with discount                                    |
| NumWebPurchases      | Number of purchases made through the company’s website                    |
| NumCatalogPurchases  | Number of purchases made using a catalog                                  |
| NumStorePurchases    | Number of purchases made directly in store                                |
| NumWebVisitsMonth    | Number of visits to the company’s website in the last month               |
| AcceptedCmp3         | 1 if the customer accepted the offer in campaign 3, 0 otherwise           |
| AcceptedCmp4         | 1 if the customer accepted the offer in campaign 4, 0 otherwise           |
| AcceptedCmp5         | 1 if the customer accepted the offer in campaign 5, 0 otherwise           |
| AcceptedCmp1         | 1 if the customer accepted the offer in campaign 1, 0 otherwise           |
| AcceptedCmp2         | 1 if the customer accepted the offer in campaign 2, 0 otherwise           |
| Complain             | 1 if the customer complained                                              |
| Z_CostContact        | Fixed cost associated with customer contact                               |
| Z_Revenue            | Estimated revenue generated from the customer                             |
| Response             | 1 if the customer accepted the offer in the last campaign, 0 otherwise    |

# Contact Information
- **Email:** [yogaapril0504@gmail.com](mailto:yogaapril0504@gmail.com)
- **LinkedIn:** [Yoga Aprila](https://www.linkedin.com/in/yoga-aprila/)

## 1. Feature Extraction
Extract some new features based on from existing features on dataset.

| New Feature's Name                       | Description                                                                                 |
|----------------------------|---------------------------------------------------------------------------------------------|
| Membership Duration        | The duration of the customer's membership calculated as the current year (2024) minus the year the customer joined (`dt_customer`). |
| Total Transaction          | The total number of transactions made by the customer, calculated as the sum of the `numpurchases` column. |
| Campaign Acceptance Total  | The total number of previous campaigns accepted by the customer.                           |
| Conversion Rate            | The percentage of purchases (Total Transaction) made through the web.                      |
| Age                        | The customer's age calculated as the current year (2024) minus the `year_birth`.           |
| Age Group                  | The customer's age group classification based on their age.                                |
| Total Kids                 | The total number of children the customer has.                                             |
| Is Parent                  | Indicates whether the customer is a parent or not.                                         |
| Total Spending             | The total amount of money spent by the customer on products, calculated from the `Mnt` columns. |


## 2. Exploratory Data Analysis (EDA)

### 2.1 Summary Statistic

### 2.2 Scatter Plot of Conversion Rate vs Age, Income, Total Spending, Total Kid, Is Parent and Campaign Acceptance Total
![Alt Link](pictures/)


### 2.3 Distribution of Conversion Rate and Age Group, Education, and Marital Status

### 2.4 Distribution Chart of Numerical Feature

### 2.5 Bar Chart of Categorical Feature

## 3. Data Preprocessing

## 4. Modeling and Evaluation

## 5. Clustering Interpretation

## 6. Business Recommendation

