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
#### 2.1.1 Summary Statistic of Numerical Feature

| Column                      | Count   | Mean           | Std            | Min     | 25%     | 50%     | 75%     | Max          |
|-----------------------------|---------|----------------|----------------|---------|---------|---------|---------|--------------|
| ID                          | 2240    | 5592.16        | 3246.66        | 0       | 2828.25 | 5458.5  | 8427.75 | 11191.0      |
| Year_Birth                  | 2240    | 1968.81        | 11.98          | 1893    | 1959    | 1970    | 1977    | 1996         |
| Income                      | 2216    | 52247254.91    | 25173084.97    | 1730000 | 35303000| 51381500| 68522000| 666666000    |
| Kidhome                     | 2240    | 0.44           | 0.54           | 0       | 0       | 0       | 1       | 2            |
| Teenhome                    | 2240    | 0.51           | 0.54           | 0       | 0       | 0       | 1       | 2            |
| Recency                     | 2240    | 49.11          | 28.96          | 0       | 24      | 49      | 74      | 99           |
| MntCoke                     | 2240    | 303935.66      | 336597.40      | 0       | 23750   | 173500  | 504250  | 1493000      |
| MntFruits                   | 2240    | 26302.23       | 39773.43       | 0       | 1000    | 8000    | 33000   | 199000       |
| MntMeatProducts             | 2240    | 166950.0       | 225715.44      | 0       | 16000   | 67000   | 232000  | 1725000      |
| MntFishProducts             | 2240    | 37525.45       | 54628.98       | 0       | 3000    | 12000   | 50000   | 259000       |
| MntSweetProducts            | 2240    | 27062.95       | 41280.50       | 0       | 1000    | 8000    | 33000   | 263000       |
| MntGoldProds                | 2240    | 44021.88       | 52167.44       | 0       | 9000    | 24000   | 56000   | 362000       |
| NumDealsPurchases           | 2240    | 2.33           | 1.93           | 0       | 1       | 2       | 3       | 15           |
| NumWebPurchases             | 2240    | 4.08           | 2.78           | 0       | 2       | 4       | 6       | 27           |
| NumCatalogPurchases         | 2240    | 2.66           | 2.92           | 0       | 0       | 2       | 4       | 28           |
| NumStorePurchases           | 2240    | 5.79           | 3.25           | 0       | 3       | 5       | 8       | 13           |
| NumWebVisitsMonth           | 2240    | 5.32           | 2.43           | 0       | 3       | 6       | 7       | 20           |
| Z_CostContact               | 2240    | 3.0            | 0.0            | 3       | 3       | 3       | 3       | 3            |
| Z_Revenue                   | 2240    | 11.0           | 0.0            | 11      | 11      | 11      | 11      | 11           |
| Conversion_Rate             | 2240    | 4.40           | 4.91           | 0.0     | 1.22    | 2.6     | 5.35    | 43.0         |
| Age                         | 2240    | 55.19          | 11.98          | 28      | 47      | 54      | 65      | 131          |
| Total_Kid                   | 2240    | 0.95           | 0.75           | 0       | 0       | 1       | 1       | 3            |
| Total_Spending              | 2240    | 605798.21      | 602249.31      | 5000    | 68750   | 396000  | 1045500 | 2525000      |
| Total_Transaction           | 2240    | 14.86          | 7.68           | 0       | 8       | 15      | 21      | 44           |
| Campaign_Acceptance_Total   | 2240    | 0.30           | 0.68           | 0       | 0       | 0       | 0       | 4            |
| Membership_Duration         | 2240    | 10.97          | 0.68           | 10      | 11      | 11      | 11      | 12           |

#### 2.2.2 Summary Statistic of Categorical Features  
| Column         | Count | Unique | Top       | Frequency |
|----------------|-------|--------|-----------|-----------|
| Education      | 2240  | 5      | S1        | 1127      |
| Marital_Status | 2240  | 6      | Menikah   | 864       |
| Age_Group      | 2240  | 3      | Dewasa    | 1447      |



### 2.2 Scatter Plot of Conversion Rate vs Age, Income, Total Spending, Total Kid, Is Parent and Campaign Acceptance Total
![Alt Link](pictures/)


### 2.3 Distribution of Conversion Rate and Age Group, Education, and Marital Status

### 2.4 Distribution Chart of Numerical Feature

### 2.5 Bar Chart of Categorical Feature

## 3. Data Preprocessing

## 4. Modeling and Evaluation

## 5. Clustering Interpretation

## 6. Business Recommendation

