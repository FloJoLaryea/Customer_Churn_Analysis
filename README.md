
# **Customer Churn Analysis**: *Unveiling the recipe of customer departure*
## Project Scenario
Every company wants to increase its profit or revenue margin and customer retention is one key area industry players focus their resources. In today's world of machine learning, most companies build classification models to perform churn analysis on their customers. 

### Overview
The purpose of this project is to develop a machine learning model for binary classification. The model will predict whether a given instance belongs to one of two classes based on a set of input features.

### Background
Vodafone, a telecommunication company want to find the likelihood of a customer leaving the organization, the key indicators of churn as well as the retention strategies that can be applied to avert this problem.

### Project Ojectives
- Develop and train a machine learning model using historical data to predict whether a customer will churn or not.
- Evaluate the model's performance using appropriate metrics : accuracy,precision,recall,f1-score.
- Fine-tune the model parameters with GridSearchCv to optimize performance.
- Perform hypothesis testing to reject or fail to reject the null hypothesis


### Data for the project
The data for this projects has been divided into 3. The first 2 data sets are for training and evaluation the machine learning model  while the last data set is for testing the model.

### Data Dictionary

| Feature           | Description                                                | Data Type|
|-------------------|------------------------------------------------------------|-----------| 
| Gender            | Whether the customer is a male or a female                 |Object|
| SeniorCitizen     | Whether a customer is a senior citizen or not              |Object|
| Partner           | Whether the customer has a partner or not (Yes, No)        |Object|
| Dependents        | Whether the customer has dependents or not (Yes, No)       |Object|
| Tenure            | Number of months the customer has stayed with the company |Int|
| Phone Service     | Whether the customer has a phone service or not (Yes, No)  |Object|
| MultipleLines     | Whether the customer has multiple lines or not            |Object|
| InternetService   | Customer's internet service provider (DSL, Fiber Optic, No)|Object|
| OnlineSecurity    | Whether the customer has online security or not (Yes, No, No Internet)|Object|
| OnlineBackup      | Whether the customer has online backup or not (Yes, No, No Internet)|Object|
| DeviceProtection  | Whether the customer has device protection or not (Yes, No, No internet service)|Object|
| TechSupport       | Whether the customer has tech support or not (Yes, No, No internet)|Object|
| StreamingTV       | Whether the customer has streaming TV or not (Yes, No, No internet service)|Object|
| StreamingMovies   | Whether the customer has streaming movies or not (Yes, No, No Internet service)|Object|
| Contract          | The contract term of the customer (Month-to-Month, One year, Two year)|Object|
| PaperlessBilling  | Whether the customer has paperless billing or not (Yes, No)|Object|
| Payment Method    | The customer's payment method (Electronic check, Mailed check, Bank transfer(automatic), Credit card(automatic))|Object|
| MonthlyCharges    | The amount charged to the customer monthly| Float|
| TotalCharges      | The total amount charged to the customer|Float|                   
| Churn             | Whether the customer churned or not (Yes or No), **target variable**          |Object|


### Business Sucess Criteria
- Model accuracy: The model's accuracy should be above 70%
- Retention Strategy Effectiveness: The implemented retention strategies should show a measurable impact on reducing customer churn rates. This can be assessed by comparing churn rates before and after implementing the strategies.
- Cost Reduction: The model should contribute to reducing the costs associated with customer acquisition by identifying at-risk customers early on and allowing targeted retention efforts.
- Customer Satisfaction: While focusing on retention, the model and strategies should also aim to maintain or improve customer satisfaction levels. High customer satisfaction leads to increased loyalty and potentially higher customer lifetime value.
- Adaptability and Scalability: The developed model should be adaptable to changing business environments and scalable to accommodate larger datasets or additional features. This ensures its long-term viability and usefulness for the company.
- Feedback and Iteration: Continuous feedback loops should be established to gather insights from the model's predictions and refine the retention strategies accordingly. This iterative process ensures ongoing improvement and optimization of the churn prediction system.

### Future Work
Deploy the model to be used in the company's mobile/web application

## 🔅Hypothesis

Initial exploration on the data gave insight to a **null hypothesis** and its **alternative hypothesis** stated below:
- **Null Hypothesis (Ho)**: There is no significant relationship between the amount of monthly customer charges and customer churn.

- **Alternative Hypothesis (Ha)**: There is a statistically significant relationship between the amount of monthly customer charges and customer churn.
For the hypothesis testing, the method choice was Wilcoxon rank-sum test for reasons of unevenly distributed data.

## ✍️Data Understanding
The data was pulled from the following various sources:
- data from a Github repository
- data from a OneDrive 
- data from a remote database

Data from the Github repository and SQL database data sets were used in this project for training and evaluation the machine
learning model while the last data set was used for testing the accuracy of the model.

The major libraries used for this project were:
 1. Data manipulation: NumPy, pandas
 2. Data visualisation : Matplotlib, Seaborn, Plotly
 3. Machine learning methods : Sklearn, imblearn

 
From a quick screening of EDA, training and evaluation datasets had common column names as stated in the data dictionary
 These were therefore concatenated and the dataframe was later investigated in the exploratory data analysis for data cleaning.
## 🏋️‍♀️Data cleaning

The first phase of the cleaning aspect checked for duplicates, nulls, mismatched columns, data coherency, datatypes and column names

There were some observations which included:

- null values present in the following columns: Multiple Lines, Online Security, Online Backup, Device Protection  ,Tech Support ,Streaming TV , Streaming Movies, Total Charges and Churn
- the datatype of total charges column assuming an object status instead of a numerical data type
- there are no duplicates in the concatenated stage
- some column value meant the same thing had synonymous names. eg. true and yes, no and false etc.
- all charge amounts are in the currency $
 


Overall, the dataset did not need much cleaning. Key highlights of the cleaning involved:

- Converting to right datatypes
- Renaming column names to be coherent with each dataset
- Mapping boolean and None values to more meaningful categories


As this project included building machine models to predict, filling of null values were paused in this stage and incorporated in the machine learning pipelines to improve thr robustness of the model






## 🔅Exploratory Data Analysis
The Exploratory Data Analysis phase delved deep into the cleaned and preprocessed datasets of the Customer Churn Analysis. Through a series of analytical techniques and visualizations, key insights were uncovered, shedding light on various aspects of what prompts churn. The following are some of the key insights gained from the EDA process:

Following a descriptive summary, these were some results observed:

***SeniorCitizen*** : a mean value of 0.162 suggesting that approximately 16.24% of the customers in the dataset are senior citizens.The standard deviation of 0.369 indicates some variability in the distribution of senior citizen status among customers and the minimum value of 0 implies that there are non-senior customers in the dataset.

***Tenure*** : On average, customers stay with the service provider for approximately 32.58 months.The minimum tenure is 0 months, which could indicate newly acquired customers and the maximum tenure is 72 months, indicating some customers have been with the provider for a significant period.The standard deviation of 24.53 suggests that there is a considerable variation in tenure lengths among customers.

***MonthlyCharges*** : The minimum monthly charge is $18.40, while the maximum is $118.65 and on average, customers are charged approximately $65.09 per month.The standard deviation of 30.07 indicates variability in monthly charges among customers.

***TotalCharges*** : On average, customers have been charged a total of approximately $2302.06 while The minimum total charge is $18.80, while the maximum is $8670.10. The standard deviation of $2269.48 suggests significant variability in total charges among customers.

Following **univariate and bivariate analysis**, it revealed:
- high positive skewness of the dataset
- an imbalanced dataset
- little/no outliers
- high range of values in the numerical columns prompting scaling



## 📉Data Analysis 
Visualisations were produced based on the analytical questions asked. These visuals were paramount to providing a solid foundation for analysising any trends or hitting patterns in analysing the factors contributing to churn analysis

 Business Questions
1. What is the rate of churn by gender?
2. How do different levels of monthly customer charges impact churn rates?
3. Do customers who pay more monthly charges tend to stay longer? 
4. What is the prefered service [Streaming movies, streaming TV] gender?
5. What impact does the prefered service has on customer churn?
6. Are customers on long-term contracts less likely to churn?
7. How does does the relationship between monthly charges and churn differ based on contract duration?
8. What impact does the ISPs have on customer churn?
9. Which Internet Service Provider accounts for the most charges?
10. What is the effect of method of payment on customer churn?


Although executed separately in jupyter notebook, these visualisations were deployed in PowerBI and the results are as shown in the dashboard below.

!["https://github.com/FloJoLaryea/Customer_Churn_Analysis/blob/main/Customer_churn_Dashboard.png"](https://github.com/FloJoLaryea/Customer_Churn_Analysis/blob/main/Datasets/Customer_churn_Dashboard.png)

[Link to dashboard]("https://app.powerbi.com/groups/me/reports/4b740771-584c-49f2-8e1b-a1f4933b22e1/ReportSection?experience=power-bi")
## 🔎Data preparation and preprocessing

The target column was the **Churn** column

⭕The Churn column was visualised to check the degree of imbalance as this could impact the accuracy of the machine learning models. It was moderately imbalanced with a ratio of 70:30

⭕A Phi-K matrix was used to compare the correlation between the churn and all other features especially the categorical features.
From the Phi-K matrix, the features most important for data modelling were:

- Tenure

- Payment method

- Monthly charges

- Paperless billing

- Dependents

- Contracts

- Total Charges



The features were then further investigated for multicollinearity. The following were flagged:

- Tenure and total charges with a correlation coefficient of 0.84
- Total charges and monthly charges with a correlation coefficient of 0.76


Further actions were to exclude totalcharges and drop the other columns not identified significant for modeling

⭕Data was split into train and test groups 80/20 respectively

⭕Several pipelines were created to impute missing values, scale numerical values and encode categorical columns


⭕ Nulls of columns such as the stage or investor were to be filled with “undisclosed” under the assumption that source of such knowledge was classified, some companies had not yet received funding, and some companies were self funding with their own capital in order to maintain the data unbiased.
## 📐Modelling

The dataset is moderately imbalanced. The modelling was done in two phases,
- first using methods known to be able to handle imbalanced datasets to train
- upscaling and then train on balanced data using Synthetic Minority Over-sampling Technique (SMOTE)

Six models were used in training the data namely:
- Decision Tree Classifier 
- K Neighbors Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- CatBoost Classifier
- Logistic Regression

The metrics used to evaluate the performance of the models were:
- accuracy 
- precision
- recall 
- f1_score
- confusion matrix

## 👀Observations
🗝️***Genders*** : male customers are slightly higher than female customrs.

🗝️***Partner***: Customers with or without partners are about the same.

🗝️***Dependents*** : Seems we have more customers without dependent members than those who have..

🗝️***PhoneService*** : Also the majority of customers who don't have a phone service are way more than those who have.

🗝️
***InternetService*** : Customers who have internet service, prefer mostly DSL or Fiber optic.

🗝️***MultipleLines, InternetService, OnlineSecurity, OnlineBackup, TechSupport*** : There is a common pattern in these features indicating that most of the customers prefer not to have access to these features than those who do.

🗝️***StreamingMovies and StreamingTV***  have similar barplots, which means there is an equal amount of customers who prefer to either have or not have these services.

🗝️***Contract*** : In general customers prefer month-to-month contracts compared to other types such as two year or one year contracts.

🗝️***PaperlessBilling*** :  most customers would rather have Paperless billing than any other form and seems they use all the different forms of banking transactions such with Electronic Check being the majority


From training with unbalanced data, the top 3 performing models based on the f1_score are: logistic regression, catboost and sgb_classifier

Confusion matrix metric revealed the machine still classifying churn incorrectly

















## ✍️Conclusion and Recommendations
From the study, it can be concluded that Financial Services is the heart of start-up businesses in India with majority of these services combined with digital or technological services.

India is one of the world leaders in Technology and IT and it's no surprise when combined with finance, it translates to the Financial Services which looking at the timeline, were funded massively in a bid to stabilise the economy after the 2020 pandemic.

In addition, location favors the amount of funding one receives. If business are set up in Mumbai or Bangalore, the probability of getting funding is high. 

It is also safe to say that the more funding of a given year, the more startups would spring up. Recommendations are therefore, to branch in finance in either of top 2 headquarters. More analysis has to be carried out on the origin of investors.

It is also recommended that the younger the startup, the more work has to be done in terms of innovation, marketing and networking in order to attract investors quickly. Otherwise, the main source of funding is self-fund, family and friends and government.


It is also recommended to enhance analysis in the future with machine learning models to predict a projection of the current years as well as integrating with real time data to monitor market trends and analyse factors influencing startup business metrics. The likelihood of success of a business or the rate of growth depending on the sector could also be studied.

Overall, a majority of startups in India seem to thrive as India is a digital/tech dominated country and about the greater portion of startups are in this sector. The Indian startup ecosystem has growm rapidly over the recent years and even more growth can be anticipated in the coming years.
## 🙋‍♀️Authors

- Florence Josephina Laryea
- florencelaryea@gmail.com


## 🤗Acknowledgements

Much of our sincere gratitude goes to our instructors Racheal Appiah-Kubi and Violette Naa Adoley Allotey for their exceptional guidance, unwavering support, and invaluable mentorship throughout the course of this project.

Their expertise, dedication, and commitment to our learning journey have been instrumental in shaping our understanding and skills in data analysis


## 📚References and bibliography
Indian startup ecosystem: https://www.startupindia.gov.in/content/sih/en/funding.html

Restructing column sectors:https://www.businessinsider.in/business/startups/news/top-10-industries-for-new-startups-in-india-as-per-hurun-list/articleshow/105651758.cms

Currency exchange rate: https://www.poundsterlinglive.com/history/USD-INR-2018
