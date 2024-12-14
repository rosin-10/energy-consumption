Project Overview:

This project focuses on time series forecasting using a dataset collected on a village "samayanallur," with measurements from 2002 to 2018. The primary objective is to predict hourly energy consumption based on historical data using the XGBoost algorithm.

Key Observations:

Data Preparation: The dataset was processed by adding features like day, month, and hour to help the model recognize time-based patterns.

Exploratory Data Analysis (EDA): Boxplots revealed trends and seasonality in the data, which are crucial for accurate predictions.

Model Performance: The XGBoost model achieved a decent fit with an R² score of around 0.65 and an RMSE of approximately 3835, indicating that the model captures a significant portion of the variance but can be improved.

Lag Features: Adding lag features (e.g., values from 24, 48, and 7 hours ago) improved the model's predictive accuracy, emphasizing the importance of past data.
Cross-Validation: Time series split cross-validation was used to ensure that the model's performance is reliable and not overfitted to any specific time period.
