import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score , mean_squared_error
plt.style.use('fivethirtyeight')
df = df.set_index("Datetime")
df
df.index = pd.to_datetime(df.index)
df.plot(style = "." , figsize=(25 , 12))
def create_features(df):
    df = df.copy()
    df["dayofmonth"] = df.index.day
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["weekofyear"] = df.index.isocalendar().week
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    return df
df = create_features(df)
df

plt.figure(figsize = (12 , 8))
sns.boxplot(x = "hour" , y = "samayanallur" , data = df)

sns.boxplot(x = "dayofweek" , y = "samayanallur" , data = df)

sns.boxplot(x = "month" , y = "samayanallur" , data = df)

df = df[df["samayanallur"] > 18000]

train = df.loc[df.index < "01-01-2015"]
test = df.loc[df.index>= "01-01-2015" ]

plt.figure(figsize = (20 , 12))
train["samayanallur"].plot(style = "." )
test["samayanallur"].plot(style = "." , color = "red")

X_train , y_train = train.drop("samayanallur" , axis = 1) , train["samayanallur"]
X_test , y_test = test.drop("samayanallur" , axis = 1) , test["samayanallur"]

y_test.index

lr = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)

lr.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
pred = lr.predict(X_test)

np.sqrt(mean_squared_error(y_test , pred))

r2_score(y_test , pred)

pred = pd.DataFrame(pred , columns = ["predicted"] , index = y_test.index)
pred

fig, ax = plt.subplots(figsize=(20, 12))
y_test.plot(style=".", label="y_test" , ax = ax)
pred.plot(style=".", color="red", label="pred" , ax = ax)
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(20, 12))
df.loc[(df.index > "01-01-2017") &(df.index < "01-08-2017") ]["samayanallur"].plot(style = "." , ax = ax)
pred.loc[(pred.index > "01-01-2017") &(pred.index < "01-08-2017") ]["predicted"].plot(style = ".", ax = ax ,color = "red")

fig, ax = plt.subplots(figsize=(20, 12))
df.loc[(df.index > "01-01-2017") &(df.index < "02-01-2017") ]["samayanallur"].plot( ax = ax)
pred.loc[(pred.index > "01-01-2017") &(pred.index < "02-01-2017") ]["predicted"].plot(style = ".", ax = ax ,color = "red")

from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits = 5 , test_size = 365 *24  , gap = 24)

df = df.sort_index()
df

for train_idx , val_idx in tss.split(df):
    break
max(val_idx)

fig , ax = plt.subplots(5 , 1 , figsize = (15 , 15), sharex=True)
fold = 0
for train_idx , val_idx in tss.split(df):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    train["samayanallur"].plot(ax = ax[fold] )
    val["samayanallur"].plot(ax = ax[fold], color = "red")
    fold+=1

target_map = df["samayanallur"].to_dict()

def add_lag(df):

    df["lag1"] = (df.index - pd.Timedelta("24 hours")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("48 hours")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("7 hours")).map(target_map)


    return df
df_lag = add_lag(df)

scores = []
for train_idx , val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    X_train = train.drop("samayanallur" , axis = 1)
    y_train = train["samayanallur"]

    X_test = test.drop("samayanallur" , axis = 1)
    y_test = test["samayanallur"]


    lr = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.3
                         , alpha = 0.5)
    lr.fit(X_train , y_train , eval_set= [(X_train , y_train) , (X_test , y_test)] , verbose=100 )
    pred = lr.predict(X_test)
    pred_y = lr.predict(X_train)
    error = np.sqrt(mean_squared_error(y_test , pred))
    r2 = r2_score(y_test , pred)
    scores.append([error , r2 ,np.sqrt(mean_squared_error(y_train, pred_y)) ])

np.set_printoptions(suppress=True, precision=6)
np.array(scores).mean(axis = 0)

feat = pd.DataFrame(lr.feature_importances_ , index = lr.feature_names_in_ , columns = ["importance"])

feat.sort_values("importance").plot(kind="barh")

X = df.drop("samayanallur" , axis = 1)
y = df["samayanallur"]

np.linspace(0.01 , 1 , 30)

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import learning_curve
train_sizes , train_scores , val_scores =  learning_curve(xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=None,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.3
                         , alpha = 0.5)
                                                          , X , y
                                                         , cv = tss
                                                         , train_sizes = np.linspace(0.1 , 1.0 , 10)
                                                         , scoring = "neg_root_mean_squared_error"
                                                          )
                                                          
train_scores_m =  - train_scores.mean(axis = 1)
val_scores_m =  - val_scores.mean(axis = 1)

train_scores_m

plt.figure(figsize = (12 , 8))
plt.plot(train_sizes , train_scores_m , "r-+" , linewidth = 0.8 , label = "training_error")
plt.plot(train_sizes , val_scores_m , "b-" , linewidth = 0.6 , label = "validation_error")
plt.legend()
plt.show()

train_sizes

X_test , y_test = test.drop("samayanallur" , axis = 1) , test["samayanallur"]

pred = pd.DataFrame(lr.predict(X_test) , index = X_test.index , columns = ["predicted"])
pred

fig , ax = plt.subplots(figsize = (40 , 12))
pred.loc[(pred.index >"2017-12-01") & (pred.index <"2017-12-30") ]["predicted"].plot(ax = ax , color = "blue")
df.loc[(df.index >"2017-12-01") & (df.index <"2017-12-30") ]["samayanallur"].plot( style = "." , ax = ax , color = "red")




# Make predictions on the test set
pred = lr.predict(X_test)

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate and print R-squared score
r2 = r2_score(y_test, pred)
print(f"R-squared: {r2}")

# Display some of the predictions (output)
print("\nPredictions (Sample):")
print(pred[:10])  # Display first 10 predictions

# Calculate and display deviation (absolute difference between actual and predicted)
deviation = np.abs(y_test - pred)
print("\nDeviation (Sample):")
print(deviation[:10])  # Display first 10 deviations

# Optional: Create a DataFrame for better visualization
output_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': pred,
    'Deviation': deviation
})

print("\nOutput DataFrame (Sample):")
print(output_df.head(10))  # Display first 10 rows of the DataFrame

# Plot Actual vs Predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', linestyle='-', marker='.')
plt.plot(y_test.index, pred, label='Predicted', linestyle='-', marker='.')
plt.title('Actual vs Predicted Values')
plt.xlabel('Datetime')
plt.ylabel('Target Variable')
plt.legend()
plt.show()

# Plot Deviations
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, deviation, label='Deviation', linestyle='-', marker='.')
plt.title('Deviation (Absolute Error)')
plt.xlabel('Datetime')
plt.ylabel('Deviation')
plt.legend()
plt.show()