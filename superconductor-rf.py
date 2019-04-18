import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

#Reading in the data and converting it to a pandas DataFrame
#It is not an exceptionally large dataset therefore it can all be loaded at once
data = pd.read_csv('data/train.csv')
df = pd.DataFrame(data)

#columns contains a list of all the column lables
columns = df.columns
#dims is a tuple containg the shape of the dataframe: (width, height)
dims = df.shape

#Check for null values, there are none in this dataset
if (df.isna().sum()).sum() == 0:

    #X is the data used to predict, aka everything other than the target variable
    X = data.iloc[:, :-1].values
    #y is the target variable
    y = data.iloc[:, -1].values

    #Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

    rf = GridSearchCV(RandomForestRegressor(), cv=5,
                       param_grid={"n_estimators": [1000, 2500, 5000],
                                   "max_depth": [1e7, 1e8]},
                        n_jobs=-1, verbose=1)

    rf.fit(X_train, y_train)

    print(rf.best_score_) # 0.9193451766366879
    print(rf.best_params_) # {'max_depth': 100000000.0, 'n_estimators': 2500}
    print(rf.score(X_test, y_test)) #0.9195991367754737

else:
    print('Script exited, null values in data')
