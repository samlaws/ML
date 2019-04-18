import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

#Reading in the data and converting it to a pandas DataFrame
#It is not an exceptionally large dataset therefore it can all be loaded at once
data = pd.read_csv(r'data\train.csv')
df = pd.DataFrame(data)

#columns contains a list of all the column lables
columns = df.columns
#dims is a tuple containg the shape of the dataframe: (width, height)
dims = df.shape

#Check for null values, there are none in this dataset
if (df.isna().sum()).sum() == 0:
    print('There are no null values in the dataset.')

#X is the data used to predict, aka everything other than the target variable
X = data.iloc[:, :-1].values
#y is the target variable
y = data.iloc[:, -1].values

#Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})

svr.fit(X_train, y_train)
score = svr.score(X_test, y_test)
