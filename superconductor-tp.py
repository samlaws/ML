import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
import numpy as np

from tpot import TPOTRegressor

#Reading in the data and converting it to a pandas DataFrame
#It is not an exceptionally large dataset therefore it can all be loaded at once
data = pd.read_csv(r'data\train.csv')
df = pd.DataFrame(data)

#columns contains a list of all the column lables
columns = df.columns
#dims is a tuple containg the shape of the dataframe: (width, height)
dims = df.shape

#X is the data used to predict, aka everything other than the target variable
X = data.iloc[:, :-1].values
#y is the target variable
y = data.iloc[:, -1].values

#Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

pipeline_optimizer = TPOTRegressor(generations=5, population_size=15, verbosity=2)


pipeline_optimizer.fit(X_train, y_train)
score = pipeline_optimizer.score(X_test, y_test)

print(score)
