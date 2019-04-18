import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from sklearn.metrics import r2_score

#Reading in the data and converting it to a pandas DataFrame
#It is not an exceptionally large dataset therefore it can all be loaded at once
data = pd.read_csv('data/train.csv')
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

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def r_sq(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_baseline():
    model = Sequential()
    model.add(Dense(units=16, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_sq])
    return model

def build_best():
    model = Sequential()
    model.add(Dense(units=64, input_dim=X.shape[1], activation='relu'))
    #model.add(Dropout(rate=0.25))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=64))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_sq])
    return model




regressor = KerasRegressor(build_fn=build_best, batch_size=10, epochs=100)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(r2_score(y_test, y_pred))
