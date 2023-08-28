import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)    

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,7)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model


df=pd.read_csv("China Merchants Bank.csv",parse_dates=["Date"],index_col=[0])
df.head()
print(df.head())
# print(df.tail())
# print(df.shape)

test_split=round(len(df)*0.20)

# print(test_split)

df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]

# print(df_for_training.shape)
# print(df_for_testing.shape)

scaler=MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.fit_transform(df_for_testing)

trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)

print("trainX Shape-- ", trainX.shape)
print("trainY Shape-- ",trainY.shape)

print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)


print("trainX[0].shape-- \n",trainX[0].shape)
# print("trainY[0]-- ",trainY[0])

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

grid_search = grid_search.fit(trainX,trainY)

print(grid_search.best_params_)

my_model=grid_search.best_estimator_.model

prediction=my_model.predict(testX)
# print("prediction\n", prediction)
# print("\nPrediction Shape-",prediction.shape)

prediction_copies_array = np.repeat(prediction,7, axis=-1)

pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),7)))[:,0]

original_copies_array = np.repeat(testY,7, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),7)))[:,0]

print("Pred Values-- " ,pred)
print("\nOriginal Values-- " ,original)

plt.plot(original, color = 'red', label = 'Real Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('China Merchants Bank Stock Price')
plt.legend()
plt.show()