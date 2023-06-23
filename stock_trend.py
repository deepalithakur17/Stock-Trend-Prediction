import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import pandas_datareader as data

start ='20-01-01'
end='2021-12-31'

df=data.DataReader('AAPL','yahoo',start,end)
df.head()
df.tail()

df=df.reset_index()
df.head()

df=df.drop(['Date','Adj Close'],axis=1)
df.head()

plt.plot(df.Close)

ma100 = df.Close.rolling(100).mean()
ma100

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')

ma200=dg.Close.rolling(200).mean()
ma200

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')

df.shape

#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len[df]*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

data_training.head()

data_testing.head()

from sklearn.preprocessing inport MiniMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array=sacler.fit_transform(data_training)
data_training_array

data_training_array.shape
x_train=[]
y_train=[]


for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
x_train, y_train=np.array(x_train),np.array(y_train)

from keras.layers import Dense,Dropout,LSTM
from keras.model import Sequential
model = Sequential()
model.add(LSTM(units=50,activation='reiu',return_sequences=True,input_shape=(x_train.shape[1],i)))
model1.add(Dropout(0,2))
model.add(LSTM(units=60,activation='reiu',return_sequences=True))
model1.add(Dropout(0,3))
model.add(LSTM(units=80,activation='reiu',return_sequences=True))
model1.add(Dropout(0,3))
model.add(LSTM(units=120,activation='reiu',return_sequences=True,input_shape=(x_train.shape[1],i)))
model1.add(Dropout(0,4))

model.sdd(Dense(units=1))
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train, ecochs=50)

model.save('keras_model.h5')
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing, ignore_index=True)
final_df.head()
