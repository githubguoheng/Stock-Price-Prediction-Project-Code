# 使用LSTM进行多变量时间序列预测股价——学习记录
## 背景
作为一个资深的A股韭菜，一切有利于解套的手段都值得尝试一下。\
偶然看到了一篇机器学习的博文，主要介绍利用深度学习方法(LSTM)进行多元时间序列预测。跟股价的模型有点相似，尝试一下，说干就干！
## 目录
### 基本概念
* 什么是时间序列分析？
* 什么是 LSTM？

**时间序列分析**：时间序列表示基于时间顺序的一系列数据。它可以是秒、分钟、小时、天、周、月、年。未来的数据将取决于它以前的值。主要有两种类型的时间序列分析——
* 单变量时间序列
* 多元时间序列

在多元时间序列数据的情况下，将有不同类型的特征值并且目标数据将依赖于这些特征。
![](/pics/price.jpg)

例如在上图中看到的，在多元变量中将有多个列来对目标值进行预测，Close列为收盘价格，不仅取决于它以前的值，还取决于其他特征。因此，要预测即将到来的Close值，我们必须考虑包括目标列在内的所有列来对目标值进行预测。

在训练时，如果我们使用 7 列 [target,feature1, feature2, feature3, feature4, feature5, feature6] 来训练模型，我们需要为即将到来的预测日提供 6 列 [feature1, feature2, feature3, feature4, feature5, feature6]

**LSTM**:是一个循环神经网络，能够处理长期依赖关系，不做过多的介绍，可以参考 [LSTM 长短期记忆人工神经网络](https://baike.baidu.com/item/%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17541107?fr=ge_ala)


### 预测招商银行股价实现思路
## 训练数据获取
从百度股市通中抓取招商银行的历史资金流向，保存文件 zhaoshang ![招商银行资金流向](/pics/history.jpg) \
使用脚本 dataprocess.py进行数据格式化处理，处理后的数据参见 \
[股价数据预处理脚本 dataprocess.py](/dataprocess.py) \
[股价数据预处理结果 China Merchants Bank.csv](/China%20Merchants%20Bank.csv)

## 预测模型实现
这里我选取了 Close 列，收盘价作为将要预测的值
### 训练数据拆分
 ```
df=pd.read_csv("China Merchants Bank.csv",parse_dates=["Date"],index_col=[0])
df.head()
print(df.head())
# print(df.tail())
# print(df.shape)

test_split=round(len(df)*0.20)

# print(test_split)

df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
 ```   

### 数据缩放
可以注意到数据范围非常大，并且它们没有在相同的范围内缩放，因此为了避免预测错误，让我们先使用MinMaxScaler缩放数据。

```
scaler=MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.fit_transform(df_for_testing)
```

### 数据拆分
将数据拆分为X和Y，这是最重要的部分

```
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)    

trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)
```

解释 \
n_past是我们在预测下一个目标值时将在过去查看的步骤数。这里使用30，意味着将使用过去的30个值(包括目标列在内的所有特性)来预测第31个目标值。因此，在trainX中我们会有所有的特征值，而在trainY中我们只有目标值。

for循环分解：对于训练，dataset = df_for_training_scaled, n_past=30 
* 当i=30时:data_X.addend (df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])，
从n_past开始的范围是30，所以第一次数据范围将是 [30 - 30,30,0:6] 相当于 [0:30,0:7]
* 因此在dataX列表中，df_for_training_scaled[0:30,0:7]数组将第一次出现。
* dataY.append(df_for_training_scaled[i,0])，i = 30，所以它将只取第30行开始的Close(因为在预测中，我们只需要Close列，所以列范围仅为0，表示Close列)。第一次在dataY列表中存储df_for_training_scaled[30,0]值。

所以包含7列的前30行存储在dataX中，只有Close列的第31行存储在dataY中。然后我们将dataX和dataY列表转换为数组，它们以数组格式在LSTM中进行训练 。

### 训练模型

```
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,7)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

grid_search = grid_search.fit(trainX,trainY)
```

### 测试模型
利用测试数据集测试模型
```
my_model=grid_search.best_estimator_.model
prediction=my_model.predict(testX)
```

### 图形化预测结果和真实数据

```
plt.plot(original, color = 'red', label = 'Real Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('China Merchants Bank Stock Price')
plt.legend()
plt.show()

```

**真实股价与预测股价对比** 
![预测结果](/pics/pred_stock.jpg)

### 未解决的问题&一些思考
正如上文介绍，利用一些多元的参数，实现了对股价的“预测”，预测的趋势折现图与实际折线图趋势基本吻合，仔细的读者会发现，“预测” 是加了引号的，也就是说，当前只实现了基于历史多元参数的历史趋势的“预测”，而历史的数据是已经发生过的，实际值真实存在。并没有真正实现对未来数据的预测。 \
而互联网上的大多数资料均出自以下参考链接中的中文翻译版本，甚至连代码和测试数据集都没有调整，并没有解决实际问题。

这里就出现了一个悖论，要对未来的股价进行预测，当前选取的多元变量也是还未发生，且未来不确定值的变量，这些数据如何产生？当前我还没搞懂，以下是几个猜测的方向，将继续探索。

* 多元变量选取的错误，不应该选取未来不确定的参数作为多元变量？
* 预测模型的入参需要基础数据，预测未来的数据，入参从哪来？入参如果也是未来的数据怎么办？

机器学习理论近几年发展较为迅速，但成功的应用于实际业务助力业务发展的案例却较少，可能受限于自己的知识范围，还没有接触到有比较成功的将机器学习应用于证券交易的案例，希望能发现学习一下

### 参考
[Stock Price Prediction – Machine Learning Project in Python](https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/)

