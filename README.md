# GRU-LSTM-ON-IMDB

## 1.LSTM 
模型训练过程中，loss函数和acc函数图像如下图：  
![](https://github.com/AaahWendy/GRU-LSTM-ON-IMDB/blob/master/pic/1.png)     
![](https://github.com/AaahWendy/GRU-LSTM-ON-IMDB/blob/master/pic/2.png)      

在测试集上测试结果：  
![](https://github.com/AaahWendy/GRU-LSTM-ON-IMDB/blob/master/pic/LSTM_test.png)  


## 2.GRU 
模型训练过程中loss函数和acc函数的图象如下图：  
![](https://github.com/AaahWendy/GRU-LSTM-ON-IMDB/blob/master/pic/3.png)  
![](https://github.com/AaahWendy/GRU-LSTM-ON-IMDB/blob/master/pic/4.png)  

在测试集上的表现：  
![](https://github.com/AaahWendy/GRU-LSTM-ON-IMDB/blob/master/pic/GRU_test.png)  


## 3.对比

可以看出，在相似的参数情况下，两个模型的训练效果相差不大，在验证集均能达到90%左右的准确率。但是在训练过程中，由于GRU模型所含参数更少，训练时间更短，收敛速度更快。在测试集和验证集上的表现，GRU的准确率稍优于LSTM。
