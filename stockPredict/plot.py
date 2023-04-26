# coding=gb2312
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from parser_my import args
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/000001SH_index.csv',index_col=0)
df.info()

plt.figure()#figsize=(15,5)表示figure 的大小为宽、长(单位为inch)
plt.subplot(2,1,1) #subplot（2,1,1）指的是在一个2行1列共2个子图的图中，定位第1个图来进行操作。最后的数字就是表示第几个子图，此数字的变化来定位不同的子图。
plt.plot(df.open.values,color='red',label='open')#红色曲线绘制开盘价变化情况
plt.plot(df.close.values,color='green',label='close')#绿色曲线绘制收盘价变化情况
plt.plot(df.low.values,color='blue',label='low')#蓝色曲线表示最低价变化情况
plt.plot(df.high.values,color='black',label='high')#黑色曲线绘制最高价变化情况
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')

plt.legend(loc = 'best')

plt.subplot(2,1,2)#定位第2个子图
plt.plot(df.vol.values,color='black',label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc = 'best')
plt.show()
#
# 这段代码主要是绘制股票数据的图表，具体解释如下：
#
# 1. 第一行代码指定编码方式为gb2312。
#
# 2. 导入必要的库，包括torch、torchvision、pandas和matplotlib.pyplot等。
#
# 3. 使用pandas库读取股票数据文件'000001SH_index.csv'，并将第一列设置为索引。
#
# 4. 使用matplotlib.pyplot库绘制股票数据的图表，包括开盘价、收盘价、最低价、最高价和成交量等信息。
#
# 5.
#
# 6. plt.subplot(2,1,1)指的是在一个2行1列共2个子图的图中，定位第1个图来进行操作。最后的数字就是表示第几个子图，此数字的变化来定位不同的子图。
#
# 7. plt.plot(df.open.values,color='red',label='open')表示在当前子图中，绘制开盘价变化情况的红色曲线。
# 同样的，plt.plot(df.close.values,color='green',label='close')表示绘制收盘价变化情况的绿色曲线，
# plt.plot(df.low.values,color='blue',label='low')表示绘制最低价变化情况的蓝色曲线，
# plt.plot(df.high.values,color='black',label='high')表示绘制最高价变化情况的黑色曲线。
#
# 8. plt.title('stock price')表示设置当前子图的标题为“stock price”，
# plt.xlabel('time [days]')表示设置X轴标签为“time [days]”，
# plt.ylabel('price')表示设置Y轴标签为“price”。
#
# 9. plt.legend(loc='best')表示在当前子图中添加图例，并将其位置设置为最佳。
#
# 10. plt.subplot(2,1,2)表示定位到第2个子图，绘制成交量的变化情况。
#
# 11. plt.plot(df.vol.values,color='black',label='volume')表示在当前子图中，绘制成交量变化情况的黑色曲线。
#
# 12. plt.title('stock volume')表示设置当前子图的标题为“stock volume”，plt.xlabel('time [days]')表示设置X轴标签为“time [days]”，plt.ylabel('volume')表示设置Y轴标签为“volume”。
#
# 13. plt.legend(loc='best')表示在当前子图中添加图例，并将其位置设置为最佳。
#
# 14. plt.show()表示显示图表。