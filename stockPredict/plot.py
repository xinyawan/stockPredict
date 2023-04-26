# coding=gb2312
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from parser_my import args
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/000001SH_index.csv',index_col=0)
df.info()

plt.figure()#figsize=(15,5)��ʾfigure �Ĵ�СΪ����(��λΪinch)
plt.subplot(2,1,1) #subplot��2,1,1��ָ������һ��2��1�й�2����ͼ��ͼ�У���λ��1��ͼ�����в������������־��Ǳ�ʾ�ڼ�����ͼ�������ֵı仯����λ��ͬ����ͼ��
plt.plot(df.open.values,color='red',label='open')#��ɫ���߻��ƿ��̼۱仯���
plt.plot(df.close.values,color='green',label='close')#��ɫ���߻������̼۱仯���
plt.plot(df.low.values,color='blue',label='low')#��ɫ���߱�ʾ��ͼ۱仯���
plt.plot(df.high.values,color='black',label='high')#��ɫ���߻�����߼۱仯���
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')

plt.legend(loc = 'best')

plt.subplot(2,1,2)#��λ��2����ͼ
plt.plot(df.vol.values,color='black',label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc = 'best')
plt.show()
#
# ��δ�����Ҫ�ǻ��ƹ�Ʊ���ݵ�ͼ������������£�
#
# 1. ��һ�д���ָ�����뷽ʽΪgb2312��
#
# 2. �����Ҫ�Ŀ⣬����torch��torchvision��pandas��matplotlib.pyplot�ȡ�
#
# 3. ʹ��pandas���ȡ��Ʊ�����ļ�'000001SH_index.csv'��������һ������Ϊ������
#
# 4. ʹ��matplotlib.pyplot����ƹ�Ʊ���ݵ�ͼ���������̼ۡ����̼ۡ���ͼۡ���߼ۺͳɽ�������Ϣ��
#
# 5.
#
# 6. plt.subplot(2,1,1)ָ������һ��2��1�й�2����ͼ��ͼ�У���λ��1��ͼ�����в������������־��Ǳ�ʾ�ڼ�����ͼ�������ֵı仯����λ��ͬ����ͼ��
#
# 7. plt.plot(df.open.values,color='red',label='open')��ʾ�ڵ�ǰ��ͼ�У����ƿ��̼۱仯����ĺ�ɫ���ߡ�
# ͬ���ģ�plt.plot(df.close.values,color='green',label='close')��ʾ�������̼۱仯�������ɫ���ߣ�
# plt.plot(df.low.values,color='blue',label='low')��ʾ������ͼ۱仯�������ɫ���ߣ�
# plt.plot(df.high.values,color='black',label='high')��ʾ������߼۱仯����ĺ�ɫ���ߡ�
#
# 8. plt.title('stock price')��ʾ���õ�ǰ��ͼ�ı���Ϊ��stock price����
# plt.xlabel('time [days]')��ʾ����X���ǩΪ��time [days]����
# plt.ylabel('price')��ʾ����Y���ǩΪ��price����
#
# 9. plt.legend(loc='best')��ʾ�ڵ�ǰ��ͼ�����ͼ����������λ������Ϊ��ѡ�
#
# 10. plt.subplot(2,1,2)��ʾ��λ����2����ͼ�����Ƴɽ����ı仯�����
#
# 11. plt.plot(df.vol.values,color='black',label='volume')��ʾ�ڵ�ǰ��ͼ�У����Ƴɽ����仯����ĺ�ɫ���ߡ�
#
# 12. plt.title('stock volume')��ʾ���õ�ǰ��ͼ�ı���Ϊ��stock volume����plt.xlabel('time [days]')��ʾ����X���ǩΪ��time [days]����plt.ylabel('volume')��ʾ����Y���ǩΪ��volume����
#
# 13. plt.legend(loc='best')��ʾ�ڵ�ǰ��ͼ�����ͼ����������λ������Ϊ��ѡ�
#
# 14. plt.show()��ʾ��ʾͼ��