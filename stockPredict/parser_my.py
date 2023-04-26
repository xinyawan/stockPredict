import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='data/000001SH_index.csv')


# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=8, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=5, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--save_file', default='model/stock.pkl') # 模型保存位置


# args = parser.parse_args()
#
# device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
# args.device = device
#
# 这段代码主要是使用 argparse 模块解析命令行参数，并且定义了一些常用的参数，
# 例如 GPU 卡号、训练轮数、LSTM 层数、输入特征维度、隐藏层维度、学习率、sequence 长度等等。这些参数可以用于训练神经网络时的超参数调节。
#
# 其中，parser.add_argument() 方法用于添加参数，包括参数名、默认值、类型等等。
# parser.parse_args() 方法用于解析命令行参数，并将解析结果存储在 args 变量中。
#
# 最后，根据是否有可用的 GPU，将设备设置为 GPU 或 CPU，并将其存储到 args.device 变量中。