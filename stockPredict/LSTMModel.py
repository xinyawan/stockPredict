import torch.nn as nn


def super(lstm, self):
    pass


class lstm(nn.Module):

    def __init__(self, input_size=8, hidden_size=32, num_layers=1 ,
                 output_size=1 , dropout=0, batch_first=True):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch, seq_len, hidden_size , hn.shape and cn.shape : num_layes * direction_numbers, batch, hidden_size
        # a, b, c = hidden.shape
        # out = self.linear(hidden.reshape(a * b, c))
        out = self.linear(hidden)
        return out

    # 以上代码定义了一个名为lstm的类，该类继承自nn.Module。该类实现了一个基于LSTM的神经网络模型。
    #
    # 在类的构造函数__init__中，定义了一些模型相关的参数，包括输入维度input_size、
    #     隐藏层维度hidden_size、LSTM层数num_layers、输出维度output_size、
    #     dropout概率dropout和batch_first标志。
    # 接着，该类初始化了一个nn.LSTM对象rnn，该对象包含了LSTM的所有参数，包括输入维度、隐藏层维度、LSTM层数、batch_first标志和dropout概率。
    # 此外，该类还初始化了一个nn.Linear对象linear，该对象用于将LSTM的输出映射到指定的输出维度上。
    #
    # 在类的forward函数中，首先将输入x传入LSTM对象rnn中进行前向计算，
    #     得到LSTM的输出out以及最终的隐藏状态和细胞状态hidden和cell。
    # 然后，将隐藏状态hidden传入线性层对象linear中，将输出映射到指定的输出维度上，
    #     并将结果作为最终的输出返回。