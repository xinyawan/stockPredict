from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import matplotlib.pyplot as plt


def enumerate(train_loader):
    pass


def range(epochs):
    pass


def print(total_loss):
    pass
def train():
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size,
                 num_layers=args.layers , output_size=1,
                 dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    # Adam梯度下降  学习率=0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    all_loss=[]
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,
                                                              args.sequence_length,args.batch_size )
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:4
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)
        all_loss.append(total_loss)
        if i % 10 == 0:
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('第%d epoch，保存模型' % i)
    # torch.save(model, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)
    plt.figure()
    plt.plot(all_loss,color='blue',label='Loss')#蓝色曲线绘制loss值变化情况
    plt.title('Loss value')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc = 'best')
    plt.show()

train()
#
#
# 这段代码是一个训练LSTM模型的程序。主要步骤如下：
#
# 1. 导入必要的库和模块，包括PyTorch的自动求导模块、神经网络模块、数据处理模块、可视化模块等。
#
# 2. 定义LSTM模型。模型的输入大小、隐藏层大小、层数、输出大小、dropout率、batch_first等都在args中设置。
#
# 3. 定义损失函数和优化器，这里使用MSE loss和Adam优化器。优化器的学习率也在args中设置。
#
# 4. 读取数据，并进行训练。训练过程中，先遍历所有的epoch，再遍历每个batch。对于每个batch，先将数据送入模型中，计算输出和损失，然后进行反向传播和参数更新。
# 在每个epoch结束时，将总的损失值记录下来，并将模型保存。最后将损失值可视化。
#
# 5. 最后调用train()函数开始训练。
#
# 总的来说，这段代码实现了一个基本的LSTM模型的训练过程，并且提供了一种可视化损失值变化的方法。