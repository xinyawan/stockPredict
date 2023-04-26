from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import matplotlib.pyplot as plt

from train import enumerate, range, print


def len(preds):
    pass


def max(error):
    pass


def min(error):
    pass


def eval(preds=None, labels=None, labels=None, preds=None):
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size,
                 num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,
                                args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())
        
    preds_1=[0]*(len(preds))
    labels_1=[0]*(len(preds))
    error=[0]*(len(preds))

    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (
        preds[i][0] * (close_max - close_min) + close_min,
        labels[i] * (close_max - close_min) + close_min))
        preds_1[i]=preds[i][0]* (close_max - close_min) + close_min
        labels_1[i]=labels[i]* (close_max - close_min) + close_min
        error[i]= (preds[i][0]-labels[i])/labels[i]￥
    plt.figure()
    #print(preds_1)
    print('最小预测误差：%.2f' % min(error))
    print('最大预测误差：%.2f' % max(error))

    plt.plot(preds_1,color='red',label='predict')#红色曲线绘制预测值变化情况
    plt.plot(labels_1,color='green',label='real')#绿色曲线绘制真实值变化情况
    plt.title('stock price')
    plt.xlabel('time [days]')
    plt.ylabel('price')
    plt.legend(loc = 'best')
    plt.show()

eval()
#
# 这段代码是用于对 LSTM 模型进行评估的。首先导入需要的模块和数据集，然后定义了一个 eval 函数。
# 在函数中，首先创建了一个 LSTM 模型，然后加载模型的参数，接着遍历测试数据集，对每个数据点进行预测，并将预测值和真实值存储到列表中。
# 然后，将预测值和真实值进行反归一化处理，计算预测误差，并绘制预测值和真实值的变化曲线。最后调用 eval 函数进行评估。