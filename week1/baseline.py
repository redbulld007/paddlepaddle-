#数据处理部分之前的代码，保持不变
import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import paddle.nn.functional as F
import gzip
import json
import load
# 定义数据集读取器
def load_data(mode='train'):
    # 加载数据
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')

    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode == 'valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")

    # 校验数据
    imgs_length = len(imgs)
    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
            len(imgs), len(labels))

    # 定义数据集每个数据的序号， 根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('float32')
            # 在使用卷积神经网络结构时，uncomment 下面两行代码
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')

            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator





# 定义多层全连接神经网络
# 多层卷积神经网络实现
import paddle.nn.functional as F


# 定义模型结构
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc = Linear(in_features=980, out_features=10)

    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    @paddle.jit.to_static
    def forward(self, inputs):

        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')


def train(model):
    model = MNIST()
    model.train()

    # 四种优化算法的设置方案，可以逐一尝试效果
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    # 调用加载数据的函数
    # train_custom_dataset = load.minist()
    train_loader = load_data('train')
    # train_loader = paddle.io.DataLoader(train_custom_dataset, batch_size=64, shuffle=True, num_workers=1,
    #                                     drop_last=True)

    EPOCH_NUM = 1
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            images, labels = data
            print(images.shape)
            images = paddle.to_tensor(images,dtype="float64")

            # 前向计算的过程，同时拿到模型输出值和分类准确率

            predicts= model(images)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist_test.pdparams')


def product(images="week1/mnist/train/imgs/9/19667.jpg"):
    model = MNIST()

    model_dict = paddle.load("mnist_test.pdparams")
    model.set_state_dict(model_dict)
    model.eval()
    image = Image.open("E:/八斗文库/paddlepaddle/week1/mnist/train/imgs/5/0.jpg")  # 替换为您的图像文件路径

    # 将图像转换为NumPy数组
    image_np = np.array(image)
    print(image_np.shape)
    images = paddle.to_tensor(image_np.reshape(1, 28, 28), dtype="float64")

    # 前向计算的过程，同时拿到模型输出值和分类准确率

    predicts= model(images)
    print(predicts)
    return predicts


# 创建模型
import time

start_time = time.time()

# 在这里写下你需要进行计时的代码
model = MNIST()
# 启动训练过程
train(model)
product()
end_time = time.time()

elapsed_time = end_time - start_time

print(f"执行时间为：{elapsed_time}秒")


print("Model has been saved.")