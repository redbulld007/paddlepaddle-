import os
import cv2
import numpy as np
from paddle.io import Dataset
from paddle.vision.transforms import Normalize

class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, data_dir, label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        with open(label_path,encoding='utf-8') as f:
            for line in f.readlines():
                image_path, label = line.strip().split('\t')
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        # 读取灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype('float32')
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

# 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 打印数据集样本数
train_custom_dataset = MyDataset('mnist/train','mnist/train/label.txt', transform)
test_custom_dataset = MyDataset('mnist/val','mnist/val/label.txt', transform)
print('train_custom_dataset images: ',len(train_custom_dataset), 'test_custom_dataset images: ',len(test_custom_dataset))
def minist():
    transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
    # 打印数据集样本数

    train_custom_dataset = MyDataset('mnist/train', 'mnist/train/label.txt', transform)
    test_custom_dataset = MyDataset('mnist/val', 'mnist/val/label.txt', transform)
    print('train_custom_dataset images: ', len(train_custom_dataset), 'test_custom_dataset images: ',
          len(test_custom_dataset))
    return train_custom_dataset
