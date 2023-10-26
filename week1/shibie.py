import cv2
import os


img1 = cv2.imread("./Test.jpg")


class AlexNet(nn.Module):
    # 构造函数
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # 提取特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input [3,224,224] output [96,55,55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # input [96,55,55] output [96,27,27]

            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # input [96,27,27] output[256,27,27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # input [256,27,27] output [256,13,13]

            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # input [256,13,13] output[384,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # input [384,13,13] output[384,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # input [384,13,13] output[256,13,13]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)  # input [256,13,13] output[256,6,6]
        )

        # 分类全连接
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, num_classes)
        )

    def forward(self, x):  # 正向传递的函数
        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        output = F.log_softmax(x, dim=1)

        return output