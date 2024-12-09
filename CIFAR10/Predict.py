# 测试图片为哪一类
from PIL import Image
import torch
import torchvision.transforms
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU

img = Image.open("img/frog4.png")

# 如果是RGBA图像，去掉透明度通道
if img.mode == 'RGBA':
    img = img.convert('RGB')

transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 数据集的归一化
])

img = transformer(img)
img = img.unsqueeze(0)  # 添加批次维度 (1, 3, 32, 32)


class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.model = Sequential(

            Conv2d(3, 32, 3, padding=1),  # Conv Layer 1
            ReLU(),
            MaxPool2d(2),

            Conv2d(32, 64, 3, padding=1),  # Conv Layer 2
            ReLU(),
            MaxPool2d(2),

            Conv2d(64, 128, 3, padding=1),  # Conv Layer 3
            ReLU(),
            MaxPool2d(2),

            Flatten(),

            Linear(128 * 4 * 4, 256),  # Fully Connected Layer 1
            ReLU(),
            Linear(256, 128),          # Fully Connected Layer 2
            ReLU(),
            Linear(128, 10)            # Fully Connected Layer 3 (Output)
        )

    def forward(self, x):
        return self.model(x)

# 加载整个模型
model1 = torch.load('model/cifar10_model.pth', weights_only=False)
model2 = torch.load('model/cifar10_ReLU_model.pth', weights_only=False)

# 检查设备是否为MPS（如果是MPS则使用MPS设备）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 将模型和图像都移到相同的设备
model1 = model1.to(device)
model2 = model2.to(device)

img = img.to(device)


model1.eval()
with torch.no_grad():
    output1 = model1(img)
    output2 = model2(img)
    print("-----------cifar10_model-----------\n")
    print(output1)
    class1 = class_names[torch.argmax(output1, dim=1).item()]
    print(class1)
    print("-----------cifar10_ReLU_model-----------\n")
    print(output2)
    class2 = class_names[torch.argmax(output2, dim=1).item()]
    print(class2)

