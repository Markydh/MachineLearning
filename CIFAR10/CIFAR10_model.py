import time

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss, ReLU
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

# 检查是否支持 MPS 加速
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理和增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# 获取数据集
train_data = torchvision.datasets.CIFAR10(root="Datasets", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root="Datasets", train=False, download=True, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # 模型结构
        self.model = Sequential(
            Conv2d(3, 32, 3, padding=1),  # Conv Layer 1
            ReLU(),
            MaxPool2d(2),

            Conv2d(32, 64, 3, padding=1),  # Conv Layer 2
            ReLU(),
            MaxPool2d(2),

            Conv2d(64, 128, 3, padding=1),  # Conv Layer 3
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



model = CIFAR10Model()
# 将模型移动到设备
model = model.to(device)

# 损失函数和优化器
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 每 30 轮学习率减小

# TensorBoard 日志记录
writer = SummaryWriter("Log/loss_log_ReLU")

start_time = time.time()

# 训练模型
epochs = 1
for epoch in range(epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        # 将数据移动到设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar("train_loss", running_loss, epoch)

    # 验证模型
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # 将数据移动到设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            val_correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()

    scheduler.step()  # 调整学习率

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, "
          f"Val Accuracy: {val_correct / len(val_data):.4f}")
    end_time = time.time()
    print(f"Run Time:{end_time - start_time}")

    writer.add_scalar("val_loss", val_loss, epoch)

writer.close()

# 测试模型
model.eval()
test_correct = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        # 将数据移动到设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        test_correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()


print(f"Test Accuracy: {test_correct / len(test_data):.4f}")


model = torch.save(model, "model/cifar10_ReLU_model.pth")


