import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

dataset_path = 'D:\\data\\SEED\\ExtractedFeatures_dataset'

class SEEDDataset(Dataset):
    def __init__(self, dataset_files):
        self.features = []
        self.labels = []
        for dataset_file in dataset_files:
            dataset = torch.load(dataset_file)
            self.features.extend(dataset.features)
            self.labels.extend(dataset.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# 卷积神经网络模型
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(62, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 2, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 超参数
batch_size = 1
learning_rate = 0.001
num_epochs = 50

# 读取数据集
# 读取所有数据集文件
dataset_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]

train_dataset = SEEDDataset(dataset_files[:30])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SEEDDataset(dataset_files[30:])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = EmotionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target + 1)
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Epoch: {epoch + 1}, Accuracy: {accuracy:.4f}')
