import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset.de_seed_dataset import SEEDDataset
import os
import torch
from tqdm import trange, tqdm
from model.eeg_swin_transformer import EEGSwinTransformer
import random
import numpy as np


def set_seed(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy


def evaluate_loss(model, dataloader, criterion):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output, attn_mat = model(data)
            loss += criterion(output, target + 1).item()

    loss /= len(dataloader)
    return loss, attn_mat


# 超参数
batch_size = 4
learning_rate = 3e-6
num_epochs = 300
set_seed(42)  # 设置随机种子为42
dataset_path = 'D:\\data\\SEED\\ExtractedFeatures_dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
# 读取所有数据集文件
dataset_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]

train_dataset = SEEDDataset(dataset_files[:30])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = SEEDDataset(dataset_files[30:])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = EEGSwinTransformer()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练和测试集上的loss和准确率
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
attn_mats = []

# 训练
for epoch in trange(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in tqdm(enumerate(train_dataloader), desc='process epoch '+str(epoch)):
        data, target = data.to(device)
        optimizer.zero_grad()
        output, channel_attn = model(data)
        loss = criterion(output, target + 1)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # 保存训练集上的loss和准确率
    train_losses.append(train_loss / len(train_dataloader))
    train_accuracy = evaluate(model, train_dataloader)
    train_accuracies.append(train_accuracy)

    # 保存测试集上的loss和准确率
    test_loss, attn_mat = evaluate_loss(model, test_dataloader, criterion)
    test_losses.append(test_loss)
    test_accuracy = evaluate(model, test_dataloader)
    test_accuracies.append(test_accuracy)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracy:.4f}')
    # 只保存在测试集上表现比较好或者后30轮训练得到的空间注意力矩阵
    if test_accuracy > 0.6 or epoch >= 270:
        attn_mats.append(attn_mat)

# 保存空间注意力矩阵
attn_mats_tensor = torch.cat(attn_mats, dim=0)
torch.save(attn_mats_tensor, 'channel-attn-mats.pt')

# 绘制并保存折线图
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
