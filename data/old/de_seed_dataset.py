import os
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm
from torch.utils.data import Dataset

# 设置数据集路径
data_path = 'D:\\data\\SEED/ExtractedFeatures/'
label_path = 'D:\\data\\SEED/ExtractedFeatures/label.mat'
save_path = 'D:\\data\\SEED/ExtractedFeatures_dataset/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


# 数据集声明
class SEEDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SEEDDataset2(SEEDDataset):
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
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

def get_dataset_parameters(opt):
    """Prepare specific parameters for different datasets"""
    dataset_input_len = {
        'seed_de': 185
    }
    dataset_input_d = {
        'seed_de': 62 * 5
    }
    dataset_typenums = {
        'seed_de': 3
    }
    dataset_ignore_zero = {
        'seed_de': False
    }
    dataset_root_path = {
        'seed_de': 'D:\data\SEED\ExtractedFeatures_dataset'
    }
    opt.input_size = dataset_input_len[opt.dataset]
    opt.d_input = dataset_input_d[opt.dataset]
    opt.ignore_zero = dataset_ignore_zero[opt.dataset]
    opt.data_root_path = dataset_root_path[opt.dataset]
    opt.num_types = dataset_typenums[opt.dataset]
    return opt


if __name__ == '__main__':
    # 读取所有特征文件
    feature_files = [file for file in os.listdir(data_path) if file.endswith('.mat')][:-1]

    # 初始化特征和标签列表
    labels_np = loadmat(label_path)['label'][0] + 1
    # 遍历特征文件
    for file in tqdm(feature_files):
        features = []
        # 读取特征和标签
        data = loadmat(os.path.join(data_path, file))
        de_features = [data['de_LDS'+str(i)] for i in range(1, 16)]

        # 预处理特征
        for de_feature in de_features:
            all_channels = []
            for channel in de_feature:
                channel = channel[:185, :]
                all_channels.append(channel)
            trial_feature = np.stack(all_channels, axis=0)
            features.append(trial_feature)
        features = np.array(features)
        # 将特征和标签转换为PyTorch张量
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels_np, dtype=torch.long)
        # 创建数据集
        seed_dataset = SEEDDataset(features, labels)

        # 保存数据集为.dataset格式
        torch.save(seed_dataset, os.path.join(save_path, file[:-3] + 'dataset'))


