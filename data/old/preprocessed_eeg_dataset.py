import os
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm
from torch.utils.data import Dataset


# 数据集声明
class SEEDDatasetMaking(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SEEDDataset(Dataset):
    def __init__(self, dataset_files):
        self.data = []
        self.labels = []
        for dataset_file in dataset_files:
            dataset = torch.load(dataset_file)
            self.data.extend(dataset.data)
            self.labels.extend(dataset.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

if __name__ == '__main__':
    # 设置数据集路径
    data_path = 'D:\\data\\SEED/Preprocessed_EEG/'
    label_path = 'D:\\data\\SEED/Preprocessed_EEG/label.mat'
    save_path = 'D:\\data\\SEED/Preprocessed_EEG_dataset/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    # 读取所有特征文件
    feature_files = [file for file in os.listdir(data_path) if file.endswith('.mat')][:-1]

    # 初始化特征和标签列表
    labels = loadmat(label_path)['label'][0]
    min_eeg_len = 47000
    # 遍历特征文件
    for file in tqdm(feature_files):
        # 读取特征和标签
        data = loadmat(os.path.join(data_path, file))
        eeg_data = list(data.values())[3:]
        eeg_list = []
        for eeg in eeg_data:
            eeg_slice = eeg[:, :37000]
            eeg_list.append(eeg_slice)

        eeg_data = np.stack(eeg_list, axis=0)
        # 将脑电数据和标签转换为PyTorch张量
        eeg = torch.tensor(eeg_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 创建数据集
        seed_dataset = SEEDDatasetMaking(eeg, labels)
        #
        # 保存数据集为.dataset格式
        torch.save(seed_dataset, os.path.join(save_path, file[:-3] + 'dataset'))
    print('处理完毕')

