import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_de_LDS_dep_train_test_dataset():
    print('loading data ...')
    # 读取数据部分
    data_files = sorted(glob.glob('D:/data/SEED/npy/de_LDS/[0-9]*.npy'))
    train_data_list = []
    test_data_list = []
    train_label_list = []
    test_label_list = []
    label = np.load('D:/data/SEED/npy/de_LDS/label.npy')
    for file in tqdm(data_files):
        data = np.load(file)
        train_data_list.append(data[:9])
        test_data_list.append(data[9:15])
        train_label_list.append(label[:9])
        test_label_list.append(label[9:15])
    train_data = np.concatenate(train_data_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)
    test_label = np.concatenate(test_label_list, axis=0)

    # 返回两个基于torch的dataset，分别表示训练集和测试集
    train_dataset = CustomDataset(train_data, train_label)
    test_dataset = CustomDataset(test_data, test_label)
    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = get_de_LDS_dep_train_test_dataset()
    train_loader = DataLoader(train_dataset, batch_size=9)
    for x, y in tqdm(train_loader):
        print(x.shape)
        print(y.shape)