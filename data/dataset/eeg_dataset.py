import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.io as scio


class SEEDEEGLoader(Dataset):
    def __init__(self, root_path, flag, sub_dep_indep, sub_id, window_size):
        super(SEEDEEGLoader, self).__init__()
        self.window_size = window_size
        self.max_seq_len = window_size
        self.load_path = root_path
        self.class_names = [0, 1, 2]
        if sub_dep_indep == 'dep':
            self.data, self.time_stamps, self.label = self.load_sub_dependent(sub_id, flag)
        elif sub_dep_indep == 'indep':
            pass

    def load_sub_dependent(self, sub_id, flag):
        labels = scio.loadmat(os.path.join(self.load_path, 'label.mat'))['label'][0]
        data_files = os.listdir(self.load_path)
        sub_files = [file for file in data_files if file.split('_')[0] == str(sub_id)]
        trials = np.arange(1, 16)
        if flag == 'TRAIN':
            trials = trials[:9]
        elif flag == 'TEST':
            trials = trials[9:]
        data_list = []
        time_stamps_list = []
        label_list = []
        for file in sub_files:
            if file.endswith('.mat') and file != 'label.mat':
                data = scio.loadmat(os.path.join(self.load_path, file))
                experiment_name = file.split('.')[0]
                for key in data.keys():
                    trial = key.split('_')[-1][3:]
                    if 'eeg' not in key or int(trial) not in trials:
                        continue
                    cur_trial_data = data[key]
                    length = len(cur_trial_data[0])
                    pos = 0
                    while pos + self.window_size <= length:
                        data_list.append(torch.from_numpy(cur_trial_data[:, pos:pos + self.window_size]))
                        raw_label = labels[int(key.split('_')[-1][3:]) - 1]  # 截取片段对应的 label，-1, 0, 1
                        label_list.append(raw_label + 1)
                        time_stamps_list.append(torch.arange(pos, pos + self.window_size) / 128)
                        pos += self.window_size
        eeg_data = torch.stack(data_list, dim=0)
        self.feature_df = eeg_data
        eeg_data = eeg_data.transpose(1, 2)
        time_stamps = torch.stack(time_stamps_list, dim=0).unsqueeze(-1)
        labels = torch.LongTensor(label_list)
        return eeg_data, labels, time_stamps

    def __getitem__(self, ind):
        return self.data[ind], self.time_stamps[ind], self.label[ind]

    def __len__(self):
        return len(self.label)


class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_eeg_dep_train_test_dataset():
    print('loading eeg subject dependent dataset...')
    # 定义训练集和测试集的数据和标签列表
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []
    
    # 读取所有以数字开头的.npy文件
    for file_name in tqdm(sorted(os.listdir('D:/data/SEED/npy/Preprocessed_EEG'))):
        if file_name[0].isdigit():
            # 读取数据
            data = np.load(os.path.join('D:/data/SEED/npy/Preprocessed_EEG', file_name))
            # 按照第0维度9:6进行分割，前9份作为训练集的部分，后6份作为测试集的部分
            train_data = data[:9]
            test_data = data[9:15]
            # 将所有训练集的部分合并作为训练集的数据，测试集部分合并作为测试集的数据
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            
            # 读取标签
            label = np.load('D:/data/SEED/npy/Preprocessed_EEG/label.npy')
            # 按照第0维度9:6进行分割，前9份作为训练集的标签，后6份作为测试集标签
            train_label = label[:9]
            test_label = label[9:15]
            # 将训练集和测试集的标签添加到列表中
            train_label_list.append(train_label)
            test_label_list.append(test_label)
    
    # 将训练集和测试集的数据和标签合并
    train_data = np.concatenate(train_data_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    test_label = np.concatenate(test_label_list, axis=0)
    
    # 将数据和标签转换为tensor类型
    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).long()
    
    # 构建训练集和测试集的数据集
    train_dataset = CustomDataset(data=train_data, label=train_label)
    test_dataset = CustomDataset(data=test_data, label=test_label)
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = get_eeg_dep_train_test_dataset()