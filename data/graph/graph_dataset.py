import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from plv import compute_PLV
from tqdm import tqdm


class EEGPLVGraphDataset(InMemoryDataset):
    def __init__(self, root='D:\data\our_data\graph-dataset', transform=None, pre_transform=None):
        super(EEGPLVGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 返回数据集中所有原始文件的文件名
        return [f for f in os.listdir(self.raw_dir) if f.endswith('_data.npy')]

    @property
    def processed_file_names(self):
        # 返回数据集中所有处理后文件的文件名
        return ['{}.pt'.format(f.split('_')[0]) for f in self.raw_file_names]

    def download(self):
        # 数据集不提供下载
        pass

    def process(self):
        for i, raw_path in tqdm(enumerate(self.raw_paths)):
            # 读取数据和标签
            data = np.load(raw_path)
            label = np.load(raw_path.replace('_data.npy', '_label.npy'))

            # 计算PLV
            plv = compute_PLV(data)

            # 过滤边权重
            num_edges = plv.shape[1] * (plv.shape[1] - 1) // 2
            top_k = num_edges // 5
            plv = torch.reshape(plv, (plv.shape[0], plv.shape[1] * plv.shape[2]))
            threshold_trails = torch.kthvalue(plv, k=top_k, dim=1).values
            # 用来保存每一个文件中的数据
            data_list = []
            for trial in range(plv.shape[0]):
                edge_index = []
                edge_attr = []
                for i in range(plv.shape[1]):
                    for j in range(i+1, plv.shape[2]):
                        if plv[trial, i, j] >= threshold_trails[trial]:
                            edge_index.append([i, j])
                            edge_attr.append(plv[trial, i, j])
                edge_index = torch.tensor(edge_index).t().contiguous()
                edge_attr = torch.tensor(edge_attr)
                # 构建图数据
                x = torch.tensor(data)
                y = torch.tensor(label)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                data_list.append(data)
            # 保存处理后的数据
            torch.save(self.collate(data_list), self.processed_paths[i])


if __name__ == '__main__':
    dataset = EEGPLVGraphDataset()