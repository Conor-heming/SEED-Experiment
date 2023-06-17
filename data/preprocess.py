import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

def de_LDS_preprocess():
# Define the path to the directory containing the .mat files
    load_path = r'D:\data\SEED\ExtractedFeatures'
    save_path = r'D:\data\SEED\npy\de_LDS'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Initialize an empty list to store the extracted da
    # Loop through each file in the directory
    for file in tqdm(os.listdir(load_path)):
        data_list = []
        # Check if the file is a .mat file
        if file.endswith('.mat') and file[0].isdigit():
            mat_data = sio.loadmat(os.path.join(load_path, file))
            # Loop through each key in the loaded data
            for key in mat_data.keys():
                # Check if the key starts with 'de_LDS' and ends with a number between 1 and 15
                if key.startswith('de_LDS'):
                    # Extract the data corresponding to the key
                    extracted_data = mat_data[key][:, :185, :]
                    # Append the extracted data to the list
                    data_list.append(extracted_data)

            # Get the filename without the extension
            filename = os.path.splitext(file)[0]

            # Concatenate the extracted data along a new dimension
            concatenated_data = np.stack(data_list, axis=0)

            # Swap the first and second dimensions
            swapped_data = np.swapaxes(concatenated_data, 1, 2)

            # Save the swapped data as a .npy file with the original filename
            np.save(os.path.join(save_path, filename + '.npy'), swapped_data)
    # 读取label.mat文件
    mat_data = sio.loadmat(os.path.join(load_path, 'label.mat'))
    # 获取键名
    key = list(mat_data.keys())[3]
    # 读取键对应的值
    labels = mat_data[key]
    # 压缩维度，去掉长度为1的维度
    labels = np.squeeze(labels) + 1
    # 将处理后的数据保存为.npy文件
    np.save(os.path.join(save_path, 'label.npy'), labels)


    print('finished.')

def eeg_preprocess():
    # 定义读取路径
    load_path = r'D:\data\SEED\Preprocessed_EEG'
    # 定义存储路径
    save_path = r'D:\data\SEED\npy\Preprocessed_EEG'
    # 如果存储路径不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 遍历读取路径下的所有文件
    for file in tqdm(os.listdir(load_path)):
        # 定义一个空列表，用于存储读取的数据
        data_list = []
        # 判断文件是否为.mat文件
        if file.endswith('.mat') and file[0].isdigit():
            # 读取.mat文件
            mat_data = sio.loadmat(os.path.join(load_path, file))
            # 遍历读取的数据中的所有键
            for key in list(mat_data.keys())[3:]:
                # 读取键对应的值
                extracted_data = mat_data[key]
                # 对第1维度进行截断处理，只保留前37000个数据点
                extracted_data = extracted_data[:, :37000]
                # 将处理后的数据添加到列表中
                data_list.append(extracted_data)
                # 输出键的名字
                print(key)
            # 获取文件名（不包含后缀）
            filename = os.path.splitext(file)[0]
            # 使用np.stack方法将所有数据合并，要求合并后有三个维度，第0维度是数据中这些键值对的个数
            concatenated_data = np.stack(data_list, axis=0)
            # 交换第1和第2维度
            swapped_data = np.swapaxes(concatenated_data, 1, 2)
            # 将处理后的数据以.npy格式保存到存储路径下，文件名与原文件名相同
            np.save(os.path.join(save_path, filename + '.npy'), swapped_data)
        # 读取label.mat文件
    mat_data = sio.loadmat(os.path.join(load_path, 'label.mat'))
    # 获取键名
    key = list(mat_data.keys())[3]
    # 读取键对应的值
    labels = mat_data[key]
    # 压缩维度，去掉长度为1的维度
    labels = np.squeeze(labels) + 1
    # 将处理后的数据保存为.npy文件
    np.save(os.path.join(save_path, 'label.npy'), labels)
    # 处理完成后输出信息
    print('处理完成')

if __name__ == '__main__':
    de_LDS_preprocess()
    eeg_preprocess()

    