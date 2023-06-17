import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch
from tqdm import trange, tqdm
from data.dataset.de_LDS_dataset import get_de_LDS_dep_train_test_dataset
from data.dataset.eeg_dataset import SEEDEEGLoader
from models.pyraformer import my_pyraformer as Pyraformer
import random
import numpy as np
import argparse

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



def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
    loss /= len(dataloader)
    return loss


def arg_parser():
    parser = argparse.ArgumentParser()

    # running mode
    parser.add_argument('-eval', action='store_true', default=False)


    # global parameters
    parser.add_argument('-task_name', type=str, default='classification')

    # dataset parameters
    parser.add_argument('-dataset', type=str, default='eeg_dep')
    parser.add_argument('-label_len', type=int, default=1)
    parser.add_argument('-pred_len', type=int, default=0)

    # Train parameters
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-lr', type=float, default=7e-5)
    parser.add_argument('-visualize_fre', type=int, default=2000)
    parser.add_argument('-pretrain', action='store_false', default=True)
    parser.add_argument('-hard_sample_mining', action='store_false', default=True)
    parser.add_argument('-seed', default=20230413)

    # Model parameters
    parser.add_argument('-model', type=str, default='Pyraformer')
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-n_heads', type=int, default=4)
    parser.add_argument('-n_layer', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    # Pyraformer parameters
    parser.add_argument('-window_size', type=str, default='[4, 4, 4]')  # # The number of children of a parent node.
    parser.add_argument('-inner_size', type=int, default=3)  # The number of ajacent nodes.
    parser.add_argument('-use_tvm', action='store_true', default=False)  # Whether to use TVM.


    opt = parser.parse_args()
    return opt


def get_dataset_parameters(opt):
    """Prepare specific parameters for different datasets"""
    dataset_seq_len = {
        'eeg_dep': 37000,
        'de_dep': 185
    }
    dataset_input_d = {
        'eeg_dep': 62,
        'de_dep': 62 * 5
    }
    dataset_typenums = {
        'eeg_dep': 3,
        'de_dep': 3
    }
    dataset_root_path = {
        'eeg_dep': r'D:\data\SEED\Preprocessed_EEG',
        'de_dep': r'D:\data\SEED\npy\ExtractedFeatures'
    }

    opt.seq_len = dataset_seq_len[opt.dataset]
    opt.enc_in = dataset_input_d[opt.dataset]
    opt.num_class = dataset_typenums[opt.dataset]
    opt.data_root_path = dataset_root_path[opt.dataset]
    return opt


if __name__ == '__main__':
    opt = arg_parser()
    opt = get_dataset_parameters(opt)
    opt.window_size = eval(opt.window_size)
    set_seed(opt.seed)

    # default device is CUDA
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    train_dataset, test_dataset = \
        SEEDEEGLoader(root_path=opt.data_root_path, flag='TRAIN', sub_dep_indep='dep', sub_id=1, window_size=200),\
        SEEDEEGLoader(root_path=opt.data_root_path, flag='TEST', sub_dep_indep='dep', sub_id=1, window_size=200)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    model = eval(opt.model).Model(opt)
    model.to(opt.device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))


    # 记录训练和测试集上的loss和准确率
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    attn_mats = []

    # 训练
    for epoch in trange(opt.epoch):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(train_dataloader), desc='process epoch ' + str(epoch)):
            data, target = data.to(opt.device), target.to(opt.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # 保存训练集上的loss和准确率
        train_losses.append(train_loss / len(train_dataloader))
        train_accuracy = evaluate(model, train_dataloader, opt.device)
        train_accuracies.append(train_accuracy)

        # 保存测试集上的loss和准确率
        test_loss = evaluate_loss(model, test_dataloader, criterion, opt.device)
        test_losses.append(test_loss)
        test_accuracy = evaluate(model, test_dataloader, opt.device)
        test_accuracies.append(test_accuracy)
        print(
            f'Epoch: {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Test Accuracy: {test_accuracy:.4f}')


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