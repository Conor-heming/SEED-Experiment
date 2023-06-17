import torch.nn as nn

import os
from torch.utils.data import DataLoader
import torch
from tqdm import trange, tqdm
from data.dataset.eeg_dataset import get_eeg_dep_train_test_dataset
from data.dataset.de_LDS_dataset import get_de_LDS_dep_train_test_dataset
import random
import numpy as np
import argparse
from models import FEDformer, Autoformer, TimesNet
from models.pyraformer import my_pyraformer as pyraformer
from itertools import chain


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
    domain_correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data, torch.arange(data.shape[1], device=opt.device), None, None)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy


# def evaluate_loss(model, test_loader, class_criterion, device):
#     model.eval()
#     loss = 0.0
#     with torch.no_grad():
#         for i, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#             output, _ = model(data, torch.ones(size=(data.shape[1],), device=device), None, None)
#             loss += class_criterion(output, target).item()
#     loss /= (i + 1)
#     return loss


def arg_parser():
    parser = argparse.ArgumentParser()

    # global parameters
    parser.add_argument('-seed', default=20230413)
    parser.add_argument('-train_sub_ids', default='list(range(1,18))')
    parser.add_argument('-test_sub_ids', default='list(range(18,23))')
    parser.add_argument('-task_name', default='classification_cross_domain')

    # dataset parameters
    parser.add_argument('-dataset', type=str, default='de_dep')
    parser.add_argument('-label_len', type=int, default=1)
    parser.add_argument('-pred_len', type=int, default=0)

    # Train parameters
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lamb', type=float, default=1.0)


    # Model parameters
    parser.add_argument('-model', type=str, default='pyraformer')
    parser.add_argument('-domain_hidden', type=int, default=512)
    # parser.add_argument('-model', type=str, default='TimesNet')
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_ff', type=int, default=256)
    parser.add_argument('-top_k', type=int, default=3)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layer', type=int, default=4)
    parser.add_argument('-d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-e_layers', type=int, default=3)
    parser.add_argument('-moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('-embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('-freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('-enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('-dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('-activation', type=str, default='gelu', help='activation')
    parser.add_argument('-c_out', type=int, default=7, help='output size')

    # autoformer
    parser.add_argument('-output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('-factor', type=int, default=1, help='attn factor')

    # Timesnet
    parser.add_argument('-num_kernels', type=int, default=6, help='for Inception')

    # Pyraformer parameters
    parser.add_argument('-window_size', type=str, default='[4, 4, 4]')  # # The number of children of a parent node.
    parser.add_argument('-inner_size', type=int, default=3)  # The number of ajacent nodes.
    parser.add_argument('-use_tvm', action='store_true', default=False)  # Whether to use TVM.

    return parser.parse_args()

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
        'eeg_dep': r'D:\data\SEED\npy\de_LDS',
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

    # 用于跨被试
    # opt.train_sub_ids = eval(opt.train_sub_ids)
    # opt.test_sub_ids = eval(opt.test_sub_ids)
    set_seed(opt.seed)
    # default device is CUDA
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    if opt.dataset == 'de_dep':
        train_dataset, test_dataset = get_de_LDS_dep_train_test_dataset()
    elif opt.dataset == 'eeg_dep':
        train_dataset, test_dataset = get_eeg_dep_train_test_dataset()
    else:
        train_dataset, test_dataset = None, None
        print('no such dataset')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    # define model
    model = eval(opt.model).Model(opt)

    # define optimizer
    distri_params = [param for name, param in model.named_parameters() if 'domain_classifier' not in name]
    domain_params = model.domain_classifier.parameters()
    optimizer_distri = torch.optim.Adam(distri_params, lr=opt.lr)
    optimizer_domain = torch.optim.Adam(domain_params, lr=opt.lr)
    

    # define loss function
   
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # move model to device
    device = opt.device
    model.to(device)

    train_loss_list = []
    test_loss_list = []
    domain_loss_list = []
    train_accs = []
    test_accs = []
    # train model
    for epoch in trange(opt.epoch, desc='Epoch'):
        running_domain_loss, running_f_loss = 0.0, 0.0
        total_hit, total_num = 0, 0
        model.train()
        for i, ((source_data, source_label), (target_data, _)) in tqdm(enumerate(zip(train_dataloader, chain(*[test_dataloader] * 10))), desc='Train'):
            source_data, source_label, target_data = source_data.to(device), source_label.to(device), target_data.to(device)
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
            # set domain label of source data to be 1.
            domain_label[:source_data.shape[0]] = 1

            # Step 1 : train domain classifier
            _, domain_logits = model(mixed_data, torch.arange(mixed_data.shape[1], device=opt.device), None, None)

            loss = domain_criterion(domain_logits, domain_label)
            running_domain_loss += loss.item()
            loss.backward()
            optimizer_domain.step()

            # Step 2 : train feature extractor and label classifier
            class_logits, domain_logits = model(mixed_data, torch.arange(mixed_data.shape[1], device=opt.device), None, None)
            class_logits = class_logits[:source_data.shape[0]]
            loss = class_criterion(class_logits, source_label) - opt.lamb * domain_criterion(domain_logits, domain_label)
            running_f_loss += loss.item()
            loss.backward()
            optimizer_distri.step()

            optimizer_domain.zero_grad()
            optimizer_distri.zero_grad()

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]

        running_f_loss = running_f_loss / (i + 1)
        running_domain_loss = running_domain_loss / (i + 1)
        # evaluate model
        test_accuracy = evaluate(model, test_dataloader, device)
        # test_loss = evaluate_loss(model, test_dataloader, class_criterion, device)
        tqdm.write('Epoch: {}, domain loss: {:.4f},  distribution loss:  {:.4f}, Train Accuracy: {:.4f},'
                   # ' test loss: {:.4f} ,'
                   ' Test Accuracy: {:.4f}'.format(epoch, running_domain_loss, running_f_loss,
                                            total_hit/total_num, test_accuracy))
        train_loss_list.append(running_f_loss)
        # test_loss_list.append(test_loss)
        domain_loss_list.append(running_domain_loss)
        train_accs.append(total_hit/total_num)
        test_accs.append(test_accuracy)

    # 绘制并保存折线图
    import matplotlib.pyplot as plt
    if not os.path.exists('./result'):
        os.mkdir('./result')
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.plot(domain_loss_list, label='domain Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./result/loss.png')
    plt.show()

    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./result/accuracy.png')
    plt.show()