import torch
import torch.nn as nn
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(self, opt):
        """
            该部分代码定义了Encoder类的初始化函数__init__()，用于对类进行初始化。该类继承了nn.Module类。

            其中，该函数接受一个参数opt，它是通过argparse解析命令行参数得到的。函数首先定义了如下变量：

            self.d_model: 模型的维度大小；
            self.model_type: 模型类型，可能是FC, Attention, Multi, ProbSparse中的一种；
            self.window_size: 窗口大小；
            self.truncate: 是否需要对序列进行截断的标志位，为True表示需要对序列进行截断；
            self.mask: 掩码，用于控制模型的注意力权重；
            self.all_size: 序列的总大小，即包括所有样本所有特征的总长度；
            self.decoder_type: 解码器类型，可能是FC或Attention；
            self.indexes: 从序列中提取参考点的下标；
            self.layers: 由多个EncoderLayer组成的nn.ModuleList；
            self.enc_embedding: 数据编码；
            self.conv_layers: 在多层编码器中使用的卷积层。
            根据参数opt的值的不同，将对以上变量进行不同的初始化操作。其中，如果opt.use_tvm为True，则采用TVM的方法对序列进行编码。
            如果opt.embed_type为CustomEmbedding，则采用自定义的嵌入方法，否则采用标准的嵌入方法。
            最后，创建CSCM卷积层的实例，其中CSCM是一个字符串，表示采用哪一种卷积层。
        """
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        if opt.decoder == 'attention':
            self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
        else:
            self.mask, self.all_size = get_mask(opt.input_size+1, opt.window_size, opt.inner_size, opt.device)
        self.decoder_type = opt.decoder
        if opt.decoder == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if opt.decoder == 'FC' else 0
            q_k_mask = get_q_k(opt.input_size + padding, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(opt.n_layer)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.n_layer)
                ])

        if opt.embed_type == 'CustomEmbedding':
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        # 对输入进行嵌入操作
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        # 获取掩码
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)

        # 卷积层
        seq_enc = self.conv_layers(seq_enc)

        # 遍历所有编码器层并传递给下一层
        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        # 如果解码器类型是FC，则采用下采样的方式，以获取全局上下文信息
        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        # 如果解码器类型是attention，则选择性截取编码器的输出
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        # 返回编码器的输出
        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()  # 继承父类初始化函数

        self.predict_step = opt.predict_step  # 预测步长
        self.d_model = opt.d_model  # 模型的维度大小
        self.input_size = opt.input_size  # 输入时间序列的长度
        self.decoder_type = opt.decoder  # 解码器类型
        self.channels = opt.enc_in  # 输入数据的通道数

        self.encoder = Encoder(opt)  # 初始化编码器

        # 如果解码器是基于注意力机制的，初始化解码器，同时构造掩码（mask）
        if opt.decoder == 'attention':
            mask = get_subsequent_mask(opt.input_size, opt.window_size, opt.predict_step, opt.truncate)
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.enc_in)

        # 如果解码器是全连接层，初始化预测器
        elif opt.decoder == 'FC':
            self.predictor = Predictor(4 * opt.d_model, opt.predict_step * opt.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain):
        """
        该函数返回隐藏表示和预测结果。对于序列 (l_1, l_2, ..., l_N)，我们要预测 (l_2, ..., l_N, l_{N+1})。
        """
        if self.decoder_type == 'attention':  # 如果解码器类型为 attention
            enc_output = self.encoder(x_enc, x_mark_enc)  # 对输入进行编码
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)  # 使用解码器生成预测

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)  # 拼接解码器输出与编码器前 input_size 个值
                pred = self.predictor(dec_enc)  # 预测拼接后的结果
            else:
                pred = self.predictor(dec_enc)  # 直接预测解码器输出

        elif self.decoder_type == 'FC':  # 如果解码器类型为 FC
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]  # 仅使用编码器最后一步的输出作为预测输入
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)  # 预测结果

        return pred


