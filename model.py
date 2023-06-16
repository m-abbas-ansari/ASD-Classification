import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from transformers import BertForSequenceClassification
import os
import json
import torch.utils.data as data
from data import CaptionDataset


class G_LSTM(nn.Module):
    """
    LSTM implementation proposed by A. Graves (2013),
    it has more parameters compared to original LSTM
    """

    def __init__(self, input_size=2048, hidden_size=512):
        super(G_LSTM, self).__init__()
        # without batch_norm
        self.input_x = nn.Linear(input_size, hidden_size, bias=True)
        self.forget_x = nn.Linear(input_size, hidden_size, bias=True)
        self.output_x = nn.Linear(input_size, hidden_size, bias=True)
        self.memory_x = nn.Linear(input_size, hidden_size, bias=True)

        self.input_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.forget_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.memory_h = nn.Linear(hidden_size, hidden_size, bias=True)

        self.input_c = nn.Linear(hidden_size, hidden_size, bias=True)
        self.forget_c = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_c = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, state):
        h, c = state
        i = torch.sigmoid(self.input_x(x) + self.input_h(h) + self.input_c(c))
        f = torch.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_c(c))
        g = torch.tanh(self.memory_x(x) + self.memory_h(h))

        next_c = torch.mul(f, c) + torch.mul(i, g)
        o = torch.sigmoid(self.output_x(x) + self.output_h(h) + self.output_c(next_c))
        h = torch.mul(o, next_c)
        state = (h, next_c)

        return state


class Sal_seq(nn.Module):
    def __init__(self, backend, seq_len, hidden_size=512, mask=False, joint=False, time_proj_dim=128):
        super(Sal_seq, self).__init__()
        self.seq_len = seq_len
        self.mask = mask
        self.joint = joint

        # defining backend
        if backend == 'resnet':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.init_resnet(resnet)
            input_size = 2048
        elif backend == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.init_vgg(vgg)
            input_size = 512
        else:
            assert 0, 'Backend not implemented'

        self.rnn = G_LSTM(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1, bias=True)  # comment for multi-modal distillation
        self.hidden_size = hidden_size

        # time projection
        if self.joint or self.mask:
            self.time_projection = nn.Linear(1, time_proj_dim)
        if self.mask:
            self.mask_projection = nn.Linear(time_proj_dim, input_size)
        if self.joint:
            self.time_embedding = nn.Linear(time_proj_dim, input_size, bias=False)

    def init_resnet(self, resnet):
        self.backend = nn.Sequential(*list(resnet.children())[:-2])

    def init_vgg(self, vgg):
        # self.backend = vgg.features
        self.backend = nn.Sequential(*list(vgg.features.children())[:-2])  # omitting the last Max Pooling

    def init_hidden(self, batch):  # initializing hidden state as all zero
        h = torch.zeros(batch, self.hidden_size)
        c = torch.zeros(batch, self.hidden_size)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return (Variable(h), Variable(c))

    def process_lengths(self, input):
        """
		Computing the lengths of sentences in current batchs
		"""
        max_length = input.size(1)
        lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
        return lengths

    def crop_seq(self, x, lengths):
        """
		Adaptively select the hidden state at the end of sentences
		"""
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        mask = Variable(mask)
        x = x.mul(mask)
        x = x.sum(1).view(batch_size, x.size(2))
        return x

    def forward(self, x, fixation, duration):
        valid_len = self.process_lengths(fixation)  # computing valid fixation lengths
        x = self.backend(x)
        batch, feat, h, w = x.size()
        x = x.view(batch, feat, -1)

        # recurrent loop
        state = self.init_hidden(batch)  # initialize hidden state
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation.to(torch.int64))
        output = []
        for i in range(self.seq_len):
            # extract features corresponding to current fixation
            cur_x = x[:, :, i].contiguous()
            if self.joint or self.mask:
                cur_t = duration[:, i].contiguous().unsqueeze(1)
                cur_t_proj = self.time_projection(cur_t)
                if self.joint:  # time-event joint embedding3
                    cur_t_enc = torch.softmax(cur_t_proj, dim=1)
                    cur_t_emb = self.time_embedding(cur_t_enc)
                    cur_x = (cur_x + cur_t_emb) / 2.0

                if self.mask:  # time mask
                    cur_t_proj = torch.relu(cur_t_proj)
                    cur_t_proj = self.mask_projection(cur_t_proj)
                    time_mask = torch.sigmoid(cur_t_proj)
                    cur_x = torch.mul(cur_x, time_mask)

            # LSTM forward
            state = self.rnn(cur_x, state)
            output.append(state[0].view(batch, 1, self.hidden_size))

        # selecting hidden states from the valid fixations without padding
        output = torch.cat(output, 1)
        output = self.crop_seq(output, valid_len)
        output = torch.sigmoid(self.decoder(output))
        return output


class SalBert(nn.Module):
    def __init__(self, backend, seq_len, hidden_size=768):
        super(SalBert, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # defining backend
        if backend == 'resnet':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.init_resnet(resnet)
            input_size = 2048
        elif backend == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.init_vgg(vgg)
            input_size = 512
        else:
            assert 0, 'Backend not implemented'

        self.input_encode = nn.Linear(input_size, hidden_size, bias=True)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
        self.bert = model.bert.encoder
        for param in self.bert.parameters(): param.requires_grad = False

        self.bert_pooler = model.bert.pooler
        self.classifier = model.classifier

    def init_resnet(self, resnet):
        self.backend = nn.Sequential(*list(resnet.children())[:-2])

    def init_vgg(self, vgg):
        self.backend = nn.Sequential(*list(vgg.features.children())[:-2])

    def forward(self, x, fixation, duration):
        x = self.backend(x)
        batch, feat, h, w = x.size()
        x = x.view(batch, feat, -1)
        x = torch.transpose(x, 1, 2)

        fixation = fixation.unsqueeze(dim=2)  # [12,14, 1]
        fixation = fixation.expand(fixation.size(0), fixation.size(1), feat)  # [12, 2048, 14]
        x = x.gather(1, fixation.to(torch.int64))

        x = self.input_encode(x)

        # bert
        out = self.bert(x)[0]
        out = self.bert_pooler(out)
        out = torch.sigmoid(self.classifier(out))
        return out


class CaptionModel(nn.Module):
    def __init__(self, seq_len, hidden_size=512):
        super(CaptionModel, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1,
                                                                     output_attentions=False,
                                                                     output_hidden_states=False).bert
        for param in self.encoder.parameters(): param.requires_grad = False

        self.rnn = G_LSTM(768, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1, bias=True)

    def init_hidden(self, batch):  # initializing hidden state as all zero
        h = torch.zeros(batch, self.hidden_size)
        c = torch.zeros(batch, self.hidden_size)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return Variable(h), Variable(c)

    def forward(self, fix_tokens):
        outs = []
        for i in range(self.seq_len):
            out = self.encoder(**{k: fix_tokens[k][:, i] for k in fix_tokens.keys()}).pooler_output
            outs.append(out)

        state = self.init_hidden(outs[0].size()[0])
        for i in range(self.seq_len):
            state = self.rnn(outs[i], state)

        output = torch.sigmoid(self.decoder(state[0]))

        return output


class VisualCaptionModel(nn.Module):
    def __init__(self, seq_len, hidden_size=512):
        super(VisualCaptionModel, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_size = 2048
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.project = nn.Linear(input_size, 768)
        self.joint_embedding = nn.Linear(768, 768, bias=False)
        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1,
                                                                     output_attentions=False,
                                                                     output_hidden_states=False).bert
        for param in self.encoder.parameters(): param.requires_grad = True

        self.rnn = G_LSTM(768, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1, bias=True)

    def init_hidden(self, batch):  # initializing hidden state as all zero
        h = torch.zeros(batch, self.hidden_size)
        c = torch.zeros(batch, self.hidden_size)
        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()
        return Variable(h), Variable(c)

    def forward(self, crops, fix_tokens):
        cap_outs = []
        for i in range(self.seq_len):
            out = self.encoder(**{k: fix_tokens[k][:, i] for k in fix_tokens.keys()}).pooler_output
            cap_outs.append(out)

        b, _, _, _ = crops[0].size()
        crop_outs = []
        for i in range(self.seq_len):
            out = self.cnn(crops[i])
            crop_outs.append(out.view(b, -1))

        state = self.init_hidden(b)
        for i in range(self.seq_len):
            cap_i = cap_outs[i]
            crop_i = crop_outs[i]

            # Create a joint embdding
            crop_i = torch.softmax(self.project(crop_i), dim=1)
            crop_i = self.joint_embedding(crop_i)
            emb_i = (cap_i + crop_i) / 2.0

            state = self.rnn(emb_i, state)

        output = torch.sigmoid(self.decoder(state[0]))

        return output


if __name__ == '__main__':
    print('Testing model')
    # model = Sal_seq(backend='resnet',seq_len=14, mask=True, joint=True)
    # model = SalBert(backend='resnet', seq_len=14)
    model = CaptionModel(seq_len=14, hidden_size=512)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model on current device: ", torch.cuda.current_device())
    else:
        print('CUDA is not available')

    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total parameters in model: {total_params:,}")

    # Sending dummy data
    # x = Variable(torch.randn(2, 3, 600, 800))
    # fixation = Variable(torch.randint(2, 5, (2, 14)).type(torch.LongTensor))
    # duration = Variable(torch.randint(2, 5, (2, 14))).type(torch.FloatTensor)
    # print('Input shape: ', x.shape)
    # print('Fixation shape: ', fixation.shape)
    # if torch.cuda.is_available():
    #     x = x.cuda()
    #     fixation = fixation.cuda()
    #     duration = duration.cuda()
    # output = model(x, fixation, duration)

    dat = {}
    with open('caption_annotations.json', 'r') as f:
        dat = json.load(f)
    dat = {int(k): dat[k] for k in dat.keys()}

    train_data = {k: dat[k] for k in range(1, 3)}
    train_set = CaptionDataset(train_data, 14)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=12, shuffle=True)
    fs, ls = next(iter(trainloader))
    for k, v in fs.items():
        fs[k] = v.cuda()
    ls = ls.cuda()

    output = model(fs)
    print('Output shape: ', output.shape)
    print('Done')
