import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable

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
    def __init__(self, backend, seq_len, im_size = (320, 512), hidden_size=512, mask=False, joint=False, time_proj_dim=128):
        super(Sal_seq, self).__init__()
        self.seq_len = seq_len
        self.im_size = im_size
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

    def get_fix_tokens(self, x, fixs):
        H,W = self.im_size
        _, feat, h, w = x.size()

        # We get fixation index in downscaled feature map for given fix coords
        fixation = (fixs[:, :, 0]*(h-1)/H).long()*w + (fixs[:, :, 1]*(w-1)/W).long()

        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)

        return x
    
    def forward(self, x, fixation, duration, valid_len):
        # valid_len = self.process_lengths(fixation)  # computing valid fixation lengths
        x = self.backend(x)
        batch, feat, h, w = x.size()

        # recurrent loop
        state = self.init_hidden(batch)  # initialize hidden state
        x = self.get_fix_tokens(x, fixation)
        
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