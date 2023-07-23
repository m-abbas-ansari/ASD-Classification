import torch
from typing import Optional
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable
from torch import optim
# from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
# from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import \
#     SelfAttentionLayer, CrossAttentionLayer, FFNLayer
    
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
'''                              
class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 20,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 6,
        query_pos_emb: Optional[Tensor] = None
    ):
        super().__init__()
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                )
            )
            
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                )
            )
            
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        self.num_queries = num_queries
        self.fix_queries = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_emb = query_pos_emb
        
    def forward(self, x, padding_mask):
        _, bs, _ = x.size()
        order_emb = self.query_pos_emb.weight.unsqueeze(1).repeat(1, bs, 1)
        out = self.fix_queries.weight.unsqueeze(1).repeat(1, bs, 1)
        for i in range(self.num_layers):
            out = self.transformer_cross_attention_layers[i](
                out, x,
                memory_key_padding_mask = padding_mask,
                query_pos = order_emb
            )
            
            out = self.transformer_self_attention_layers[i](
                out,
                query_pos = order_emb
            )
            
            out = self.transformer_ffn_layers[i](out)
    
        return self.decoder_norm(out)

class HATFormer(nn.Module):
    def __init__(self, num_fix = 20, hidden_dim = 256, device = "cuda:0"):
        super().__init__()
        self.num_fix = num_fix
        self.hidden_dim = hidden_dim
        self.device = device
        self.backend = torch.load("backend.pt").to(device)
        for param in self.backend.parameters(): param.requires_grad = False
        self.pixel_decoder = torch.load("pixel_decoder.pt").to(device)
        for param in self.pixel_decoder.parameters(): param.requires_grad = False
        
        # self.pos_embs = PositionEmbeddingSine(hidden_dim // 2, normalize=True)(torch.rand((1, 3, 320, 512))).to(device).flatten(2)
        self.per_pos_embs = self.get_per_pos_embs(self.pos_embs, (320, 512), (10, 16))
        self.order_emb = nn.Embedding(num_fix, hidden_dim).to(device)
        self.scale_emb = nn.Embedding(2, hidden_dim).to(device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8).to(device)
        encoder_norm = nn.LayerNorm(256).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3, encoder_norm).to(device)

        self.decoder = Decoder(query_pos_emb=self.order_emb).to(device)
        
        self.token_predictor = nn.Linear(hidden_dim, 2).to(device) # [valid fix, padding]
        self.fix_loc_predictor = nn.Linear(hidden_dim, 2).to(device) # [y, x]
        self.softmax = nn.LogSoftmax(dim=-1).to(device)
    
    def get_fix_tokens(self, x, fixs, im_size):
        H, W = im_size
        _, feat, h, w = x.size()
        
        # We get fixation index in downscaled feature map for given fix coords
        fixation = (fixs[:, :, 0]*(h/H)).to(torch.int64)*w + (fixs[:, :, 1]*(w/W)).to(torch.int64)
        
        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)
        
        return x

    def get_per_pos_embs(self, pos_emb, im_size, feat_size):
        H, W = im_size
        h, w = feat_size
        center_coords = torch.LongTensor([int(H/(2*h)*(2*y + 1)) * W + int(W/(2*w)*(2*x + 1)) 
                                      for y in range(h) 
                                          for x in range(w)]).to(self.device)
            
        bs, dim, _ = pos_emb.size()
        coords = center_coords.expand(bs, dim, center_coords.size()[0])
        x = pos_emb.gather(2, coords)
        
        return x
    
    def get_fov_pos_embs(self, pos_emb, fixs, im_size):
        H, W = im_size
        
        # flatten fix coords on original im_size
        fixation = (fixs[:, :, 0]*W + fixs[:, :, 1]).to(torch.int64)
        
        bs, l = fixation.size()
        _, feat, num = pos_emb.size()
        pos_emb = pos_emb.expand(bs, feat, num)
        fixation = fixation.view(bs, 1, l)
        fixation = fixation.expand(bs, feat, l)
        x = pos_emb.gather(2, fixation)
        
        return x
    
    def forward(self, im, fixs, padding_mask):
        bs, _, H, W = im.size()
        #pos_emb = self.pe_layer(im).flatten(2)
        
        out = self.backend(im) 
        out = self.pixel_decoder.forward_features(out)
        p1 = out[1] # (N, 256, H/32, W/32) peripheral features
        p4 = out[0] # (N, 256, H/4, W/4)   foveal features
        
        _, _, h, w = p1.size()
        per_tokens = p1.flatten(2)
        fov_tokens = self.get_fix_tokens(p4, fixs, (H, W))
        
        #per_pos_embs = self.get_per_pos_embs(pos_emb, (H, W), (h, w))
        fov_pos_embs = self.get_fov_pos_embs(self.pos_embs, fixs, (H, W))
        
        per_scale_emb = self.scale_emb(torch.tensor(0).to(self.device)).unsqueeze(-1)
        fov_scale_emb = self.scale_emb(torch.tensor(1).to(self.device)).unsqueeze(-1)
        
        order_idx = torch.LongTensor([[i for i in range(self.num_fix)]]).to(self.device)
        fix_order_emb = self.order_emb(order_idx)
        
        per_tokens = (per_tokens + self.per_pos_embs + per_scale_emb).permute(2, 0, 1)
        fov_tokens = ((fov_tokens + fov_pos_embs + fov_scale_emb).transpose(1,2) + fix_order_emb).transpose(0,1)
        
        tokens = torch.cat([per_tokens, fov_tokens], axis=0)
        encoded_tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        
        out = self.decoder(encoded_tokens, padding_mask).transpose(0,1) # get batch first
        
        return self.softmax(self.token_predictor(out)), F.relu(self.fix_loc_predictor(out)) 
    
class HATLastFormer(nn.Module):
    def __init__(self, num_fix = 20, hidden_dim = 256, device = "cuda:0"):
        super().__init__()
        self.num_fix = num_fix
        self.hidden_dim = hidden_dim
        self.device = device
        self.backend = torch.load("backend.pt").to(device)
        for param in self.backend.parameters(): param.requires_grad = False
        self.pixel_decoder = torch.load("pixel_decoder.pt").to(device)
        for param in self.pixel_decoder.parameters(): param.requires_grad = False
        
        self.pos_embs = PositionEmbeddingSine(hidden_dim // 2, normalize=True)(torch.rand((1, 3, 320, 512))).to(device).flatten(2)
        self.per_pos_embs = self.get_per_pos_embs(self.pos_embs, (320, 512), (10, 16))
        self.order_emb = nn.Embedding(num_fix, hidden_dim).to(device)
        self.scale_emb = nn.Embedding(2, hidden_dim).to(device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8).to(device)
        encoder_norm = nn.LayerNorm(256).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3, encoder_norm).to(device)

        self.decoder = Decoder(query_pos_emb=self.order_emb).to(device)
        
        self.project = nn.Linear(20*hidden_dim, hidden_dim).to(device) # [valid fix, padding]
        self.fix_loc_predictor = nn.Linear(hidden_dim, 2).to(device) # [y, x]
        self.softmax = nn.LogSoftmax(dim=-1).to(device)
    
    def get_fix_tokens(self, x, fixs, im_size):
        H, W = im_size
        _, feat, h, w = x.size()
        
        # We get fixation index in downscaled feature map for given fix coords
        fixation = (fixs[:, :, 0]*(h/H)).to(torch.int64)*w + (fixs[:, :, 1]*(w/W)).to(torch.int64)
        
        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)
        
        return x

    def get_per_pos_embs(self, pos_emb, im_size, feat_size):
        H, W = im_size
        h, w = feat_size
        center_coords = torch.LongTensor([int(H/(2*h)*(2*y + 1)) * W + int(W/(2*w)*(2*x + 1)) 
                                      for y in range(h) 
                                          for x in range(w)]).to(self.device)
            
        bs, dim, _ = pos_emb.size()
        coords = center_coords.expand(bs, dim, center_coords.size()[0])
        x = pos_emb.gather(2, coords)
        
        return x
    
    def get_fov_pos_embs(self, pos_emb, fixs, im_size):
        H, W = im_size
        
        # flatten fix coords on original im_size
        fixation = (fixs[:, :, 0]*W + fixs[:, :, 1]).to(torch.int64)
        
        bs, l = fixation.size()
        _, feat, num = pos_emb.size()
        pos_emb = pos_emb.expand(bs, feat, num)
        fixation = fixation.view(bs, 1, l)
        fixation = fixation.expand(bs, feat, l)
        x = pos_emb.gather(2, fixation)
        
        return x
    
    def forward(self, im, fixs, padding_mask):
        bs, _, H, W = im.size()
        #pos_emb = self.pe_layer(im).flatten(2)
        
        out = self.backend(im) 
        out = self.pixel_decoder.forward_features(out)
        p1 = out[1] # (N, 256, H/32, W/32) peripheral features
        p4 = out[0] # (N, 256, H/4, W/4)   foveal features
        
        _, _, h, w = p1.size()
        per_tokens = p1.flatten(2)
        fov_tokens = self.get_fix_tokens(p4, fixs, (H, W))
        
        #per_pos_embs = self.get_per_pos_embs(pos_emb, (H, W), (h, w))
        fov_pos_embs = self.get_fov_pos_embs(self.pos_embs, fixs, (H, W))
        
        per_scale_emb = self.scale_emb(torch.tensor(0).to(self.device)).unsqueeze(-1)
        fov_scale_emb = self.scale_emb(torch.tensor(1).to(self.device)).unsqueeze(-1)
        
        order_idx = torch.LongTensor([[i for i in range(self.num_fix)]]).to(self.device)
        fix_order_emb = self.order_emb(order_idx)
        
        per_tokens = (per_tokens + self.per_pos_embs + per_scale_emb).permute(2, 0, 1)
        fov_tokens = ((fov_tokens + fov_pos_embs + fov_scale_emb).transpose(1,2) + fix_order_emb).transpose(0,1)
        
        tokens = torch.cat([per_tokens, fov_tokens], axis=0)
        encoded_tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        
        out = self.decoder(encoded_tokens, padding_mask).transpose(0,1).flatten(1) # get batch first
        out = self.project(out)
        return F.relu(self.fix_loc_predictor(out)) 
    
class HATFormerTensor(nn.Module):
    def __init__(self, num_fix = 20, hidden_dim = 256, dropout=0.4, mask_ratio = 0.2, device = "cuda:0"):
        super().__init__()
        self.num_fix = num_fix
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.device = device

        self.order_emb = nn.Embedding(num_fix, hidden_dim).to(device)
        self.order_idx = torch.LongTensor([[i for i in range(self.num_fix)]]).to(device)
        self.scale_emb = nn.Embedding(2, hidden_dim).to(device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8).to(device)
        encoder_norm = nn.LayerNorm(256).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, 3, encoder_norm).to(device)

        self.decoder = Decoder(query_pos_emb=self.order_emb).to(device)
        
        #classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, 2).to(device)
        #Gaussian parameters for x,y
        self.generator_y_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_y_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_logvar = nn.Linear(self.hidden_dim, 1).to(device)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1).to(device)
    
    #reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, fixs, padding_mask, per_tokens, fov_tokens):
        
        per_scale_emb = self.scale_emb(torch.tensor(0).to(self.device)).unsqueeze(-1)
        fov_scale_emb = self.scale_emb(torch.tensor(1).to(self.device)).unsqueeze(-1)
        
        
        fix_order_emb = self.order_emb(self.order_idx)
        
        per_tokens = (per_tokens + per_scale_emb).permute(2, 0, 1)
        fov_tokens = ((fov_tokens + fov_scale_emb).transpose(1,2) + fix_order_emb).transpose(0,1)
        
        tokens = torch.cat([per_tokens, fov_tokens], axis=0)
        encoded_tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        
        outs = self.decoder(encoded_tokens, padding_mask)
        outs = self.dropout(outs)
        
        y_mu, y_logvar, x_mu, x_logvar = self.generator_y_mu(outs),self.generator_y_logvar(outs), self.generator_x_mu(outs), self.generator_x_logvar(outs)

        
        return self.softmax(self.token_predictor(outs)), self.activation(self.reparameterize(y_mu, y_logvar)),self.activation(self.reparameterize(x_mu, x_logvar))
    
    
class HATClassifier(nn.Module):
    def __init__(self, num_fix = 20, hidden_dim = 256, pretrained=False, device = "cuda:0"):
        super().__init__()
        self.num_fix = num_fix
        self.hidden_dim = hidden_dim
        self.device = device
        self.backend = torch.load("backend.pt").to(device)
        for param in self.backend.parameters(): param.requires_grad = False
        self.pixel_decoder = torch.load("pixel_decoder.pt").to(device)
        for param in self.pixel_decoder.parameters(): param.requires_grad = False
        
        if pretrained:
            self.HAT = torch.load("ssl-masking-0.2.pt").to(device)
        else:
            self.HAT = HATFormerTensor()
        #for param in self.HAT.parameters(): param.requires_grad = False
        
        self.pos_embs = PositionEmbeddingSine(hidden_dim // 2, normalize=True)(
            torch.rand((1, 3, 320, 512))).to(device).flatten(2)
        
        self.per_pos_embs = self.get_per_pos_embs(self.pos_embs, (320, 512), (10, 16))
        self.order_emb = self.HAT.order_emb
        self.order_idx = torch.LongTensor([[i for i in range(self.num_fix)]]).to(device)

        self.scale_emb = self.HAT.scale_emb
        
        self.encoder = self.HAT.encoder
        self.decoder = self.HAT.decoder
        self.dropout = self.HAT.dropout
        
        self.project = nn.Linear(20*hidden_dim, hidden_dim).to(device) # [valid fix, padding]
        self.token_predictor = nn.Linear(self.hidden_dim, 2).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)


    def get_fix_tokens(self, x, fixs, im_size):
        H, W = im_size
        _, feat, h, w = x.size()
        
        # We get fixation index in downscaled feature map for given fix coords
        fixation = (fixs[:, :, 0]*(h/H)).long()*w + (fixs[:, :, 1]*(w/W)).long()
        
        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)
        
        return x

    def get_per_pos_embs(self, pos_emb, im_size, feat_size):
        H, W = im_size
        h, w = feat_size
        center_coords = torch.LongTensor([int(H/(2*h)*(2*y + 1)) * W + int(W/(2*w)*(2*x + 1)) 
                                        for y in range(h) 
                                            for x in range(w)]).to(self.device)
            
        bs, dim, _ = pos_emb.size()
        coords = center_coords.expand(bs, dim, center_coords.size()[0])
        x = pos_emb.gather(2, coords)
        
        return x

    def get_fov_pos_embs(self, pos_emb, fixs, im_size):
        H, W = im_size
        
        # flatten fix coords on original im_size
        fixation = (fixs[:, :, 0]*W + fixs[:, :, 1]).long()
        
        bs, l = fixation.size()
        _, feat, num = pos_emb.size()
        pos_emb = pos_emb.expand(bs, feat, num)
        fixation = fixation.view(bs, 1, l)
        fixation = fixation.expand(bs, feat, l)
        x = pos_emb.gather(2, fixation)
        
        return x

    def forward(self, im, fixs, padding_mask):
        bs, _, H, W = im.size()
        #pos_emb = self.pe_layer(im).flatten(2)
        
        out = self.backend(im) 
        out = self.pixel_decoder.forward_features(out)
        p1 = out[1] # (N, 256, H/32, W/32) peripheral features
        p4 = out[0] # (N, 256, H/4, W/4)   foveal features
        
        _, _, h, w = p1.size()
        per_tokens = p1.flatten(2)
        fov_tokens = self.get_fix_tokens(p4, fixs, (H, W))
        
        fov_pos_embs = self.get_fov_pos_embs(self.pos_embs, fixs, (H, W))
        
        per_scale_emb = self.scale_emb(torch.tensor(0).to(self.device)).unsqueeze(-1)
        fov_scale_emb = self.scale_emb(torch.tensor(1).to(self.device)).unsqueeze(-1)
        
        fix_order_emb = self.order_emb(self.order_idx)
        
        per_tokens = (per_tokens + self.per_pos_embs + per_scale_emb).permute(2, 0, 1)
        fov_tokens = ((fov_tokens + fov_pos_embs + fov_scale_emb).transpose(1,2) + fix_order_emb).transpose(0,1)
        
        tokens = torch.cat([per_tokens, fov_tokens], axis=0)
        encoded_tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        
        out = self.dropout(self.decoder(encoded_tokens, padding_mask))
        out = out.transpose(0,1).flatten(1) # get batch first
        out = F.relu(self.project(out))
        
        return self.softmax(self.token_predictor(out)) 
'''
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
    def __init__(self, 
                 backend="resnet50",
                 seq_len=20, 
                 im_size = (320, 512), 
                 hidden_size=512, 
                 pretrained=False,
                 weights_loc=None,
                 device="cuda"):
        super(Sal_seq, self).__init__()
        self.seq_len = seq_len
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.device = device

        if pretrained:
            network = torch.load(weights_loc).to(device)
            self.backend = network.backend
            self.rnn = network.rnn
        else:
            # defining backend
            if backend == "resnet50":
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.init_resnet(resnet)
                input_size=2048
            elif backend == "resnet18":
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.init_resnet(resnet)
                input_size=512
            self.rnn = G_LSTM(input_size, hidden_size).to(device)
            
        self.decoder = nn.Linear(hidden_size, 2, bias=True).to(device)  
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def init_resnet(self, resnet):
        self.backend = nn.Sequential(*list(resnet.children())[:-2]).to(self.device)

    def init_vgg(self, vgg):
        # self.backend = vgg.features
        self.backend = nn.Sequential(*list(vgg.features.children())[:-2])  # omitting the last Max Pooling

    def init_hidden(self, batch):  # initializing hidden state as all zero
        h = torch.zeros(batch, self.hidden_size).to(self.device)
        c = torch.zeros(batch, self.hidden_size).to(self.device)
        return (Variable(h), Variable(c))

    def process_lengths(self, pads):
        """
        Computing the lengths of sentences in current batchs
		"""
        max_length = pads.size(1)
        lengths = list(max_length - pads.data.sum(1).squeeze())
        return lengths
    
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

    def forward(self, img, fixation, padding_mask):
        valid_len = self.process_lengths(padding_mask[:,-self.seq_len:])  # computing valid fixation lengths
        x = self.backend(img)
        batch, feat, h, w = x.size()
        # recurrent loop
        state = self.init_hidden(batch)  # initialize hidden state
        x = self.get_fix_tokens(x, fixation)
        
        output = []
        for i in range(self.seq_len):
            # extract features corresponding to current fixation
            cur_x = x[:, :, i].contiguous()
            # LSTM forward
            state = self.rnn(cur_x, state)
            output.append(state[0].view(batch, 1, self.hidden_size))

        # selecting hidden states from the valid fixations without padding
        output = torch.cat(output, 1)
        output = self.crop_seq(output, valid_len)
        #output = self.softmax(self.decoder(output))
        return output

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class SSL(nn.Module):
    def __init__(self,
                 method="barlow-twins",
                 visual_backend="resnet50",
                 im_size=(320, 512),
                 seq_len=20,
                 batch_size=32,
                 hidden_dim=512,
                 device="cuda"):
        
        super().__init__()        
        self.method = method
        self.batch_size = batch_size
        self.backbone = Sal_seq(backend=visual_backend, im_size=im_size, seq_len=seq_len, hidden_size=hidden_dim, device=device)
        self.lambd = 0.0051
        self.num_features = hidden_dim * 4
        # projector
        sizes = [hidden_dim] + [hidden_dim*4, hidden_dim*4, hidden_dim*4]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers).to(device)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False).to(device)

    def forward(self, view1, view2):
        img1, fix1, pad1 = view1
        z1 = self.projector(self.backbone(img1, fix1, pad1))
        img2, fix2, pad2 = view2
        z2 = self.projector(self.backbone(img2, fix2, pad2))

        if self.method == "barlow-twins":
            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            c.div_(self.batch_size)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambd * off_diag
            
        elif self.method == "vic-reg":
            x, y = z1, z2
            repr_loss = F.mse_loss(x, y)

            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            cov_x = (x.T @ x) / (self.batch_size - 1)
            cov_y = (y.T @ y) / (self.batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                self.num_features
            ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

            loss = (
                25.0 * repr_loss
                + 25.0 * std_loss
                + 1.0 * cov_loss
            )
            
        return loss

class Sal_seq_pre(nn.Module):
    def __init__(self, 
                 backend='resnet', 
                 seq_len=20, 
                 im_size = (320, 512),
                 mask_ratio=0.2,
                 hidden_size=512,
                 device="cuda"):
        
        super(Sal_seq_pre, self).__init__()
        self.seq_len = seq_len
        self.im_size = im_size
        self.hidden_size = hidden_size
        self.device = device
        self.mask_ratio = mask_ratio
        
        # defining backend
        if backend == 'resnet':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
            self.init_resnet(resnet)
            input_size = 2048
        elif backend == 'vgg':
            vgg = models.vgg19(pretrained=True)
            self.init_vgg(vgg)
            input_size = 512
        else:
            assert 0, 'Backend not implemented'

        self.rnn = G_LSTM(input_size, hidden_size).to(device)
        
        #classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_size, 2).to(device)
        #Gaussian parameters for x,y
        self.generator_y_mu = nn.Linear(self.hidden_size, 1).to(device)
        self.generator_x_mu = nn.Linear(self.hidden_size, 1).to(device)
        self.generator_y_logvar = nn.Linear(self.hidden_size, 1).to(device)
        self.generator_x_logvar = nn.Linear(self.hidden_size, 1).to(device)
        
        self.activation = F.relu
        self.softmax = nn.LogSoftmax(dim=-1).to(device)

    def init_resnet(self, resnet):
        self.backend = nn.Sequential(*list(resnet.children())[:-2])

    def init_vgg(self, vgg):
        # self.backend = vgg.features
        self.backend = nn.Sequential(*list(vgg.features.children())[:-2])  # omitting the last Max Pooling

    def init_hidden(self, batch):  # initializing hidden state as all zero
        h = torch.zeros(batch, self.hidden_size).to(self.device)
        c = torch.zeros(batch, self.hidden_size).to(self.device)

        return (Variable(h), Variable(c))

    def process_lengths(self, pads):
        """
        Computing the lengths of sentences in current batchs
		"""
        max_length = pads.size(1)
        lengths = list(max_length - pads.data.sum(1).squeeze())
        return lengths
    
    def get_fix_tokens(self, x, fixs):
        H, W = self.im_size
        _, feat, h, w = x.size()

        # We get fixation index in downscaled feature map for given fix coords
        fixation = (fixs[:, :, 0]*(h/H)).long()*w + (fixs[:, :, 1]*(w/W)).long()

        x = x.flatten(2)
        fixation = fixation.view(fixation.size(0), 1, fixation.size(1))
        fixation = fixation.expand(fixation.size(0), feat, fixation.size(2))
        x = x.gather(2, fixation)

        return x

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
    
    def random_mask_tokens(self, fov_tokens, padding_mask):
        batch_size, dim, seq_length = fov_tokens.size()
        
        # Create the mask tensor
        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool).to(self.device)
        # Set 0s at the randomly selected positions
        for i in range(batch_size):
            
            num_fix = int(20 - padding_mask[i].sum()) # find number of fixations [excluding padding]
            # Calculate the number of tokens to mask per batch
            num_tokens_to_mask = int(num_fix * self.mask_ratio)
            if num_tokens_to_mask == 0: 
                num_tokens_to_mask = 1  # atleast mask one token
            # Generate random mask positions for each batch
            mask_positions = torch.randint(num_fix, (num_tokens_to_mask,))
            mask[i, mask_positions] = 1
        # Broadcast the mask to match the dimensions of fov_tokens
        mask = mask.unsqueeze(1).expand(-1, dim, -1)
        # Apply the mask to fov_tokens
        fov_tokens_masked = fov_tokens.masked_fill(mask, 0)
        
        return fov_tokens_masked
    
    #reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, img, fixation, padding_mask):
        valid_len = self.process_lengths(padding_mask[:,-20:])  # computing valid fixation lengths
        x = self.backend(img)
        batch, feat, h, w = x.size()
        # recurrent loop
        state = self.init_hidden(batch)  # initialize hidden state
        x = self.get_fix_tokens(x, fixation)
        x = self.random_mask_tokens(x, padding_mask) # apply masking
        
        output = []
        for i in range(self.seq_len):
            # extract features corresponding to current fixation
            cur_x = x[:, :, i].contiguous()
            # LSTM forward
            state = self.rnn(cur_x, state)
            output.append(state[0].view(batch, 1, self.hidden_size))

        # selecting hidden states from the valid fixations without padding
        outs = torch.cat(output, 1)
        y_mu, y_logvar = self.generator_y_mu(outs),self.generator_y_logvar(outs) 
        x_mu, x_logvar = self.generator_x_mu(outs), self.generator_x_logvar(outs)
        
        return self.softmax(self.token_predictor(outs)), \
               self.activation(self.reparameterize(y_mu, y_logvar)), \
               self.activation(self.reparameterize(x_mu, x_logvar))
               
