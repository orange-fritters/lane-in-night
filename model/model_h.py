import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
 
# general libs
import math
import gc
import utils.helper as helper
 
class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, stride=1):
        super(ResBlock, self).__init__()
        if output_dim == None:
            output_dim = input_dim
        if input_dim == output_dim and stride==1:
            self.down_sample = None
        else:
            self.down_sample = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        residual = self.conv1(F.relu(x))
        residual = self.conv2(F.relu(residual))

        if self.down_sample is not None:
            x = self.down_sample(x)
         
        return x + residual 

class Encoder_Memory(nn.Module):
    def __init__(self):
        super(Encoder_Memory, self).__init__()
        self.conv1_mask = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_frame, in_mask):
        normalized_frame = (in_frame - self.mean) / self.std

        x = self.conv1(normalized_frame) + self.conv1_mask(in_mask.float())
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, normalized_frame
 

class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        normalized_frame = (in_f - self.mean) / self.std

        x = self.conv1(normalized_frame) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, normalized_frame


class Refine(nn.Module):
    def __init__(self, 
                inplanes : int,
                planes : int,
                scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor
        self.transposed_conv = nn.ConvTranspose2d(planes, planes, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        pm = self.transposed_conv(pm)
        m = s + pm
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim, scale_rate):
        super(Decoder, self).__init__()
        self.backbone = "resnet50"
        self.convFM = nn.Conv2d(1024//scale_rate, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine( 512 // scale_rate, mdim) # 1/8 -> 1/4
        self.RF2 = Refine( 256 // scale_rate, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

        self.transposed_conv = nn.ConvTranspose2d(2, mdim, kernel_size=4, stride=4)


    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        p = self.transposed_conv(p2)

        return p #, p2, p3, p4  


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o, c, t, h, w
        # m4, viz = self.Memory(keys[0], values[0], k4e, v4e)
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        scale_rate = 1

        self.Encoder_Memory = Encoder_Memory() 
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        self.KV_Q_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)

        self.Memory = Memory()
        self.Decoder = Decoder(256, scale_rate)

    def Pad_memory(self, mems, B):
        pad_mems = []
        for mem in mems: # [keys, values]
            pad_mem = helper.ToCuda(torch.zeros(B, 1, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            for b in range(B):
                pad_mem[b, :, :, 0] = mem[b]
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks): 
        # memorize a frame 
        B, _, H, W = masks.shape # B = 1

        (frame, masks), _ = helper.pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        r4, _, _, _, _ = self.Encoder_Memory(frame, masks)
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], B)
        return k4, v4

    def segment(self, frame, keys, values): 
        # 찍어보니까 B, 1, 128, 1, 14, 14 나옴
        B, _, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = helper.pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))
        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # B, dim, H/16, W/16
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(B,-1,-1,-1), v4.expand(B,-1,-1,-1) 
        r3e, r2e = r3.expand(B,-1,-1,-1), r2.expand(B,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        # k4e.size() = 4, 128, 14, 14 // v4e.size() = 4, 512, 14, 14
        m4, viz = self.Memory(keys[:, 0, ...], values[:, 0, ...], k4e, v4e)
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # no(1), h, w  
        
        #ps = indipendant possibility to belong to each object
        logit = self.soft_aggregation(ps) # 1, K, H, W

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        return logit    

    def soft_aggregation(self, ps):
        B, H, W = ps.shape
        em = helper.ToCuda(torch.zeros(B, 1, H, W))  # Single channel for object probabilities
        em[:, 0] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit
    
    def forward(self, mode, *args, **kwargs):
        if mode == "segment": # keys
            return self.segment(*args, **kwargs)
        elif mode == "memorize":
            return self.memorize(*args, **kwargs)
        else:
            raise NotImplementedError