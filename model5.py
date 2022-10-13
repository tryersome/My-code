import warnings
warnings.filterwarnings("ignore")
import os
import sys
from builtins import print
import math
from collections import deque
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models 
from fuse_util import ChannelAttention, Three_fusion ,Two_fusion,SCPM
#新模型，只用前几次补语义
#—————————————————————杂项函数———————————————————————————————————
class ConvBnrelu2d_3(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        
class ConvBnrelu2d_1(nn.Module):
    # convolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(ConvBnrelu2d_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class TransConvBnLeakyRelu2d(nn.Module):
    # deconvolution
    # batch normalization
    # Lrelu
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(TransConvBnLeakyRelu2d, self).__init__()      
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)  
        for m in self.modules():            
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()        
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)   
                               
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))  
#————————————————————————————————————————第一阶段编码—————————————————————————————————————————————————————————————————————————————————
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        resnet1 = models.resnet50(pretrained=True)
        self.encoder0 = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)
        self.encoder1 = nn.Sequential(resnet1.maxpool, resnet1.layer1)
        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        #[64, 240, 320][256,120,160][512,60,80][1024,30,40][2048,15,20]resnet50
        
    def forward(self,tensor):
        res0 = self.encoder0(tensor)  #[64, 240, 320]
        res1 = self.encoder1(res0)   #[256, 120, 160]
        res2 = self.encoder2(res1)   #[512, 60, 80]
        res3 = self.encoder3(res2)   #[1024, 30, 40] 
        return res0,res1,res2,res3

class Encoder_E(nn.Module):
    def __init__(self):
        super(Encoder_E,self).__init__()
        resnet1 = models.resnet50(pretrained=True)
        #resnet1.conv1= nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,bias=False) #
        self.encoder0 = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)
        self.encoder1 = nn.Sequential(resnet1.maxpool, resnet1.layer1)
        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        #resnet50 [64, 240, 320]  [256,120,160]  [512,60,80]   [1024,30,40]  [2048,15,20]
        #decoder   [512 60 80]    [256 120 160]  [64 240 320]  [64 480 640]  [9 480 640]  

        self.conv0 = ConvBnrelu2d_1(64+64, 64)
        self.conv1 = ConvBnrelu2d_1(256+256+256, 256)
        self.conv2 = ConvBnrelu2d_1(512+512+512, 512)
        
    def forward(self,r0,r1,r2,r3,f0,f1,f2,f3,):
        res0=torch.cat((r0,f2),dim=1)    #64+64   
        res0=self.conv0(res0)    #128---64   

        res1 = self.encoder1(res0)   #[256, 120, 160]

        # res1=torch.cat((r1,f1,res1),dim=1)  #256+256+256
        # res1=self.conv1(res1)    #256+64+256----256
        
        res2 = self.encoder2(res1)   #[512, 60, 80]

        # res2=torch.cat((r2,f0,res2),dim=1)  #512+256+512
        # res2=self.conv2(res2)    #512+256+512----512

        res3 = self.encoder3(res2)   #[1024, 30, 40] 

        return res0,res1,res2,res3
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        
        self.fusion_decoder0 = TransConvBnLeakyRelu2d(1024,512)
        self.fusion_conv1 = ConvBnrelu2d_3(512, 512)
        self.fusion_decoder1 = TransConvBnLeakyRelu2d(512,256) #1/4        
        self.fusion_conv2 = ConvBnrelu2d_3(256, 256)
        self.fusion_decoder2 = TransConvBnLeakyRelu2d(256,64) #1/2       
        self.fusion_conv3 = ConvBnrelu2d_3(64, 64)
        self.fusion_decoder3 = TransConvBnLeakyRelu2d(64,64) #1
        self.fusion_conv4 = ConvBnrelu2d_3(64, 64)
        self.fusion_conv5 = nn.Conv2d(64, 9, kernel_size=1, padding=0, stride=1,bias=False)
        nn.init.xavier_uniform_(self.fusion_conv5.weight.data)
        
        self.high1 = ConvBnrelu2d_1(1024, 512)
        self.high2 = ConvBnrelu2d_1(1024, 256)
        self.high3 = ConvBnrelu2d_1(1024, 64)
        self.high4 = ConvBnrelu2d_1(1024, 64)

        self.fuse = ConvBnrelu2d_3(1024, 1024)

    def forward(self, m1,m2,m3,final):  
      
        fusion = self.fuse(final)  #1024 30 40
        
        high = fusion  #1024 30 40

        fusion = self.fusion_decoder0(fusion)  #512 60 80
        high1 = F.interpolate(self.high1(high), scale_factor=2, mode='bilinear',align_corners = True) #512 60 80
        fusion = fusion + m3 + high1  #512 60 80
        fusion = self.fusion_conv1(fusion) #混合

        fusion1=fusion #512 60 80

        fusion = self.fusion_decoder1(fusion)
        high2 = F.interpolate(self.high2(high), scale_factor=4, mode='bilinear',align_corners = True)
        fusion = fusion + m2 + high2 #256 120 160
        fusion = self.fusion_conv2(fusion)

        fusion2=fusion #256 120 160

        fusion = self.fusion_decoder2(fusion)
        high3 = F.interpolate(self.high3(high), scale_factor=8, mode='bilinear',align_corners = True)
        fusion = fusion + m1 + high3
        fusion = self.fusion_conv3(fusion)

        fusion3=fusion #64 240 320

        fusion = self.fusion_decoder3(fusion)
        fusion = self.fusion_conv4(fusion) #64 480 640

        fusion4=fusion

        fusion_loss = self.fusion_conv5(fusion) #9 480 640     


        return fusion1,fusion2,fusion3,fusion4,fusion_loss #512 60 80   #256 120 160  #64 240 320  #64 480 640  #9 480 640 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class TFNet(nn.Module):
    # resnet50   [64,240,320]   [256,120,160]  [512,60,80]   [1024,30,40]  
    def __init__(self):
        super(TFNet,self).__init__()

        self.Encoder_rgb= Encoder()
        self.Encoder_t= Encoder()

        self.Encoder_rgb_E= Encoder_E()
        self.Encoder_t_E= Encoder_E()

        self.Decoder_RGBT=Decoder()
        self.Decoder_Final=Decoder()

        #self.SCPM = SCPM(1024)

    def forward(self, input_rgb,input_t):           

        r0,r1,r2,r3=self.Encoder_rgb(input_rgb)
        t0,t1,t2,t3=self.Encoder_t(input_t)

        p0=r0+t0
        p1=r1+t1
        p2=r2+t2
        p3=r3+t3

        #p3=self.SCPM(rt3)                              

        f0,f1,f2,f3,f_loss=self.Decoder_RGBT(p0,p1,p2,p3)   #第一段解码     
        # 512 60 80    256 120 160   64 240 320   64 480 640   9 480 640  

        re0,re1,re2,re3=self.Encoder_rgb_E(r0,r1,r2,r3,   f0,f1,f2,f3)
        te0,te1,te2,te3=self.Encoder_t_E(t0,t1,t2,t3,   f0,f1,f2,f3)

        m0=re0+te0
        m1=re1+te1
        m2=re2+te2
        m3=re3+te3
    
        pre0,pre1,pre2,pre3,pre_loss=self.Decoder_Final(m0,m1,m2,m3)   #最终语义解码

        return pre_loss,f_loss
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__=='__main__':
    
    input_rgb=torch.rand(2,3,480,640)
    input_t=torch.rand(2,3,480,640)
    
    text_net=TFNet()
    
    pre_loss,fusion_loss=text_net(input_rgb,input_t)

    print(pre_loss.size(),fusion_loss.size())
    
