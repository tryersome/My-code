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
class Encoder_1(nn.Module):
    def __init__(self):
        super(Encoder_1,self).__init__()
        resnet1 = models.resnet50(pretrained=True)
        self.encoder0 = nn.Sequential(resnet1.conv1, resnet1.bn1, resnet1.relu)
        self.encoder1 = nn.Sequential(resnet1.maxpool, resnet1.layer1)
        self.encoder2 = resnet1.layer2
        self.encoder3 = resnet1.layer3
        self.encoder4 = resnet1.layer4#————————————————————————————————RGB encoder
        #[64, 240, 320][256,120,160][512,60,80][1024,30,40][2048,15,20]resnet50
        
    def forward(self,tensor):
        res0 = self.encoder0(tensor)  #[64, 240, 320]
        res1 = self.encoder1(res0)   #[256, 120, 160]
        res2 = self.encoder2(res1)   #[512, 60, 80]
        res3 = self.encoder3(res2)   #[1024, 30, 40] 
        return res0,res1,res2,res3

class Encoder_2(nn.Module):
    def __init__(self):
        super(Encoder_2,self).__init__()
        resnet3 = models.resnet50(pretrained=True)
        resnet3.conv1= nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,bias=False) #
        self.encoder0rt = nn.Sequential(resnet3.conv1, resnet3.bn1, resnet3.relu)
        self.encoder1rt = nn.Sequential(resnet3.maxpool, resnet3.layer1)
        self.encoder2rt = resnet3.layer2
        self.encoder3rt = resnet3.layer3
        self.encoder4rt = resnet3.layer4 #————————————————————————————————RGBT encoder
        #[64, 240, 320][256,120,160][512,60,80][1024,30,40][2048,15,20]resnet50
        
    def forward(self,tensor):
        res0 = self.encoder0(tensor)  #[64, 240, 320]
        res1 = self.encoder1(res0)   #[256, 120, 160]
        res2 = self.encoder2(res1)   #[512, 60, 80]
        res3 = self.encoder3(res2)   #[1024, 30, 40] 
        return res0,res1,res2,res3
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class ALL_Decoder(nn.Module):
    def __init__(self):
        super(ALL_Decoder,self).__init__()
        
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

    def forward(self, m1,m2,m3,final):  
      
        fusion = final  #1024 30 40
        
        high = final  #1024 30 40

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
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————
class TFNet(nn.Module):
    # resnet50   [64,240,320]   [256,120,160]  [512,60,80]   [1024,30,40]  
    def __init__(self):
        super(TFNet,self).__init__()

        self.Encoder_1= Encoder_1()
        self.Encoder_2= Encoder_2()

        self.Decoder1=ALL_Decoder()
        self.Decoder2=ALL_Decoder()

        self.fusion_conv1 = ConvBnrelu2d_3(6, 3) 
        self.fusion_conv2 = ConvBnrelu2d_3(70, 64)             

    def forward(self, input_rgb,input_t):

        input_rgbt= torch.cat((input_rgb,input_t),dim=1)
        input_rgbt=self.fusion_conv1(input_rgbt)                      

        p0,p1,p2,p3=self.Encoder_1(input_rgbt)    # [64,240,320]   [256,120,160]  [512,60,80]   [1024,30,40] 
        f0,f1,f2,f3,mid_loss=self.Decoder1(p0,p1,p2,p3)

        input_rgbt=torch.cat((f3,input_rgb,input_t),dim=1)
        input_rgbt= self.fusion_conv2(input_rgbt)

        rt0,rt1,rt2,rt3=self.Encoder_2(input_rgbt)    
                                                                  
        fusion0,fusion1,fusion2,fusion3,pre=self.Decoder2(rt0,rt1,rt2,rt3)   #第一段解码     
      
        return mid_loss,pre
#—————————————————————————————————————————————————————————————————————————————————————————————————————————————
if __name__=='__main__':
    
    input_rgb=torch.rand(2,3,480,640)
    input_t=torch.rand(2,3,480,640)
    
    text_net=TFNet()
    
    pre=text_net(input_rgb,input_t)

    print(pre.size())
    
